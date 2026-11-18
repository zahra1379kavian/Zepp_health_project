"""Fold-local data views and exercise-conditioned, station-ignored scoring."""

from argparse import Namespace
import copy
import json
from pathlib import Path

import numpy as np

from ..constants import CLASSES, DT_MS, FEATURES
from ..data import (dense_phase_targets, load_arrays, load_sensor, phase_label_targets,
                    runs, subject_folds, training_statistics, write_json)
from ..metrics import aggregate_repetitions, count_rows, match_repetitions, repetition_metrics
from ..runner import digest, evaluate, train, tune


def pilot(args):
    root, out = Path(args.data).resolve(), Path(args.output).resolve()
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite {out}")
    manifest = json.loads((root / "manifest.json").read_text())
    if (manifest.get("version", 0) < 4 or manifest.get("reference_mode") != "labels"
            or not manifest.get("phase_order_validated")):
        raise ValueError("Analyses require version-4 wrist data with validated phase labels")
    if manifest["features"] != FEATURES or manifest.get("analysis"):
        raise ValueError("Start these analyses from the original nine-channel wrist dataset")
    if args.station_only or args.repetition_head not in ("auto", "phase"):
        raise ValueError("These analyses require dense phase1/phase2 targets")
    splits = subject_folds(manifest["sessions"], args.folds, args.seed)
    if not 1 <= args.fold <= len(splits):
        raise ValueError("fold must be between 1 and folds")
    return root, out, manifest, splits[args.fold - 1]


def waist_on_grid(source, row, t):
    files = sorted((Path(source) / row["source_session"]).glob("*waist*.csv"))
    x, available = np.zeros((len(t), 6), np.float32), np.zeros(len(t), bool)
    if not files:
        return x, available, {"selected": None, "candidates": []}
    candidates, selected = [], None
    for path in files:
        clocks, values = load_sensor(path, FEATURES[:6], allow_waist=True)
        indices = np.searchsorted(t, clocks)
        inside = indices < len(t)
        inside[inside] &= t[indices[inside]] == clocks[inside]
        if not inside.all():
            raise ValueError(f"{path}: waist clock does not exactly match the wrist label grid")
        candidates.append({"file": path.name, "aligned_samples": len(clocks)})
        # Clock availability only; never examine labels or model performance.
        # Strict greater-than preserves lexicographic filename tie-breaking.
        if selected is None or len(clocks) > len(selected[1]):
            selected = path, indices, values
    path, indices, values = selected
    x[indices], available[indices] = values, True
    return x, available, {"selected": path.name, "candidates": candidates}


def make_view(root, dest, manifest, split, analysis, sensor_set="all", waist=False,
              paired_control=False, held_out=None, source=None):
    """Share immutable files; write changed arrays only inside a new experiment.

    Held-out frames become gaps in train AND validation. Their original y/raw
    labels are preserved. Thus neither loss, smoothing, normalization, context,
    early stopping nor threshold tuning uses the held-out exercise.
    """
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=False)
    view = copy.deepcopy(manifest)
    view.update(split=split, analysis=analysis, parent_manifest_sha256=digest(root / "manifest.json"))
    view["features"] = FEATURES[:] if sensor_set == "all" else FEATURES[:6]
    if waist:
        view["features"] = [f"{site}_{f}" for site in ("wrist", "waist") for f in FEATURES[:6]]
    if waist or paired_control:
        view["waist_source"] = str(Path(source or manifest["source"]).resolve())
        view["waist_selection"] = "Most aligned samples per source session; filename breaks ties. Clock-only selection, no device stitching."
    view["wrist_selection"] = ("paired wrist + waist signals; wrist labels only" if waist else
                                "wrist signals; matched waist availability" if paired_control else
                                manifest["wrist_selection"])
    masked_names = set(split["train"] + split["validation"])
    audit = []
    cached_waist = None
    for row in view["sessions"]:
        name = row["session"]
        original = root / name
        arrays = load_arrays(original)
        valid = np.array(arrays["valid"], copy=True)
        updates = {}
        if sensor_set == "acc_gyro":
            updates["x"] = np.array(arrays["x"][:, :6])
        waist_available = None
        waist_info = None
        if waist or paired_control:
            if cached_waist is None or cached_waist[0] != row["source_session"]:
                cached_waist = (row["source_session"],
                                waist_on_grid(source or manifest["source"], row, arrays["t"]))
            waist_x, waist_available, waist_info = cached_waist[1]
            valid &= waist_available
            if waist:
                updates["x"] = np.concatenate((arrays["x"][:, :6], waist_x), axis=1)
        removed = np.zeros(len(valid), bool)
        if held_out and name in masked_names:
            removed = arrays["y"] == CLASSES.index(held_out)
            valid &= ~removed
        target = dest / name
        target.mkdir()
        refs = json.loads((original / "references.json").read_text())
        if not np.array_equal(valid, arrays["valid"]):
            raw = np.load(original / "raw.npy", allow_pickle=False)
            boundary, coverage, refs, rejected = phase_label_targets(arrays["t"], raw, valid)
            phase, phase_mask = dense_phase_targets(arrays["t"], raw, valid, coverage, refs)
            updates.update(valid=valid, boundary=boundary, coverage=coverage, phase=phase, phase_mask=phase_mask)
            write_json(target / "references.json", refs)
            write_json(target / "reference_audit.json", {"analysis": analysis, "label_column": row["label_column"],
                       "accepted_repetitions": len(refs), "rejected": rejected})
        for key, array in updates.items():
            np.save(target / f"{key}.npy", array, allow_pickle=False)
        for path in original.iterdir():
            if path.is_file() and not (target / path.name).exists():
                (target / path.name).symlink_to(path.resolve())
        row.update(valid_samples=int(valid.sum()), reference_repetitions=len(refs),
                   reference_samples=int(updates.get("coverage", arrays["coverage"]).sum()))
        audit.append({"session": name, "split": next(k for k, names in split.items() if name in names),
                      "held_out_masked_frames": int((removed & arrays["valid"]).sum()),
                      "waist_unavailable_frames": int((arrays["valid"] & ~waist_available).sum())
                      if waist_available is not None else 0,
                      "waist_sensor": waist_info,
                      "eligible_frames": int(valid.sum()), "reference_repetitions": len(refs)})
    view.update(training_statistics(dest, split["train"]))
    write_json(dest / "mask_audit.json", audit)
    write_json(dest / "manifest.json", view)
    return view


def fit_and_score(args, folder, report_scope="single_fold_analysis_pilot", **overrides):
    params = {**vars(args), **overrides, "repetition_head": "phase",
              "data": folder / "data", "output": folder / "model"}
    checkpoint = train(Namespace(**params))
    params.update(checkpoint=checkpoint, output=folder / "decoder.json")
    tune(Namespace(**params))
    params.update(decoder=folder / "decoder.json", split="test", output=folder / "test")
    report = evaluate(Namespace(**params))
    report["report_scope"] = report_scope
    write_json(folder / "test/report.json", report)
    return report


def exercise_predictions(predictions, references, t, coverage, tolerance):
    """Keep matches and every unmatched prediction overlapping scored coverage.

    Do not filter by predicted station: a wrong or unknown station still counts,
    and an overlong cycle crossing the exercise boundary cannot hide an extra.
    """
    matched = {p for p, _ in match_repetitions(predictions, references, tolerance, require_station=False)}
    intervals = [(int(t[a]), int(t[b-1] + DT_MS)) for a, b in runs(coverage)]
    return [r for i, r in enumerate(predictions) if i in matched or
            any(r["start_ms"] < b and r["end_ms"] > a for a, b in intervals)]


def score_exercise(data, predictions, exercise, tolerances, primary):
    """Use exercise labels only after full-session inference and decoding."""
    manifest = json.loads((Path(data) / "manifest.json").read_text())
    metadata = {row["session"]: row for row in manifest["sessions"]}
    records = []
    for name in manifest["split"]["test"]:
        a = load_arrays(Path(data) / name)
        coverage = a["coverage"] & a["valid"] & (a["y"] == CLASSES.index(exercise))
        refs = [r for r in json.loads((Path(data) / name / "references.json").read_text())
                if r["station"] == exercise]
        if not coverage.any() and not refs:
            continue
        reps = json.loads((Path(predictions) / f"{name}.json").read_text())["repetitions"]
        row = {k: metadata[name][k] for k in ("session", "subject", "side", "source_session")}
        row["scores"] = {}
        for tolerance in tolerances:
            scored = exercise_predictions(reps, refs, a["t"], coverage, tolerance)
            counts = count_rows(scored, refs, name, require_station=False)
            row["scores"][str(tolerance)] = {
                "metrics": repetition_metrics(scored, refs, tolerance, require_station=False), "counts": counts}
        records.append(row)

    def pool(rows):
        return {str(tol): aggregate_repetitions([r["scores"][str(tol)]["metrics"] for r in rows],
                                               [c for r in rows for c in r["scores"][str(tol)]["counts"]])
                for tol in tolerances}

    by_tolerance = pool(records)
    per_subject = {subject: pool([r for r in records if r["subject"] == subject])
                   for subject in sorted({r["subject"] for r in records})}
    for tol in tolerances:
        f1 = [s[str(tol)]["f1"] for s in per_subject.values() if s[str(tol)]["status"] == "scored"]
        by_tolerance[str(tol)]["subject_macro_f1"] = float(np.mean(f1)) if f1 else None
    return {"exercise": exercise, "require_station": False, "streams": len(records),
            "subjects": len(per_subject), "segmentation": by_tolerance[str(primary)],
            "by_tolerance": by_tolerance, "per_subject": per_subject,
            "per_side": {side: pool([r for r in records if r["side"] == side])
                         for side in ("left_wrist", "right_wrist")},
            "scoring_rule": "Decode complete test sessions without true exercise labels. Match endpoints ignoring station; count unmatched predictions that overlap certified held-out exercise frames. Other-exercise/rest-only predictions are outside this exercise-conditioned score.",
            "per_stream": records}
