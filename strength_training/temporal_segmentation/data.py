"""Exact-clock inputs, exclusion masks, subject splits and validated references."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from .constants import CLASSES, DT_MS, EXCLUDED_SCENES, EXCLUDED_SESSIONS, FEATURES, station


WRIST_LOCATIONS = ("wrist", "left_wrist", "right_wrist")
PHASE_CONVENTION = (
    "User-validated full repetition: phase1 then phase2, within each continuous "
    "same-exercise bout. An initial phase2 is incomplete. Same-exercise transitions "
    "between phases are inside the repetition; transitions after the second phase "
    "are outside it. Repeated same-phase runs separated by transitions are a "
    "paused phase. Never pair across rest, another exercise, exclusions, or gaps. "
    "Discard unpaired and recording-edge-truncated cycles. Plank is a hold. "
    "Boundaries are first-phase start and exclusive second-phase end."
)


def select_wrists(folder, location="wrist"):
    """Return every requested wrist as a separate stream, without reading labels."""
    if location not in WRIST_LOCATIONS:
        raise ValueError("This method uses wrist signals only; waist input is not supported")
    choices = ("left_wrist", "right_wrist") if location == "wrist" else (location,)
    return [(side, Path(folder) / f"{side}.csv") for side in choices
            if (Path(folder) / f"{side}.csv").is_file()]


def wrist_reference(entry, side):
    """External references are side-specific; never copy labels across wrists."""
    if entry is None:
        return None
    if "wrists" in entry:
        if set(entry["wrists"]) - {"left_wrist", "right_wrist"}:
            raise ValueError("External reference wrists must be left_wrist or right_wrist")
        return entry["wrists"].get(side)
    if entry.get("location") not in ("left_wrist", "right_wrist"):
        raise ValueError("External wrist references need an explicit left_wrist or right_wrist location")
    return entry if entry["location"] == side else None


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def runs(mask):
    """Half-open runs of True; gaps must reset both the network and decoder."""
    edges = np.diff(np.r_[False, mask, False].astype(np.int8))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def load_sensor(path):
    if "waist" in Path(path).stem.lower():
        raise ValueError("This method uses wrist signals only; received a waist file")
    # Explicit usecols excludes every embedded label, including duplicate headers.
    frame = pd.read_csv(path, usecols=["sync_timestamp", *FEATURES])
    t = frame.sync_timestamp.to_numpy(dtype=np.int64)
    x = frame[FEATURES].to_numpy(dtype=np.float32)
    if not len(t) or np.any(np.diff(t) <= 0) or not np.isfinite(x).all():
        raise ValueError(f"{path}: empty, nonfinite or unordered sensor input")
    if np.any((t - t[0]) % DT_MS):
        raise ValueError(f"{path}: timestamps are not on a 25 Hz grid")
    return t, x


def log_epoch(date, clock):
    day = datetime.strptime(str(date), "%Y/%m/%d").replace(tzinfo=timezone.utc)
    h, m, s = map(float, str(clock).split(":"))
    if not (0 <= h < 24 and 0 <= m < 60 and 0 <= s < 60):
        raise ValueError("invalid log time")
    return int(day.timestamp() * 1000 + (h * 3600 + m * 60 + s) * 1000)


def log_masks(t, rows, alignment):
    """Fail closed when a relevant interval cannot be placed on the sensor clock."""
    valid, probe = np.ones(len(t), bool), np.zeros(len(t), bool)
    intervals = []
    if rows.empty:
        raise ValueError("missing workout log; exclusion coverage cannot be checked")
    affected = rows[rows.scene_id.str.lower().str.strip().isin(EXCLUDED_SCENES)]
    if affected.empty:
        return valid, probe, intervals
    if alignment is None or not np.isfinite(alignment.get("offset_hours", np.nan)):
        raise ValueError("unresolved log-to-sensor offset")
    if (alignment.get("named_station_match_fraction", 0) <= .5 or
            alignment.get("named_station_match_fraction", 0) -
            alignment.get("runner_up_fraction", 1) <= .1):
        raise ValueError("insufficient log alignment evidence")
    offset = float(alignment["offset_hours"]) * 3600000
    for row in affected.itertuples():
        try:
            a = int(log_epoch(row.collect_date, row.device_start_time) - offset)
            b = int(log_epoch(row.collect_date, row.device_end_time) - offset)
        except (ValueError, TypeError, OverflowError) as error:
            raise ValueError("invalid affected log interval; review needed") from error
        if b <= a:
            raise ValueError("non-positive affected log interval; review needed")
        scene = row.scene_id.strip().lower()
        mask = (t >= a) & (t < b)
        valid[mask] = False
        if scene == "spull":
            probe[mask] = True
        intervals.append({"scene": scene, "start_ms": a, "end_ms": b,
                          "overlap_samples": int(mask.sum())})
    # Failed recordings and calibration take priority over the unseen probe.
    for interval in intervals:
        if interval["scene"] != "spull":
            probe[(t >= interval["start_ms"]) & (t < interval["end_ms"])] = False
    return valid, probe, intervals


def reference_targets(t, valid, entry, location):
    """Coverage certifies exhaustive dynamic-repetition annotations, including zeros.

    Intervals and repetitions use [start_ms, end_ms). End impulses are placed at
    the exclusive end clock tick, so end decisions require no future samples.
    """
    boundary = np.zeros((len(t), 2), np.float32)
    covered = np.zeros(len(t), bool)
    if entry is None:
        return boundary, covered, [], []
    if (entry.get("validated") is not True or entry.get("location") != location or
            not entry.get("source") or not entry.get("convention")):
        raise ValueError("References need validated=true, matching location, source and convention")
    intervals = sorted(entry.get("coverage", []), key=lambda r: r["start_ms"])
    for i, row in enumerate(intervals):
        a, b = int(row["start_ms"]), int(row["end_ms"])
        if (a < t[0] or b > t[-1] + DT_MS or b <= a or
                (a - t[0]) % DT_MS or (b - t[0]) % DT_MS or
                (i and a < intervals[i-1]["end_ms"])):
            raise ValueError("Reference coverage must be ordered, disjoint and on the label grid")
        covered |= (t >= a) & (t < b)
    repetitions, removed = [], []
    ordered = sorted(entry.get("repetitions", []), key=lambda r: r["start_ms"])
    previous_end = -1
    for row in ordered:
        a, b, name = int(row["start_ms"]), int(row["end_ms"]), row["station"]
        if (name not in CLASSES[1:] or name == "plank" or b <= a or a < previous_end or
                (a - t[0]) % DT_MS or (b - t[0]) % DT_MS):
            raise ValueError(f"Invalid or overlapping full repetition: {row}")
        previous_end = b
        # End tick must itself be in certified coverage for boundary supervision.
        if not any(c["start_ms"] <= a < b < c["end_ms"] for c in intervals):
            raise ValueError("Each repetition, including its end tick, must be inside coverage")
        i, j = np.searchsorted(t, [a, b])
        if not valid[i:j+1].all():
            # Do not turn pieces of a cut repetition into boundary negatives.
            covered[i:j+1] = False
            removed.append({**row, "reason": "overlaps exclusion or missing sensor"})
            continue
        boundary[i, 0] = 1
        boundary[j, 1] = 1
        repetitions.append({"station": name, "start_ms": a, "end_ms": b})
    # A removed cycle may share its start/end with a retained neighbour.
    # Keep the neighbour's known event supervised after all coverage cuts.
    covered |= boundary.any(axis=1) & valid
    return boundary, covered & valid, repetitions, removed


def phase_label_targets(t, raw, valid):
    """Convert wrist labels with the user-validated phase1 -> phase2 order."""
    boundary = np.zeros((len(t), 2), np.float32)
    coverage = np.array(valid, dtype=bool, copy=True)
    repetitions, rejected = [], []
    starts = np.r_[0, np.flatnonzero(raw[1:] != raw[:-1]) + 1]
    ends = np.r_[starts[1:], len(raw)]
    blocks, bout_station = [], None

    def finish():
        k = 0
        while k < len(blocks):
            first = blocks[k]
            second = (blocks[k+1] if first[0] == "phase1" and k+1 < len(blocks)
                      and blocks[k+1][0] == "phase2" else None)
            k += 2 if second else 1
            a, b = first[1], second[2] if second else first[2]
            reason = None
            if second is None:
                reason = "unpaired phase"
            elif a == 0 or b == len(t):
                reason = "recording-edge-truncated cycle"
            elif not valid[a:b+1].all():
                reason = "overlaps exclusion or missing wrist sample"
            if reason:
                coverage[a:b] = False
                rejected.append({"station": bout_station, "start_ms": int(t[a]),
                                 "end_ms": int(t[b-1] + DT_MS), "reason": reason})
            else:
                boundary[a, 0], boundary[b, 1] = 1, 1
                repetitions.append({"station": bout_station, "start_ms": int(t[a]),
                                    "end_ms": int(t[b]), "phase_order": f"{first[0]}-{second[0]}"})

    for a, b in zip(starts, ends):
        label = str(raw[a])
        name = station(label)
        if name != bout_station:
            finish()
            blocks, bout_station = [], name
        if label.endswith(("_phase1", "_phase2")):
            phase = label.rsplit("_", 1)[1]
            if blocks and blocks[-1][0] == phase:
                # Same phase resumes after a same-exercise transition/pause.
                blocks[-1][2] = int(b)
            else:
                blocks.append([phase, int(a), int(b)])
    finish()
    # An accepted cycle's exclusive end may coincide with an incomplete next
    # cycle's start. Keep that known end impulse without supervising the orphan.
    coverage |= boundary.any(axis=1) & valid
    return boundary, coverage & valid, repetitions, rejected


def dense_phase_targets(t, raw, valid, coverage, repetitions):
    """NULL / phase1 / phase2 on certified cycles and reference negatives.

    Within a retained cycle, same-exercise transition pauses continue the prior
    phase. Outside cycles transitions are NULL. Unpaired/cut phases are masked.
    External references without corresponding phase labels remain boundary-only.
    """
    phase = np.zeros(len(t), np.int64)
    mask = np.array(coverage & valid, copy=True)
    phase[np.char.endswith(raw.astype(str), "_phase1")] = 1
    phase[np.char.endswith(raw.astype(str), "_phase2")] = 2
    for rep in repetitions:
        a, b = np.searchsorted(t, [rep["start_ms"], rep["end_ms"]])
        last = 0
        for i in range(a, b):
            if phase[i]:
                last = phase[i]
            elif str(raw[i]) == rep["station"] + "_transition":
                phase[i] = last
    # A restored shared boundary may also be an orphan phase's first frame.
    # Boundary supervision remains valid, but it does not certify that phase.
    accepted = np.zeros(len(t), bool)
    for rep in repetitions:
        a, b = np.searchsorted(t, [rep["start_ms"], rep["end_ms"]])
        accepted[a:b] = True
    mask &= (phase == 0) | accepted
    return phase, mask


def subject_folds(sessions, folds=5, seed=42):
    """Outer test folds with a separate rotating validation fold."""
    subjects = sorted({s["subject"] for s in sessions})
    if not 3 <= folds <= len(subjects):
        raise ValueError("Use 3..number-of-subjects folds for disjoint train/validation/test")
    np.random.default_rng(seed).shuffle(subjects)
    groups = [set(g) for g in np.array_split(subjects, folds)]
    result = []
    for i in range(folds):
        test, validation = groups[i], groups[(i + 1) % folds]
        train = set(subjects) - test - validation
        result.append({key: [s["session"] for s in sessions if s["subject"] in ids]
                       for key, ids in (("train", train), ("validation", validation), ("test", test))})
    return result


def subject_split(sessions, seed=42):
    subjects = sorted({s["subject"] for s in sessions})
    if len(subjects) < 3:
        raise ValueError("At least three eligible subjects are needed for train/validation/test")
    np.random.default_rng(seed).shuffle(subjects)
    n = max(1, int(round(len(subjects) * .15)))
    groups = {"test": subjects[:n], "validation": subjects[n:2*n], "train": subjects[2*n:]}
    return {key: [s["session"] for s in sessions if s["subject"] in ids]
            for key, ids in groups.items()}


def prepare(source, output, alignment_path, location="wrist", references_path=None, seed=42,
            reference_mode="labels"):
    if location not in WRIST_LOCATIONS:
        raise ValueError("This method uses wrist signals only")
    if reference_mode not in ("labels", "none"):
        raise ValueError("reference_mode must be labels or none")
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("Source and output must be separate, non-nested directories")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    align = pd.read_csv(alignment_path).set_index("session").to_dict("index")
    log_files = list(source.glob("*Sheet1.csv"))
    if len(log_files) != 1:
        raise ValueError("Expected exactly one workout-log *Sheet1.csv")
    log = pd.read_csv(log_files[0])
    log["session"] = pd.to_datetime(log.collect_date, format="%Y/%m/%d").dt.strftime("%Y%m%d") + "_" + log.masked_id
    refs = json.loads(Path(references_path).read_text()) if references_path else {"version": 1, "sessions": {}}
    if refs.get("version") != 1:
        raise ValueError("Unsupported references schema")
    manifest = {"version": 4, "source": str(source), "location": location,
                "classes": CLASSES, "features": FEATURES, "sample_period_ms": DT_MS,
                "alignment_source": str(Path(alignment_path).resolve()),
                "references_source": str(Path(references_path).resolve()) if references_path else
                ("label.csv, corresponding selected wrist column" if reference_mode == "labels" else None),
                "reference_mode": "external" if references_path else reference_mode,
                "reference_convention": "See external reference entries" if references_path else
                (PHASE_CONVENTION if reference_mode == "labels" else None),
                "label_reliability": "User confirmed label.csv files are reliable references",
                "phase_order_validated": reference_mode == "labels" and not references_path,
                "phase_validation_source": "User confirmed phase1 then phase2 in code review",
                "seed": seed, "sessions": [], "excluded": [],
                "wrist_selection": "every available wrist as a separate stream; split by subject" if location == "wrist" else location,
                "stream_id_format": "<source_session>__<wrist_side>"}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".tas-prepare-", dir=output.parent) as temp:
        dest = Path(temp) / "dataset"
        dest.mkdir()
        for folder in sorted(p for p in source.iterdir() if p.is_dir() and not p.name.startswith(".")):
            name = folder.name
            if name in EXCLUDED_SESSIONS:
                manifest["excluded"].append({"session": name, "reason": "plan whole-session exclusion"})
                continue
            wrists = select_wrists(folder, location)
            if not wrists:
                manifest["excluded"].append({"session": name, "reason": f"no {location} sensor"})
                continue
            labels = pd.read_csv(folder / "label.csv")
            t = labels.sync_timestamp.to_numpy(np.int64)
            if not len(t) or np.any(np.diff(t) != DT_MS):
                raise ValueError(f"{folder}: label clock must be a complete 25 Hz grid")
            try:
                eligible, probe, exclusions = log_masks(t, log[log.session == name], align.get(name))
            except ValueError as error:
                manifest["excluded"].append({"session": name, "reason": str(error)})
                continue
            for selected_location, sensor_path in wrists:
                if selected_location not in labels:
                    raise ValueError(f"{folder}: missing authoritative {selected_location} labels")
                sensor_t, sensor_x = load_sensor(sensor_path)
                idx = np.searchsorted(t, sensor_t)
                if np.any(idx >= len(t)) or not np.array_equal(t[idx], sensor_t):
                    raise ValueError(f"{sensor_path}: sensor clock does not match label grid exactly")
                x, available = np.zeros((len(t), len(FEATURES)), np.float32), np.zeros(len(t), bool)
                x[idx], available[idx] = sensor_x, True
                raw = labels[selected_location].to_numpy(dtype=str)
                mapping = {value: CLASSES.index(station(str(value))) for value in np.unique(raw)}
                y = np.array([mapping[value] for value in raw], dtype=np.int64)
                valid = eligible & available
                if references_path or reference_mode == "none":
                    entry = wrist_reference(refs["sessions"].get(name), selected_location)
                    boundary, coverage, reps, cut = reference_targets(t, valid, entry, selected_location)
                else:
                    boundary, coverage, reps, cut = phase_label_targets(t, raw, valid)
                phase, phase_mask = dense_phase_targets(t, raw, valid, coverage, reps)
                stream = f"{name}__{selected_location}"
                path = dest / stream
                path.mkdir()
                for key, array in dict(x=x, t=t, y=y, raw=raw, valid=valid, available=available,
                                       eligible=eligible, probe=probe & available,
                                       boundary=boundary, coverage=coverage,
                                       phase=phase, phase_mask=phase_mask).items():
                    np.save(path / f"{key}.npy", array, allow_pickle=False)
                write_json(path / "references.json", reps)
                write_json(path / "reference_audit.json", {"source": manifest["references_source"],
                           "convention": manifest["reference_convention"], "label_column": selected_location,
                           "accepted_repetitions": len(reps), "rejected": cut})
                record = {"session": stream, "source_session": name, "subject": name.split("_", 1)[1],
                          "side": selected_location, "sensor": sensor_path.name, "label_column": selected_location,
                          "samples": len(t), "available_samples": int(available.sum()),
                          "valid_samples": int(valid.sum()), "masked_log_samples": int((~eligible).sum()),
                          "probe_samples": int((probe & available).sum()), "exclusions": exclusions,
                          "reference_samples": int(coverage.sum()), "reference_repetitions": len(reps),
                          "removed_references": cut, "raw_label_counts": dict(Counter(map(str, raw)))}
                manifest["sessions"].append(record)
                print(f"Prepared {stream}: {valid.sum():,} eligible samples; {len(reps)} reference reps", flush=True)
        unknown = set(refs["sessions"]) - {p.name for p in source.iterdir() if p.is_dir()}
        if unknown:
            raise ValueError(f"References name unknown sessions: {sorted(unknown)}")
        manifest["split"] = subject_split(manifest["sessions"], seed)
        manifest["subjects"] = len({s["subject"] for s in manifest["sessions"]})
        manifest["source_sessions"] = len({s["source_session"] for s in manifest["sessions"]})
        manifest["streams_per_side"] = dict(Counter(s["side"] for s in manifest["sessions"]))
        manifest.update(training_statistics(dest, manifest["split"]["train"]))
        write_json(dest / "manifest.json", manifest)
        dest.rename(output)
    return manifest


def load_arrays(path):
    arrays = {key: np.load(Path(path) / f"{key}.npy", mmap_mode="r", allow_pickle=False)
              for key in ("x", "t", "y", "valid", "boundary", "coverage", "probe")}
    for key in ("phase", "phase_mask"):
        if (Path(path) / f"{key}.npy").exists():
            arrays[key] = np.load(Path(path) / f"{key}.npy", mmap_mode="r", allow_pickle=False)
    return arrays


def training_statistics(root, sessions):
    """Fit normalization and all loss weights on this fold's training subjects."""
    sums, squares, count = np.zeros(len(FEATURES)), np.zeros(len(FEATURES)), 0
    class_counts, phase_counts = np.zeros(len(CLASSES), np.int64), np.zeros(3, np.int64)
    boundary_positives, boundary_frames = np.zeros(2), 0
    for name in sessions:
        arrays = load_arrays(Path(root) / name)
        selected = arrays["x"][arrays["valid"]].astype(np.float64)
        sums += selected.sum(0)
        squares += (selected ** 2).sum(0)
        count += len(selected)
        class_counts += np.bincount(arrays["y"][arrays["valid"]], minlength=len(CLASSES))
        mask = arrays["coverage"]
        boundary_positives += arrays["boundary"][mask].sum(0)
        boundary_frames += int(mask.sum())
        if "phase" in arrays:
            phase_counts += np.bincount(arrays["phase"][arrays["phase_mask"]], minlength=3)
    if count == 0:
        raise ValueError("No eligible training samples")
    mean = sums / count
    std = np.sqrt(np.maximum(squares / count - mean ** 2, 1e-8))
    return {"normalization": {"mean": mean.tolist(), "std": std.tolist(), "fit_samples": count},
            "train_class_counts": class_counts.tolist(), "train_phase_counts": phase_counts.tolist(),
            "train_boundary_positives": boundary_positives.tolist(), "train_boundary_frames": boundary_frames}
