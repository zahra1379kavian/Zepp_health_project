"""Extend the phase-only exercise holdout study to all exercises and participant folds.

Run from the project root with python -m baseline.mstcn.analyses.leave_one_exercise_out.all_folds.
Each fold has a separate worker and log. Restarting the command reuses completed fits.
"""
from argparse import ArgumentParser, Namespace
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stderr, redirect_stdout
import csv
import fcntl
import json
import multiprocessing
from pathlib import Path
from tempfile import mkdtemp

import torch

from ..common import fit_and_score, make_view, score_exercise
from ...constants import CLASSES
from ...data import subject_folds, write_json
from ...runner import decoder_config, digest, load_checkpoint

EXERCISES = [name for name in CLASSES[1:] if name != "plank"]
CONTROL = "seen_exercise_control"
RUNS = Path("baseline/MS-TCN/experiment_runs")
TOLERANCES = [200, 400]


def read(path):
    return json.loads(path.read_text())


def publish(path, value):
    temporary = path.with_suffix(".tmp")
    write_json(temporary, value)
    temporary.replace(path)


def verify_pilot_fit(folder, exercise, split, parent_hash, settings):
    data = folder / "data"
    view = read(data / "manifest.json")
    model_path = folder / "model/best.pt"
    model, checkpoint = load_checkpoint(model_path, data=data)
    decoder_config(folder / "decoder.json", model_path, settings["tolerance_ms"])
    decoder = read(folder / "decoder.json")
    analysis = view["analysis"]
    expected_kind = "leave_one_exercise_out" if exercise else CONTROL
    if not (view["parent_manifest_sha256"] == parent_hash and view["split"] == checkpoint["split"] == split
            and analysis["kind"] == expected_kind and analysis.get("held_out_exercise") == exercise
            and (not exercise or set(analysis["masked_splits"]) == {"train", "validation"})
            and checkpoint["analysis"] == analysis and checkpoint["seed"] == settings["seed"]
            and analysis["subject_fold"] == 1 and model.config.classes == 0 and model.config.repetition_head == "phase"
            and checkpoint["partial_epoch_max_steps"] == settings["max_steps"]
            and decoder["selected_on"] == "validation" and decoder["sessions"] == split["validation"]
            and decoder["tolerance_ms"] == settings["tolerance_ms"]
            and all((folder / "test" / f"{stream}.json").is_file() for stream in split["test"])):
        raise ValueError(f"Incompatible pilot fit at {folder}")


def get_fit(folder, exercise, fold, split, data, manifest, settings, pilot_source):
    if pilot_source:
        if not folder.exists():
            folder.symlink_to(pilot_source, target_is_directory=True)
        if folder.resolve() != pilot_source:
            raise ValueError(f"Unexpected source for {folder}")
        return
    if (folder / "test/report.json").is_file():
        return
    # Only publish a new fit once all validation and test outputs have been written.
    work = Path(mkdtemp(prefix=f".{folder.name}-", dir=folder.parent))
    analysis = {"kind": "leave_one_exercise_out" if exercise else CONTROL,
                "subject_fold": fold, "subject_folds": 5, "seed": settings["seed"]}
    if exercise:
        analysis.update(held_out_exercise=exercise, masked_splits=["train", "validation"],
                        exercise_fold=EXERCISES.index(exercise) + 1)
    print(f"Training fold {fold}, {folder.name}", flush=True)
    make_view(data, work / "data", manifest, split, analysis, held_out=exercise)
    scope = "exercise_fold_test" if exercise else "seen_exercise_control_test"
    fit_and_score(Namespace(**settings), work, report_scope=scope, repetition_head="phase", phase_only=True)
    work.rename(folder)


def exercise_counts(folder, exercise):
    result = score_exercise(folder / "data", folder / "test", exercise, TOLERANCES, 400)
    counts = {}
    for tol, row in result["by_tolerance"].items():
        counts[tol] = {key: row[key] for key in ("tp", "extra", "missed")}
        counts[tol]["absolute_error_sum"] = sum(c["absolute_error"] for c in row["count_rows"])
        counts[tol]["streams"] = len(row["count_rows"])
    return counts


def run_fold(fold, out, data, pilot, plan):
    fold_dir = out / f"fold_{fold}"
    fold_dir.mkdir(exist_ok=True)
    split = plan["splits"][fold - 1]
    settings = dict(plan["training"], fold=fold)
    torch.set_num_threads(settings["threads"])
    manifest = read(data / "manifest.json")
    old = plan["pilot_exercises"] if fold == 1 else []
    report_path = fold_dir / "report.json"
    report = read(report_path) if report_path.exists() else {"subject_fold": fold, "split": split, "results": {}}
    with (fold_dir / "training.log").open("a", buffering=1) as log, redirect_stdout(log), redirect_stderr(log):
        control = fold_dir / CONTROL
        get_fit(control, None, fold, split, data, manifest, settings, pilot / CONTROL if fold == 1 else None)
        for exercise in dict.fromkeys([*old, *EXERCISES]):
            if exercise in report["results"]:
                continue
            folder = fold_dir / exercise
            get_fit(folder, exercise, fold, split, data, manifest, settings, pilot / exercise if exercise in old else None)
            row = {name: exercise_counts(path, exercise) for name, path in (("seen", control), ("held_out", folder))}
            for tol in map(str, TOLERANCES):
                ref = plan["test_reference_counts"][fold - 1][exercise]
                if any(cond[tol]["tp"] + cond[tol]["missed"] != ref for cond in row.values()):
                    raise ValueError(f"Reference mismatch at {folder} for {tol} ms")
            report["results"][exercise] = row
            publish(report_path, report)
            print(f"Scored fold {fold}, {exercise} ({len(report['results'])}/{len(EXERCISES)})", flush=True)


def pool_reports(reports):
    rows = []
    for exercise in EXERCISES:
        for tol in TOLERANCES:
            row = {"exercise": exercise, "tolerance_ms": tol}
            for condition in ("seen", "held_out"):
                parts = [report["results"][exercise][condition][str(tol)] for report in reports]
                pooled = {k: sum(p[k] for p in parts) for k in parts[0]}
                tp, fp, fn = (pooled[k] for k in ("tp", "extra", "missed"))
                pooled.update(f1=2 * tp / (2 * tp + fp + fn),
                              count_mae=pooled["absolute_error_sum"] / pooled["streams"])
                row.update({f"{condition}_{k}": v for k, v in pooled.items()})
            rows.append(row)
    return rows


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", type=Path, default=RUNS / "leave_one_exercise_out_phase_only")
    parser.add_argument("--output", type=Path, default=RUNS / "leave_one_exercise_out_phase_only_all_folds")
    parser.add_argument("--workers", type=int, choices=range(1, 6), default=5)
    args = parser.parse_args()
    out, pilot = args.output.resolve(), args.pilot.resolve()
    seed_report = read(pilot / "report.json")
    settings = seed_report["training"]
    if not (settings["folds"] == 5 and settings["fold"] == 1 and settings["tolerance_ms"] == 400
            and settings["report_tolerances_ms"] == TOLERANCES and settings["station_only"] is False
            and settings["repetition_head"] in ("auto", "phase") and settings["max_steps"] == 0):
        raise ValueError("Expected the fold-1 phase-only pilot with 200 and 400 ms scores")
    data = Path(settings["data"]).resolve()
    manifest = read(data / "manifest.json")
    parent_hash = digest(data / "manifest.json")
    splits = subject_folds(manifest["sessions"], 5, settings["seed"])
    if splits[0] != seed_report["split"]:
        raise ValueError("Pilot participant assignment differs from fold 1")
    old = list(seed_report["results"])
    for name in [CONTROL, *old]:
        verify_pilot_fit(pilot / name, None if name == CONTROL else name, splits[0], parent_hash, settings)
    from collections import Counter
    refs = {s["session"]: Counter(r["station"] for r in read(data / s["session"] / "references.json"))
            for s in manifest["sessions"]}
    tests = [{e: sum(refs[n][e] for n in split["test"]) for e in EXERCISES} for split in splits]
    if any(min(counts.values()) == 0 for counts in tests):
        raise ValueError("An exercise has no labeled test reps in a fold")
    settings = {k: v for k, v in settings.items() if k not in ("data", "output", "exercises", "fold")}
    plan = {"analysis": "phase_only_leave_one_exercise_out", "exercises": EXERCISES, "splits": splits,
            "data": str(data), "manifest_sha256": parent_hash, "training": settings,
            "pilot": str(pilot), "pilot_exercises": old, "test_reference_counts": tests,
            "pilot_checkpoint_sha256": {n: digest(pilot / n / "model/best.pt") for n in [CONTROL, *old]}}
    out.mkdir(parents=True, exist_ok=True)
    with (out / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (out / "plan.json").exists() and read(out / "plan.json") != plan:
            raise ValueError(f"{out} already contains a different experiment")
        publish(out / "plan.json", plan)
        print(f"Starting 5 participant folds × {len(EXERCISES)} exercise holdouts. Logs in {out}/fold_*/training.log", flush=True)
        with ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            futures = {pool.submit(run_fold, f, out, data, pilot, plan): f for f in range(1, 6)}
            for future in as_completed(futures):
                future.result()
                print(f"Completed participant fold {futures[future]}/5", flush=True)
        reports = [read(out / f"fold_{f}/report.json") for f in range(1, 6)]
        rows = pool_reports(reports)
        from .all_folds_figures import plot
        plot(rows, out / "figures")
        with (out / "pooled.tmp").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        (out / "pooled.tmp").replace(out / "pooled.csv")
        publish(out / "report.json", {"participant_folds": 5, "exercise_holdouts": 70, "per_exercise": rows})
        print(f"Finished all 70 exercise holdouts. Results in {out}", flush=True)


if __name__ == "__main__":
    main()
