"""Run the three independent step-2 screens, then confirm and benchmark the winner."""

import argparse
import json
from pathlib import Path

import numpy as np

from . import PAIRS, SCREEN
from ...__main__ import main as mstcn
from ...data import write_json
from ...orientation import INPUT_MODES


def summarize(reports, exercises, folds):
    """Pool held-out subjects per exercise, then give each exercise equal weight."""
    by_pair = {}
    for report in reports:
        for exercise, score in report["results"].items():
            pair = report["subject_fold"], exercise
            if pair in by_pair:
                raise ValueError(f"Duplicate holdout {pair}")
            by_pair[pair] = score
    if set(by_pair) != {(fold, exercise) for fold in folds for exercise in exercises}:
        raise ValueError("Incomplete exercise/subject fold comparison")
    f1 = {}
    for exercise in exercises:
        rows = [by_pair[fold, exercise]["by_tolerance"]["400"] for fold in folds]
        tp, extra, missed = (sum(r[k] for r in rows) for k in ("tp", "extra", "missed"))
        f1[exercise] = 2 * tp / (2 * tp + extra + missed)
    guards = [s["guard_rails"] for s in by_pair.values()]
    hours = sum(g["certified_rest_hours"] for g in guards)
    counts = sum(g["false_reps_during_rest"] for g in guards)

    def recall(field):
        total = sum(g[field]["total"] for g in guards)
        return sum(g[field]["matched"] for g in guards) / total if total else None

    if not hours or any(g[k] is None for g in guards for k in ("seen_f1_at_400", "crunch_f1_at_400")):
        raise ValueError("The study needs certified rest, seen exercise references and seen crunch references")
    return {"per_exercise_f1_at_400": f1, "mean_unseen_f1_at_400": float(np.mean(list(f1.values()))),
            "mean_seen_f1_at_400": float(np.mean([g["seen_f1_at_400"] for g in guards])),
            "mean_crunch_f1_at_400": float(np.mean([g["crunch_f1_at_400"] for g in guards])),
            "false_reps_per_rest_hour": counts / hours if hours else None,
            "first_rep_recall": recall("first_rep_recall"), "unseen_first_rep_recall": recall("unseen_first_rep_recall")}


def select(summaries):
    baseline = summaries["original"]
    passing = [mode for mode, s in summaries.items() if mode != "original"
               and s["mean_unseen_f1_at_400"] >= baseline["mean_unseen_f1_at_400"] + .05
               and s["mean_seen_f1_at_400"] >= baseline["mean_seen_f1_at_400"] - .02
               and s["false_reps_per_rest_hour"] <= 1.2 * baseline["false_reps_per_rest_hour"]]
    return max(passing, key=lambda m: summaries[m]["mean_unseen_f1_at_400"], default="original")


def fit_options(args, mode):
    return ["--data", str(args.data), "--repetition-head", "complete", "--rep-inputs", mode,
            "--rest-reviews", str(args.rest_reviews), "--threads", str(args.threads), "--device", args.device]


def holdout(args, mode, fold, exercises, output):
    mstcn(["leave-one-exercise-out", *fit_options(args, mode), "--protocol", "paired",
           "--subject-fold", str(fold), "--exercises", *exercises, "--output", str(output)])
    return json.loads((output / "report.json").read_text())


def finish(args):
    work = args.output
    for mode in INPUT_MODES:
        marker = work / f"screen_{mode}.status"
        if not marker.exists():
            raise ValueError(f"The {mode} screen has not finished")
        if marker.read_text().strip() != "complete":
            raise RuntimeError(f"The {mode} screen failed. See its training log.")
    reports = {mode: json.loads((work / f"screen_{mode}/report.json").read_text()) for mode in INPUT_MODES}
    screen = {mode: summarize([r], SCREEN, [1]) for mode, r in reports.items()}
    winner = select(screen)
    write_json(work / "screen_decision.json", {"variants": screen, "selected": winner})

    confirmed = {}
    for mode in dict.fromkeys(["original", winner]):
        remaining = [e for e in PAIRS if e not in SCREEN]
        fold1 = holdout(args, mode, 1, remaining, work / f"confirm_{mode}_fold_1_remaining")
        fold2 = holdout(args, mode, 2, list(PAIRS), work / f"confirm_{mode}_fold_2")
        confirmed[mode] = summarize([reports[mode], fold1, fold2], PAIRS, [1, 2])
    selected = select(confirmed)
    write_json(work / "confirmed_decision.json", {"variants": confirmed, "selected": selected})

    cv = work / "all_exercise_cv"
    mstcn(["cross-validate", *fit_options(args, selected), "--folds", "5", "--output", str(cv)])
    benchmark = work / "selected_benchmark"
    mstcn(["reference-benchmark", "--run-dir", str(cv), "--output", str(benchmark),
           "--source", str(json.loads((args.data / "manifest.json").read_text())["source"]),
           "--model-name", f"Complete rep ({selected})", "--threads", str(args.threads), "--device", args.device])
    from baseline.rep_benchmark.compare import compare
    compare([args.phase_benchmark, benchmark], work / "phase_comparison")


def main():
    runs = Path("baseline/MS-TCN/experiment_runs")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job", choices=[*INPUT_MODES, "finish"])
    parser.add_argument("--output", required=True, type=Path, help="Shared study directory")
    parser.add_argument("--data", type=Path, default=Path("baseline/MS-TCN/prepared_datasets/separate_wrist_streams"))
    parser.add_argument("--rest-reviews", type=Path, default=runs / "complete_rep_review/reviewed.json")
    parser.add_argument("--phase-benchmark", type=Path, default=runs / "feishu_wrist_rep_comparison/phase_sequence_model")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.job == "finish":
        finish(args)
        return
    args.output.mkdir(parents=True, exist_ok=True)
    folder = args.output / f"screen_{args.job}"
    marker = folder.with_suffix(".status")
    if folder.exists() or marker.exists():
        raise FileExistsError(folder)
    try:
        holdout(args, args.job, 1, SCREEN, folder)
    except BaseException:
        status = "failed"
        raise
    else:
        status = "complete"
    finally:
        pending = marker.with_suffix(".tmp")
        pending.write_text(status + "\n")
        pending.replace(marker)


if __name__ == "__main__":
    main()
