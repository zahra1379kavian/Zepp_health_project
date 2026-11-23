"""Exercise holdouts with participant-disjoint training, selection and testing."""

import json
import numpy as np

from ..common import fit_and_score, make_view, pilot, score_exercise
from ...constants import CLASSES, DT_MS
from ...data import load_arrays, write_json
from ...metrics import reporting_tolerances


PAIRS = {"biceps_curls": "bench_press", "lat_pulldown": "lunges", "push-up": "rowing",
         "overhead_press_(ohp)": "deadlift", "squat": "pull-up", "hip_thrust": "dips"}
SCREEN = ["biceps_curls", "push-up", "squat"]
LEGACY = ["leg_raise", "squat", "push-up", "biceps_curls", "crunch"]


def paired_guards(folder, full_report, seen, score, rest_hours):
    def scored(exercises):
        return score_exercise(folder / "data", folder / "test", exercises, [400], 400)["segmentation"]

    full = full_report["by_tolerance"]["400"]["segmentation"]
    extras = full["extra_breakdown"]["during_rest"]
    return {"seen_exercises": seen, "seen_f1_at_400": scored(seen).get("f1"),
            "crunch_f1_at_400": scored("crunch").get("f1"),
            "false_reps_during_rest": extras, "certified_rest_hours": rest_hours,
            "false_reps_per_rest_hour": extras / rest_hours if rest_hours else None,
            "first_rep_recall": full["detection_by_position"]["first"],
            "unseen_first_rep_recall": score["by_tolerance"]["400"]["detection_by_position"]["first"]}


def run(args):
    root, out, manifest, split = pilot(args)
    dynamic = [exercise for exercise in CLASSES[1:] if exercise != "plank"]
    head = "phase" if args.repetition_head == "auto" else args.repetition_head
    preset = args.exercises in (["screen"], ["confirm"])
    paired = (args.protocol == "paired" if args.protocol else
              head == "complete" or args.rep_inputs != "original" or preset)
    exercises = args.exercises or (list(PAIRS) if paired else LEGACY)
    if preset:
        exercises = SCREEN if exercises == ["screen"] else list(PAIRS)
    if exercises == ["all"]:
        exercises = dynamic
    if paired and (set(exercises) - PAIRS.keys() or args.tolerance_ms != 400):
        raise ValueError("Paired evaluation requires the predefined exercise pairs and 400 ms selection tolerance")
    if preset and not paired:
        raise ValueError("The screen and confirm presets require the paired protocol")
    if not exercises or len(set(exercises)) != len(exercises) or set(exercises) - set(dynamic):
        raise ValueError("Choose unique dynamic exercise names from the fixed vocabulary; plank is a hold")
    # Every probe must be supported in both the seen control and the held-out test.
    selectors = [PAIRS[e] for e in exercises] if paired else []
    counts = {exercise: {part: 0 for part in split} for exercise in exercises + selectors}
    rest_samples = 0
    for part, names in split.items():
        for name in names:
            if paired and part == "test":
                a = load_arrays(root / name, include_phase=False)
                rest_samples += np.count_nonzero(a["valid"] & a["coverage"] & (a["y"] == 0))
            for rep in json.loads((root / name / "references.json").read_text()):
                if rep["station"] in counts:
                    counts[rep["station"]][part] += 1
    for exercise in exercises:
        parts = counts[exercise]
        if not parts["train"] or not parts["test"]:
            raise ValueError(f"{exercise}: the participant split needs training and test repetitions: {parts}")
    if any(not counts[e]["validation"] for e in selectors):
        raise ValueError(f"Missing validation references for paired selection: {counts}")
    tolerances = reporting_tolerances(args.report_tolerances_ms, args.tolerance_ms)
    result = {"analysis": "leave_one_exercise_out", "report_scope": "exercise_wise_holdouts",
              "subject_fold": args.fold, "subject_folds": args.folds, "seed": args.seed, "split": split,
              "exercise_fold_count": len(exercises), "exercises": exercises, "reference_counts_before_masking": counts,
              "training": {k: str(v) if hasattr(v, "__fspath__") else v for k, v in vars(args).items()},
              "protocol": "paired" if paired else "legacy", "repetition_head": head, "rep_inputs": args.rep_inputs,
              "selection_exercises": dict(zip(exercises, selectors)),
              "results": {}}
    out.mkdir(parents=True)
    write_json(out / "plan.json", result)

    split_metadata = {"subject_fold": args.fold, "subject_folds": args.folds, "seed": args.seed}
    control = out / "seen_exercise_control"
    options = dict(repetition_head=head, phase_only=head == "phase")
    if not paired:
        print("Training the seen-exercise control", flush=True)
        make_view(root, control / "data", manifest, split,
                  {"kind": "seen_exercise_control", **split_metadata}, rep_inputs=args.rep_inputs)
        fit_and_score(args, control, report_scope="seen_exercise_control_test", **options)

    for index, exercise in enumerate(exercises, 1):
        print(f"Exercise fold {index}/{len(exercises)}: {exercise}", flush=True)
        folder = out / exercise
        analysis = {"kind": "leave_one_exercise_out", "held_out_exercise": exercise,
                    "masked_splits": ["train", "validation"], "exercise_fold": index, **split_metadata}
        selector = PAIRS[exercise] if paired else None
        analysis["selection_exercise"] = selector
        view = make_view(root, folder / "data", manifest, split, analysis, held_out=exercise,
                         selection_exercise=selector, rep_inputs=args.rep_inputs)
        full = fit_and_score(args, folder, report_scope="exercise_fold_test", **options)
        score = score_exercise(folder / "data", folder / "test", exercise, tolerances, args.tolerance_ms)
        if paired:
            seen = [e for e in dynamic if view["train_class_counts"][CLASSES.index(e)]]
            score.update(exercise_fold=index, selection_exercise=selector,
                         guard_rails=paired_guards(folder, full, seen, score, rest_samples * DT_MS / 3600000))
            write_json(folder / "held_out_report.json", score)
            result["results"][exercise] = score
            if len(result["results"]) == len(exercises):
                result["mean_unseen_f1_at_400"] = float(np.mean(
                    [s["by_tolerance"]["400"]["f1"] for s in result["results"].values()]))
            write_json(out / "report.json", result)
            continue
        seen = score_exercise(control / "data", control / "test", exercise, tolerances, args.tolerance_ms)
        write_json(folder / "seen_exercise_report.json", seen)
        comparison = {}
        for tol in map(str, tolerances):
            unseen_metrics, seen_metrics = score["by_tolerance"][tol], seen["by_tolerance"][tol]
            seen_mae = seen_metrics["count_mae_all_station_sessions"]
            unseen_mae = unseen_metrics["count_mae_all_station_sessions"]
            comparison[tol] = {
                "seen_f1": seen_metrics["f1"], "unseen_f1": unseen_metrics["f1"],
                "unseen_minus_seen_f1": unseen_metrics["f1"] - seen_metrics["f1"],
                "seen_count_mae": seen_mae, "unseen_count_mae": unseen_mae,
                "unseen_minus_seen_count_mae": unseen_mae - seen_mae,
            }
        score.update(exercise_fold=index, seen_exercise_comparison=comparison)
        write_json(folder / "held_out_report.json", score)
        result["results"][exercise] = score
        write_json(out / "report.json", result)
    print(f"Leave-one-exercise-out run complete: {out / 'report.json'}", flush=True)
    return result
