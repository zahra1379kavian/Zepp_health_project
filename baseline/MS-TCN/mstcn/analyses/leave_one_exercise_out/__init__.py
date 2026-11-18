"""Evaluate phase-only repetition counting on unseen exercise identities."""

import json

from ..common import fit_and_score, make_view, pilot, score_exercise
from ...constants import CLASSES
from ...data import write_json
from ...metrics import reporting_tolerances


def run(args):
    root, out, manifest, split = pilot(args)
    dynamic = [exercise for exercise in CLASSES[1:] if exercise != "plank"]
    exercises = dynamic if args.exercises == ["all"] else args.exercises
    if not exercises or len(set(exercises)) != len(exercises) or set(exercises) - set(dynamic):
        raise ValueError("Choose unique dynamic exercise names from the fixed vocabulary; plank is a hold")
    # Every probe must be supported in both the seen control and the held-out test.
    counts = {exercise: {part: 0 for part in split} for exercise in exercises}
    for part, names in split.items():
        for name in names:
            for rep in json.loads((root / name / "references.json").read_text()):
                if rep["station"] in counts:
                    counts[rep["station"]][part] += 1
    for exercise, parts in counts.items():
        if not parts["train"] or not parts["test"]:
            raise ValueError(f"{exercise}: the participant split needs training and test repetitions: {parts}")
    tolerances = reporting_tolerances(args.report_tolerances_ms, args.tolerance_ms)
    result = {"analysis": "leave_one_exercise_out", "report_scope": "exercise_wise_holdouts",
              "subject_fold": args.fold, "subject_folds": args.folds, "seed": args.seed, "split": split,
              "exercise_fold_count": len(exercises), "exercises": exercises, "reference_counts_before_masking": counts,
              "training": {k: str(v) if hasattr(v, "__fspath__") else v for k, v in vars(args).items()},
              "protocol": ("Every exercise fold shares one participant-disjoint split. Train a phase-only model "
                           "per removed exercise, masking its frames in training and validation. Refit training "
                           "statistics per model. Compare with one phase-only model trained with all exercises. "
                           "Select checkpoints and decoder thresholds on validation. Decode full test sessions "
                           "from sensor inputs and apply exercise labels only to select repetitions for scoring."),
              "results": {}}
    out.mkdir(parents=True)
    write_json(out / "plan.json", result)

    split_metadata = {"subject_fold": args.fold, "subject_folds": args.folds, "seed": args.seed}
    control = out / "seen_exercise_control"
    print("Training the seen-exercise control", flush=True)
    make_view(root, control / "data", manifest, split,
              {"kind": "seen_exercise_control", **split_metadata})
    fit_and_score(args, control, report_scope="seen_exercise_control_test", phase_only=True)

    for index, exercise in enumerate(exercises, 1):
        print(f"Exercise fold {index}/{len(exercises)}: {exercise}", flush=True)
        folder = out / exercise
        analysis = {"kind": "leave_one_exercise_out", "held_out_exercise": exercise,
                    "masked_splits": ["train", "validation"], "exercise_fold": index, **split_metadata}
        make_view(root, folder / "data", manifest, split, analysis, held_out=exercise)
        fit_and_score(args, folder, report_scope="exercise_fold_test", phase_only=True)
        score = score_exercise(folder / "data", folder / "test", exercise, tolerances, args.tolerance_ms)
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
