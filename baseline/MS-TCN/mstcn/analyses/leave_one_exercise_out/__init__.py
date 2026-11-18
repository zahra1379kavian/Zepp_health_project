"""Exercise-disjoint fitting with subject-disjoint held-out evaluation."""

import json

from ..common import fit_and_score, make_view, pilot, score_exercise
from ...constants import CLASSES
from ...data import write_json
from ...metrics import reporting_tolerances


def run(args):
    root, out, manifest, split = pilot(args)
    if (not args.exercises or len(set(args.exercises)) != len(args.exercises)
            or set(args.exercises) - (set(CLASSES[1:]) - {"plank"})):
        raise ValueError("Choose unique dynamic exercise names from the fixed vocabulary; plank is a hold")
    # Fail before training if this fold cannot test one of the requested probes.
    counts = {exercise: {part: 0 for part in split} for exercise in args.exercises}
    for part, names in split.items():
        for name in names:
            for rep in json.loads((root / name / "references.json").read_text()):
                if rep["station"] in counts:
                    counts[rep["station"]][part] += 1
    for exercise, parts in counts.items():
        if not parts["train"] or not parts["test"]:
            raise ValueError(f"{exercise}: this fold needs training and held-out-subject test repetitions: {parts}")
    tolerances = reporting_tolerances(args.report_tolerances_ms, args.tolerance_ms)
    result = {"analysis": "leave_one_exercise_out", "report_scope": "single_fold_analysis_pilot",
              "fold": args.fold, "folds": args.folds, "seed": args.seed, "split": split,
              "exercises": args.exercises, "reference_counts_before_masking": counts,
              "training": {k: str(v) if hasattr(v, "__fspath__") else v for k, v in vars(args).items()},
              "protocol": "One independently trained model per removed exercise. Mask its frames in train and validation; retain original labels. Refit statistics on remaining training frames. Select checkpoints and decoder on known exercises in validation subjects. Decode full test sessions, then score only held-out exercise repetitions ignoring predicted station.",
              "results": {}}
    out.mkdir(parents=True)
    write_json(out / "plan.json", result)
    for exercise in args.exercises:
        folder = out / exercise
        analysis = {"kind": "leave_one_exercise_out", "held_out_exercise": exercise,
                    "masked_splits": ["train", "validation"], "fold": args.fold,
                    "folds": args.folds, "seed": args.seed}
        make_view(root, folder / "data", manifest, split, analysis, held_out=exercise)
        fit_and_score(args, folder)
        score = score_exercise(folder / "data", folder / "test", exercise, tolerances, args.tolerance_ms)
        write_json(folder / "held_out_report.json", score)
        result["results"][exercise] = score
        write_json(out / "report.json", result)
    print(f"Leave-one-exercise-out pilot complete: {out / 'report.json'}", flush=True)
    return result
