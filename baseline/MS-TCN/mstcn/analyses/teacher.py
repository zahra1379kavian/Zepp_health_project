"""Compare the offline teacher with its matched step-2 streaming student."""

import json
from pathlib import Path
from statistics import mean

import torch

from .leave_one_exercise_out import PAIRS
from .leave_one_exercise_out.study import unseen_f1_by_exercise


def ceiling_scores(cv, holdouts, acausal):
    report = json.loads((cv / "report.json").read_text())
    if (report["report_scope"] != "subject_wise_cross_validation" or report["timing_tolerance_ms"] != 400
            or report.get("acausal", False) != acausal):
        raise ValueError("Expected all-exercise subject cross-validation selected at 400 ms")
    folds = {f["fold"]: f["split"] for f in report["folds"]}
    if set(folds) != set(range(1, 6)):
        raise ValueError("The teacher ceiling requires all five subject folds")
    manifests = {}
    recipes = []
    for fold, split in folds.items():
        # Read the saved recipe, including for student fits predating --acausal.
        ckpt = torch.load(cv / f"fold_{fold}/model/best.pt", weights_only=True, map_location="cpu")
        config = ckpt["config"]
        if (config.get("acausal", False) != acausal or config.get("lookahead_samples", 0)
                or config["repetition_head"] != "complete" or ckpt["split"] != split):
            raise ValueError(f"Wrong checkpoint or split in {cv}, fold {fold}")
        recipes.append(dict(rep_inputs=ckpt["normalization"].get("rep_inputs", "original"),
                            causal_similarity=config.get("causal_similarity", False),
                            synthetic_reps=ckpt.get("synthetic_reps", False), seed=ckpt["seed"],
                            stages=config["stages"], layers=config["layers"], channels=config["channels"],
                            max_steps=ckpt["partial_epoch_max_steps"]))
        manifests[fold] = (ckpt["data_manifest_sha256"], ckpt.get("rest_reviews_sha256"))
    recipe = recipes[0]
    if any(r != recipe for r in recipes):
        raise ValueError("All five folds must use the same step-2 settings")

    by_pair, paired_settings = {}, {}
    for path in holdouts:
        r = json.loads(path.read_text())
        fold = r["subject_fold"]
        settings = {"acausal": False, "causal_similarity": False, "synthetic_reps": False, **r["training"]}
        if (r["protocol"] != "paired" or r["repetition_head"] != "complete" or settings["acausal"] != acausal
                or settings.get("lookahead_ms", 0) or r["split"] != folds.get(fold)
                or settings["tolerance_ms"] != 400 or set(r["exercises"]) != set(r["results"])
                or any(settings[k] != v for k, v in {**recipe, **report.get("fit_settings", {})}.items())):
            raise ValueError(f"Holdout settings do not match the all-exercise fit: {path}")
        for exercise, score in r["results"].items():
            pair = fold, exercise
            if pair in by_pair or exercise not in PAIRS or r["selection_exercises"][exercise] != PAIRS[exercise]:
                raise ValueError(f"Duplicate or incorrect exercise pair: {pair}")
            by_pair[pair] = score
            metrics = score["by_tolerance"]["400"]
            paired_settings[pair] = (metrics["tp"] + metrics["missed"], Path(settings["data"]).resolve(),
                                     Path(settings["rest_reviews"]).resolve() if settings["rest_reviews"] else None,
                                     {k: settings[k] for k in ("batch_size", "window", "epochs",
                                                               "learning_rate", "patience", "augment")})
    holdout_folds = sorted({fold for fold, _ in by_pair})
    if len(holdout_folds) < 2:
        raise ValueError("Use the six confirmed exercise pairs on at least two subject folds")
    unseen = unseen_f1_by_exercise(by_pair, PAIRS, holdout_folds)
    seg = report["by_tolerance"]["400"]["segmentation"]
    positions = seg["detection_by_position"]
    scores = {"first_rep_recall": positions["first"]["recall"], "last_rep_recall": positions["last"]["recall"],
              "f1_at_400": seg["f1"], "mean_unseen_f1_at_400": mean(unseen.values())}
    signature = (folds, manifests, recipe, paired_settings, seg["tp"] + seg["missed"],
                 positions["first"]["total"], positions["last"]["total"])
    return scores, unseen, holdout_folds, signature


def compare_teacher(args):
    student, student_exercises, folds, student_signature = ceiling_scores(args.student_cv, args.student_holdouts, False)
    teacher, teacher_exercises, _, teacher_signature = ceiling_scores(args.teacher_cv, args.teacher_holdouts, True)
    if student_signature != teacher_signature:
        raise ValueError("Compare the same inputs, training settings, subject splits and exercise pairs")
    gains = {key: round(teacher[key] - student[key], 12) for key in student}
    decision = {"tolerance_ms": 400, "holdout_subject_folds": folds, "student": student, "teacher": teacher,
                "teacher_minus_student": gains,
                "unseen_f1_by_exercise": {"student": student_exercises, "teacher": teacher_exercises},
                "proceed_to_step_3_3": gains["first_rep_recall"] >= .15 or gains["mean_unseen_f1_at_400"] >= .05}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        print(json.dumps(decision, indent=2, allow_nan=False), file=handle)
    return decision
