"""One-factor ablations, combined model, and matched wrist/waist comparison."""

from ..common import fit_and_score, make_view, pilot, score_exercise
from ...constants import CLASSES
from ...data import write_json
from ...metrics import reporting_tolerances


VARIANTS = {
    "baseline": dict(sensor_set="all", lookahead_ms=0, augment=False),
    "lookahead": dict(sensor_set="all", lookahead_ms=500, augment=False),
    "acc_gyro": dict(sensor_set="acc_gyro", lookahead_ms=0, augment=False),
    "augmentation": dict(sensor_set="all", lookahead_ms=0, augment=True),
    "combined": dict(sensor_set="acc_gyro", lookahead_ms=500, augment=True),
    "combined_paired_wrist": dict(sensor_set="acc_gyro", lookahead_ms=500, augment=True, paired_control=True),
    "combined_waist": dict(sensor_set="acc_gyro", lookahead_ms=500, augment=True, waist=True),
}


def run(args):
    root, out, manifest, split = pilot(args)
    if len(set(args.variants)) != len(args.variants):
        raise ValueError("variants must not contain duplicates")
    tolerances = reporting_tolerances(args.report_tolerances_ms, args.tolerance_ms)
    plan = {"analysis": "sensor_ablation", "fold": args.fold, "folds": args.folds,
            "seed": args.seed, "split": split, "parent_data": str(root),
            "variants": {name: VARIANTS[name] for name in args.variants},
            "training": {k: str(v) if hasattr(v, "__fspath__") else v for k, v in vars(args).items()},
            "report_scope": "single_fold_analysis_pilot", "results": {}}
    out.mkdir(parents=True)
    write_json(out / "plan.json", plan)
    for name in args.variants:
        settings = VARIANTS[name]
        folder = out / name
        analysis = {"kind": "sensor_ablation", "variant": name, "fold": args.fold,
                    "folds": args.folds, "seed": args.seed, **settings}
        make_view(root, folder / "data", manifest, split, analysis,
                  sensor_set=settings["sensor_set"], waist=settings.get("waist", False),
                  paired_control=settings.get("paired_control", False), source=args.source)
        report = fit_and_score(args, folder, lookahead_ms=settings["lookahead_ms"], augment=settings["augment"])
        exercises = {exercise: score_exercise(folder / "data", folder / "test", exercise,
                                               tolerances, args.tolerance_ms)
                     for exercise in CLASSES[1:] if exercise != "plank"}
        write_json(folder / "exercise_scores.json", exercises)
        plan["results"][name] = {"segmentation": report["segmentation"], "by_tolerance": report["by_tolerance"],
                                  "leg_raise": exercises["leg_raise"], "model_lookahead_ms": report["model_lookahead_ms"]}
        write_json(out / "report.json", plan)
    print(f"Sensor ablation pilot complete: {out / 'report.json'}", flush=True)
    return plan
