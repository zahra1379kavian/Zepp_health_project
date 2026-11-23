"""Run with python -m baseline.mstcn --help."""

import argparse
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m baseline.mstcn",
                                     description="Causal MS-TCN: station/NULL + dense repetition phases")
    sub = parser.add_subparsers(dest="command", required=True)
    compare = sub.add_parser("reference-benchmark", help="Replay wrist checkpoints on the Feishu reference clips and KPIs")
    compare.add_argument("--run-dir", type=Path, required=True, help="Directory containing fold_1 through fold_5")
    compare.add_argument("--source", type=Path, default=Path("strength_training/strength_training_data_clean"))
    compare.add_argument("--output", type=Path, required=True)
    compare.add_argument("--model-name", default="MS-TCN")
    compare.add_argument("--start-rule", choices=["learned", "previous_end"])
    compare.add_argument("--refine-end", action="store_true")
    compare.add_argument("--threads", type=int, default=1)
    compare.add_argument("--device", default="cpu")
    prepare = sub.add_parser("prepare", help="Exact-clock joins, exclusions, references and subject splits")
    prepare.add_argument("--source", type=Path, default=Path("strength_training/strength_training_data_clean"))
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--alignment", type=Path, default=Path("strength_training/data_demographics/tables/log_clock_alignment.csv"))
    prepare.add_argument("--location", choices=["wrist", "left_wrist", "right_wrist"], default="wrist")
    prepare.add_argument("--references", type=Path)
    prepare.add_argument("--reference-mode", choices=["labels", "none", "endpoints"], default="labels",
                         help="endpoints uses --references and binary station presence, with no phase conversion")
    prepare.add_argument("--seed", type=int, default=42)

    train = sub.add_parser("train", help="Train both heads, or an explicit station-only diagnostic")
    cv = sub.add_parser("cross-validate", help="Subject-wise folds with separate validation and pooled held-out scores")
    ablation = sub.add_parser("sensor-ablation", help="One-fold lookahead, inertial augmentation and waist comparisons")
    loeo = sub.add_parser("leave-one-exercise-out", help="Train and score a repetition model per held-out exercise")
    diagnostic = sub.add_parser("biceps-diagnostic", help="Inspect threshold sensitivity of the frozen biceps phase holdout")
    diagnostic.add_argument("--run-dir", type=Path, required=True, help="Existing biceps_curls holdout folder")
    diagnostic.add_argument("--output", type=Path, required=True, help="Diagnostic JSON, separate from scored results")
    diagnostic.add_argument("--threads", type=int, default=2)
    diagnostic.add_argument("--device", default="cpu")
    ablation.add_argument("--folds", type=int, default=5)
    ablation.add_argument("--fold", type=int, default=1, help="One-based pilot fold; only this fold is run")
    loeo.add_argument("--subject-folds", "--folds", dest="folds", type=int, default=5,
                      help="Number of groups used to define the fixed participant split")
    loeo.add_argument("--subject-fold", "--fold", dest="fold", type=int, default=1,
                      help="Participant split shared by every exercise holdout")
    ablation.add_argument("--source", type=Path, help="Raw sensor root for waist fusion; defaults to dataset source")
    ablation.add_argument("--variants", nargs="+", default=["baseline", "lookahead", "acc_gyro", "augmentation",
                                                           "combined", "combined_paired_wrist", "combined_waist"],
                          choices=["baseline", "lookahead", "acc_gyro", "augmentation", "combined",
                                   "combined_paired_wrist", "combined_waist"])
    loeo.add_argument("--exercises", nargs="+", help="Exercise names, 'screen' (three), or 'confirm' (six). Legacy also accepts 'all'.")
    loeo.add_argument("--protocol", choices=["paired", "legacy"],
                      help="Complete heads, step 2 inputs and screen/confirm presets default to paired validation")
    train.add_argument("--phase-only", action="store_true",
                       help="Train a repetition model on phase targets without an exercise classification head")
    cv.add_argument("--folds", type=int, default=5)
    cv.add_argument("--resume-completed", action="store_true",
                    help="Reuse verified completed folds in an existing output; incomplete folds must be moved aside")
    for command in (train, cv, ablation, loeo):
        command.add_argument("--data", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--station-only", action="store_true")
        heads = ["auto", "phase", "boundary", "complete"] if command in (train, cv, loeo) else ["auto", "phase", "boundary"]
        command.add_argument("--repetition-head", choices=heads, default="auto")
        command.add_argument("--epochs", type=int, default=30)
        command.add_argument("--batch-size", type=int, default=8)
        command.add_argument("--window", type=int, default=256)
        command.add_argument("--stages", type=int, default=3)
        command.add_argument("--layers", type=int, default=6)
        command.add_argument("--channels", type=int, default=64)
        command.add_argument("--learning-rate", type=float, default=1e-3)
        command.add_argument("--patience", type=int, default=6)
        command.add_argument("--seed", type=int, default=42)
        command.add_argument("--tolerance-ms", type=int, default=400)
        command.add_argument("--max-steps", type=int, default=0, help="Diagnostic cap per epoch/split; 0 means all batches")
        if command is not ablation:
            command.add_argument("--lookahead-ms", type=int, default=0,
                                 help="Fixed output delay, rounded up to 40 ms samples; 500 gives 520 ms")
            command.add_argument("--augment", action="store_true", help="Training-only rotations, time warp and scaling (phase head)")

    for name in ("evaluate", "tune"):
        command = sub.add_parser(name)
        command.add_argument("--data", type=Path, required=True)
        command.add_argument("--checkpoint", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--tolerance-ms", type=int, default=400,
                             help="Declare before test evaluation; tune stores this fixed matching tolerance")
        if name == "evaluate":
            command.add_argument("--split", choices=["validation", "test"], default="test")
            command.add_argument("--decoder", type=Path)
            command.add_argument("--start-rule", choices=["learned", "previous_end"],
                                 help="Override the start rule for complete-rep evaluation")
    mine = sub.add_parser("mine-rest", help="Find runs of 3+ predicted reps in held-out, labeled rest for review")
    mine.add_argument("--predictions", type=Path, required=True, help="Baseline cross-validation folder")
    mine.add_argument("--data", type=Path, required=True)
    mine.add_argument("--output", type=Path, required=True)
    rescore = sub.add_parser("rescore", help="Regenerate an evaluation report from saved decoded predictions")
    rescore.add_argument("--predictions", type=Path, required=True, help="Evaluation or cross-validation output directory")
    rescore.add_argument("--data", type=Path, help="Original prepared dataset when fold-local data is unavailable")
    rescore.add_argument("--output", type=Path, required=True, help="Path for the new JSON report")
    predict = sub.add_parser("predict", help="Sensor-only CSV replay with online cached inference")
    predict.add_argument("--input", type=Path, required=True)
    predict.add_argument("--waist-input", type=Path, help="Required for a paired wrist+waist checkpoint")
    predict.add_argument("--checkpoint", type=Path, required=True)
    predict.add_argument("--output", type=Path, required=True)
    predict.add_argument("--decoder", type=Path)
    predict.add_argument("--chunk", type=int, default=1, help="1..20 samples; default is one sample (40 ms)")
    for command in (sub.choices["evaluate"], predict):
        command.add_argument("--refine-end", action="store_true",
                             help="Refine complete-rep endpoints with a probability-weighted ±4-frame window")
    demo = sub.add_parser("demo", help="Run the whole pipeline with generated, validated synthetic boundaries")
    demo.add_argument("--output", type=Path, default=Path("baseline/MS-TCN/experiment_runs/demo"))
    run = sub.add_parser("run", help="Prepare every wrist stream → subject-wise cross-validation")
    run.add_argument("--output", type=Path, default=Path("baseline/MS-TCN/experiment_runs/wrist_cross_validation"))
    run.add_argument("--data", type=Path, help="Reuse a prepared wrist dataset instead of preparing it again")
    run.add_argument("--references", type=Path, help="Validated full-rep references when preparing a new dataset")
    run.add_argument("--repetition-head", choices=["auto", "phase", "boundary", "complete"], default="auto")
    for command in (run, train, cv, loeo):
        command.add_argument("--rest-reviews", type=Path, help="Complete mode: reviewed mine-rest JSON")
        command.add_argument("--rep-inputs", choices=["original", "gravity", "rotated"], default="original",
                             help="Existing sensors, six gravity features, or acc+gyro with uniform training rotations")
    run.add_argument("--source", type=Path, default=Path("strength_training/strength_training_data_clean"))
    run.add_argument("--alignment", type=Path, default=Path("strength_training/data_demographics/tables/log_clock_alignment.csv"))
    run.add_argument("--location", choices=["wrist", "left_wrist", "right_wrist"], default="wrist")
    run.add_argument("--epochs", type=int, default=30)
    run.add_argument("--folds", type=int, default=5)
    run.add_argument("--batch-size", type=int, default=8)
    run.add_argument("--seed", type=int, default=42)
    run.add_argument("--tolerance-ms", type=int, default=400)
    for command in (cv, ablation, loeo, sub.choices["evaluate"], run):
        command.add_argument("--report-tolerances-ms", type=int, nargs="+", default=[200, 400],
                             help="Score the same decoded predictions at each tolerance; selection tolerance is always included")
    for command in (train, cv, ablation, loeo, sub.choices["evaluate"], sub.choices["tune"], predict, demo, run):
        command.add_argument("--device", default="cpu")
        command.add_argument("--threads", type=int, default=2)
    args = parser.parse_args(argv)
    if hasattr(args, "threads"):
        import torch
        torch.set_num_threads(args.threads)
    if args.command == "reference-benchmark":
        from .analyses.reference import benchmark
        benchmark(args)
    elif args.command == "biceps-diagnostic":
        from .analyses.leave_one_exercise_out.diagnostic import run
        run(args)
    elif args.command == "prepare":
        from .data import prepare as prepare_dataset
        result = prepare_dataset(args.source, args.output, args.alignment, args.location,
                                 args.references, args.seed, args.reference_mode)
        print(f"Saved {len(result['sessions'])} wrist streams from {result['subjects']} subjects; "
              f"excluded {len(result['excluded'])} source sessions: {args.output}")
    elif args.command == "demo":
        from .demo import run_demo
        run_demo(args.output, args.device)
    elif args.command == "run":
        from .runner import run_pipeline
        run_pipeline(args)
    elif args.command == "mine-rest":
        from .complete import mine_rest
        mine_rest(args)
    elif args.command in ("sensor-ablation", "leave-one-exercise-out"):
        from .analyses import run_analysis
        run_analysis(args)
    else:
        from . import runner
        result = getattr(runner, args.command.replace("-", "_"))(args)
        print(f"{args.command} complete: {result if args.command == 'train' else args.output}")


if __name__ == "__main__":
    main()
