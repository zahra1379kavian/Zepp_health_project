"""Run with python -m strength_training.temporal_segmentation --help."""

import argparse
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description="Causal MS-TCN: station/NULL + dense repetition phases")
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare", help="Exact-clock joins, exclusions, references and subject splits")
    prepare.add_argument("--source", type=Path, default=Path("strength_training/strength_training_data_clean"))
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--alignment", type=Path, default=Path("strength_training/data_demographics/tables/log_clock_alignment.csv"))
    prepare.add_argument("--location", choices=["wrist", "left_wrist", "right_wrist"], default="wrist")
    prepare.add_argument("--references", type=Path)
    prepare.add_argument("--reference-mode", choices=["labels", "none"], default="labels",
                         help="Default: validated phase1 then phase2 cycles from label.csv wrist phases")
    prepare.add_argument("--seed", type=int, default=42)

    train = sub.add_parser("train", help="Train both heads, or an explicit station-only diagnostic")
    cv = sub.add_parser("cross-validate", help="Subject-wise folds with separate validation and pooled held-out scores")
    cv.add_argument("--folds", type=int, default=5)
    for command in (train, cv):
        command.add_argument("--data", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--station-only", action="store_true")
        command.add_argument("--repetition-head", choices=["auto", "phase", "boundary"], default="auto")
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
    predict = sub.add_parser("predict", help="Sensor-only CSV replay with online cached inference")
    predict.add_argument("--input", type=Path, required=True)
    predict.add_argument("--checkpoint", type=Path, required=True)
    predict.add_argument("--output", type=Path, required=True)
    predict.add_argument("--decoder", type=Path)
    predict.add_argument("--chunk", type=int, default=1, help="1..20 samples; default is one sample (40 ms)")
    demo = sub.add_parser("demo", help="Run the whole pipeline with generated, validated synthetic boundaries")
    demo.add_argument("--output", type=Path, default=Path("strength_training/tas_runs/demo"))
    run = sub.add_parser("run", help="Prepare every wrist stream → subject-wise cross-validation")
    run.add_argument("--output", type=Path, default=Path("strength_training/tas_runs/wrist_cv"))
    run.add_argument("--data", type=Path, help="Reuse a prepared wrist dataset instead of preparing it again")
    run.add_argument("--source", type=Path, default=Path("strength_training/strength_training_data_clean"))
    run.add_argument("--alignment", type=Path, default=Path("strength_training/data_demographics/tables/log_clock_alignment.csv"))
    run.add_argument("--location", choices=["wrist", "left_wrist", "right_wrist"], default="wrist")
    run.add_argument("--epochs", type=int, default=30)
    run.add_argument("--folds", type=int, default=5)
    run.add_argument("--batch-size", type=int, default=8)
    run.add_argument("--seed", type=int, default=42)
    run.add_argument("--tolerance-ms", type=int, default=400)
    for command in (cv, sub.choices["evaluate"], run):
        command.add_argument("--report-tolerances-ms", type=int, nargs="+", default=[200, 400],
                             help="Score the same decoded predictions at each tolerance; selection tolerance is always included")
    for command in (train, cv, sub.choices["evaluate"], sub.choices["tune"], predict, demo, run):
        command.add_argument("--device", default="cpu")
        command.add_argument("--threads", type=int, default=2)
    args = parser.parse_args(argv)
    if hasattr(args, "threads"):
        import torch
        torch.set_num_threads(args.threads)
    if args.command == "prepare":
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
    else:
        from . import runner
        result = getattr(runner, args.command.replace("-", "_"))(args)
        print(f"{args.command} complete: {result if args.command == 'train' else args.output}")


if __name__ == "__main__":
    main()
