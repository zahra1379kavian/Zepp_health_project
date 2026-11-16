"""Run with python -m strength_training.temporal_segmentation --help."""

import argparse
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description="Causal MS-TCN: station/NULL + repetition boundaries")
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare", help="Exact-clock joins, exclusions, references and subject splits")
    prepare.add_argument("--source", type=Path, default=Path("strength_training/strength_training_data_clean"))
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--alignment", type=Path, default=Path("strength_training/data_demographics/tables/log_clock_alignment.csv"))
    prepare.add_argument("--location", choices=["waist", "left_wrist", "right_wrist"], default="waist")
    prepare.add_argument("--references", type=Path)
    prepare.add_argument("--seed", type=int, default=42)

    train = sub.add_parser("train", help="Train both heads, or an explicit station-only diagnostic")
    train.add_argument("--data", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--station-only", action="store_true")
    train.add_argument("--epochs", type=int, default=30)
    train.add_argument("--batch-size", type=int, default=8)
    train.add_argument("--window", type=int, default=256)
    train.add_argument("--stages", type=int, default=3)
    train.add_argument("--layers", type=int, default=6)
    train.add_argument("--channels", type=int, default=64)
    train.add_argument("--learning-rate", type=float, default=1e-3)
    train.add_argument("--patience", type=int, default=6)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--max-steps", type=int, default=0, help="Diagnostic cap per epoch/split; 0 means all batches")

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
    for command in (train, sub.choices["evaluate"], sub.choices["tune"], predict, demo):
        command.add_argument("--device", default="cpu")
        command.add_argument("--threads", type=int, default=2)
    args = parser.parse_args(argv)
    if hasattr(args, "threads"):
        import torch
        torch.set_num_threads(args.threads)
    if args.command == "prepare":
        from .data import prepare as prepare_dataset
        result = prepare_dataset(args.source, args.output, args.alignment, args.location, args.references, args.seed)
        print(f"Saved {len(result['sessions'])} eligible sessions; excluded {len(result['excluded'])}: {args.output}")
    elif args.command == "demo":
        from .demo import run_demo
        run_demo(args.output, args.device)
    else:
        from . import runner
        result = getattr(runner, args.command)(args)
        print(f"{args.command} complete: {result if args.command == 'train' else args.output}")


if __name__ == "__main__":
    main()
