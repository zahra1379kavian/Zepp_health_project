"""Train/evaluate the RecoFit-style baseline without a neural-model dependency."""

import argparse
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m baseline.RecoFit",
                                     description="RecoFit-style wrist baseline at 25 Hz")
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare", help="Reuse exact-clock preparation, exclusions and wrist references")
    prepare.add_argument("--source", type=Path, default=Path("strength_training/strength_training_data_clean"))
    prepare.add_argument("--alignment", type=Path, default=Path("strength_training/data_demographics/tables/log_clock_alignment.csv"))
    prepare.add_argument("--location", choices=["wrist", "left_wrist", "right_wrist"], default="wrist")
    prepare.add_argument("--output", type=Path, default=Path("baseline/RecoFit/data"))
    prepare.add_argument("--seed", type=int, default=42)
    prepare.add_argument("--references", type=Path)
    prepare.add_argument("--reference-mode", choices=["labels", "none"], default="labels")

    train = sub.add_parser("train", help="Fit SVMs, period bounds and boundary offsets on training subjects")
    cv = sub.add_parser("cross-validate", help="Disjoint subject folds and pooled test metrics")
    run = sub.add_parser("run", help="Prepare if needed, then run subject-wise cross-validation")
    for p in (train, cv, run):
        p.add_argument("--data", type=Path, default=Path("strength_training/tas_data_both_wrists"))
        p.add_argument("--output", type=Path, required=True)
        p.add_argument("--window-seconds", type=float, default=5.)
        p.add_argument("--train-step-ms", type=int, default=1000)
        p.add_argument("--prediction-step-ms", type=int, default=200)
        p.add_argument("--max-windows-per-class", type=int, default=3000)
        p.add_argument("--c-values", type=float, nargs="+", default=[.1, 1., 10.])
        p.add_argument("--confirm-steps", type=int, default=3,
                       help="decisions to confirm exercise/rest and a new segment-wide station vote leader")
        p.add_argument("--seed", type=int, default=42)
    for p in (cv, run):
        p.add_argument("--folds", type=int, default=5)
    run.add_argument("--source", type=Path, default=Path("strength_training/strength_training_data_clean"))
    run.add_argument("--alignment", type=Path, default=Path("strength_training/data_demographics/tables/log_clock_alignment.csv"))
    run.add_argument("--location", choices=["wrist", "left_wrist", "right_wrist"], default="wrist")

    evaluate = sub.add_parser("evaluate", help="Score all eligible samples and validated references")
    evaluate.add_argument("--data", type=Path, default=Path("strength_training/tas_data_both_wrists"))
    evaluate.add_argument("--split", choices=["test", "validation"], default="test")
    prediction = sub.add_parser("predict", help="Sensor-only wrist CSV prediction; embedded labels ignored")
    prediction.add_argument("--input", type=Path, required=True)
    prediction.add_argument("--side", choices=["left_wrist", "right_wrist"],
                            help="sensor wrist; inferred from left_wrist.csv/right_wrist.csv or a single-wrist model")
    for p in (evaluate, prediction):
        p.add_argument("--model", type=Path, required=True)
        p.add_argument("--output", type=Path, required=True)
    for p in (evaluate, cv, run):
        p.add_argument("--tolerance-ms", type=int, default=400)
        p.add_argument("--report-tolerances-ms", type=int, nargs="+", default=[200, 400])
    for p in (evaluate, prediction, cv, run):
        p.add_argument("--mode", choices=["causal", "offline"], default="causal",
                       help="causal: immutable prefix-only interval events; offline: batch counter diagnostic")
    demo = sub.add_parser("demo", help="Small synthetic integration check; not a real-data benchmark")
    demo.add_argument("--output", type=Path, default=Path("baseline/RecoFit/runs/demo"))
    args = parser.parse_args(argv)
    if args.command == "prepare":
        from strength_training.temporal_segmentation.data import prepare as prepare_data
        prepare_data(args.source, args.output, args.alignment, args.location, args.references,
                     args.seed, args.reference_mode)
    elif args.command == "demo":
        from .demo import run_demo
        run_demo(args.output)
    else:
        from . import pipeline
        if args.command == "predict":
            pipeline.predict(args.model, args.input, args.output, mode=args.mode, side=args.side)
        elif args.command == "evaluate":
            pipeline.evaluate(args.data, args.model, args.output, split_name=args.split, mode=args.mode,
                              tolerance_ms=args.tolerance_ms, report_tolerances_ms=args.report_tolerances_ms)
        else:
            kwargs = {key: getattr(args, key) for key in ("window_seconds", "train_step_ms", "prediction_step_ms",
                       "max_windows_per_class", "c_values", "confirm_steps", "seed")}
            if args.command == "train":
                pipeline.fit(args.data, args.output, **kwargs)
            else:
                if args.command == "run":
                    if args.output.exists():
                        raise FileExistsError(f"Refusing to overwrite {args.output}")
                    if not args.data.exists():
                        from strength_training.temporal_segmentation.data import prepare as prepare_data
                        prepare_data(args.source, args.output/"data", args.alignment, args.location, seed=args.seed)
                        args.data = args.output/"data"
                    args.output = args.output/"cross_validation"
                pipeline.cross_validate(args.data, args.output, folds=args.folds, mode=args.mode,
                                         tolerance_ms=args.tolerance_ms,
                                         report_tolerances_ms=args.report_tolerances_ms, **kwargs)
    print(f"{args.command} complete: {args.output}")


if __name__ == "__main__":
    main()
