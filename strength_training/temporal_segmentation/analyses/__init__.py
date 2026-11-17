"""Separate, reproducible MS-TCN pilot analyses."""


def run_analysis(args):
    if args.command == "sensor-ablation":
        from .sensor_ablation import run
    else:
        from .leave_one_exercise_out import run
    return run(args)
