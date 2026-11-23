"""Feishu wrist tables and figures from per-clip counts, with no model dependencies."""

from contextlib import redirect_stdout
from pathlib import Path
import io
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .protocol import REFERENCE, rep


# August 31 wrist/activity-agnostic release, PDF pp. 7 and 8, in printed order.
# Column order within each triple is TP%, FN%, FP%.
PUBLISHED = {
    "Overall": ((92.7, 7.3, 4.4), (95.1, 4.9, 4.1)),
    "Overall minus leg raise, crunch": ((96.3, 3.7, 4.5), (97.4, 2.6, 3.9)),
    "push-up": ((92.7, 7.3, 2.7), (94.3, 5.7, 3.1)),
    "deadlift": ((97.5, 2.5, 5.8), (99.4, .6, 4.6)),
    "squat": ((98.3, 1.7, 2.6), (98.6, 1.4, 3.1)),
    "leg_raise": ((51.3, 48.7, 3.8), (62.6, 37.4, 3.2)),
    "lat_pulldown": ((99.1, .9, 4.6), (100., 0., 4.)),
    "overhead_press_(ohp)": ((95.3, 4.7, 4.1), (97.6, 2.4, 1.8)),
    "lunges": ((98.5, 1.5, 2.7), (99.2, .8, 1.7)),
    "hip_thrust": ((95.1, 4.9, 3.4), (95.7, 4.3, 4.1)),
    "dips": ((91.8, 8.2, 8.8), (92.4, 7.6, 8.3)),
    "pull-up": ((92.6, 7.4, 10.1), (94.3, 5.7, 8.0)),
    "bench_press": ((98.3, 1.7, 5.6), (99.4, .6, 3.6)),
    "biceps_curls": ((96.6, 3.4, 3.8), (97.8, 2.2, 4.1)),
    "rowing": ((97.2, 2.8, 4.0), (97.7, 2.3, 3.4)),
    "crunch": ((75.3, 24.7, 2.7), (94.5, 5.5, 13.3)),
}
COUNTS = ["TP", "FN", "FP", "FP_rest", "FP_activity", "total", "pred", "cnt_error"]
OUTCOMES = ["TP", "FN", "FP_rest", "FP_activity"]


def reference_plots():
    # The release plot module uses top-level src imports. Scope its import path.
    bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(REFERENCE))
    try:
        from src.utils import plot
        if Path(plot.__file__).resolve() != REFERENCE / "src/utils/plot.py":
            raise ImportError("A different src.utils.plot is already loaded")
        return plot
    finally:
        sys.path.remove(str(REFERENCE))
        sys.dont_write_bytecode = bytecode


def summaries(rows):
    groups = [rows, rows[~rows.activity.isin(["leg_raise", "crunch"])]]
    groups.extend(rows[rows.activity == act] for act in list(PUBLISHED)[2:])
    values = []
    for label, group in zip(PUBLISHED, groups):
        counts = group[COUNTS].sum().to_dict()
        total = counts["total"]
        rates = {k + "_pct": 100 * counts[k] / total if total else np.nan
                 for k in [*OUTCOMES, "FP", "cnt_error"]}
        values.append(dict(activity=label, **counts, **rates, dur_error=group.dur_error.mean()))
    return pd.DataFrame(values)


def save(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def table_figure(summary, title, out):
    fig, ax = plt.subplots(figsize=(7.4, 6))
    ax.axis("off")
    cells = [[r.activity, *(f"{p:.1f}" for p in (r.TP_pct, r.FN_pct, r.FP_pct))]
             for r in summary.itertuples()]
    table = ax.table(cellText=cells, colLabels=["Stations", "TP %", "FN %", "FP %"],
                     loc="center", cellLoc="left", colLoc="left", colWidths=[.53, .15, .15, .15])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.55)
    for (i, j), cell in table.get_celld().items():
        cell.set_edgecolor("#d5d8db")
        if i <= 2:
            cell.set_text_props(weight="bold")
        if i == 0:
            cell.set_facecolor("#edf1f3")
    ax.set_title(title)
    save(fig, out / "table_results.png")


def example_plot(path):
    meta = json.loads(path.read_text())
    gt, pred = meta["ground_truth"], meta["predictions"]
    pairs, misses, extras = rep.match_reps_greedy(gt, pred)
    with np.load(path.with_suffix(".npz")) as data:
        size = len(data["labels"])
        events = misses + extras if path.stem.endswith("_bad") else [g for g, _ in pairs]
        centers = np.array([(r["start"] + r["end"]) // 2 for r in events])
        width = 25 * 30
        starts = list(range(0, max(1, size - width + 1), 25)) + [max(0, size - width)]
        first = max(starts, key=lambda i: np.sum((centers >= i) & (centers < i + width)))
        last = min(size, first + width)
        t = np.arange(first, last) / 25
        fig, axes = plt.subplots(4, sharex=True, figsize=(10, 5.3),
                                 gridspec_kw={"height_ratios": [2, 2, 1, 1]}, layout="constrained")
        for key in data.files:
            if key not in ("labels", "acc_norm", "gyro_norm"):
                axes[0].plot(t, data[key][first:last], label=key)
        if axes[0].lines:
            axes[0].legend(loc="upper right", ncols=3, fontsize=8)
        axes[0].set(ylabel="Probability", ylim=(-.05, 1.05))
        axes[1].plot(t, data["acc_norm"][first:last], color="tab:blue", lw=.9)
        gyro = axes[1].twinx()
        gyro.plot(t, data["gyro_norm"][first:last], color="tab:orange", lw=.9)
        gyro.set_ylabel("Gyro norm", color="tab:orange")
        axes[1].set_ylabel("Acc norm", color="tab:blue")
        for ax, intervals, label in zip(axes[2:], (gt, pred), ("Labeled reps", "Predicted reps")):
            for r in intervals:
                if r["end"] > first and r["start"] < last:
                    ax.axvspan(r["start"] / 25, r["end"] / 25, ymin=.15, ymax=.85,
                               facecolor="tab:blue" if ax is axes[2] else "tab:green", edgecolor="white")
            ax.set(yticks=[], ylabel=label)
        axes[-1].set(xlabel="Time from clip start (s)", xlim=(first / 25, last / 25))
        fig.suptitle(f"{path.stem} · {meta['session']} · {meta['device']} · clip F1 {meta['f1']:.3f}")
        save(fig, path.with_suffix(".png"))


def render(stage):
    stage = Path(stage)
    frame = pd.read_csv(stage / "test_metrics_detailed.csv")
    manifest = json.loads((stage / "manifest.json").read_text())
    delays = {i: [] for i in range(1, 6)}
    with (stage / "intervals.jsonl").open() as f:
        for line in f:
            record = json.loads(line)
            delays[record["fold"]].extend(record["matched_delay_ms"])
    for scope, idx, label in [("pooled", 0, "Five folds pooled"), ("fold_5", 1, "Local fold 5")]:
        out = stage / scope
        out.mkdir(exist_ok=True)
        rows = frame if idx == 0 else frame[frame.fold == 5]
        summary = summaries(rows)
        summary.to_csv(out / "test_metrics_summary.csv", index=False)
        timing = delays[5] if idx else [v for fold in delays.values() for v in fold]
        latency = dict(matched_reps=int(rows.TP.sum()), timed_matches=len(timing),
                       clock="25 Hz replay; emitted frame minus labeled exclusive end, excluding compute and transport")
        if timing:
            latency.update(median_ms=float(np.median(timing)), p95_ms=float(np.percentile(timing, 95)),
                           max_ms=int(max(timing)), early_fraction=float(np.mean(np.array(timing) < 0)))
        (out / "reporting_delay.json").write_text(json.dumps(latency, indent=2) + "\n")
        compare = summary[["activity", "TP_pct", "FN_pct", "FP_pct"]].copy()
        compare[["published_TP_pct", "published_FN_pct", "published_FP_pct"]] = [
            row[idx] for row in PUBLISHED.values()]
        compare.to_csv(out / "published_comparison.csv", index=False)
        title = f"{manifest['model']} · {label}"
        table_figure(summary, title, out)
        plot = reference_plots()
        detailed = rows.sort_values("cnt_error_pct", ascending=False)
        csv = out / "test_metrics_detailed.csv"
        detailed.to_csv(csv, index=False)
        by_activity = summary.iloc[2:].sort_values("cnt_error_pct", ascending=False)
        by_activity.to_csv(out / "test_metrics_by_activity.csv", index=False)
        summary.iloc[:1].to_csv(out / "test_metrics_combined.csv", index=False)
        plot.plot_result_table(by_activity.copy(), out / "table_results_online.png")
        plot.plot_metrics(rows, out / "test_metrics.png")
        plot.plot_outliers_by_session(rows, out / "fn_outliers_by_session.png")
        for activity, data in detailed.groupby("activity"):
            folder = out / "activities" / activity
            folder.mkdir(parents=True, exist_ok=True)
            data.to_csv(folder / "test_metrics.csv", index=False)
            plot.plot_metrics(data, folder / "test_metrics.png")
            plot.plot_outliers_by_session(data, folder / "fn_outliers_by_session.png", activity)
            with redirect_stdout(io.StringIO()):
                plot.plot_error_by_session_device(csv, activity=activity, save_dir=folder)
        for path in sorted((out / "examples").glob("*.json")):
            example_plot(path)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Render the wrist reports from a saved benchmark run")
    parser.add_argument("run", type=Path)
    render(parser.parse_args().run)
