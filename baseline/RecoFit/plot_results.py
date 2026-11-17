"""Render an auditable interim report from completed RecoFit test folds.

Run from the repository root with a Python environment containing matplotlib:
    .venv/bin/python -m baseline.RecoFit.plot_results

This only reads saved predictions; it does not fit models or run inference.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch
import numpy as np

from strength_training.temporal_segmentation.constants import CLASSES
from strength_training.temporal_segmentation.data import subject_folds
from strength_training.temporal_segmentation.metrics import grouped_stream_report


BLUE = "#315E88"
TEAL = "#21857D"
ORANGE = "#C17B35"
PURPLE = "#7C6494"
INK = "#263541"
GRAY = "#DAE0E4"
LIGHT = "#F3F6F8"
LABELS = {
    "NULL": "Rest (NULL)", "bench_press": "Bench press", "biceps_curls": "Biceps curls",
    "crunch": "Crunch", "deadlift": "Deadlift", "dips": "Dips", "hip_thrust": "Hip thrust",
    "lat_pulldown": "Lat pulldown", "leg_raise": "Leg raise", "lunges": "Lunges",
    "overhead_press_(ohp)": "Overhead press", "plank": "Plank", "pull-up": "Pull-up",
    "push-up": "Push-up", "rowing": "Rowing", "squat": "Squat",
}


def read_json(path):
    return json.loads(path.read_text())


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def same_values(actual, expected, location="report"):
    """Check reconstructed fold aggregates against their saved reports."""
    if isinstance(actual, dict):
        for key, value in actual.items():
            same_values(value, expected[key], f"{location}.{key}")
    elif isinstance(actual, list):
        if len(actual) != len(expected):
            raise ValueError(f"Length mismatch at {location}")
        for index, (left, right) in enumerate(zip(actual, expected)):
            same_values(left, right, f"{location}[{index}]")
    elif isinstance(actual, float):
        if not np.isclose(actual, expected, rtol=1e-12, atol=1e-12):
            raise ValueError(f"Metric mismatch at {location}: {actual} != {expected}")
    elif actual != expected:
        raise ValueError(f"Value mismatch at {location}: {actual} != {expected}")


def load_results(args):
    manifest = read_json(args.data / "manifest.json")
    metadata = {row["session"]: row for row in manifest["sessions"]}
    planned_splits = subject_folds(manifest["sessions"], args.planned_folds, args.seed)
    reports, records, source_paths, tested_subjects = [], [], [], set()
    source_paths.append(args.data / "manifest.json")
    modes = set()
    for number in args.folds:
        fold = args.run / f"fold_{number}"
        report_path, training_path = fold / "test/report.json", fold / "model/training.json"
        report, training = read_json(report_path), read_json(training_path)
        if training["split"] != planned_splits[number - 1]:
            raise ValueError(f"Fold {number} does not match the declared subject split")
        if report["split"] != "test" or report["report_tolerances_ms"] != [200, 400]:
            raise ValueError(f"Unexpected evaluation configuration in fold {number}")
        if report["timing_tolerance_ms"] != 400:
            raise ValueError("This figure set requires the primary 400 ms tolerance")
        modes.add(report["mode"])
        split_subjects = {
            key: {metadata[name]["subject"] for name in training["split"][key]}
            for key in ("train", "validation", "test")
        }
        for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
            if split_subjects[left] & split_subjects[right]:
                raise ValueError(f"Subject overlap within fold {number}")
        if tested_subjects & split_subjects["test"]:
            raise ValueError("Repeated test subjects across selected folds")
        tested_subjects |= split_subjects["test"]
        fold_records = []
        for name in sorted(training["split"]["test"]):
            path = fold / "test" / f"{name}.json"
            result = read_json(path)
            record = result["scoring"]
            if record["session"] != name or result["mode"] != report["mode"]:
                raise ValueError(f"Unexpected prediction metadata: {path}")
            for key in ("subject", "side", "source_session"):
                if record[key] != metadata[name][key]:
                    raise ValueError(f"Stream metadata mismatch: {path}, {key}")
            fold_records.append(record)
            source_paths.append(path)
        same_values(grouped_stream_report(fold_records, [200, 400], 400), report)
        records.extend(fold_records)
        reports.append((number, report))
        source_paths.extend([report_path, training_path])
    if modes != {"causal"}:
        raise ValueError("These timing captions require causal evaluation")
    pooled = grouped_stream_report(records, [200, 400], 400)
    if pooled["subjects"] != len(tested_subjects):
        raise ValueError("Incorrect held-out subject total")
    for tolerance in ("200", "400"):
        for key in ("repetitions", "segmentation"):
            for count in ("tp", "extra", "missed"):
                expected = sum(report["by_tolerance"][tolerance][key][count] for _, report in reports)
                if pooled["by_tolerance"][tolerance][key][count] != expected:
                    raise ValueError("Pooled event counts do not sum to fold counts")
    return pooled, reports, records, source_paths, manifest


def set_style(font_path):
    if font_path:
        fm.fontManager.addfont(str(font_path))
        family = fm.FontProperties(fname=str(font_path)).get_name()
    else:
        try:
            fm.findfont("Times New Roman", fallback_to_default=False)
            family = "Times New Roman"
        except ValueError:
            family = "DejaVu Serif"
    resolved = fm.findfont(family, fallback_to_default=False)
    plt.rcParams.update({
        "font.family": family, "font.size": 10, "text.color": INK,
        "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
        "axes.titlesize": 11, "axes.titleweight": "bold", "axes.labelsize": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#AAB6BF", "axes.linewidth": .6,
        "xtick.labelsize": 9, "ytick.labelsize": 9,
        "legend.fontsize": 9, "legend.frameon": False,
        "figure.facecolor": "white", "savefig.facecolor": "white",
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "path",
        "axes.unicode_minus": False,
    })
    return family, resolved


def style_axis(ax, axis="y"):
    ax.set_axisbelow(True)
    ax.grid(axis=axis, color="#E6EBEF", linewidth=.6)
    ax.tick_params(length=0, pad=5)


def heading(fig, title, subtitle):
    fig.text(.035, .967, title, fontsize=18, weight="bold", va="top")
    fig.text(.035, .920, subtitle, fontsize=10, color="#536572", va="top")


def footer(fig, text):
    fig.text(.035, .026, text, fontsize=8.2, va="bottom", linespacing=1.45, color="#536572")


def card(fig, x, y, width, value, label, color=BLUE):
    box = FancyBboxPatch((x, y), width, .115, boxstyle="round,pad=0.008,rounding_size=0.008",
                         transform=fig.transFigure, facecolor=LIGHT, edgecolor="none")
    box.set_path_effects([pe.SimplePatchShadow(offset=(1.4, -1.4), alpha=.09), pe.Normal()])
    fig.add_artist(box)
    fig.text(x+.014, y+.070, value, fontsize=23, color=color, weight="bold", va="center")
    fig.text(x+.014, y+.025, label, fontsize=8.4, va="center", linespacing=1.25)


def focal_note(ax, text, xy=(.97, .95), ha="right", size=9):
    note = ax.text(*xy, text, transform=ax.transAxes, va="top", ha=ha, fontsize=size,
                   bbox={"boxstyle": "round,pad=0.4", "fc": LIGHT, "ec": "none"})
    note.get_bbox_patch().set_path_effects([
        pe.SimplePatchShadow(offset=(1.2, -1.2), alpha=.10), pe.Normal()])


def overview(pooled, reports, stage):
    fig = plt.figure(figsize=(12.6, 7.1))
    heading(fig, "RecoFit | Interim held-out performance", stage)
    rep = pooled["repetitions"]
    cards = [
        (f"{100*pooled['classification']['macro_f1_all_16']:.1f}%", "Frame macro F1 · all 16 classes", BLUE),
        (f"{100*pooled['classification']['exercise_rest_f1']:.1f}%", "Exercise / rest frame F1", PURPLE),
        (f"{100*rep['f1']:.1f}%", "Repetition F1 · ±400 ms", TEAL),
        (f"{rep['count_mae_active_station_sessions']:.2f}", "Count MAE (repetitions)\nPer active exercise / wrist stream", ORANGE),
    ]
    for i, (value, label, color) in enumerate(cards):
        card(fig, .043+i*.236, .748, .216, value, label, color)
    ax = fig.add_axes([.075, .235, .49, .405])
    groups = [report for _, report in reports] + [pooled]
    series = [
        ("Frame macro F1 (16 classes)", BLUE, lambda d: d["classification"]["macro_f1_all_16"]),
        ("Repetition F1 (±400 ms)", TEAL, lambda d: d["repetitions"]["f1"]),
        ("Boundary-only F1 (±400 ms)", ORANGE, lambda d: d["segmentation"]["f1"]),
    ]
    x, width = np.arange(len(groups)), .235
    ax.axvspan(len(groups)-1.47, len(groups)-.53, color=LIGHT, zorder=0)
    for j, (label, color, metric) in enumerate(series):
        bars = ax.bar(x+(j-1)*width, [100*metric(d) for d in groups], width,
                      color=color, label=label)
        ax.bar_label(bars, fmt="%.1f", fontsize=8, padding=3)
    ax.set(xticks=x, xticklabels=[f"Fold {n}" for n, _ in reports]+["Pooled"],
           ylim=(0, 65), ylabel="F1 (%)", yticks=np.arange(0, 61, 20))
    ax.set_title("A   Performance across completed folds", loc="left", pad=14)
    style_axis(ax)
    ax.legend(loc="upper left", bbox_to_anchor=(-.03, -.17), ncol=3, handlelength=1.2,
              columnspacing=1.0, fontsize=7.4)
    ax = fig.add_axes([.655, .235, .31, .405])
    x, width = np.arange(3), .32
    for j, (tol, color) in enumerate((("200", "#98AAA9"), ("400", TEAL))):
        values = [100*pooled["by_tolerance"][tol]["repetitions"][m] for m in ("precision", "recall", "f1")]
        bars = ax.bar(x+(j-.5)*width, values, width, color=color, label=f"±{tol} ms")
        ax.bar_label(bars, fmt="%.1f", padding=4, fontsize=9)
    ax.set(xticks=x, xticklabels=["Precision", "Recall", "F1"], ylim=(0, 50),
           ylabel="Repetition score (%)", yticks=np.arange(0, 51, 10))
    ax.set_title("B   Endpoint tolerance", loc="left", pad=14)
    style_axis(ax)
    ax.legend(loc="upper left", bbox_to_anchor=(0, -.17), ncol=2)
    footer(fig, "Pooled scores recomputed from all selected test streams; they are not averages of fold percentages.\n"
           "Repetition matching requires the correct exercise and both endpoints within tolerance; boundary-only matching ignores exercise identity.\n"
           "Paired wrists are separate observations of the same workout. Plank holds are excluded from repetition metrics.")
    return fig


def recognition(pooled, stage):
    fig = plt.figure(figsize=(15.2, 9.0))
    heading(fig, "RecoFit | Exercise recognition", stage)
    ax = fig.add_axes([.14, .265, .49, .585])
    cm = np.asarray(pooled["classification"]["confusion_matrix"])
    normalized = np.divide(100*cm, cm.sum(axis=1, keepdims=True),
                           out=np.zeros_like(cm, dtype=float), where=cm.sum(axis=1, keepdims=True)>0)
    cmap = LinearSegmentedColormap.from_list("recofit_blues", ["#FAFCFD", "#DCE8F0", "#6694B5", "#224F75"])
    im = ax.imshow(normalized, vmin=0, vmax=100, cmap=cmap, aspect="auto")
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            value = normalized[i, j]
            if value >= 5 or i == j:
                ax.text(j, i, f"{value:.0f}", va="center", ha="center", fontsize=7.8,
                        color="white" if value >= 60 else INK, weight="bold" if i == j else "normal")
    names = [LABELS[name] for name in CLASSES]
    ax.set(xticks=range(16), yticks=range(16), xticklabels=names, yticklabels=names,
           xlabel="Predicted class", ylabel="Reference class")
    plt.setp(ax.get_xticklabels(), rotation=55, ha="right", rotation_mode="anchor")
    ax.tick_params(length=0, labelsize=8.5)
    ax.set_title("A   Confusion matrix · row-normalized (%)", loc="left", pad=15)
    ax.spines[["left", "bottom"]].set_visible(False)
    cax = fig.add_axes([.64, .265, .011, .585])
    cb = fig.colorbar(im, cax=cax)
    cb.set_ticks([0, 25, 50, 75, 100])
    cb.ax.tick_params(labelsize=8, length=0)
    cb.outline.set_visible(False)
    ax = fig.add_axes([.79, .265, .18, .585])
    classes = pooled["classification"]["per_class"]
    order = sorted(CLASSES, key=lambda name: classes[name]["f1"], reverse=True)
    vals = [100*classes[name]["f1"] for name in order]
    bars = ax.barh(np.arange(16), vals, height=.66,
                   color=["#85949F" if name == "NULL" else BLUE for name in order])
    ax.bar_label(bars, fmt="%.1f", padding=4, fontsize=8)
    ax.set(yticks=range(16), yticklabels=[LABELS[name] for name in order],
           xlim=(0, 104), xticks=[0, 25, 50, 75, 100], xlabel="Frame F1 (%)")
    ax.invert_yaxis()
    ax.set_title("B   F1 by class", loc="left", pad=15)
    style_axis(ax, "x")
    rest_pct = 100*cm[0].sum()/cm.sum()
    focal_note(ax, f"Macro F1: {100*pooled['classification']['macro_f1_all_16']:.1f}%",
               xy=(.98, -.20), size=9)
    footer(fig, f"All eligible test timestamps are scored. Rest accounts for {rest_pct:.1f}% of reference frames; macro F1 weights all 16 classes equally.\n"
           "Rows sum to 100%; diagonal cells and off-diagonal cells ≥5% are labeled. Small confusions remain visible in the color scale.")
    return fig


def exercise_results(pooled, stage):
    fig = plt.figure(figsize=(13.5, 7.8))
    heading(fig, "RecoFit | Repetition performance by exercise", stage + "  ·  ±400 ms")
    classes = pooled["per_exercise"]
    order = sorted(classes, key=lambda name: classes[name]["f1"], reverse=True)
    y = np.arange(len(order))
    ax = fig.add_axes([.15, .19, .24, .64])
    bars = ax.barh(y, [100*classes[name]["f1"] for name in order], color=TEAL, height=.65)
    ax.bar_label(bars, fmt="%.1f", fontsize=9, padding=4)
    ax.set(yticks=y, yticklabels=[LABELS[name] for name in order], xlim=(0, 100),
           xticks=[0, 25, 50, 75, 100], xlabel="Repetition F1 (%)")
    ax.invert_yaxis()
    ax.set_title("A   Exercise + boundaries", loc="left", pad=14)
    style_axis(ax, "x")
    ax = fig.add_axes([.445, .19, .215, .64])
    bars = ax.barh(y, [classes[name]["count_mae_active_station_sessions"] for name in order],
                   color=ORANGE, height=.65)
    ax.bar_label(bars, fmt="%.1f", fontsize=9, padding=4)
    ax.set(yticks=y, yticklabels=[], xlim=(0, 32), xticks=[0, 10, 20, 30],
           xlabel="Count MAE (repetitions)")
    ax.invert_yaxis()
    ax.set_title("B   Count error per active stream", loc="left", pad=14, fontsize=10.4)
    style_axis(ax, "x")
    ax = fig.add_axes([.715, .19, .25, .64])
    matched = np.array([classes[name]["tp"] for name in order])
    missed = np.array([classes[name]["missed"] for name in order])
    ax.barh(y, matched, color=TEAL, label="Matched", height=.65)
    ax.barh(y, missed, left=matched, color=GRAY, label="Missed", height=.65)
    for i, total in enumerate(matched+missed):
        ax.text(total+45, i, f"{total:,}", va="center", fontsize=8.5)
    ax.set(yticks=y, yticklabels=[], xlim=(0, max(matched+missed)*1.18),
           xticks=[0, 1000, 2000, 3000, 4000], xlabel="Reference repetitions (wrist observations)")
    ax.invert_yaxis()
    ax.set_title("C   Reference coverage", loc="left", pad=14)
    style_axis(ax, "x")
    ax.legend(loc="upper left", bbox_to_anchor=(0, -.105), ncol=2)
    focal_note(fig.axes[0], f"Pooled repetition F1: {100*pooled['repetitions']['f1']:.1f}%",
               xy=(0, -.115), ha="left", size=9)
    footer(fig, "Ordered by pooled repetition F1. Matching requires correct exercise identity and both endpoints within ±400 ms.\n"
           "Count MAE averages |predicted − reference| per exercise and wrist stream where either count is nonzero; it is not error per set.\n"
           "Plank is scored as a frame class but its holds are not included in repetition evaluation. Paired wrists can observe the same physical repetition.")
    return fig


def diagnostics(pooled, records, stage):
    fig = plt.figure(figsize=(13.5, 9.0))
    heading(fig, "RecoFit | Counting and reporting diagnostics", stage + "  ·  ±400 ms")
    rep = pooled["repetitions"]
    rows = rep["count_rows"]
    active = [row for row in rows if row["reference"] or row["predicted"]]
    delays = np.array([v for record in records for v in record["repetition_scores"]["400"]["repetitions"]["metrics"]["report_delays_ms"]])
    ax = fig.add_axes([.08, .56, .385, .265])
    limit = max(max(row["reference"], row["predicted"]) for row in active)*1.06
    ax.plot([0, limit], [0, limit], color="#92A0AA", lw=1, linestyle="--", label="Perfect count", zorder=0)
    for side, color, marker in (("left_wrist", BLUE, "o"), ("right_wrist", ORANGE, "^")):
        subset = [row for row in active if row["side"] == side]
        ax.scatter([row["reference"] for row in subset], [row["predicted"] for row in subset],
                   s=17, alpha=.4, color=color, marker=marker, edgecolors="none",
                   label=side.replace("_", " ").capitalize())
    ax.set(xlim=(0, limit), ylim=(0, limit), xlabel="Reference repetitions", ylabel="Predicted repetitions")
    ax.set_title("A   Count agreement per stream-exercise", loc="left", pad=13)
    style_axis(ax, "both")
    focal_note(ax, f"{len(active):,} active observations\nMAE = {rep['count_mae_active_station_sessions']:.2f} reps\n"
               f"Exact count = {100*rep['exact_count_accuracy_active_station_sessions']:.1f}%", size=8.5)
    ax.legend(loc="upper left", fontsize=8, markerscale=1.2)
    ax = fig.add_axes([.585, .56, .38, .265])
    metrics = [("Frame macro F1", lambda d: d["classification"]["macro_f1_all_16"]),
               ("Repetition F1", lambda d: d["repetitions"]["f1"]),
               ("Boundary-only F1", lambda d: d["segmentation"]["f1"])]
    x, width = np.arange(3), .32
    for j, (side, color) in enumerate((("left_wrist", BLUE), ("right_wrist", ORANGE))):
        d = pooled["per_side"][side]
        bars = ax.bar(x+(j-.5)*width, [100*fn(d) for _, fn in metrics], width, color=color,
                      label=f"{side.replace('_', ' ').capitalize()} (n={d['streams']} streams)")
        ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=8.5)
    ax.set(xticks=x, xticklabels=[label for label, _ in metrics], ylim=(0, 76),
           yticks=[0, 20, 40, 60], ylabel="F1 (%)")
    ax.set_title("B   Performance by wrist", loc="left", pad=13)
    style_axis(ax)
    ax.legend(loc="upper left", ncol=1, fontsize=8.2)
    ax = fig.add_axes([.08, .165, .385, .265])
    for key, label, color in (("start_errors_ms", "Start", BLUE), ("end_errors_ms", "End", TEAL)):
        values = np.array([v for record in records for v in record["repetition_scores"]["400"]["repetitions"]["metrics"][key]])
        unique, counts = np.unique(values, return_counts=True)
        ax.step(unique, 100*np.cumsum(counts)/len(values), where="post", color=color,
                linewidth=2, label=f"{label} · MAE {np.mean(values):.0f} ms")
    ax.set(xlim=(0, 400), ylim=(0, 105), xticks=[0, 100, 200, 300, 400], yticks=[0, 25, 50, 75, 100],
           xlabel="Absolute endpoint error (ms)", ylabel="Matched repetitions at or below error (%)")
    ax.set_title(f"C   Boundary error · {rep['tp']:,} matched repetitions", loc="left", pad=13)
    style_axis(ax, "both")
    ax.legend(loc="lower right", fontsize=8.5)
    ax = fig.add_axes([.585, .165, .38, .265])
    counts = [rep["missed"], int(np.count_nonzero((delays >= 0) & (delays < 1000))),
              int(np.count_nonzero(delays < 0)), int(np.count_nonzero(delays >= 1000))]
    reference_total = rep["tp"]+rep["missed"]
    if sum(counts) != reference_total:
        raise ValueError("Timing outcome categories must cover all reference repetitions")
    values = np.array(counts)*100/reference_total
    bars = ax.barh(range(4), values, color=[GRAY, TEAL, ORANGE, PURPLE], height=.57)
    ax.bar_label(bars, labels=[f"{v:.1f}% ({n:,})" for v, n in zip(values, counts)], padding=4, fontsize=8.7)
    ax.set(yticks=range(4), yticklabels=["Unmatched", "Matched: 0–999 ms", "Matched: <0 ms", "Matched: ≥1,000 ms"],
           xlim=(0, 86), xticks=[0, 20, 40, 60, 80], xlabel="Share of all reference repetitions (%)")
    ax.tick_params(axis="y", labelsize=8)
    ax.invert_yaxis()
    ax.set_title("D   Reporting time relative to reference end", loc="left", pad=13, fontsize=10.5)
    style_axis(ax, "x")
    footer(fig, "Boundary errors are conditional on a correct match at ±400 ms; unmatched repetitions are excluded from panel C.\n"
           f"Panel D includes all {reference_total:,} reference wrist observations: 0–999 ms reports cover {values[1]:.1f}% of references "
           f"({100*rep['matched_reported_0_to_999ms_fraction']:.1f}% of matches).\n"
           "Reporting time uses causal sample-time replay and excludes compute / transport delay. Negative values are reports before the reference end.")
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("baseline/RecoFit/runs/recofit_cv/cross_validation"))
    parser.add_argument("--data", type=Path, default=Path("strength_training/tas_data_both_wrists"))
    parser.add_argument("--output", type=Path, default=Path("baseline/RecoFit/figures/folds_1-4"))
    parser.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--planned-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--font", type=Path, help="Optional local Times New Roman font file")
    args = parser.parse_args()
    if len(set(args.folds)) != len(args.folds) or not all(1 <= n <= args.planned_folds for n in args.folds):
        parser.error("Select distinct fold numbers within the planned fold range")
    args.folds.sort()
    pooled, reports, records, source_paths, manifest = load_results(args)
    family, font_path = set_style(args.font)
    args.output.mkdir(parents=True, exist_ok=True)
    total_subjects = len({row["subject"] for row in manifest["sessions"]})
    references = pooled["repetitions"]["tp"]+pooled["repetitions"]["missed"]
    stage = (f"{len(args.folds)}/{args.planned_folds} folds · causal replay · "
             f"{pooled['subjects']}/{total_subjects} subjects · {pooled['streams']} wrist streams · "
             f"{references:,} reference repetition observations")
    captions = {
        "01_overview": "Pooled classification, repetition, and count metrics; variation across the four held-out folds; endpoint-tolerance sensitivity.",
        "02_recognition": "Row-normalized frame confusion matrix and per-class F1 for 15 exercises plus rest.",
        "03_exercise_repetitions": "Per-exercise repetition F1, count MAE per active wrist stream, and matched / missed reference counts at ±400 ms.",
        "04_counting_timing": "Count agreement, wrist-specific performance, matched boundary errors, and reporting-time outcomes relative to all references.",
    }
    figures = [overview(pooled, reports, stage), recognition(pooled, stage),
               exercise_results(pooled, stage), diagnostics(pooled, records, stage)]
    with PdfPages(args.output / "recofit_interim_report.pdf") as pdf:
        for (name, caption), fig in zip(captions.items(), figures):
            fig.savefig(args.output / f"{name}.png", dpi=300)
            fig.savefig(args.output / f"{name}.svg")
            pdf.savefig(fig)
            plt.close(fig)
    fold_rows = []
    for number, d in reports + [("pooled", pooled)]:
        fold_rows.append({"fold": number, "subjects": d["subjects"], "wrist_streams": d["streams"],
                          "frame_macro_f1": d["classification"]["macro_f1_all_16"],
                          "exercise_rest_f1": d["classification"]["exercise_rest_f1"],
                          "repetition_f1_400ms": d["repetitions"]["f1"],
                          "boundary_only_f1_400ms": d["segmentation"]["f1"],
                          "repetition_precision_400ms": d["repetitions"]["precision"],
                          "repetition_recall_400ms": d["repetitions"]["recall"],
                          "count_mae_active_stream_exercises": d["repetitions"]["count_mae_active_station_sessions"]})
    write_csv(args.output / "fold_metrics.csv", fold_rows)
    write_csv(args.output / "exercise_metrics.csv", [
        {"exercise": name, "frame_f1": pooled["classification"]["per_class"][name]["f1"],
         **{k: d[k] for k in ("precision", "recall", "f1", "tp", "extra", "missed", "count_mae_active_station_sessions")}}
        for name, d in pooled["per_exercise"].items()])
    write_csv(args.output / "count_observations.csv", pooled["repetitions"]["count_rows"])
    repo = Path(__file__).resolve().parents[2]
    source_paths.extend([Path(__file__).resolve(),
                         repo / "strength_training/temporal_segmentation/metrics.py",
                         repo / "strength_training/temporal_segmentation/data.py",
                         repo / "strength_training/temporal_segmentation/constants.py"])
    provenance = {
        "created_at": datetime.now(ZoneInfo("America/Vancouver")).isoformat(),
        "status": "interim" if len(args.folds) < args.planned_folds else "complete",
        "included_folds": args.folds, "planned_folds": args.planned_folds,
        "completed_fold_reports_validated": True, "subject_split_verified": True,
        "seed": args.seed, "font_family": family, "font_file": font_path,
        "font_note": "Times New Roman unavailable; DejaVu Serif used." if family != "Times New Roman" else "Times New Roman verified.",
        "png_dpi": 300, "captions": captions,
        "sources": [{"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                    for path in dict.fromkeys(source_paths)],
    }
    write_json(args.output / "provenance.json", provenance)
    write_json(args.output / "interim_report.json", {
        "report_scope": "interim_subject_wise_cross_validation", "included_folds": args.folds,
        "planned_folds": args.planned_folds, "mode": "causal", "timing_tolerance_ms": 400,
        "report_tolerances_ms": [200, 400],
        **{key: reports[0][1][key] for key in ("coverage_note", "latency_note", "boundary_note", "reference_convention")},
        **pooled})
    readme = f"""# RecoFit interim figures — folds {', '.join(map(str, args.folds))}

{stage}. This is an interim snapshot of the selected folds, not the final five-fold result.

## Figures

""" + "\n".join(f"- [{name}.png]({name}.png): {caption}" for name, caption in captions.items()) + f"""

Each figure is available as a 300 dpi PNG and a vector SVG with outlined text. All four figures are also in [recofit_interim_report.pdf](recofit_interim_report.pdf), with embedded fonts.

## Main values

- Frame macro F1 (all 16 classes): {100*pooled['classification']['macro_f1_all_16']:.2f}%.
- Exercise/rest frame F1: {100*pooled['classification']['exercise_rest_f1']:.2f}%.
- Repetition F1, precision, recall at ±400 ms: {100*pooled['repetitions']['f1']:.2f}%, {100*pooled['repetitions']['precision']:.2f}%, {100*pooled['repetitions']['recall']:.2f}%.
- Boundary-only repetition F1 at ±400 ms: {100*pooled['segmentation']['f1']:.2f}%.
- Count MAE: {pooled['repetitions']['count_mae_active_station_sessions']:.2f} repetitions per active exercise/wrist-stream observation.
- Matched start/end MAE: {pooled['repetitions']['start_mae_ms']:.1f}/{pooled['repetitions']['end_mae_ms']:.1f} ms.

## Interpretation and reproducibility

Pooled metrics are reconstructed using the repository's `grouped_stream_report` on saved test-stream scoring records. All reconstructed fold metrics were checked against the original fold reports, and the five-fold subject split (seed {args.seed}) was verified against the prepared manifest. Selected test subjects are disjoint and do not overlap training or validation subjects within their fold. Models were not retrained or rerun.

Repetition matching requires both endpoints within the stated tolerance and correct exercise identity. Boundary-only matching ignores exercise identity. Count MAE uses scored complete interval estimates, grouped by exercise and wrist stream; it is not per-set count error or offline peak-count error. An active observation has a nonzero reference or predicted count. Plank holds are excluded from repetition metrics.

Paired wrist streams are observations of the same workout, so reference repetition totals and durations do not count unique physical repetitions or independent participants. Timing errors apply only to matched repetitions. Replay delays exclude compute/transport time; the share reported within one second among matches is not overall recall. Panel D therefore uses all reference repetitions as its denominator.

Typography: {provenance['font_note']} The exact resolved font path, source hashes, and creation time are in [provenance.json](provenance.json). A local Times New Roman font can be supplied with `--font /path/to/times.ttf`.

Data: [fold_metrics.csv](fold_metrics.csv), [exercise_metrics.csv](exercise_metrics.csv), [count_observations.csv](count_observations.csv), and [interim_report.json](interim_report.json). The pooled report includes per-wrist and per-subject summaries.

From the repository root, regenerate this fixed snapshot with:

```bash
.venv/bin/python -m baseline.RecoFit.plot_results --folds {' '.join(map(str, args.folds))} --output {args.output}
```

Only the listed folds are included, even if another fold completes later. Choose a new output directory when reporting a later stage. The script overwrites only its own named report artifacts in the requested output directory.
"""
    (args.output / "README.md").write_text(readme)
    print(json.dumps({"output": str(args.output), "font": family, "validated_folds": args.folds,
                      "subjects": pooled["subjects"], "streams": pooled["streams"],
                      "repetition_f1": pooled["repetitions"]["f1"], "figure_count": len(figures)}))


if __name__ == "__main__":
    main()
