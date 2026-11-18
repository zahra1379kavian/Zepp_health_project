"""Render an auditable report from completed RecoFit test folds.

Run from the repository root with a Python environment containing matplotlib:
    .venv/bin/python -m baseline.RecoFit.plot_results

Use --compare-run and --compare-data to include a matched wrist + waist run.
By default, include every completed fold and label the report final only when
all planned folds are present. This only reads saved predictions; it does not
fit models or run inference.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shlex
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
from matplotlib.patches import FancyBboxPatch, Patch
import numpy as np

from baseline.mstcn.constants import CLASSES
from baseline.mstcn import WRIST_DATASET
from baseline.mstcn.data import subject_folds
from baseline.mstcn.metrics import grouped_stream_report


BLUE = "#315E88"
TEAL = "#21857D"
ORANGE = "#C17B35"
PURPLE = "#7C6494"
INK = "#263541"
GRAY = "#DAE0E4"
LIGHT = "#F3F6F8"
SENSOR_SERIES = (("wrist", "Wrist only", BLUE), ("wrist-waist", "Wrist + waist", TEAL))
COUNT_MAE = "count_mae_active_station_sessions"
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


def rest_frame_false_alarm_pct(report):
    rest_index = CLASSES.index("NULL")
    rest_predictions = report["classification"]["confusion_matrix"][rest_index]
    rest_frames = sum(rest_predictions)
    return 100 * (rest_frames - rest_predictions[rest_index]) / rest_frames if rest_frames else None


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
        if report.get("sensors", "wrist") != training.get("sensors", "wrist"):
            raise ValueError(f"Sensor configuration differs between training and test in fold {number}")
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
    if len(args.folds) == args.planned_folds:
        if {record["session"] for record in records} != set(metadata):
            raise ValueError("Completed cross-validation must test every prepared stream")
        final_report_path = args.run / "report.json"
        if final_report_path.exists():
            same_values(pooled, read_json(final_report_path), "final_report")
            source_paths.append(final_report_path)
    return pooled, reports, records, source_paths, manifest


def validate_comparison(primary, paired):
    """Check the saved scoring populations before comparing sensor configurations."""
    _, wrist_reports, wrist_records, _, _ = primary
    _, paired_reports, paired_records, _, _ = paired
    routing = paired_reports[0][1].get("counting_sensors")
    if (any(report.get("sensors", "wrist") != "wrist" for _, report in wrist_reports)
            or not routing
            or any(report.get("sensors") != "wrist-waist" or report.get("counting_sensors") != routing
                   for _, report in paired_reports)):
        raise ValueError("Expected wrist-only and wrist + waist runs with consistent counting sensor routing")
    for wrist, waist in zip(wrist_records, paired_records, strict=True):
        for key in ("session", "source_session", "subject", "side", "rest_hours"):
            if wrist[key] != waist[key]:
                raise ValueError(f"Comparison streams differ for {wrist['session']}: {key}")
        reference_frames = [np.asarray(record["confusion_matrix"]).sum(axis=1) for record in (wrist, waist)]
        if not np.array_equal(*reference_frames):
            raise ValueError(f"Reference frame totals differ for {wrist['session']}")
        reference_counts = [
            {row["station"]: row["reference"]
             for row in record["repetition_scores"].get("400", {}).get("repetitions", {}).get("counts", [])}
            for record in (wrist, waist)
        ]
        if reference_counts[0] != reference_counts[1]:
            raise ValueError(f"Reference repetition totals differ for {wrist['session']}")
    return routing


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


def overview(pooled, reports, stage, complete=False):
    fig = plt.figure(figsize=(12.6, 7.1))
    heading(fig, f"RecoFit | {'Final' if complete else 'Interim'} held-out performance", stage)
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


def confusion_panel(ax, cm, *, show_reference_labels=True, cell_font=7.8):
    normalized = np.divide(100*cm, cm.sum(axis=1, keepdims=True),
                           out=np.zeros_like(cm, dtype=float), where=cm.sum(axis=1, keepdims=True)>0)
    cmap = LinearSegmentedColormap.from_list("recofit_blues", ["#FAFCFD", "#DCE8F0", "#6694B5", "#224F75"])
    im = ax.imshow(normalized, vmin=0, vmax=100, cmap=cmap, aspect="auto")
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            value = normalized[i, j]
            if value >= 5 or i == j:
                ax.text(j, i, f"{value:.0f}", va="center", ha="center", fontsize=cell_font,
                        color="white" if value >= 60 else INK, weight="bold" if i == j else "normal")
    names = [LABELS[name] for name in CLASSES]
    ax.set(xticks=range(16), yticks=range(16), xticklabels=names, yticklabels=names,
           xlabel="Predicted class", ylabel="Reference class")
    if not show_reference_labels:
        ax.tick_params(labelleft=False)
        ax.set_ylabel("")
    plt.setp(ax.get_xticklabels(), rotation=55, ha="right", rotation_mode="anchor")
    ax.tick_params(length=0, labelsize=8.5)
    ax.spines[["left", "bottom"]].set_visible(False)
    return im


def recognition(pooled, stage):
    fig = plt.figure(figsize=(15.2, 9.0))
    heading(fig, "RecoFit | Exercise recognition", stage)
    ax = fig.add_axes([.14, .265, .49, .585])
    cm = np.asarray(pooled["classification"]["confusion_matrix"])
    im = confusion_panel(ax, cm)
    ax.set_title("A   Confusion matrix · row-normalized (%)", loc="left", pad=15)
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


def comparison_legend(fig, y=.88):
    handles = [Patch(color=color, label=label) for _, label, color in SENSOR_SERIES]
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(.973, y),
               ncol=2, handlelength=1.3)


def comparison_bars(ax, series, *, horizontal=False, decimals=None):
    positions, width = np.arange(len(series[0])), .34
    for index, (values, (_, _, color)) in enumerate(zip(series, SENSOR_SERIES, strict=True)):
        centers = positions + (index-.5)*width
        values = [np.nan if value is None else value for value in values]
        bars = (ax.barh(centers, values, height=width, color=color) if horizontal
                else ax.bar(centers, values, width=width, color=color))
        if decimals is not None:
            ax.bar_label(bars, labels=[f"{value:.{decimals}f}" if np.isfinite(value) else ""
                                       for value in values], fontsize=8, padding=3)
    style_axis(ax, "x" if horizontal else "y")


def bar_limit(series, minimum, padding=1.15):
    largest = max((value for row in series for value in row if value is not None), default=0)
    return max(minimum, largest * padding)


def comparison_overview(runs, stage, complete):
    fig = plt.figure(figsize=(13.6, 7.2))
    heading(fig, f"RecoFit | {'Final' if complete else 'Interim'} held-out performance", stage)
    metrics = (
        ("Frame macro F1 (%)", lambda d: 100*d["classification"]["macro_f1_all_16"], ".1f"),
        ("Repetition F1 at ±400 ms (%)", lambda d: 100*d["repetitions"]["f1"], ".1f"),
        ("Count MAE (repetitions)", lambda d: d["repetitions"][COUNT_MAE], ".2f"),
        ("False alarm rate (% of rest frames)", rest_frame_false_alarm_pct, ".2f"),
    )
    for index, (label, metric, precision) in enumerate(metrics):
        x, y, width = .043+index*.234, .695, .213
        box = FancyBboxPatch((x, y), width, .106, boxstyle="round,pad=0.008,rounding_size=0.008",
                             transform=fig.transFigure, facecolor=LIGHT, edgecolor="none")
        if index == 1:
            box.set_path_effects([pe.SimplePatchShadow(offset=(1.4, -1.4), alpha=.10), pe.Normal()])
        fig.add_artist(box)
        for offset, (run, (_, _, color)) in zip((.014, .115), zip(runs, SENSOR_SERIES, strict=True)):
            value = metric(run[0])
            text = "—" if value is None else format(value, precision)
            fig.text(x+offset, y+.063, text, fontsize=21, color=color, weight="bold", va="center")
        fig.text(x+.014, y+.020, label, fontsize=9, va="center")
    comparison_legend(fig, .872)
    reports = [run[1] for run in runs]
    labels = [f"Fold {n}" for n, _ in reports[0]]
    fold_scores = [[100*report["repetitions"]["f1"] for _, report in fold_reports] for fold_reports in reports]
    tolerances = ("200", "400")
    names = ("precision", "recall", "f1")
    tolerance_scores = [[100*run[0]["by_tolerance"][tol]["repetitions"][metric]
                         for tol in tolerances for metric in names] for run in runs]
    limit = bar_limit(fold_scores + tolerance_scores, 53)
    ticks = np.arange(0, min(101, limit), 10)
    ax = fig.add_axes([.075, .235, .42, .345])
    comparison_bars(ax, fold_scores, decimals=1)
    ax.set(xticks=range(len(labels)), xticklabels=labels, ylim=(0, limit), ylabel="Repetition F1 (%)", yticks=ticks)
    ax.set_title("A   Repetition F1 across held-out folds at ±400 ms", loc="left", pad=13)
    ax = fig.add_axes([.570, .235, .39, .345])
    comparison_bars(ax, tolerance_scores, decimals=1)
    ax.axvline(2.5, color=GRAY, lw=.8)
    ax.set(xticks=range(6), xticklabels=["Precision", "Recall", "F1"]*2,
           ylim=(0, limit), ylabel="Repetition score (%)", yticks=ticks)
    for x, tol in zip((.25, .75), tolerances):
        ax.text(x, -.22, f"±{tol} ms", ha="center", va="top", transform=ax.transAxes, fontsize=9)
    ax.set_title("B   Pooled scores by endpoint tolerance", loc="left", pad=13)
    footer(fig, "Both runs use the same held-out streams. Pooled metrics use all test streams together.\n"
           "A repetition match requires the correct exercise and both endpoints within tolerance. Count MAE uses active exercise / wrist-stream pairs.\n"
           "Left and right wrists can observe the same workout. Plank holds are excluded from repetition metrics.")
    return fig


def comparison_recognition(runs, stage):
    fig = plt.figure(figsize=(15.6, 8.1))
    heading(fig, "RecoFit | Exercise recognition by sensor configuration", stage)
    schemes = [run[0]["classification"] for run in runs]
    for index, (scheme, (_, label, _)) in enumerate(zip(schemes, SENSOR_SERIES, strict=True)):
        ax = fig.add_axes([.103+index*.260, .28, .239, .54])
        im = confusion_panel(ax, np.asarray(scheme["confusion_matrix"]),
                             show_reference_labels=index == 0, cell_font=7.1)
        ax.set_title(f"{'AB'[index]}   {label} · row-normalized (%)", loc="left", pad=12)
    cb = fig.colorbar(im, cax=fig.add_axes([.612, .28, .010, .54]), ticks=[0, 25, 50, 75, 100])
    cb.ax.tick_params(labelsize=8, length=0)
    cb.outline.set_visible(False)
    order = sorted(CLASSES, key=lambda name: schemes[0]["per_class"][name]["f1"], reverse=True)
    ax = fig.add_axes([.748, .28, .219, .54])
    y = np.arange(len(order))
    values = [[100*scheme["per_class"][name]["f1"] for name in order] for scheme in schemes]
    ax.hlines(y, *values, color="#AAB6BF", lw=1.4, zorder=2)
    for scores, (_, label, color) in zip(values, SENSOR_SERIES, strict=True):
        ax.scatter(scores, y, color=color, label=label, s=26, zorder=3)
    ax.set(yticks=y, yticklabels=[LABELS[name] for name in order], xlim=(0, 103),
           xticks=[0, 25, 50, 75, 100], xlabel="Frame F1 (%)")
    ax.invert_yaxis()
    for name, tick in zip(order, ax.get_yticklabels(), strict=True):
        if name == "leg_raise":
            tick.set_weight("bold")
    ax.set_title("C   Frame F1 by class", loc="left", pad=12)
    style_axis(ax, "x")
    ax.legend(loc="upper right", bbox_to_anchor=(1, -.15), ncol=2, columnspacing=.9, handletextpad=.3)
    rest_pct = 100*schemes[0]["per_class"]["NULL"]["support"] / sum(
        metrics["support"] for metrics in schemes[0]["per_class"].values())
    rest_f1 = " and ".join(f"{100*scheme['exercise_rest_f1']:.1f}%" for scheme in schemes)
    footer(fig, f"Rest accounts for {rest_pct:.1f}% of reference frames. Exercise/rest F1 is {rest_f1} for wrist only and wrist + waist, respectively.\n"
           "Confusion matrices show percentages within each reference class. Diagonal cells and off-diagonal cells ≥5% are labeled.")
    return fig


def comparison_exercises(runs, stage, counting_sensors):
    fig = plt.figure(figsize=(13.7, 8.0))
    heading(fig, "RecoFit | Repetition performance by exercise", stage + "  ·  ±400 ms")
    schemes = [run[0]["per_exercise"] for run in runs]
    order = sorted(schemes[0], key=lambda name: schemes[0][name]["f1"], reverse=True)
    leg_raise_uses_waist = counting_sensors.get("leg_raise") == "waist"
    comparison_legend(fig, .886)
    percent_limit = bar_limit([[100*scheme[name][key] for name in order for key in ("f1", "recall")]
                              for scheme in schemes], 89)
    for index, (title, label, x, metric, scale, tick_step) in enumerate((
        ("A   Repetition F1", "F1 (%)", .130, "f1", 100, 20),
        ("B   Count error", "Count MAE (repetitions)", .423, COUNT_MAE, 1, 10),
        ("C   Reference repetitions matched", "Recall (%)", .716, "recall", 100, 20),
    )):
        ax = fig.add_axes([x, .176, .25, .625])
        if leg_raise_uses_waist:
            row = order.index("leg_raise")
            ax.axhspan(row-.49, row+.49, color=LIGHT, zorder=0)
        scores = [[scheme[name].get(metric) for name in order] for scheme in schemes]
        values = [[scale*value if value is not None else None for value in row] for row in scores]
        comparison_bars(ax, values, horizontal=True, decimals=1)
        limit = percent_limit if scale == 100 else bar_limit(values, 32)
        ticks = np.arange(0, min(101, limit) if scale == 100 else limit, tick_step)
        ax.set(yticks=range(len(order)), yticklabels=[LABELS[name] for name in order] if index == 0 else [],
               ylim=(len(order)-.5, -.5), xlim=(0, limit), xticks=ticks, xlabel=label)
        ax.set_title(title, loc="left", pad=12)
    waist_note = "The paired counter uses waist acceleration for leg raises. " if leg_raise_uses_waist else ""
    footer(fig, "A match requires the correct exercise and both endpoints within ±400 ms. Recall uses all reference repetitions for each exercise.\n"
           "Count MAE averages the absolute count difference per exercise and wrist stream with a nonzero reference or predicted count.\n"
           f"{waist_note}Plank holds are excluded from repetition metrics.")
    return fig


def repetition_samples(records, field):
    return np.array([value for record in records if "400" in record["repetition_scores"]
                     for value in record["repetition_scores"]["400"]["repetitions"]["metrics"][field]])


def comparison_timing(runs, stage):
    fig = plt.figure(figsize=(14.7, 9.0))
    heading(fig, "RecoFit | Count agreement and reporting time", stage + "  ·  ±400 ms")
    comparison_legend(fig, .888)
    reps = [run[0]["repetitions"] for run in runs]
    active = [[row for row in rep["count_rows"] if row["reference"] or row["predicted"]] for rep in reps]
    limit = max(max(row["reference"], row["predicted"]) for rows in active for row in rows)*1.06
    for index, (rows, rep, (_, label, color)) in enumerate(zip(active, reps, SENSOR_SERIES, strict=True)):
        ax = fig.add_axes([.067+index*.326, .555, .245, .256])
        ax.plot([0, limit], [0, limit], color="#92A0AA", lw=1, linestyle="--", zorder=0)
        ax.scatter([row["reference"] for row in rows], [row["predicted"] for row in rows],
                   s=14, alpha=.34, color=color, edgecolors="none", zorder=2)
        ax.set(xlim=(0, limit), ylim=(0, limit), xlabel="Reference repetitions", ylabel="Predicted repetitions")
        ax.set_title(f"{'AB'[index]}   {label}", loc="left", pad=12)
        style_axis(ax, "both")
        if rep[COUNT_MAE] is not None:
            ax.text(.965, .94, f"MAE {rep[COUNT_MAE]:.2f} reps\n"
                    f"Exact count {100*rep['exact_count_accuracy_active_station_sessions']:.1f}%",
                    transform=ax.transAxes, ha="right", va="top", fontsize=8.5,
                    bbox={"boxstyle": "round,pad=.3", "fc": LIGHT, "ec": "none"})
    ax = fig.add_axes([.718, .555, .245, .256])
    scores = [[100*metric(run[0]["per_side"][side]) for side in ("left_wrist", "right_wrist")
               for metric in (lambda d: d["classification"]["macro_f1_all_16"],
                              lambda d: d["repetitions"]["f1"])] for run in runs]
    comparison_bars(ax, scores, decimals=1)
    ax.axvline(1.5, color=GRAY, lw=.8)
    limit = bar_limit(scores, 82)
    ax.set(xticks=range(4), xticklabels=["Frame", "Rep."]*2, ylim=(0, limit),
           yticks=np.arange(0, min(101, limit), 20), ylabel="F1 (%)")
    for x, side in zip((.25, .75), ("Left wrist", "Right wrist")):
        ax.text(x, -.20, side, ha="center", va="top", transform=ax.transAxes, fontsize=9)
    ax.set_title("C   Performance by wrist", loc="left", pad=12)
    ax = fig.add_axes([.067, .17, .39, .233])
    for run, (_, label, color) in zip(runs, SENSOR_SERIES, strict=True):
        for key, linestyle in (("start", "-"), ("end", "--")):
            errors = repetition_samples(run[2], f"{key}_errors_ms")
            if errors.size:
                unique, counts = np.unique(errors, return_counts=True)
                ax.step(unique, 100*np.cumsum(counts)/errors.size, where="post",
                        color=color, linestyle=linestyle, linewidth=1.7,
                        label=f"{label} · {key} ({np.mean(errors):.0f} ms MAE)")
    ax.set(xlim=(0, 400), ylim=(0, 102), xticks=[0, 100, 200, 300, 400], yticks=[0, 25, 50, 75, 100],
           xlabel="Absolute endpoint error (ms)", ylabel="Matches at or below error (%)")
    ax.set_title("D   Endpoint error on matched repetitions", loc="left", pad=12)
    style_axis(ax, "both")
    ax.legend(loc="lower right", fontsize=8.1)
    ax = fig.add_axes([.620, .17, .343, .233])
    outcomes = []
    for run, rep in zip(runs, reps, strict=True):
        delays = repetition_samples(run[2], "report_delays_ms")
        counts = [rep["missed"], np.count_nonzero(delays < 1000), np.count_nonzero(delays >= 1000)]
        reference_total = rep["tp"]+rep["missed"]
        if sum(counts) != reference_total:
            raise ValueError("Timing outcome categories must cover all reference repetitions")
        outcomes.append([100*count/reference_total for count in counts])
    comparison_bars(ax, outcomes, horizontal=True, decimals=1)
    ax.set(yticks=range(3), yticklabels=["Unmatched", "Matched before 1 s deadline", "Matched at or after deadline"],
           ylim=(2.55, -.55), xlim=(0, max(map(max, outcomes)) + 10),
           xlabel="Share of all reference repetitions (%)")
    ax.set_title("E   Reporting time relative to reference end", loc="left", pad=12)
    footer(fig, "Each dot is an active exercise / wrist-stream count. The dashed diagonal indicates equal predicted and reference counts.\n"
           "Endpoint errors use matched repetitions. Reporting outcomes use all annotated repetitions, including misses.\n"
           "The reporting deadline is 1 s after the annotated rep end. Sample-time replay excludes computation and transport time.")
    return fig


def comparison_summary(runs):
    metrics = (
        ("Frame macro F1", lambda d: 100*d["classification"]["macro_f1_all_16"], ".2f", "%"),
        ("Exercise/rest frame F1", lambda d: 100*d["classification"]["exercise_rest_f1"], ".2f", "%"),
        ("Repetition F1 at ±400 ms", lambda d: 100*d["repetitions"]["f1"], ".2f", "%"),
        ("Repetition precision at ±400 ms", lambda d: 100*d["repetitions"]["precision"], ".2f", "%"),
        ("Repetition recall at ±400 ms", lambda d: 100*d["repetitions"]["recall"], ".2f", "%"),
        ("Boundary-only F1 at ±400 ms", lambda d: 100*d["segmentation"]["f1"], ".2f", "%"),
        ("Count MAE", lambda d: d["repetitions"][COUNT_MAE], ".2f", " reps"),
        ("False alarm rate (% of rest frames)", rest_frame_false_alarm_pct, ".2f", "%"),
        ("Leg-raise frame F1", lambda d: 100*d["classification"]["per_class"]["leg_raise"]["f1"], ".2f", "%"),
        ("Leg-raise repetition F1 at ±400 ms", lambda d: 100*d["per_exercise"]["leg_raise"]["f1"], ".2f", "%"),
        ("Leg-raise count MAE", lambda d: d["per_exercise"]["leg_raise"][COUNT_MAE], ".2f", " reps"),
    )
    rows = ["| Metric | Wrist only | Wrist + waist |", "| --- | ---: | ---: |"]
    for name, metric, precision, unit in metrics:
        values = [metric(run[0]) for run in runs]
        cells = ["—" if value is None else format(value, precision)+unit for value in values]
        rows.append("| " + " | ".join([name, *cells]) + " |")
    return "\n".join(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("baseline/RecoFit/runs/recofit_cv/cross_validation"))
    parser.add_argument("--data", type=Path, default=WRIST_DATASET)
    parser.add_argument("--compare-run", type=Path, help="Wrist + waist cross-validation output")
    parser.add_argument("--compare-data", type=Path, help="Prepared wrist + waist dataset")
    parser.add_argument("--output", type=Path, help="Defaults to figures/folds_<selected folds>")
    parser.add_argument("--folds", type=int, nargs="+", help="Defaults to all completed test folds")
    parser.add_argument("--planned-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--font", type=Path, help="Optional local Times New Roman font file")
    args = parser.parse_args()
    if bool(args.compare_run) != bool(args.compare_data):
        parser.error("--compare-run and --compare-data are required together")
    if args.folds is None:
        args.folds = [n for n in range(1, args.planned_folds + 1)
                      if (args.run / f"fold_{n}/test/report.json").is_file()]
    if not args.folds:
        parser.error("No completed test folds found")
    if len(set(args.folds)) != len(args.folds) or not all(1 <= n <= args.planned_folds for n in args.folds):
        parser.error("Select distinct fold numbers within the planned fold range")
    args.folds.sort()
    complete = len(args.folds) == args.planned_folds
    if args.output is None:
        fold_tag = (f"{args.folds[0]}-{args.folds[-1]}"
                    if len(args.folds) > 1 and args.folds == list(range(args.folds[0], args.folds[-1]+1))
                    else "_".join(map(str, args.folds)))
        args.output = Path("baseline/RecoFit/figures") / f"folds_{fold_tag}"
    primary = load_results(args)
    pooled, reports, records, source_paths, manifest = primary
    runs = (primary,)
    counting_sensors = None
    if args.compare_run:
        compare_args = argparse.Namespace(**{**vars(args), "run": args.compare_run, "data": args.compare_data})
        paired = load_results(compare_args)
        counting_sensors = validate_comparison(primary, paired)
        runs = (primary, paired)
        source_paths = [*source_paths, *paired[3]]
    family, font_path = set_style(args.font)
    args.output.mkdir(parents=True, exist_ok=True)
    total_subjects = len({row["subject"] for row in manifest["sessions"]})
    references = pooled["repetitions"]["tp"]+pooled["repetitions"]["missed"]
    stage = (f"{len(args.folds)}/{args.planned_folds} folds · causal replay · "
             f"{pooled['subjects']}/{total_subjects} subjects · {pooled['streams']} wrist streams · "
             f"{references:,} reference repetition observations")
    captions = {
        "01_overview": f"Pooled classification, repetition, and count metrics; variation across the {len(args.folds)} held-out folds; endpoint-tolerance sensitivity.",
        "02_recognition": "Row-normalized frame confusion matrix and per-class F1 for 15 exercises plus rest.",
        "03_exercise_repetitions": "Per-exercise repetition F1, count MAE per active wrist stream, and matched / missed reference counts at ±400 ms.",
        "04_counting_timing": "Count agreement, wrist-specific performance, matched boundary errors, and reporting-time outcomes relative to all references.",
    }
    if args.compare_run:
        figures = [comparison_overview(runs, stage, complete), comparison_recognition(runs, stage),
                   comparison_exercises(runs, stage, counting_sensors), comparison_timing(runs, stage)]
        captions = {
            "01_overview": "Pooled classification, repetition, count error, and rest-frame false alarm rates for both sensor configurations, with held-out fold scores and endpoint-tolerance sensitivity.",
            "02_recognition": "Side-by-side row-normalized confusion matrices and frame F1 for 15 exercises plus rest.",
            "03_exercise_repetitions": "Repetition F1, count MAE per active exercise and wrist stream, and recall by exercise at ±400 ms.",
            "04_counting_timing": "Count agreement by sensor configuration, wrist-specific performance, matched endpoint errors, and reporting-time outcomes across all reference repetitions.",
        }
    else:
        figures = [overview(pooled, reports, stage, complete), recognition(pooled, stage),
                   exercise_results(pooled, stage), diagnostics(pooled, records, stage)]
    with PdfPages(args.output / "recofit_report.pdf") as pdf:
        for name, fig in zip(captions, figures, strict=True):
            fig.savefig(args.output / f"{name}.png", dpi=300)
            fig.savefig(args.output / f"{name}.svg")
            pdf.savefig(fig)
            plt.close(fig)
    fold_rows, exercise_rows, count_rows = [], [], []
    for (result, fold_reports, _, _, _), (sensors, _, _) in zip(runs, SENSOR_SERIES):
        model_field = {"sensors": sensors} if args.compare_run else {}
        for number, d in fold_reports + [("pooled", result)]:
            fold_rows.append({**model_field, "fold": number, "subjects": d["subjects"], "wrist_streams": d["streams"],
                              "frame_macro_f1": d["classification"]["macro_f1_all_16"],
                              "exercise_rest_f1": d["classification"]["exercise_rest_f1"],
                              "repetition_f1_400ms": d["repetitions"]["f1"],
                              "boundary_only_f1_400ms": d["segmentation"]["f1"],
                              "repetition_precision_400ms": d["repetitions"]["precision"],
                              "repetition_recall_400ms": d["repetitions"]["recall"],
                              "count_mae_active_stream_exercises": d["repetitions"][COUNT_MAE],
                              "rest_frame_false_alarm_pct": rest_frame_false_alarm_pct(d)})
        exercise_rows.extend(
            {**model_field, "exercise": name, "frame_f1": result["classification"]["per_class"][name]["f1"],
             **{k: d[k] for k in ("precision", "recall", "f1", "tp", "extra", "missed", COUNT_MAE)}}
            for name, d in result["per_exercise"].items())
        count_rows.extend({**model_field, **row} for row in result["repetitions"]["count_rows"])
    write_csv(args.output / "fold_metrics.csv", fold_rows)
    write_csv(args.output / "exercise_metrics.csv", exercise_rows)
    write_csv(args.output / "count_observations.csv", count_rows)
    repo = Path(__file__).resolve().parents[2]
    source_paths.extend([Path(__file__).resolve(),
                         repo / "baseline/MS-TCN/mstcn/metrics.py",
                         repo / "baseline/MS-TCN/mstcn/data.py",
                         repo / "baseline/MS-TCN/mstcn/constants.py"])
    final_reports = [run / "report.json" for run in (args.run, args.compare_run) if run is not None]
    font_note = ("Times New Roman verified." if family == "Times New Roman" else
                 f"Times New Roman unavailable. Figures use {family}." if not args.font else
                 f"Figures use the supplied {family} font.")
    provenance = {
        "created_at": datetime.now(ZoneInfo("America/Vancouver")).isoformat(),
        "status": "complete" if complete else "interim",
        "included_folds": args.folds, "planned_folds": args.planned_folds,
        "completed_fold_reports_validated": True, "subject_split_verified": True,
        "final_pooled_report_validated": all(path in source_paths for path in final_reports),
        "seed": args.seed, "font_family": family, "font_file": font_path,
        "font_note": font_note,
        "png_dpi": 300, "captions": captions,
        "sources": [{"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                    for path in dict.fromkeys(source_paths)],
    }
    report_keys = ("coverage_note", "latency_note", "boundary_note", "reference_convention")
    pooled_false_alarm_pct = rest_frame_false_alarm_pct(pooled)
    report_output = {
        "report_scope": "subject_wise_cross_validation" if complete else "interim_subject_wise_cross_validation",
        "status": "complete" if complete else "interim", "included_folds": args.folds,
        "planned_folds": args.planned_folds, "mode": "causal", "timing_tolerance_ms": 400,
        "report_tolerances_ms": [200, 400],
        **{key: reports[0][1][key] for key in report_keys},
        **pooled,
        "rest_frame_false_alarm_pct": pooled_false_alarm_pct,
    }
    if args.compare_run:
        paired_pooled, paired_folds, _, _, _ = runs[1]
        report_output["sensors"] = "wrist"
        report_output["comparison"] = {
            "sensors": "wrist-waist", "counting_sensors": counting_sensors,
            **{key: paired_folds[0][1][key] for key in report_keys},
            **paired_pooled,
            "rest_frame_false_alarm_pct": rest_frame_false_alarm_pct(paired_pooled),
        }
        provenance["comparison"] = {
            "wrist_run": str(args.run.resolve()), "wrist_waist_run": str(args.compare_run.resolve()),
            "reference_totals_verified_per_stream": True, "counting_sensors": counting_sensors,
        }
    write_json(args.output / "provenance.json", provenance)
    write_json(args.output / "report.json", report_output)
    scope_note = (f"This report covers all {args.planned_folds} planned cross-validation folds."
                  if complete else "This is an interim snapshot of the selected folds.")
    validation_note = (" The pooled metrics were also checked against the saved cross-validation reports."
                       if provenance["final_pooled_report_validated"] else "")
    false_alarm_display = "—" if pooled_false_alarm_pct is None else f"{pooled_false_alarm_pct:.2f}%"
    main_values = f"""- Frame macro F1 (all 16 classes) is {100*pooled['classification']['macro_f1_all_16']:.2f}%.
- Exercise/rest frame F1 is {100*pooled['classification']['exercise_rest_f1']:.2f}%.
- Repetition F1, precision, and recall at ±400 ms are {100*pooled['repetitions']['f1']:.2f}%, {100*pooled['repetitions']['precision']:.2f}%, and {100*pooled['repetitions']['recall']:.2f}%.
- Boundary-only repetition F1 at ±400 ms is {100*pooled['segmentation']['f1']:.2f}%.
- Count MAE is {pooled['repetitions'][COUNT_MAE]:.2f} repetitions per active exercise and wrist stream.
- False alarm rate (% of rest frames) is {false_alarm_display}.
- Matched start and end MAE are {pooled['repetitions']['start_mae_ms']:.1f} and {pooled['repetitions']['end_mae_ms']:.1f} ms."""
    comparison_note = ""
    command = [".venv/bin/python", "-m", "baseline.RecoFit.plot_results",
               "--run", str(args.run), "--data", str(args.data), "--folds", *map(str, args.folds),
               "--planned-folds", str(args.planned_folds), "--seed", str(args.seed), "--output", str(args.output)]
    if args.compare_run:
        main_values = comparison_summary(runs)
        waist_exercises = ", ".join(LABELS[station].lower() for station, sensor in counting_sensors.items()
                                   if sensor == "waist")
        counting_note = (f" The counter uses waist acceleration for these predicted exercises ({waist_exercises})."
                         if waist_exercises else "")
        comparison_note = (" Both runs were checked for the same held-out streams, class support, rest time, "
                           "and reference repetition totals per stream. The paired classifiers use both IMUs."
                           + counting_note)
        command.extend(["--compare-run", str(args.compare_run), "--compare-data", str(args.compare_data)])
    if args.font:
        command.extend(["--font", str(args.font)])
    readme = f"""# RecoFit {'final' if complete else 'interim'} figures — folds {', '.join(map(str, args.folds))}

{stage}. {scope_note}

## Figures

""" + "\n".join(f"- [{name}.png]({name}.png) · [SVG]({name}.svg)\n  {caption}"
                 for name, caption in captions.items()) + f"""

Each figure is available as a 300 dpi PNG and a vector SVG with outlined text. All four figures are also in [recofit_report.pdf](recofit_report.pdf), with embedded fonts.

## Main values

{main_values}

## Interpretation and reproducibility

Pooled metrics were reconstructed from saved test-stream scoring records. The reconstructed fold metrics match the saved fold reports, and the {args.planned_folds}-fold subject split (seed {args.seed}) was verified against the prepared manifest.{validation_note}{comparison_note}

Repetition matching requires both endpoints within the stated tolerance and correct exercise identity. Boundary-only matching ignores exercise identity. Count MAE uses scored complete intervals grouped by exercise and wrist stream. An active observation has a nonzero reference or predicted count. Plank holds are excluded from repetition metrics.

The false alarm rate is the percentage of annotated rest frames predicted as exercise, pooled over the held-out wrist streams.

Left and right wrist streams can observe the same workout. Timing errors apply to matched repetitions. Reporting-time outcome percentages use all reference repetitions and replay delays exclude computation and transport time.

{provenance['font_note']} The font path, source hashes, and creation time are in [provenance.json](provenance.json). A local Times New Roman font can be supplied with `--font /path/to/times.ttf`.

The pooled report includes per-wrist and per-subject summaries. [Fold metrics](fold_metrics.csv) · [Exercise metrics](exercise_metrics.csv) · [Count observations](count_observations.csv) · [Pooled report](report.json)

Run the following command from the repository root to regenerate this snapshot.

```bash
{shlex.join(command)}
```

The explicit fold list reproduces the cohort in this report. Omit `--folds` to discover completed test folds from the primary run.
"""
    (args.output / "README.md").write_text(readme)
    print(json.dumps({"output": str(args.output), "font": family, "validated_folds": args.folds,
                      "subjects": pooled["subjects"], "streams": pooled["streams"],
                      "repetition_f1": pooled["repetitions"]["f1"], "figure_count": len(figures),
                      **({"wrist_waist_repetition_f1": runs[1][0]["repetitions"]["f1"]} if args.compare_run else {})}))


if __name__ == "__main__":
    main()
