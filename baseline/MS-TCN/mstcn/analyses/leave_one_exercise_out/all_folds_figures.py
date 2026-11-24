"""Figures for the participant-pooled phase-only holdout report."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.lines import Line2D


def plot(rows, output):
    output = Path(output)
    output.mkdir(exist_ok=True, parents=True)
    try:
        fm.findfont("Times New Roman", fallback_to_default=False)
        family = "Times New Roman"
    except ValueError:
        family = "DejaVu Serif"
    primary = sorted((r for r in rows if r["tolerance_ms"] == 400), key=lambda r: -r["seen_f1"])
    secondary = {r["exercise"]: r for r in rows if r["tolerance_ms"] == 200}
    labels = [r["exercise"].replace("_", " ").capitalize().replace("(ohp)", "(OHP)") for r in primary]
    colors = ("#315d70", "#a34e27")
    markers = [Line2D([], [], marker="o", linestyle="", markerfacecolor=fc, markeredgecolor=c, label=label)
               for c, fc, label in zip(colors, ("white", colors[1]), ("Seen in training", "Held out"))]

    def frame(title):
        fig, axes = plt.subplots(1, 2, figsize=(11, 6.5), sharey=True, layout="constrained")
        fig.suptitle(title + "\n14 exercise holdouts pooled across 5 participant folds", fontsize=16)
        for ax in axes:
            ax.set_yticks(range(len(labels)), labels=labels)
            ax.set_ylim(len(labels) - 0.5, -0.5)
            ax.spines[["top", "right", "left"]].set_visible(False)
            ax.tick_params(axis="y", length=0)
            ax.grid(axis="x", alpha=0.18)
            ax.set_axisbelow(True)
        axes[1].set_facecolor("#f6f9f9")
        return fig, axes

    def pairs(ax, scores, metric, scale):
        for i, r in enumerate(scores):
            seen, unseen = [r[f"{s}_{metric}"] * scale for s in ("seen", "held_out")]
            ax.plot([seen, unseen], [i, i], color="#aabbbf", lw=1.7)
            ax.scatter(seen, i, edgecolor=colors[0], facecolor="white", zorder=3)
            ax.scatter(unseen, i, color=colors[1], s=21, zorder=4)

    def save(fig, filename):
        temporary = output / (filename + ".tmp")
        fig.savefig(temporary, format="png", dpi=180, metadata={"Description": f"Typeface: {family}"})
        plt.close(fig)
        temporary.replace(output / filename)

    with plt.rc_context({"font.family": family, "font.size": 10}):
        fig, axes = frame("Repetition segmentation on seen and unseen exercises")
        for ax, tol in zip(axes, (200, 400)):
            scores = [secondary[r["exercise"]] for r in primary] if tol == 200 else primary
            pairs(ax, scores, "f1", 100)
            ax.set(xlim=(-3, 102), title=f"Start and end within ±{tol} ms", xlabel="Rep segmentation F1 (%)")
        axes[0].legend(handles=markers, loc="upper left", bbox_to_anchor=(0, -0.08), ncol=2, frameon=False)
        save(fig, "seen_vs_unseen_f1.png")

        fig, axes = frame("Repetition count error and missed repetitions")
        pairs(axes[0], primary, "count_mae", 1)
        axes[0].set(title="Count error by exercise", xlabel="Mean absolute count error (reps per wrist recording)")
        axes[0].legend(handles=markers, loc="upper left", bbox_to_anchor=(0, -0.08), ncol=2, frameon=False)
        hits = [r["held_out_tp"] for r in primary]
        misses = [r["held_out_missed"] for r in primary]
        y = range(len(primary))
        axes[1].barh(y, hits, height=0.55, color="#247275", label="Matched")
        axes[1].barh(y, misses, left=hits, height=0.55, color="#dbe5e7", label="Missed")
        axes[1].set(title="Held-out model at ±400 ms", xlabel="Labeled repetitions")
        axes[1].legend(loc="upper left", bbox_to_anchor=(0, -0.08), ncol=2, frameon=False)
        save(fig, "rep_counts_and_misses.png")
