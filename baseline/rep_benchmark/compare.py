"""Combine completed, commonly scored wrist runs into a CSV and a review PDF."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import textwrap

from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image
import pandas as pd

from . import report_directory
from .report import PUBLISHED, plt


def picture(ax, image_path):
    ax.axis("off")
    with Image.open(image_path) as image:
        image.thumbnail((1500, 950))
        ax.imshow(image.convert("RGB"))


def rates_table(ax, labels, cells, widths):
    ax.axis("off")
    table = ax.table(cellText=cells, colLabels=labels, cellLoc="center",
                     loc="center", colWidths=widths)
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.35)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#d9dfe4")
        if row == 0:
            cell.set_facecolor("#e7eef5")
            cell.set_text_props(weight="bold")
        if col == 0:
            cell.set_text_props(ha="left")


def compare(runs, output):
    runs = [Path(r) for r in runs]
    manifests = [json.loads((p / "manifest.json").read_text()) for p in runs]
    if not runs or len({m["protocol_sha256"] for m in manifests}) != 1:
        raise ValueError("Runs need identical session, fold, wrist, timestamp and label signatures")
    if any((m["raw_inputs_sha256"], m["reference_sha256"]) !=
           (manifests[0]["raw_inputs_sha256"], manifests[0]["reference_sha256"]) for m in manifests):
        raise ValueError("Raw recordings or reference versions differ between runs")
    if len({m["model"] for m in manifests}) != len(runs):
        raise ValueError("Use a distinct model name for each result")
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frames = []
    for run, m in zip(runs, manifests):
        for scope in ("pooled", "fold_5"):
            folder = report_directory(run, scope)
            frame = pd.read_csv(folder / "published_comparison.csv")
            counts = pd.read_csv(folder / "test_metrics_summary.csv")
            frame = frame.merge(counts[["activity", "total", "TP", "FN", "FP", "FP_rest", "FP_activity"]],
                                on="activity", validate="one_to_one")
            frames.append(frame.assign(model=m["model"], scope=scope))
    all_rows = pd.concat(frames, ignore_index=True)
    with TemporaryDirectory(dir=output.parent, prefix=".compare-") as tmp:
        stage = Path(tmp) / "report"
        stage.mkdir()
        all_rows.to_csv(stage / "model_comparison.csv", index=False)
        with PdfPages(stage / "wrist_model_comparison.pdf") as pdf:
            fig = plt.figure(figsize=(8.27, 8))
            fig.text(.07, .92, "Activity-agnostic wrist repetition detection", fontsize=19, weight="bold")
            fig.text(.07, .86, "Five-fold results across 14 exercises", fontsize=13)
            overview = all_rows[(all_rows.scope == "pooled") & (all_rows.activity == "Overall")]
            cells = [["Published wrist reference", "41,363", *[f"{v:.1f}" for v in PUBLISHED["Overall"][0]]]]
            for r in overview.itertuples():
                cells.append([r.model, f"{r.total:,}", f"{r.TP_pct:.1f}", f"{r.FN_pct:.1f}", f"{r.FP_pct:.1f}"])
            rates_table(fig.add_axes([.07, .50, .86, .32]), ["Model", "Labeled reps", "TP %", "FN %", "FP %"],
                        cells, [.4, .19, .13, .13, .13])
            m = manifests[0]
            sessions = m["sessions"]
            notes = [
                f"{len(sessions)} local sessions checked. {sum(s['clips'] > 0 for s in sessions)} "
                f"contributed {sum(s['clips'] for s in sessions):,} session–wrist–exercise clips.",
                "Each rate is per 100 labeled repetitions. The reference counts each labeled interval once "
                "and matches the unused prediction with the greatest positive overlap.",
                "Published values come from the August 31 Feishu wrist release, pp. 7–8. "
                "Its pooled figure gives 38,351 TP and 3,012 FN. The local checkpoints use their original "
                "subject splits. The local fifth fold and published final fold have different provenance.",
                "Unavailable release sessions: " + ", ".join(m["missing_sessions"])]
            y = .46
            for note in notes:
                lines = textwrap.wrap(note, 92)
                fig.text(.07, y, "\n".join(lines), fontsize=10, va="top", linespacing=1.4)
                y -= .029 * len(lines) + .025
            pdf.savefig(fig)
            plt.close(fig)
            for (run, m) in zip(runs, manifests):
                for scope, reference_scope in (("pooled", "five-fold"), ("fold_5", "final fold")):
                    data = all_rows[(all_rows.model == m["model"]) & (all_rows.scope == scope)]
                    fig = plt.figure(figsize=(8.27, 10.5))
                    title = "Five folds pooled" if scope == "pooled" else "Local fold 5"
                    fig.text(.08, .95, f"{m['model']} · {title}", fontsize=17, weight="bold")
                    fig.text(.08, .912, f"Reference columns show the published {reference_scope} values.", fontsize=10)
                    cols = ["Exercise", "TP %", "FN %", "FP %", "Ref TP %", "Ref FN %", "Ref FP %"]
                    cells = [[r.activity, *[f"{v:.1f}" for v in (r.TP_pct, r.FN_pct, r.FP_pct,
                              r.published_TP_pct, r.published_FN_pct, r.published_FP_pct)]] for r in data.itertuples()]
                    rates_table(fig.add_axes([.055, .38, .89, .52]), cols, cells, [.36, .10, .10, .10, .11, .11, .11])
                    picture(fig.add_axes([.12, .045, .76, .32]), report_directory(run, scope) / "test_metrics.png")
                    pdf.savefig(fig)
                    plt.close(fig)
                examples_dir = report_directory(report_directory(run, "pooled"), "examples")
                for act in ("push-up", "deadlift", "leg_raise"):
                    examples = [examples_dir / f"{act}_{kind}.png" for kind in ("good", "bad")]
                    if all(p.is_file() for p in examples):
                        fig, axes = plt.subplots(2, figsize=(9, 10), layout="constrained")
                        fig.suptitle(f"{m['model']} · {act} prediction examples")
                        for ax, p in zip(axes, examples):
                            picture(ax, p)
                        pdf.savefig(fig)
                        plt.close(fig)
        stage.rename(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    compare(args.runs, args.output)
