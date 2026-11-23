"""Run a zero-lookahead wrist / fused / gated repetition comparison."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from baseline.rep_benchmark.protocol import (Prediction, clip_signature, clips, rep,
                                             score, session_plan, utils)
from baseline.rep_benchmark.report import reference_plots, summaries
from ...constants import FEATURES
from ...data import load_sensor, write_json
from ...runner import digest
from ..reference import WristModel
from .fit import fit
from .model import ARMS, align_waist, predict_clip


def read(path):
    return json.loads(path.read_text())


def report(rows, delays, folder):
    folder.mkdir(exist_ok=True)
    tables = []
    for arm in ARMS:
        data = rows[rows.arm == arm]
        table = summaries(data)
        table.insert(0, "arm", arm)
        totals = table.TP + table.FP + table.FN + table.TP
        table["rep_f1_pct"] = 200 * table.TP / totals.replace(0, np.nan)
        tables.append(table)
        path = folder / arm
        path.mkdir(exist_ok=True)
        reference_plots().plot_result_table(table.iloc[2:].copy(), path / "table_results_online.png")
        d = delays[arm]
        write_json(path / "reporting_delay.json", dict(matched=len(d),
                   median_ms=float(np.median(d)) if d else None,
                   p95_ms=float(np.percentile(d, 95)) if d else None,
                   clock="Emission minus labeled end on 25 Hz replay, excluding compute and transport"))
    summary = pd.concat(tables)
    summary.to_csv(folder / "comparison.csv", index=False)
    print(summary[summary.activity == "Overall"][["arm", "TP_pct", "FN_pct", "FP_pct", "cnt_error_pct",
                                                  "rep_f1_pct"]].to_string(index=False), flush=True)


def test_fold(base, expert, gate, raw, sessions, paired, expected, out):
    rows, devices = [], []
    delays = {arm: [] for arm in ARMS}
    assigned = {r["session"].split("__")[0]: r["waist_sensor"]["selected"]
                for r in read(paired / "mask_audit.json")}
    seen = set()
    parity_diffs = 0
    with (out / "intervals.jsonl").open("w") as log:
        for item in sessions:
            session = item["session"]
            folder = raw / session
            # Use the training-data device assignment if present. For extra release
            # sessions choose the first waist by filename, before reading any labels.
            files = sorted(folder.glob("*waist*.csv"))
            filename = assigned.get(session, files[0].name if files else None)
            wt, wx = (load_sensor(folder / filename, FEATURES[:6], allow_waist=True) if filename
                      else (np.empty(0, dtype=np.int64), np.empty((0, 6), np.float32)))
            count = 0
            for side, activity, t, x, y in clips(folder):
                key = (session, side, activity)
                sig = clip_signature(t, y)
                if key in seen or expected[key]["clip_sha256"] != sig:
                    raise ValueError(f"Reference clip mismatch for {key}")
                seen.add(key)
                z, present = align_waist(t, wt, wx)
                events, weights = predict_clip(base, expert, gate, t, x, z, present)
                parity_diffs += int(events["wrist"] != expected[key]["predictions"])
                count += 1
                info = dict(session=session, device=side, activity=activity, fold=item["fold"],
                            release_fold=item["release_fold"], clip_sha256=sig)
                record = dict(**info, waist=filename, frames=len(t), waist_frames=int(present.sum()),
                              mean_fused_weight=float(weights.mean()), predictions=events, matched_delay_ms={})
                for arm in ARMS:
                    gt, pred, metrics, delay = score(Prediction(events[arm]), y)
                    matches, _, _ = rep.match_reps_greedy(gt, pred)
                    matched = {(g["start"], g["end"]) for g, _ in matches}
                    if arm == "wrist":
                        baseline_matches = matched
                        record["ground_truth"] = gt
                    rows.append(dict(**info, arm=arm, **metrics, frames=len(t),
                                     waist_frames=int(present.sum()), mean_gated_fused_weight=float(weights.mean()),
                                     recovered_vs_wrist=len(matched - baseline_matches),
                                     lost_vs_wrist=len(baseline_matches - matched)))
                    record["matched_delay_ms"][arm] = delay
                    delays[arm].extend(delay)
                log.write(json.dumps(record, separators=(",", ":")) + "\n")
            if count != item["clips"]:
                raise ValueError(f"Expected {item['clips']} release clips in {session}, got {count}")
            devices.append(dict(session=session, waist=filename, clips=count,
                                waist_sha256=digest(folder / filename) if filename else None))
            print(f"Scored {session}: {count} clips", flush=True)
    if seen != set(expected):
        raise ValueError("Held-out session plan differs from the original phase benchmark")
    write_json(out / "coverage.json", dict(sessions=devices, clips=len(seen),
               wrist_replay_clips_differing_from_saved_intervals=parity_diffs))
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "clips.csv", index=False)
    report(frame, delays, out)
    return frame, delays


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path("baseline/MS-TCN/experiment_runs"))
    parser.add_argument("--output", type=Path, required=True, help="New experiment directory")
    parser.add_argument("--folds", type=int, nargs="+", choices=range(1, 6), default=[5, 1, 2, 3, 4])
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.threads < 1 or len(set(args.folds)) != len(args.folds):
        parser.error("Use positive threads and distinct folds")
    torch.set_num_threads(args.threads)
    if list(utils.REQUIRED_COLS) != FEATURES:
        raise ValueError("Unexpected release sensor order")
    runs = args.runs.resolve()
    reference = runs / "feishu_wrist_rep_comparison/phase_sequence_model"
    manifest = read(reference / "manifest.json")
    raw = Path(manifest["raw_source"])
    parts = {int(i): p for i, p in manifest["fold_subjects"].items()}
    plan, missing = session_plan(raw, parts)
    if plan != [{k: v for k, v in s.items() if k != "clips"} for s in manifest["sessions"]]:
        raise ValueError("Release subject assignments differ from the saved benchmark")
    expected = {}
    with (reference / "intervals.jsonl").open() as handle:
        for line in handle:
            r = json.loads(line)
            expected.setdefault(r["fold"], {})[r["session"], r["device"], r["activity"]] = r
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "plan.json", dict(folds=args.folds, seed=args.seed, missing_sessions=missing,
               reference_manifest_sha256=digest(reference / "manifest.json"), raw_source=str(raw), subjects=parts,
               sessions=manifest["sessions"], arms=list(ARMS), model_lookahead_ms=0,
               unavailable_waist="Use the wrist probabilities on these frames in every arm"))
    frames, pooled_delays = [], {arm: [] for arm in ARMS}
    for i in args.folds:
        print(f"Starting fold {i}", flush=True)
        out = args.output / f"fold_{i}"
        out.mkdir()
        wrist = runs / f"wrist_cross_validation/cross_validation/fold_{i}"
        ablation = runs / ("sensor_ablation_fold_1" if i == 1 else f"sensor_ablation_five_folds/fold_{i}")
        paired = ablation / "combined_waist/data"
        base = WristModel(wrist, "cpu")
        original = manifest["models"][str(i)]
        if (base.head != "phase" or base.lag or base.model.config.acausal or
                any(base.metadata[k] != original[k] for k in ("checkpoint_sha256", "decoder_sha256"))):
            raise ValueError("Expected the original causal phase model and decoder")
        expert, gate = fit(base, wrist / "data", paired, parts[i], out, args.seed + i)
        sessions = [s for s in manifest["sessions"] if s["fold"] == i]
        frame, delay = test_fold(base, expert, gate, raw, sessions, paired, expected[i], out)
        frames.append(frame)
        for arm in ARMS:
            pooled_delays[arm].extend(delay[arm])
    if len(args.folds) == 5:
        report(pd.concat(frames), pooled_delays, args.output / "all_five_folds")
    write_json(args.output / "completed.json", dict(folds=args.folds))


if __name__ == "__main__":
    main()
