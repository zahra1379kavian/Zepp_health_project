"""Threshold sensitivity on a frozen legacy model. Never used for model selection."""

from dataclasses import replace
import json

import numpy as np

from ..common import exercise_predictions
from ...constants import CLASSES, DT_MS
from ...data import load_arrays, write_json
from ...decode import decode_sequence
from ...metrics import match_repetitions
from ...runner import decoder_config, digest, load_checkpoint, prediction_mask, probabilities


def sweep(folder, model, checkpoint, config, split, exercise, thresholds, device):
    totals = {v: dict(matched=0, extra=0, missed=0, missed_without_overlap=0,
                      overlap_start_errors=[], overlap_end_errors=[]) for v in thresholds}
    for name in checkpoint["split"][split]:
        stream = folder / "data" / name
        a = load_arrays(stream, include_phase=False)
        coverage = a["valid"] & a["coverage"] & (a["y"] == CLASSES.index(exercise))
        refs = [r for r in json.loads((stream / "references.json").read_text()) if r["station"] == exercise]
        if not coverage.any() and not refs:
            continue
        p, q = probabilities(model, a["x"], a["valid"], checkpoint["normalization"], device)
        lag = model.config.lookahead_samples
        valid = prediction_mask(a["valid"], lag)
        for threshold, counts in totals.items():
            reps = decode_sequence(a["t"], p, q, valid, replace(config, phase_threshold=threshold),
                                   lookahead_ms=lag * DT_MS)["repetitions"]
            scored = exercise_predictions(reps, refs, a["t"], coverage, 400)
            hits = {r for _, r in match_repetitions(scored, refs, 400, require_station=False)}
            counts["matched"] += len(hits)
            counts["extra"] += len(scored) - len(hits)
            counts["missed"] += len(refs) - len(hits)
            for i, ref in enumerate(refs):
                if i in hits:
                    continue
                errors = [(abs(r["start_ms"] - ref["start_ms"]), abs(r["end_ms"] - ref["end_ms"]))
                          for r in reps if r["start_ms"] < ref["end_ms"] and r["end_ms"] > ref["start_ms"]]
                if errors:
                    start, end = min(errors, key=lambda pair: (max(pair), sum(pair)))
                    counts["overlap_start_errors"].append(start)
                    counts["overlap_end_errors"].append(end)
                else:
                    counts["missed_without_overlap"] += 1
        print(f"Inspected {split} stream {name}", flush=True)
    if not any(v["matched"] + v["missed"] for v in totals.values()):
        raise ValueError(f"No reference repetitions for {exercise} on {split}")
    for counts in totals.values():
        for key in ("start", "end"):
            errors = counts.pop(f"overlap_{key}_errors")
            counts[f"missed_with_overlap_median_abs_{key}_error_ms"] = float(np.median(errors)) if errors else None
        tp = counts["matched"]
        counts["f1_at_400"] = 2 * tp / (2 * tp + counts["extra"] + counts["missed"])
    return totals


def run(args):
    folder = args.run_dir
    if args.output.exists():
        raise FileExistsError(args.output)
    path = folder / "model/best.pt"
    model, ckpt = load_checkpoint(path, args.device, folder / "data")
    if model.config.repetition_head != "phase" or ckpt["analysis"].get("held_out_exercise") != "biceps_curls":
        raise ValueError("Use the existing biceps-curl leave-one-out phase checkpoint")
    config, _, _ = decoder_config(folder / "decoder.json", path)
    thresholds = sorted({.2, .35, .45, .5, .55, .65, .75, .85, .95, config.phase_threshold})
    validation = sweep(folder, model, ckpt, config, "validation", "bench_press", thresholds, args.device)
    selected = max(thresholds, key=lambda v: validation[v]["f1_at_400"])
    biceps = sweep(folder, model, ckpt, config, "test", "biceps_curls", thresholds, args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, {"scope": "diagnostic_only", "checkpoint_sha256": digest(path),
                             "legacy_threshold": config.phase_threshold, "bench_validation_sweep": validation,
                             "bench_selected_threshold": selected, "biceps_test_sweep": biceps,
                             "biceps_at_bench_selected_threshold": biceps[selected],
                             "note": "Bench press was seen in legacy training. Paired holdouts are required for independent transfer results."})
