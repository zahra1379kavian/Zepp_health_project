"""Training, validation-only decoder selection, evaluation and sensor-only inference."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .constants import CLASSES, DT_MS, FEATURES
from .data import (WRIST_LOCATIONS, load_arrays, load_sensor, runs, write_json,
                   subject_folds, training_statistics)
from .decode import DecoderConfig, RepDecoder, decode_sequence, group_sets
from .metrics import (aggregate_repetitions, classification_metrics, confusion_matrix,
                      count_rows, covered_repetitions, false_exercise_events, repetition_metrics,
                      reporting_tolerances, score_stream, aggregate_streams, grouped_stream_report)
from .model import DenseMSTCN, ModelConfig, configuration, dense_loss


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


class Windows(Dataset):
    def __init__(self, root, sessions, norm, history, width, repetition_head="boundary"):
        if width < 1:
            raise ValueError("Window width must be positive")
        self.arrays = [load_arrays(Path(root) / name) for name in sessions]
        self.mean = np.asarray(norm["mean"], np.float32)
        self.std = np.asarray(norm["std"], np.float32)
        self.history, self.width = history, width
        self.repetition_head = repetition_head
        self.windows = []
        for i, arrays in enumerate(self.arrays):
            for a, b in runs(arrays["valid"]):
                for start in range(a, b, width):
                    self.windows.append((i, a, start, min(start + width, b)))

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, i):
        session, run_start, start, end = self.windows[i]
        arrays = self.arrays[session]
        left, length = max(run_start, start - self.history), self.history + self.width
        offset = self.history - (start - left)
        stop = offset + end - left
        x = np.zeros((length, len(FEATURES)), np.float32)
        x[offset:stop] = (arrays["x"][left:end] - self.mean) / self.std
        mask = np.zeros((1, length), np.float32)
        mask[:, offset:stop] = 1
        target_mask = np.zeros(length, bool)
        target_mask[self.history:stop] = True
        y = np.zeros(length, np.int64)
        y[offset:stop] = arrays["y"][left:end]
        boundary = np.zeros((length, 2), np.float32)
        boundary[offset:stop] = arrays["boundary"][left:end]
        reference = np.zeros(length, bool)
        reference[offset:stop] = arrays["coverage"][left:end]
        if self.repetition_head == "phase":
            boundary = np.zeros(length, np.int64)
            boundary[offset:stop] = arrays["phase"][left:end]
            reference[offset:stop] = arrays["phase_mask"][left:end]
        return tuple(torch.from_numpy(a.copy()) for a in (x.T, mask, y, boundary, target_mask, reference))


def train(args):
    root, out = Path(args.data), Path(args.output)
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest["location"] not in WRIST_LOCATIONS:
        raise ValueError("Prepare a wrist-only dataset; waist datasets are not supported")
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite {out}")
    if (args.epochs < 1 or args.batch_size < 1 or args.max_steps < 0 or args.patience < 1
            or getattr(args, "tolerance_ms", 400) <= 0):
        raise ValueError("epochs/batch-size must be positive; max-steps must be nonnegative")
    if not args.station_only:
        if manifest.get("reference_mode") == "labels" and (
                manifest.get("version", 0) < 3 or not manifest.get("phase_order_validated")):
            raise ValueError("Reprepare label references with the validated phase1 -> phase2 convention")
        if min(manifest["train_boundary_positives"]) == 0:
            raise ValueError("No training repetitions. Prepare with --reference-mode labels or supply references.")
        val_names = manifest["split"]["validation"]
        if not sum(s["reference_repetitions"] for s in manifest["sessions"] if s["session"] in val_names):
            raise ValueError("No validation repetitions; cannot select a repetition model")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    head = getattr(args, "repetition_head", "auto")
    if head == "auto":
        head = "phase" if manifest.get("reference_mode") == "labels" and not args.station_only else "boundary"
    if head == "phase" and not manifest.get("phase_order_validated"):
        raise ValueError("Dense phase training requires the validated phase convention; reprepare data")
    config = ModelConfig(stages=args.stages, layers=args.layers, channels=args.channels, repetition_head=head)
    model = DenseMSTCN(config).to(args.device)
    datasets = {split: Windows(root, manifest["split"][split], manifest["normalization"],
                               model.history_samples, args.window, head)
                for split in ("train", "validation")}
    if any(not len(d) for d in datasets.values()):
        raise ValueError("Training/validation partition has no eligible windows")
    loaders = {split: DataLoader(data, batch_size=args.batch_size, shuffle=split == "train", num_workers=0)
               for split, data in datasets.items()}
    counts = np.asarray(manifest["train_class_counts"], dtype=float)
    weights = np.where(counts > 0, 1 / np.sqrt(np.maximum(counts, 1)), 0)
    weights /= weights[weights > 0].mean()
    weights = torch.tensor(weights, dtype=torch.float32, device=args.device)
    positives = np.asarray(manifest["train_boundary_positives"])
    positive_weights = np.clip((manifest["train_boundary_frames"] - positives) / np.maximum(positives, 1), 1, 100)
    positive_weights = torch.tensor(positive_weights, dtype=torch.float32, device=args.device)
    phase_counts = np.asarray(manifest.get("train_phase_counts", [0, 0, 0]), dtype=float)
    if head == "phase" and not args.station_only and min(phase_counts[1:]) == 0:
        raise ValueError("No supervised phase1/phase2 training samples; reprepare phase targets")
    phase_weights = np.where(phase_counts > 0, 1 / np.sqrt(np.maximum(phase_counts, 1)), 0)
    phase_weights /= max(phase_weights[phase_weights > 0].mean() if phase_weights.any() else 0, 1e-12)
    phase_weights = torch.tensor(phase_weights, dtype=torch.float32, device=args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    out.mkdir(parents=True)
    history, best, stale = [], None, 0
    selection_decoder = DecoderConfig()
    tolerance = getattr(args, "tolerance_ms", 400)
    for epoch in range(1, args.epochs + 1):
        record = {"epoch": epoch}
        for split, loader in loaders.items():
            model.train(split == "train")
            loss_sum, frames, batches = 0., 0, 0
            with torch.set_grad_enabled(split == "train"):
                for batch in loader:
                    x, mask, y, boundary, valid, reference = [a.to(args.device) for a in batch]
                    outputs = model(x, mask)
                    loss = dense_loss(outputs, y, boundary, valid, reference, weights, positive_weights,
                                      boundary_weight=0 if args.station_only else 1, phase_weights=phase_weights)
                    if not torch.isfinite(loss):
                        raise RuntimeError("Nonfinite training loss")
                    if split == "train":
                        optimizer.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
                        optimizer.step()
                    n = int(valid.sum())
                    loss_sum += float(loss.detach()) * n
                    frames += n
                    batches += 1
                    if batches % 100 == 0:
                        print(f"Epoch {epoch}/{args.epochs} {split}: batch {batches}/{len(loader)}; "
                              f"mean loss {loss_sum / max(frames, 1):.4f}", flush=True)
                    if args.max_steps and batches >= args.max_steps:
                        break
            record[f"{split}_loss"] = loss_sum / max(frames, 1)
            record[f"{split}_batches"] = batches
        if args.station_only:
            key = (-record["validation_loss"],)
        else:
            cache = validation_cache(model, root, manifest, args.device)
            summary = score_cache(cache, selection_decoder, tolerance, require_station=False)
            record["validation_segmentation"] = summary
            key = checkpoint_key(summary)
        history.append(record)
        print(json.dumps(record), flush=True)
        if best is None or key > best:
            best, stale = key, 0
            checkpoint = {"version": 2, "model": model.state_dict(), "config": configuration(model),
                          "normalization": manifest["normalization"], "classes": CLASSES, "features": FEATURES,
                          "location": manifest["location"], "station_only": args.station_only,
                          "sensor_policy": manifest.get("wrist_selection"),
                          "training_streams": [{key: row.get(key) for key in ("session", "source_session", "subject", "side")}
                                               for row in manifest["sessions"] if row["session"] in manifest["split"]["train"]],
                          "reference_mode": manifest.get("reference_mode"),
                          "reference_convention": manifest.get("reference_convention"),
                          "data_manifest_sha256": digest(root / "manifest.json"), "split": manifest["split"],
                          "epoch": epoch, "validation_loss": record["validation_loss"], "seed": args.seed,
                          "selection_rule": "validation loss" if args.station_only else
                          "maximize station-independent repetition F1; tie-break by session count MAE",
                          "selection_key": list(best), "selection_decoder": asdict(selection_decoder),
                          "tolerance_ms": tolerance,
                          "partial_epoch_max_steps": args.max_steps}
            torch.save(checkpoint, out / "best.pt")
        else:
            stale += 1
        write_json(out / "history.json", history)
        if stale >= args.patience:
            break
    return out / "best.pt"


def load_checkpoint(path, device="cpu", data=None):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint["location"] not in WRIST_LOCATIONS:
        raise ValueError("This method now uses wrist signals only; retrain the old waist checkpoint")
    if checkpoint["classes"] != CLASSES or checkpoint["features"] != FEATURES:
        raise ValueError("Checkpoint vocabulary/features are incompatible")
    if data is not None and digest(Path(data) / "manifest.json") != checkpoint["data_manifest_sha256"]:
        raise ValueError("Dataset manifest differs from the checkpoint's training dataset")
    configuration = dict(checkpoint["config"])
    configuration.setdefault("repetition_head", "boundary")
    model = DenseMSTCN(ModelConfig(**configuration)).to(device).eval()
    model.load_state_dict(checkpoint["model"])
    return model, checkpoint


@torch.no_grad()
def probabilities(model, x, valid, norm, device="cpu", chunk=512):
    """Offline replay of exact causal outputs, resetting all caches at each gap."""
    p = np.zeros((len(x), len(CLASSES)), np.float32)
    q = np.zeros((len(x), model.config.repetition_classes), np.float32)
    mean, std = np.array(norm["mean"], np.float32), np.array(norm["std"], np.float32)
    for a, b in runs(valid):
        state = None
        for left in range(a, b, chunk):
            right = min(b, left + chunk)
            features = torch.tensor(((x[left:right] - mean) / std).T[None], device=device)
            station, boundary, state = model.stream(features, state)
            p[left:right] = station.softmax(1)[0].T.cpu().numpy()
            q[left:right] = model.repetition_probabilities(boundary)[0].T.cpu().numpy()
    return p, q


def checkpoint_key(summary):
    return (summary["f1"], -summary["count_mae_all_station_sessions"])


def validation_cache(model, root, manifest, device):
    cache = []
    model.eval()
    for name in manifest["split"]["validation"]:
        a = load_arrays(Path(root) / name)
        if not a["coverage"].any():
            continue
        p, q = probabilities(model, a["x"], a["valid"], manifest["normalization"], device)
        refs = json.loads((Path(root) / name / "references.json").read_text())
        cache.append((name, a, p, q, refs))
    return cache


def score_cache(cache, config, tolerance, require_station=True):
    scores, rows = [], []
    for name, a, p, q, refs in cache:
        result = decode_sequence(a["t"], p, q, a["valid"], config)
        scored, _ = covered_repetitions(result["repetitions"], a["t"], a["coverage"],
                                        refs, tolerance, require_station)
        scores.append(repetition_metrics(scored, refs, tolerance, require_station))
        rows.extend(count_rows(scored, refs, name, require_station))
    return aggregate_repetitions(scores, rows)


def decoder_config(path, checkpoint_path, default_tolerance=400):
    if path is None:
        return DecoderConfig(), default_tolerance, "defaults_not_validation_selected"
    data = json.loads(Path(path).read_text())
    if data.get("checkpoint_sha256") != digest(checkpoint_path) or data.get("selected_on") != "validation":
        raise ValueError("Decoder configuration must be selected on validation for this checkpoint")
    return DecoderConfig(**data["config"]), data["tolerance_ms"], "validation_selected"


def evaluate(args):
    root, out = Path(args.data), Path(args.output)
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite {out}")
    model, ckpt = load_checkpoint(args.checkpoint, args.device, root)
    config, tolerance, selection = decoder_config(args.decoder, args.checkpoint, args.tolerance_ms)
    tolerances = reporting_tolerances(getattr(args, "report_tolerances_ms", None), tolerance)
    manifest = json.loads((root / "manifest.json").read_text())
    metadata = {s["session"]: s for s in manifest["sessions"]}
    out.mkdir(parents=True)
    records, sessions = [], []
    for name in manifest["split"][args.split]:
        arrays = load_arrays(root / name)
        p, q = probabilities(model, arrays["x"], arrays["valid"], ckpt["normalization"], args.device)
        if ckpt["station_only"]:
            q[:] = 0
        decoded = decode_sequence(arrays["t"], p, q, arrays["valid"], config)
        predicted = p.argmax(1)
        cm = confusion_matrix(arrays["y"], predicted, arrays["valid"])
        info = metadata[name]
        record = {"session": name, "source_session": info.get("source_session", name),
                  "subject": info["subject"], "side": info.get("side", info["label_column"]),
                  "confusion_matrix": cm.tolist(), "repetition_scores": {},
                  **false_exercise_events(arrays["y"], predicted, arrays["valid"]),
                  "eligible_samples": int(arrays["valid"].sum()),
                  "reference_samples": int(arrays["coverage"].sum())}
        references = json.loads((root / name / "references.json").read_text())
        scored, unscored = covered_repetitions(decoded["repetitions"], arrays["t"], arrays["coverage"],
                                               references, tolerance)
        if arrays["coverage"].any() and not ckpt["station_only"]:
            record["repetition_scores"] = score_stream(decoded["repetitions"], references,
                                                        arrays["t"], arrays["coverage"], record, tolerances)
        result = {key: record[key] for key in ("session", "source_session", "subject", "side",
                                              "eligible_samples", "reference_samples")}
        result.update(aggregate_streams([record], tolerances, tolerance))
        if ckpt["station_only"]:
            result["repetitions"] = {"status": "station_only"}
        decoded["scored_repetitions"], decoded["unscored_repetitions"] = scored, unscored
        write_json(out / f"{name}.json", {**decoded, "metrics": result, "scoring": record})
        records.append(record)
        sessions.append(result)
        print(f"Evaluated {name}", flush=True)
    report = {**grouped_stream_report(records, tolerances, tolerance),
              "split": args.split, "decoder_selection": selection, "timing_tolerance_ms": tolerance,
              "report_tolerances_ms": tolerances,
              "report_scope": "fold_test" if "cross_validation" in manifest else "single_split_diagnostic",
              "reference_mode": manifest.get("reference_mode"),
              "reference_convention": manifest.get("reference_convention"),
              "phase_order_validated": manifest.get("phase_order_validated", False),
              "sensor_policy": manifest.get("wrist_selection"),
              "per_subject_session": sessions,
              "latency_note": "Sample-time replay includes decoder confirmation lag and timing error, but not compute/transport. Matched early reports meet the deadline and are reported separately; matching still requires endpoint tolerance. Missed reps are excluded from matched-delay fractions.",
              "coverage_note": "Counts cover certified, available wrist streams. Paired wrists observe the same workout; pooled repetitions and stream-hours are not distinct workout totals. Both wrists stay in the same subject fold.",
              "limitations": "One session per subject; uneven exercise and demographic coverage. NULL is category-relative rest, not a learned unknown-exercise class.",
              "excluded_sessions": manifest["excluded"]}
    write_json(out / "report.json", report)
    return report


def tune(args):
    if Path(args.output).exists():
        raise FileExistsError(args.output)
    root = Path(args.data)
    model, ckpt = load_checkpoint(args.checkpoint, args.device, root)
    if ckpt["station_only"]:
        raise ValueError("A station-only checkpoint cannot tune repetition boundaries")
    manifest = json.loads((root / "manifest.json").read_text())
    cache = validation_cache(model, root, manifest, args.device)
    if not cache or not sum(len(c[-1]) for c in cache):
        raise ValueError("No validated validation repetitions available for threshold selection")
    trials, best_key, best_config = [], None, None
    for event in (.35, .5, .65):
        for station in (.3, .5, .7):
            config = DecoderConfig(start_threshold=event, end_threshold=event,
                                   phase_threshold=event, station_threshold=station)
            summary = score_cache(cache, config, args.tolerance_ms, require_station=False)
            labelled = score_cache(cache, config, args.tolerance_ms)
            key = (*checkpoint_key(summary), labelled["f1"])
            trials.append({"event_threshold": event, "station_threshold": station,
                           "f1": summary["f1"], "count_mae": -key[1], "station_labelled_f1": labelled["f1"]})
            if best_key is None or key > best_key:
                best_key, best_config = key, config
    result = {"selected_on": "validation", "checkpoint_sha256": digest(args.checkpoint),
              "sessions": ckpt["split"]["validation"], "tolerance_ms": args.tolerance_ms,
              "selection_rule": "maximize station-independent repetition F1, then session count MAE, then station-labelled F1; fixed tolerance",
              "config": asdict(best_config), "trials": trials}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, result)
    return result


@torch.no_grad()
def predict(args):
    """A label-free CSV path; event timestamps are recorded at chunk arrival."""
    from collections import Counter
    import csv

    out = Path(args.output)
    if out.exists():
        raise FileExistsError(out)
    if not 1 <= args.chunk <= 20:
        raise ValueError("Online chunk size must be 1..20 samples (at most 760 ms buffering)")
    model, ckpt = load_checkpoint(args.checkpoint, args.device)
    input_side = Path(args.input).stem
    if (ckpt["location"] in ("left_wrist", "right_wrist") and
            input_side in ("left_wrist", "right_wrist") and input_side != ckpt["location"]):
        raise ValueError("Input wrist side differs from the checkpoint's training location")
    config, _, selection = decoder_config(args.decoder, args.checkpoint)
    t, x = load_sensor(args.input)
    norm = ckpt["normalization"]
    x = (x - np.array(norm["mean"], np.float32)) / np.array(norm["std"], np.float32)
    out.mkdir(parents=True)
    decoder, repetitions, latencies = RepDecoder(config), [], []
    all_p = np.zeros((len(t), len(CLASSES)), np.float32)
    all_q = np.zeros((len(t), model.config.repetition_classes), np.float32)
    # Break at actual missing clocks; no interpolation and no phantom cycles.
    edges = np.r_[0, np.flatnonzero(np.diff(t) != DT_MS) + 1, len(t)]
    with (out / "events.jsonl").open("w") as event_file:
        for a, b in zip(edges[:-1], edges[1:]):
            state = None
            decoder.reset()
            for left in range(a, b, args.chunk):
                right = min(b, left + args.chunk)
                features = torch.tensor(x[left:right].T[None], device=args.device)
                if str(args.device).startswith("cuda"):
                    torch.cuda.synchronize()
                started = time.perf_counter()
                station, boundary, state = model.stream(features, state)
                p = station.softmax(1)[0].T.cpu().numpy()
                q = model.repetition_probabilities(boundary)[0].T.cpu().numpy()
                if ckpt["station_only"]:
                    q[:] = 0
                all_p[left:right], all_q[left:right] = p, q
                batch_events = []
                for i in range(right-left):
                    batch_events.extend(decoder.step(t[left+i], p[i], q[i], reported_at_ms=int(t[right-1])))
                elapsed_ms = (time.perf_counter() - started) * 1000
                latencies.append(elapsed_ms)
                for event in batch_events:
                    event["compute_ms"] = elapsed_ms
                    event["decoder_delay_ms"] = config.lag_frames * DT_MS
                    event["buffer_delay_ms"] = event["reported_at_ms"] - event["end_ms"] - event["decoder_delay_ms"]
                    event["report_delay_from_predicted_end_ms"] = event["reported_at_ms"] - event["end_ms"] + elapsed_ms
                    event_file.write(json.dumps(event) + "\n")
                if batch_events:
                    event_file.flush()
                repetitions.extend(batch_events)
    with (out / "frames.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        rep_columns = (["phase_null_probability", "phase1_probability", "phase2_probability"]
                       if model.config.repetition_head == "phase" else ["start_probability", "end_probability"])
        writer.writerow(["sync_timestamp", "predicted_station", "confidence", *rep_columns])
        for clock, p, q in zip(t, all_p, all_q):
            writer.writerow([int(clock), CLASSES[int(p.argmax())], float(p.max()), *map(float, q)])
    holds = decode_sequence(t, all_p, np.zeros_like(all_q), np.ones(len(t), bool), config)["holds"]
    summary = {"repetitions": repetitions, "counts": dict(Counter(r["station"] for r in repetitions)),
               "holds": holds, "sets": group_sets(repetitions, config.set_gap_ms),
               "station_only": ckpt["station_only"], "decoder_selection": selection,
               "sensor_location_required": ckpt["location"], "chunk_samples": args.chunk,
               "compute_ms_p50": float(np.percentile(latencies, 50)),
               "compute_ms_p95": float(np.percentile(latencies, 95)),
               "compute_ms_max": max(latencies), "maximum_buffer_delay_ms": (args.chunk - 1) * DT_MS,
               "decoder_confirmation_delay_ms": config.lag_frames * DT_MS,
               "latency_note": "Compute+buffer delay is relative to predicted boundaries. Real <1 s performance requires validated end-time references and sensor/transport measurements. CSV replay is not a live hardware driver."}
    write_json(out / "predictions.json", summary)
    return summary


def run_pipeline(args):
    """Prepare all requested wrist streams and report subject-wise cross-validation."""
    from .__main__ import main
    out = Path(args.output)
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite {out}; choose another --output")
    if args.epochs < 1 or args.batch_size < 1 or args.tolerance_ms <= 0:
        raise ValueError("epochs, batch-size and tolerance-ms must be positive")
    root = Path(args.data) if args.data else out / "data"
    if args.data:
        manifest = json.loads((root / "manifest.json").read_text())
        if manifest.get("version", 0) < 4 or manifest["location"] != args.location:
            raise ValueError("Reprepare --data as version 4 with every requested wrist stream")
        if manifest.get("reference_mode") != "labels":
            raise ValueError("The run command expects references derived from label.csv")
    else:
        main(["prepare", "--source", str(args.source), "--alignment", str(args.alignment),
              "--location", args.location, "--reference-mode", "labels",
              "--seed", str(args.seed), "--output", str(root)])
    common = ["--device", args.device, "--threads", str(args.threads)]
    main(["cross-validate", "--data", str(root), "--output", str(out / "cross_validation"),
          "--folds", str(args.folds),
          "--epochs", str(args.epochs), "--batch-size", str(args.batch_size),
          "--seed", str(args.seed), "--tolerance-ms", str(args.tolerance_ms),
          "--report-tolerances-ms", *map(str, args.report_tolerances_ms), *common])
    print(f"Wrist cross-validation complete. Report: {out / 'cross_validation' / 'report.json'}", flush=True)


def completed_fold_report(args, fold, expected_manifest, tolerances):
    """Verify a completed fold before reusing its held-out predictions."""
    required = [fold / name for name in ("data/manifest.json", "model/best.pt", "model/history.json",
                                         "decoder.json", "test/report.json")]
    if any(not path.is_file() for path in required):
        raise FileExistsError(f"Incomplete fold: {fold}. Move it aside before restarting this fold.")
    saved_manifest = json.loads((fold / "data/manifest.json").read_text())
    if saved_manifest != expected_manifest:
        raise ValueError(f"Dataset, split or normalization changed for {fold}")
    _, checkpoint = load_checkpoint(fold / "model/best.pt", "cpu", fold / "data")
    head = args.repetition_head
    if head == "auto":
        head = "phase" if expected_manifest.get("reference_mode") == "labels" else "boundary"
    expected_config = {"stages": args.stages, "layers": args.layers, "channels": args.channels,
                       "repetition_head": head}
    if (any(checkpoint["config"].get(key) != value for key, value in expected_config.items())
            or checkpoint["seed"] != args.seed or checkpoint["station_only"]
            or checkpoint["tolerance_ms"] != args.tolerance_ms
            or checkpoint["partial_epoch_max_steps"] != args.max_steps):
        raise ValueError(f"Training configuration changed for {fold}")
    history = json.loads((fold / "model/history.json").read_text())
    if (not history or [row["epoch"] for row in history] != list(range(1, len(history) + 1))
            or history[-1]["epoch"] > args.epochs or not 1 <= checkpoint["epoch"] <= len(history)):
        raise ValueError(f"Incompatible training history for {fold}")
    _, tolerance, _ = decoder_config(fold / "decoder.json", fold / "model/best.pt")
    decoder = json.loads((fold / "decoder.json").read_text())
    if tolerance != args.tolerance_ms or decoder["sessions"] != expected_manifest["split"]["validation"]:
        raise ValueError(f"Decoder selection changed for {fold}")
    report = json.loads((fold / "test/report.json").read_text())
    if (report.get("split") != "test" or report.get("report_scope") != "fold_test"
            or report.get("decoder_selection") != "validation_selected"
            or report.get("timing_tolerance_ms") != args.tolerance_ms
            or report.get("report_tolerances_ms") != tolerances):
        raise ValueError(f"Evaluation configuration changed for {fold}")
    records = []
    for name in expected_manifest["split"]["test"]:
        record = json.loads((fold / "test" / f"{name}.json").read_text())["scoring"]
        if record["session"] != name or set(record["repetition_scores"]) != set(map(str, tolerances)):
            raise ValueError(f"Incomplete held-out scores for {fold}: {name}")
        records.append(record)
    pooled = grouped_stream_report(records, tolerances, args.tolerance_ms)
    if any(report[key] != value for key, value in pooled.items()):
        raise ValueError(f"Held-out scores differ from the saved report for {fold}")
    return report


def cross_validate(args):
    """Each subject is tested once; fitting and threshold tuning stay within folds."""
    from argparse import Namespace

    root, out = Path(args.data).resolve(), Path(args.output)
    resume_completed = getattr(args, "resume_completed", False)
    if out.exists() and not resume_completed:
        raise FileExistsError(out)
    if args.station_only:
        raise ValueError("Repetition cross-validation requires a repetition head")
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("version", 0) < 4:
        raise ValueError("Reprepare as version 4 so cross-validation includes every requested wrist stream")
    tolerances = reporting_tolerances(getattr(args, "report_tolerances_ms", None), args.tolerance_ms)
    splits = subject_folds(manifest["sessions"], args.folds, args.seed)
    reports, records = [], []
    for index, split in enumerate(splits):
        fold = out / f"fold_{index + 1}"
        data = fold / "data"
        fold_manifest = {**manifest, "split": split, "seed": args.seed,
                         "cross_validation": {"fold": index + 1, "folds": args.folds}}
        fold_manifest.update(training_statistics(root, split["train"]))
        if fold.exists():
            if not resume_completed:
                raise FileExistsError(fold)
            report = completed_fold_report(args, fold, fold_manifest, tolerances)
            print(f"Fold {index + 1}/{args.folds}: reusing completed results", flush=True)
        else:
            print(f"Fold {index + 1}/{args.folds}: training from epoch 1", flush=True)
            data.mkdir(parents=True)
            # Immutable arrays are shared; normalization and fitting stay within this fold.
            for session in manifest["sessions"]:
                (data / session["session"]).symlink_to(root / session["session"], target_is_directory=True)
            write_json(data / "manifest.json", fold_manifest)
            params = {**vars(args), "data": data, "output": fold / "model"}
            checkpoint = train(Namespace(**params))
            params.update(checkpoint=checkpoint, output=fold / "decoder.json")
            tune(Namespace(**params))
            params.update(decoder=fold / "decoder.json", split="test", output=fold / "test")
            report = evaluate(Namespace(**params))
        reports.append({"fold": index + 1, "split": split,
                        **{key: report[key] for key in ("repetitions", "segmentation", "by_tolerance", "per_side",
                                                       "streams", "subjects", "source_sessions")}})
        for name in split["test"]:
            # Reuse scores from the same once-decoded test predictions at every
            # tolerance. Never rerun inference or tune thresholds on test data.
            records.append(json.loads((fold / "test" / f"{name}.json").read_text())["scoring"])
        write_json(out / "folds.json", reports)
    result = {**grouped_stream_report(records, tolerances, args.tolerance_ms), "folds": reports,
              "report_scope": "subject_wise_cross_validation", "report_tolerances_ms": tolerances,
              "seed": args.seed, "timing_tolerance_ms": args.tolerance_ms,
              "protocol": "Subject-disjoint outer test folds; next fold validates; remaining folds train. Both wrists of a subject always stay together. Each subject tested exactly once. Normalization and loss weights refit on training subjects. Checkpoint and decoder selected only on validation at the primary tolerance; the same test predictions are scored at every reporting tolerance.",
              "coverage_note": "Pooled counts and hours are wrist-stream observations, not distinct workout totals. Report per-side metrics and subject-macro F1 alongside pooled metrics.",
              "sensor_policy": manifest["wrist_selection"], "reference_convention": manifest["reference_convention"]}
    for target, fold_groups in [(result, reports), *[(result["per_side"][side], [r["per_side"][side] for r in reports])
                                                     for side in ("left_wrist", "right_wrist")]]:
        for tolerance in tolerances:
            f1 = [r["by_tolerance"][str(tolerance)]["segmentation"]["f1"] for r in fold_groups
                  if r["by_tolerance"][str(tolerance)]["segmentation"]["status"] == "scored"]
            stats = {"fold_segmentation_f1_mean": float(np.mean(f1)) if f1 else None,
                     "fold_segmentation_f1_std": float(np.std(f1, ddof=1)) if len(f1) > 1 else None}
            target["by_tolerance"][str(tolerance)].update(stats)
            if tolerance == args.tolerance_ms:
                target.update(stats)
    write_json(out / "report.json", result)
    return result
