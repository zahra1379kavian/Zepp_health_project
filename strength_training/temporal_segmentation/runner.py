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
from .data import WRIST_LOCATIONS, load_arrays, load_sensor, runs, write_json
from .decode import DecoderConfig, RepDecoder, decode_sequence, group_sets
from .metrics import (aggregate_repetitions, classification_metrics, confusion_matrix,
                      count_rows, covered_repetitions, false_exercise_events, repetition_metrics)
from .model import DenseMSTCN, ModelConfig, configuration, dense_loss


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


class Windows(Dataset):
    def __init__(self, root, sessions, norm, history, width):
        if width < 1:
            raise ValueError("Window width must be positive")
        self.arrays = [load_arrays(Path(root) / name) for name in sessions]
        self.mean = np.asarray(norm["mean"], np.float32)
        self.std = np.asarray(norm["std"], np.float32)
        self.history, self.width = history, width
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
        return tuple(torch.from_numpy(a.copy()) for a in (x.T, mask, y, boundary, target_mask, reference))


def train(args):
    root, out = Path(args.data), Path(args.output)
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest["location"] not in WRIST_LOCATIONS:
        raise ValueError("Prepare a wrist-only dataset; waist datasets are not supported")
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite {out}")
    if args.epochs < 1 or args.batch_size < 1 or args.max_steps < 0:
        raise ValueError("epochs/batch-size must be positive; max-steps must be nonnegative")
    if not args.station_only:
        if min(manifest["train_boundary_positives"]) == 0:
            raise ValueError("No training repetitions. Prepare with --reference-mode labels or supply references.")
        val_names = manifest["split"]["validation"]
        if not sum(s["reference_repetitions"] for s in manifest["sessions"] if s["session"] in val_names):
            raise ValueError("No validation repetitions; cannot select a repetition model")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    config = ModelConfig(stages=args.stages, layers=args.layers, channels=args.channels)
    model = DenseMSTCN(config).to(args.device)
    datasets = {split: Windows(root, manifest["split"][split], manifest["normalization"],
                               model.history_samples, args.window)
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    out.mkdir(parents=True)
    history, best, stale = [], float("inf"), 0
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
                                      boundary_weight=0 if args.station_only else 1)
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
        history.append(record)
        print(json.dumps(record), flush=True)
        if record["validation_loss"] < best:
            best, stale = record["validation_loss"], 0
            checkpoint = {"version": 1, "model": model.state_dict(), "config": configuration(model),
                          "normalization": manifest["normalization"], "classes": CLASSES, "features": FEATURES,
                          "location": manifest["location"], "station_only": args.station_only,
                          "reference_mode": manifest.get("reference_mode"),
                          "reference_convention": manifest.get("reference_convention"),
                          "data_manifest_sha256": digest(root / "manifest.json"), "split": manifest["split"],
                          "epoch": epoch, "validation_loss": best, "seed": args.seed,
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
    model = DenseMSTCN(ModelConfig(**checkpoint["config"])).to(device).eval()
    model.load_state_dict(checkpoint["model"])
    return model, checkpoint


@torch.no_grad()
def probabilities(model, x, valid, norm, device="cpu", chunk=512):
    """Offline replay of exact causal outputs, resetting all caches at each gap."""
    p, q = np.zeros((len(x), len(CLASSES)), np.float32), np.zeros((len(x), 2), np.float32)
    mean, std = np.array(norm["mean"], np.float32), np.array(norm["std"], np.float32)
    for a, b in runs(valid):
        state = None
        for left in range(a, b, chunk):
            right = min(b, left + chunk)
            features = torch.tensor(((x[left:right] - mean) / std).T[None], device=device)
            station, boundary, state = model.stream(features, state)
            p[left:right] = station.softmax(1)[0].T.cpu().numpy()
            q[left:right] = boundary.sigmoid()[0].T.cpu().numpy()
    return p, q


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
    manifest = json.loads((root / "manifest.json").read_text())
    out.mkdir(parents=True)
    pooled_cm = np.zeros((len(CLASSES), len(CLASSES)), np.int64)
    sessions, rep_results, counts, exercise_results = [], [], [], {name: [] for name in CLASSES[1:] if name != "plank"}
    for name in manifest["split"][args.split]:
        arrays = load_arrays(root / name)
        p, q = probabilities(model, arrays["x"], arrays["valid"], ckpt["normalization"], args.device)
        if ckpt["station_only"]:
            q[:] = 0
        decoded = decode_sequence(arrays["t"], p, q, arrays["valid"], config)
        predicted = p.argmax(1)
        cm = confusion_matrix(arrays["y"], predicted, arrays["valid"])
        pooled_cm += cm
        result = {"session": name, "subject": name.split("_", 1)[1],
                  "classification": classification_metrics(cm),
                  **false_exercise_events(arrays["y"], predicted, arrays["valid"]),
                  "eligible_samples": int(arrays["valid"].sum()),
                  "reference_samples": int(arrays["coverage"].sum())}
        scored, unscored = covered_repetitions(decoded["repetitions"], arrays["t"], arrays["coverage"])
        if arrays["coverage"].any() and not ckpt["station_only"]:
            references = json.loads((root / name / "references.json").read_text())
            metric = repetition_metrics(scored, references, tolerance)
            rows = count_rows(scored, references, name)
            result["repetitions"] = aggregate_repetitions([metric], rows)
            rep_results.append(metric)
            counts.extend(rows)
            for station in exercise_results:
                metric = repetition_metrics([r for r in scored if r["station"] == station],
                                            [r for r in references if r["station"] == station], tolerance)
                exercise_results[station].append(metric)
        else:
            result["repetitions"] = {"status": "station_only" if ckpt["station_only"] else "no_validated_references"}
        decoded["scored_repetitions"], decoded["unscored_repetitions"] = scored, unscored
        write_json(out / f"{name}.json", {**decoded, "metrics": result})
        sessions.append(result)
        print(f"Evaluated {name}", flush=True)
    hours = sum(r["rest_hours"] for r in sessions)
    events = sum(r["false_exercise_events"] for r in sessions)
    report = {"split": args.split, "decoder_selection": selection, "timing_tolerance_ms": tolerance,
              "reference_mode": manifest.get("reference_mode"),
              "reference_convention": manifest.get("reference_convention"),
              "sensor_policy": manifest.get("wrist_selection"),
              "classification": classification_metrics(pooled_cm),
              "false_exercise_events": events, "rest_hours": hours,
              "false_exercise_events_per_rest_hour": events / hours if hours else None,
              "repetitions": aggregate_repetitions(rep_results, counts), "per_subject_session": sessions,
              "per_exercise": {s: aggregate_repetitions(r, [c for c in counts if c["station"] == s])
                               for s, r in exercise_results.items()},
              "latency_note": "Causal sample-time replay; matched delays include model timing error, not compute or sensor transport. Early reports and missed reps do not satisfy the <1 s target.",
              "coverage_note": "Repetition counts refer only to certified, available intervals, not unannotated whole sessions.",
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
    cache = []
    for name in ckpt["split"]["validation"]:
        a = load_arrays(root / name)
        if not a["coverage"].any():
            continue
        p, q = probabilities(model, a["x"], a["valid"], ckpt["normalization"], args.device)
        refs = json.loads((root / name / "references.json").read_text())
        cache.append((name, a, p, q, refs))
    if not cache or not sum(len(c[-1]) for c in cache):
        raise ValueError("No validated validation repetitions available for threshold selection")
    trials, best_key, best_config = [], None, None
    for event in (.35, .5, .65):
        for station in (.3, .5, .7):
            config = DecoderConfig(start_threshold=event, end_threshold=event, station_threshold=station)
            scores, rows = [], []
            for name, a, p, q, refs in cache:
                result = decode_sequence(a["t"], p, q, a["valid"], config)
                scored, _ = covered_repetitions(result["repetitions"], a["t"], a["coverage"])
                scores.append(repetition_metrics(scored, refs, args.tolerance_ms))
                rows.extend(count_rows(scored, refs, name))
            summary = aggregate_repetitions(scores, rows)
            key = (summary["f1"], -summary["count_mae_all_station_sessions"])
            trials.append({"event_threshold": event, "station_threshold": station,
                           "f1": summary["f1"], "count_mae": -key[1]})
            if best_key is None or key > best_key:
                best_key, best_config = key, config
    result = {"selected_on": "validation", "checkpoint_sha256": digest(args.checkpoint),
              "sessions": ckpt["split"]["validation"], "tolerance_ms": args.tolerance_ms,
              "selection_rule": "maximize repetition F1, break ties with count MAE; fixed declared tolerance",
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
    all_p, all_q = np.zeros((len(t), len(CLASSES)), np.float32), np.zeros((len(t), 2), np.float32)
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
                q = boundary.sigmoid()[0].T.cpu().numpy()
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
                    event["buffer_delay_ms"] = event["reported_at_ms"] - event["end_ms"]
                    event["report_delay_from_predicted_end_ms"] = event["buffer_delay_ms"] + elapsed_ms
                    event_file.write(json.dumps(event) + "\n")
                if batch_events:
                    event_file.flush()
                repetitions.extend(batch_events)
    with (out / "frames.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sync_timestamp", "predicted_station", "confidence", "start_probability", "end_probability"])
        for clock, p, q in zip(t, all_p, all_q):
            writer.writerow([int(clock), CLASSES[int(p.argmax())], float(p.max()), float(q[0]), float(q[1])])
    holds = decode_sequence(t, all_p, np.zeros_like(all_q), np.ones(len(t), bool), config)["holds"]
    summary = {"repetitions": repetitions, "counts": dict(Counter(r["station"] for r in repetitions)),
               "holds": holds, "sets": group_sets(repetitions, config.set_gap_ms),
               "station_only": ckpt["station_only"], "decoder_selection": selection,
               "sensor_location_required": ckpt["location"], "chunk_samples": args.chunk,
               "compute_ms_p50": float(np.percentile(latencies, 50)),
               "compute_ms_p95": float(np.percentile(latencies, 95)),
               "compute_ms_max": max(latencies), "maximum_buffer_delay_ms": (args.chunk - 1) * DT_MS,
               "latency_note": "Compute+buffer delay is relative to predicted boundaries. Real <1 s performance requires validated end-time references and sensor/transport measurements. CSV replay is not a live hardware driver."}
    write_json(out / "predictions.json", summary)
    return summary


def run_pipeline(args):
    """One command for the real wrist experiment; no station-only fallback."""
    from .__main__ import main
    out = Path(args.output)
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite {out}; choose another --output")
    if args.epochs < 1 or args.batch_size < 1 or args.tolerance_ms <= 0:
        raise ValueError("epochs, batch-size and tolerance-ms must be positive")
    root = Path(args.data) if args.data else out / "data"
    if args.data:
        manifest = json.loads((root / "manifest.json").read_text())
        if manifest.get("version", 0) < 2 or manifest["location"] != args.location:
            raise ValueError("--data must be a current wrist dataset matching --location")
        if manifest.get("reference_mode") != "labels":
            raise ValueError("The run command expects references derived from label.csv")
    else:
        main(["prepare", "--source", str(args.source), "--alignment", str(args.alignment),
              "--location", args.location, "--reference-mode", "labels",
              "--seed", str(args.seed), "--output", str(root)])
    common = ["--device", args.device, "--threads", str(args.threads)]
    main(["train", "--data", str(root), "--output", str(out / "model"),
          "--epochs", str(args.epochs), "--batch-size", str(args.batch_size),
          "--seed", str(args.seed), *common])
    checkpoint = str(out / "model" / "best.pt")
    main(["tune", "--data", str(root), "--checkpoint", checkpoint,
          "--tolerance-ms", str(args.tolerance_ms), "--output", str(out / "decoder.json"), *common])
    main(["evaluate", "--data", str(root), "--checkpoint", checkpoint,
          "--decoder", str(out / "decoder.json"), "--split", "test",
          "--output", str(out / "test"), *common])
    print(f"Wrist experiment complete. Report: {out / 'test' / 'report.json'}", flush=True)
