"""Subject-safe fitting, sensor-only inference and shared repetition scoring."""

from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import time

import joblib
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from strength_training.temporal_segmentation.constants import CLASSES, DT_MS, FEATURES
from strength_training.temporal_segmentation.data import (load_arrays, load_sensor, runs,
                                                         subject_folds, write_json)
from strength_training.temporal_segmentation.metrics import (
    confusion_matrix, count_rows, false_exercise_events, grouped_stream_report, score_stream,
    reporting_tolerances,
)
from .counter import CounterConfig, decode
from .features import FS, feature_batches, filtered_inputs, window_ends


def manifest_at(root):
    manifest = json.loads((Path(root)/"manifest.json").read_text())
    if (manifest.get("version") != 4 or manifest.get("features") != FEATURES or
            manifest.get("classes") != CLASSES or manifest.get("sample_period_ms") != DT_MS):
        raise ValueError("Expected the project's version-4 prepared wrist dataset; run prepare first")
    return manifest


def check_split(manifest, split):
    by_name = {s["session"]: s for s in manifest["sessions"]}
    seen_names, seen_subjects = set(), set()
    for key in ("train", "validation", "test"):
        names = split[key]
        if not names or len(names) != len(set(names)) or set(names)-by_name.keys():
            raise ValueError(f"Invalid or empty {key} split")
        subjects = {by_name[n]["subject"] for n in names}
        if seen_names.intersection(names) or seen_subjects.intersection(subjects):
            raise ValueError("Subject leakage across train/validation/test (including paired wrists)")
        seen_names.update(names)
        seen_subjects.update(subjects)
    if seen_names != by_name.keys():
        raise ValueError("Split must include every prepared stream exactly once")


def pure_targets(y, ends, window):
    """Recognition training uses only complete single-exercise windows."""
    changes = np.r_[0, np.cumsum(y[1:] != y[:-1])]
    pure = changes[ends] == changes[ends-window+1]
    return np.where(pure & (y[ends] != 0), y[ends], -1)


def sampled_features(root, names, window, step, cap, seed):
    """Select windows before costly extraction; label-based sampling is fit-only.

    Caps are global per endpoint station and per pure recognition station, not per
    subject. A second selection prevents endpoint sampling losing pure windows.
    """
    candidates = []
    for stream_index, name in enumerate(names):
        arrays = load_arrays(Path(root)/name)
        ends = window_ends(arrays["valid"], window, step)
        if len(ends):
            candidates.append(np.column_stack((np.full(len(ends), stream_index), ends,
                                               arrays["y"][ends], pure_targets(arrays["y"], ends, window))))
    if not candidates:
        raise ValueError("No full eligible windows; decrease --window-seconds or check masks")
    table = np.concatenate(candidates)
    rng, selected = np.random.default_rng(seed), set()
    for column in (2, 3):
        for label in np.unique(table[:, column]):
            if label < 0:
                continue
            choices = np.flatnonzero(table[:, column] == label)
            selected.update(rng.choice(choices, min(cap, len(choices)), replace=False).tolist())
    table = table[sorted(selected)]
    segmentation, recognition, targets, pure = [], [], [], []
    for stream_index, name in enumerate(names):
        rows = table[table[:, 0] == stream_index]
        if not len(rows):
            continue
        arrays = load_arrays(Path(root)/name)
        filtered = filtered_inputs(arrays["x"], arrays["valid"])
        for _, seg, rec in feature_batches(filtered, rows[:, 1], window):
            segmentation.append(seg)
            recognition.append(rec)
        targets.append(rows[:, 2])
        pure.append(rows[:, 3])
    return tuple(np.concatenate(v) for v in (segmentation, recognition, targets, pure))


def classifier(x, y, c, seed):
    estimator = (DummyClassifier(strategy="constant", constant=int(y[0])) if len(np.unique(y)) == 1 else
                 LinearSVC(C=c, class_weight="balanced", dual=False, max_iter=20000, random_state=seed))
    return make_pipeline(StandardScaler(), estimator).fit(x, y)


def fit_period_bounds(root, names):
    durations = {name: [] for name in CLASSES[1:] if name != "plank"}
    for stream in names:
        for r in json.loads((Path(root)/stream/"references.json").read_text()):
            durations[r["station"]].append((r["end_ms"]-r["start_ms"])/1000)
    pooled = [v for values in durations.values() for v in values]
    result = {}
    for name, values in durations.items():
        source = values or pooled
        low, high = np.quantile(source, [.01, .99]) if source else (.5, 8.)
        minimum = max(.24, float(low)*.8)
        maximum = max(minimum+.08, float(high)*1.25)
        result[name] = {"min_s": minimum, "max_s": maximum, "training_references": len(values),
                        "source": "station_training_references" if values else
                                  "pooled_training_references" if pooled else "fixed_fallback"}
    return result


def new_directory(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    return path


def peak_count_summary(records):
    """Paper-style offline peak counts, separately from interval-based scoring."""
    rows = [row for record in records for row in record.get("offline_peak_count_rows", [])]
    active = [row for row in rows if row["predicted"] or row["reference"]]
    return {"status": "scored" if rows else "not_applicable",
            "count_mae_all_station_sessions": float(np.mean([r["absolute_error"] for r in rows])) if rows else None,
            "exact_count_accuracy_all_station_sessions": float(np.mean([r["exact"] for r in rows])) if rows else None,
            "count_mae_active_station_sessions": float(np.mean([r["absolute_error"] for r in active])) if active else None,
            "exact_count_accuracy_active_station_sessions": float(np.mean([r["exact"] for r in active])) if active else None,
            "within_one_accuracy_active_station_sessions": float(np.mean([r["absolute_error"] <= 1 for r in active])) if active else None,
            "count_rows": rows,
            "note": "Offline peak counts use certified peak timestamps; they do not evaluate repetition start/end boundaries."}


def fit(root, output, *, split=None, window_seconds=5., train_step_ms=1000,
        prediction_step_ms=200, max_windows_per_class=3000, c_values=(.1, 1., 10.),
        seed=42, confirm_steps=3, counter_config=CounterConfig()):
    root, output = Path(root), Path(output)
    manifest = manifest_at(root)
    split = split or manifest["split"]
    check_split(manifest, split)
    if (not np.isfinite(window_seconds) or window_seconds < .5 or train_step_ms < DT_MS or train_step_ms % DT_MS or
            prediction_step_ms < DT_MS or prediction_step_ms % DT_MS or
            max_windows_per_class < 1 or confirm_steps < 1 or
            not c_values or any(not np.isfinite(c) or c <= 0 for c in c_values)):
        raise ValueError("Invalid window, step, sample cap, SVM C or confirmation settings")
    window = int(round(window_seconds*FS))
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    print("Extracting training features", flush=True)
    seg, rec, target, pure = sampled_features(root, split["train"], window,
                                              train_step_ms//DT_MS, max_windows_per_class, seed)
    mask = pure > 0
    if not mask.any():
        raise ValueError("No complete exercise windows for recognition; reduce --window-seconds")
    print("Extracting validation features", flush=True)
    vs, vr, vy, _ = sampled_features(root, split["validation"], window,
                                     train_step_ms//DT_MS, max_windows_per_class, seed+1)
    trials, best, best_f1 = [], None, -1.
    for c in c_values:
        binary = classifier(seg, (target > 0).astype(int), c, seed)
        recognition = classifier(rec[mask], pure[mask], c, seed)
        predicted = np.where(binary.predict(vs) > 0, recognition.predict(vr), 0)
        score = float(f1_score(vy, predicted, average="macro", zero_division=0))
        trials.append({"C": c, "validation_window_macro_f1": score})
        print(f"C={c:g}: validation window macro F1={score:.4f}", flush=True)
        if score > best_f1:
            best_f1, best = score, (binary, recognition, c)
    binary, recognition, c = best
    model = {"format": "recofit_wrist_v1", "binary": binary, "recognition": recognition,
             "window_samples": window, "prediction_step": prediction_step_ms//DT_MS,
             "confirm_steps": confirm_steps, "period_bounds": fit_period_bounds(root, split["train"]),
             "counter_config": asdict(counter_config), "classes": CLASSES, "features": FEATURES,
             "sample_period_ms": DT_MS, "split": split, "seed": seed,
             "reference_convention": manifest.get("reference_convention"),
             "reference_mode": manifest.get("reference_mode"),
             "phase_order_validated": manifest.get("phase_order_validated", False),
             "training_source": str(root.resolve())}
    output = new_directory(output)
    joblib.dump(model, output/"model.joblib", compress=3)
    write_json(output/"training.json", {"selected_C": c, "selection": "validation_window_macro_f1",
               "trials": trials, "training_windows": len(target), "recognition_windows": int(mask.sum()),
               "training_window_class_counts": {CLASSES[int(k)]: int(v) for k, v in Counter(target).items()},
               "missing_training_stations": sorted(set(CLASSES[1:])-{CLASSES[int(k)] for k in np.unique(pure[mask])}),
               "train_step_ms": train_step_ms, "max_windows_per_class": max_windows_per_class,
               "window_seconds": window/FS, "prediction_step_ms": prediction_step_ms,
               "confirm_steps": confirm_steps, "counter_config": asdict(counter_config),
               "period_bounds": model["period_bounds"], "split": split,
               "reference_convention": model["reference_convention"],
               "note": "Feature scaling and period bounds fit training subjects only; C selected on validation windows. No session exercise lists."})
    return output/"model.joblib"


def load_model(path):
    model = joblib.load(path)
    if (model.get("format") != "recofit_wrist_v1" or model.get("classes") != CLASSES or
            model.get("features") != FEATURES or model.get("sample_period_ms") != DT_MS):
        raise ValueError("Unsupported RecoFit model")
    return model


def classify(model, x, valid):
    """Trailing-window SVMs + causal saturating binary accumulator.

    The accumulator's default is 3 decisions (600 ms), rather than the paper's
    six seconds. Recognition is updated every decision, allowing station changes.
    Initial windows after gaps remain NULL until a complete history is available.
    """
    prediction = np.zeros(len(x), np.int64)
    filtered = filtered_inputs(x, valid)
    window, step = model["window_samples"], model["prediction_step"]
    ends = window_ends(valid, window, step)
    decisions, stations = [], []
    for _, seg, rec in feature_batches(filtered, ends, window):
        decisions.extend(model["binary"].predict(seg).tolist())
        stations.extend(model["recognition"].predict(rec).tolist())
    accumulator, active, previous = 0, False, None
    run_index, valid_runs = 0, runs(valid)
    for end, exercise, station_id in zip(ends, decisions, stations):
        while valid_runs[run_index][1] <= end:
            run_index += 1
        a, b = valid_runs[run_index]
        if previous is None or end-previous != step:
            accumulator, active = 0, False
        accumulator = int(np.clip(accumulator+(1 if exercise else -1), 0, model["confirm_steps"]))
        if accumulator == model["confirm_steps"]:
            active = True
        elif accumulator == 0:
            active = False
        prediction[end:min(end+step, b)] = station_id if active else 0
        previous = end
    return prediction


def infer(model, t, x, valid, mode="causal"):
    t, x, valid = np.asarray(t), np.asarray(x), np.asarray(valid, bool).copy()
    if (not len(t) or x.shape != (len(t), len(FEATURES)) or valid.shape != t.shape or
            not np.isfinite(x[valid]).all() or np.any(np.diff(t) != DT_MS)):
        raise ValueError("Inference needs finite sensor inputs on the complete 40 ms grid, plus a validity mask")
    start = time.perf_counter()
    labels = classify(model, x, valid)
    result = decode(t, x, labels, valid, model["period_bounds"], mode,
                    model["prediction_step"], CounterConfig(**model["counter_config"]))
    result["mode"] = mode
    result["runtime"] = {"compute_seconds": time.perf_counter()-start,
                         "sensor_seconds": len(t)*DT_MS/1000}
    result["boundary_note"] = "Intervals are symmetric local-period estimates around peaks, not phase boundaries from the original paper."
    result["latency_note"] = (
        "Causal sample-time replay; reported_at_ms includes classifier/counter waiting but excludes compute/transport. Deadline must be measured against reference ends."
        if mode == "causal" else "Offline full-session diagnostic; every event is marked available at the final observed timestamp.")
    return labels, result


def evaluate(root, model_path, output, *, split_name="test", mode="causal", tolerance_ms=400,
             report_tolerances_ms=(200, 400), model=None):
    root = Path(root)
    manifest = manifest_at(root)
    model = load_model(model_path) if model is None else model
    check_split(manifest, model["split"])
    if Path(model["training_source"]).resolve() != root.resolve():
        raise ValueError("Evaluation must use the same prepared dataset as the fitted split")
    if split_name not in ("test", "validation"):
        raise ValueError("Evaluate validation or test only")
    tolerances = reporting_tolerances(report_tolerances_ms, tolerance_ms)
    out = new_directory(output)
    metadata, records = {s["session"]: s for s in manifest["sessions"]}, []
    for name in model["split"][split_name]:
        arrays = load_arrays(root/name)
        labels, result = infer(model, arrays["t"], arrays["x"], arrays["valid"], mode)
        info = metadata[name]
        record = {"session": name, "source_session": info["source_session"],
                  "subject": info["subject"], "side": info["side"],
                  "confusion_matrix": confusion_matrix(arrays["y"], labels, arrays["valid"]).tolist(),
                  "repetition_scores": {}, **false_exercise_events(arrays["y"], labels, arrays["valid"])}
        references = json.loads((root/name/"references.json").read_text())
        if np.any(arrays["coverage"]):
            record["repetition_scores"] = score_stream(result["repetitions"], references, arrays["t"],
                                                         arrays["coverage"], record, tolerances)
            if mode == "offline":
                peak_indices = [int(np.searchsorted(arrays["t"], p["peak_ms"])) for p in result["offline_peaks"]]
                certified = [p for p, i in zip(result["offline_peaks"], peak_indices) if arrays["coverage"][i]]
                record["offline_peak_count_rows"] = count_rows(certified, references, name)
                for row in record["offline_peak_count_rows"]:
                    row.update({k: record[k] for k in ("subject", "source_session", "side")})
        result["scoring"] = record
        write_json(out/f"{name}.json", result)
        records.append(record)
        print(f"Evaluated {name}: {len(result['repetitions'])} complete interval estimates", flush=True)
    report = {**grouped_stream_report(records, tolerances, tolerance_ms),
              "method": "RecoFit-style adapted to wrist IMUs at 25 Hz", "mode": mode,
              "split": split_name, "report_scope": "single_split_diagnostic",
              "timing_tolerance_ms": tolerance_ms, "report_tolerances_ms": tolerances,
              "offline_peak_counts": peak_count_summary(records),
              "reference_convention": model["reference_convention"],
              "phase_order_validated": model["phase_order_validated"],
              "excluded_sessions": manifest.get("excluded", []),
              "coverage_note": "Each wrist is an observation of the same workout; pooled counts/hours are wrist-stream observations, not distinct physical repetitions. See per_side and per_subject.",
              "latency_note": result["latency_note"], "boundary_note": result["boundary_note"],
              "limitations": "Device X is not anatomical arm X. NULL is category-relative rest; no unseen-exercise identification. Period estimates cannot certify completion or resolve pauses. First reps after resets may be missed; no <1s guarantee."}
    write_json(out/"report.json", report)
    return report, records


def predict(model_path, input_path, output, *, mode="causal"):
    model = load_model(model_path)
    sensor_t, sensor_x = load_sensor(input_path)  # only clocks + nine channels; labels ignored
    t = np.arange(sensor_t[0], sensor_t[-1]+DT_MS, DT_MS, dtype=np.int64)
    indices = ((sensor_t-t[0])//DT_MS).astype(int)
    x, valid = np.zeros((len(t), len(FEATURES)), np.float32), np.zeros(len(t), bool)
    x[indices], valid[indices] = sensor_x, True
    out = new_directory(output)
    labels, result = infer(model, t, x, valid, mode)
    result["input"] = str(Path(input_path).resolve())
    result["reference_scoring"] = "unscored_sensor_only_prediction"
    write_json(out/"predictions.json", result)
    with (out/"events.jsonl").open("w") as handle:
        for r in result["repetitions"]:
            handle.write(json.dumps(r)+"\n")
    import csv
    with (out/"frames.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("sync_timestamp", "available", "predicted_station"))
        writer.writerows((int(clock), int(v), CLASSES[int(label)]) for clock, v, label in zip(t, valid, labels))
    return result


def cross_validate(root, output, *, folds=5, seed=42, mode="causal", tolerance_ms=400,
                   report_tolerances_ms=(200, 400), **fit_kwargs):
    root = Path(root)
    manifest = manifest_at(root)
    splits = subject_folds(manifest["sessions"], folds, seed)
    out = new_directory(output)
    all_records, summaries = [], []
    for number, split in enumerate(splits, 1):
        print(f"Fold {number}/{folds}", flush=True)
        fold = out/f"fold_{number}"
        model = fit(root, fold/"model", split=split, seed=seed, **fit_kwargs)
        report, records = evaluate(root, model, fold/"test", mode=mode, tolerance_ms=tolerance_ms,
                                    report_tolerances_ms=report_tolerances_ms)
        summaries.append({"fold": number, "split": split,
                           "segmentation_f1": report["segmentation"].get("f1"),
                           "repetition_f1": report["repetitions"].get("f1")})
        all_records.extend(records)
    tolerances = reporting_tolerances(report_tolerances_ms, tolerance_ms)
    scores = [s["segmentation_f1"] for s in summaries if s["segmentation_f1"] is not None]
    pooled = {**grouped_stream_report(all_records, tolerances, tolerance_ms),
              "offline_peak_counts": peak_count_summary(all_records),
              "report_scope": "subject_wise_cross_validation", "folds": summaries,
              "fold_segmentation_f1_mean": float(np.mean(scores)) if scores else None,
              "fold_segmentation_f1_std": float(np.std(scores)) if scores else None,
              **{key: report[key] for key in ("method", "mode", "timing_tolerance_ms", "report_tolerances_ms",
                                              "reference_convention", "phase_order_validated", "excluded_sessions",
                                              "coverage_note", "latency_note", "boundary_note", "limitations")}}
    write_json(out/"report.json", pooled)
    return pooled
