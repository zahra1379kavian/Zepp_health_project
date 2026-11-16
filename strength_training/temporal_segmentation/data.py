"""Exact-clock inputs, exclusion masks, subject splits and validated references."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from .constants import CLASSES, DT_MS, EXCLUDED_SCENES, EXCLUDED_SESSIONS, FEATURES, station


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def runs(mask):
    """Half-open runs of True; gaps must reset both the network and decoder."""
    edges = np.diff(np.r_[False, mask, False].astype(np.int8))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def load_sensor(path):
    # Explicit usecols excludes every embedded label, including duplicate headers.
    frame = pd.read_csv(path, usecols=["sync_timestamp", *FEATURES])
    t = frame.sync_timestamp.to_numpy(dtype=np.int64)
    x = frame[FEATURES].to_numpy(dtype=np.float32)
    if not len(t) or np.any(np.diff(t) <= 0) or not np.isfinite(x).all():
        raise ValueError(f"{path}: empty, nonfinite or unordered sensor input")
    if np.any((t - t[0]) % DT_MS):
        raise ValueError(f"{path}: timestamps are not on a 25 Hz grid")
    return t, x


def log_epoch(date, clock):
    day = datetime.strptime(str(date), "%Y/%m/%d").replace(tzinfo=timezone.utc)
    h, m, s = map(float, str(clock).split(":"))
    if not (0 <= h < 24 and 0 <= m < 60 and 0 <= s < 60):
        raise ValueError("invalid log time")
    return int(day.timestamp() * 1000 + (h * 3600 + m * 60 + s) * 1000)


def log_masks(t, rows, alignment):
    """Fail closed when a relevant interval cannot be placed on the sensor clock."""
    valid, probe = np.ones(len(t), bool), np.zeros(len(t), bool)
    intervals = []
    if rows.empty:
        raise ValueError("missing workout log; exclusion coverage cannot be checked")
    affected = rows[rows.scene_id.str.lower().str.strip().isin(EXCLUDED_SCENES)]
    if affected.empty:
        return valid, probe, intervals
    if alignment is None or not np.isfinite(alignment.get("offset_hours", np.nan)):
        raise ValueError("unresolved log-to-sensor offset")
    if (alignment.get("named_station_match_fraction", 0) <= .5 or
            alignment.get("named_station_match_fraction", 0) -
            alignment.get("runner_up_fraction", 1) <= .1):
        raise ValueError("insufficient log alignment evidence")
    offset = float(alignment["offset_hours"]) * 3600000
    for row in affected.itertuples():
        try:
            a = int(log_epoch(row.collect_date, row.device_start_time) - offset)
            b = int(log_epoch(row.collect_date, row.device_end_time) - offset)
        except (ValueError, TypeError, OverflowError) as error:
            raise ValueError("invalid affected log interval; review needed") from error
        if b <= a:
            raise ValueError("non-positive affected log interval; review needed")
        scene = row.scene_id.strip().lower()
        mask = (t >= a) & (t < b)
        valid[mask] = False
        if scene == "spull":
            probe[mask] = True
        intervals.append({"scene": scene, "start_ms": a, "end_ms": b,
                          "overlap_samples": int(mask.sum())})
    # Failed recordings and calibration take priority over the unseen probe.
    for interval in intervals:
        if interval["scene"] != "spull":
            probe[(t >= interval["start_ms"]) & (t < interval["end_ms"])] = False
    return valid, probe, intervals


def reference_targets(t, valid, entry, location):
    """Coverage certifies exhaustive dynamic-repetition annotations, including zeros.

    Intervals and repetitions use [start_ms, end_ms). End impulses are placed at
    the exclusive end clock tick, so end decisions require no future samples.
    """
    boundary = np.zeros((len(t), 2), np.float32)
    covered = np.zeros(len(t), bool)
    if entry is None:
        return boundary, covered, [], []
    if (entry.get("validated") is not True or entry.get("location") != location or
            not entry.get("source") or not entry.get("convention")):
        raise ValueError("References need validated=true, matching location, source and convention")
    intervals = sorted(entry.get("coverage", []), key=lambda r: r["start_ms"])
    for i, row in enumerate(intervals):
        a, b = int(row["start_ms"]), int(row["end_ms"])
        if (a < t[0] or b > t[-1] + DT_MS or b <= a or
                (a - t[0]) % DT_MS or (b - t[0]) % DT_MS or
                (i and a < intervals[i-1]["end_ms"])):
            raise ValueError("Reference coverage must be ordered, disjoint and on the label grid")
        covered |= (t >= a) & (t < b)
    repetitions, removed = [], []
    ordered = sorted(entry.get("repetitions", []), key=lambda r: r["start_ms"])
    previous_end = -1
    for row in ordered:
        a, b, name = int(row["start_ms"]), int(row["end_ms"]), row["station"]
        if (name not in CLASSES[1:] or name == "plank" or b <= a or a < previous_end or
                (a - t[0]) % DT_MS or (b - t[0]) % DT_MS):
            raise ValueError(f"Invalid or overlapping full repetition: {row}")
        previous_end = b
        # End tick must itself be in certified coverage for boundary supervision.
        if not any(c["start_ms"] <= a < b < c["end_ms"] for c in intervals):
            raise ValueError("Each repetition, including its end tick, must be inside coverage")
        i, j = np.searchsorted(t, [a, b])
        if not valid[i:j+1].all():
            # Do not turn pieces of a cut repetition into boundary negatives.
            covered[i:j+1] = False
            removed.append({**row, "reason": "overlaps exclusion or missing sensor"})
            continue
        boundary[i, 0] = 1
        boundary[j, 1] = 1
        repetitions.append({"station": name, "start_ms": a, "end_ms": b})
    return boundary, covered & valid, repetitions, removed


def phase_candidates(t, raw):
    """Audit only: adjacent phase1→phase2 runs are NOT validated repetitions."""
    starts = np.r_[0, np.flatnonzero(raw[1:] != raw[:-1]) + 1]
    ends = np.r_[starts[1:], len(raw)]
    result = []
    for i in range(len(starts) - 1):
        label = str(raw[starts[i]])
        if label.endswith("_phase1") and raw[starts[i+1]] == label[:-7] + "_phase2":
            result.append({"station": station(label), "start_ms": int(t[starts[i]]),
                           "end_ms": int(t[ends[i+1]-1] + DT_MS), "validated": False})
    return result


def subject_split(sessions, seed=42):
    subjects = sorted({s["subject"] for s in sessions})
    if len(subjects) < 3:
        raise ValueError("At least three eligible subjects are needed for train/validation/test")
    np.random.default_rng(seed).shuffle(subjects)
    n = max(1, int(round(len(subjects) * .15)))
    groups = {"test": subjects[:n], "validation": subjects[n:2*n], "train": subjects[2*n:]}
    return {key: [s["session"] for s in sessions if s["subject"] in ids]
            for key, ids in groups.items()}


def prepare(source, output, alignment_path, location="waist", references_path=None, seed=42):
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("Source and output must be separate, non-nested directories")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    align = pd.read_csv(alignment_path).set_index("session").to_dict("index")
    log_files = list(source.glob("*Sheet1.csv"))
    if len(log_files) != 1:
        raise ValueError("Expected exactly one workout-log *Sheet1.csv")
    log = pd.read_csv(log_files[0])
    log["session"] = pd.to_datetime(log.collect_date, format="%Y/%m/%d").dt.strftime("%Y%m%d") + "_" + log.masked_id
    refs = json.loads(Path(references_path).read_text()) if references_path else {"version": 1, "sessions": {}}
    if refs.get("version") != 1:
        raise ValueError("Unsupported references schema")
    manifest = {"version": 1, "source": str(source), "location": location,
                "classes": CLASSES, "features": FEATURES, "sample_period_ms": DT_MS,
                "alignment_source": str(Path(alignment_path).resolve()),
                "references_source": str(Path(references_path).resolve()) if references_path else None,
                "seed": seed, "sessions": [], "excluded": [],
                "waist_selection": "lexicographically first waist filename; independent of labels"}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".tas-prepare-", dir=output.parent) as temp:
        dest = Path(temp) / "dataset"
        dest.mkdir()
        for folder in sorted(p for p in source.iterdir() if p.is_dir() and not p.name.startswith(".")):
            name = folder.name
            if name in EXCLUDED_SESSIONS:
                manifest["excluded"].append({"session": name, "reason": "plan whole-session exclusion"})
                continue
            paths = sorted(folder.glob("*_waist.csv")) if location == "waist" else list(folder.glob(f"{location}.csv"))
            if not paths:
                manifest["excluded"].append({"session": name, "reason": f"no {location} sensor"})
                continue
            labels = pd.read_csv(folder / "label.csv")
            if location not in labels:
                raise ValueError(f"{folder}: missing authoritative {location} labels")
            t = labels.sync_timestamp.to_numpy(np.int64)
            if not len(t) or np.any(np.diff(t) != DT_MS):
                raise ValueError(f"{folder}: label clock must be a complete 25 Hz grid")
            try:
                eligible, probe, exclusions = log_masks(t, log[log.session == name], align.get(name))
            except ValueError as error:
                manifest["excluded"].append({"session": name, "reason": str(error)})
                continue
            sensor_t, sensor_x = load_sensor(paths[0])
            idx = np.searchsorted(t, sensor_t)
            if np.any(idx >= len(t)) or not np.array_equal(t[idx], sensor_t):
                raise ValueError(f"{paths[0]}: sensor clock does not match label grid exactly")
            x, available = np.zeros((len(t), len(FEATURES)), np.float32), np.zeros(len(t), bool)
            x[idx], available[idx] = sensor_x, True
            raw = labels[location].to_numpy(dtype=str)
            mapping = {value: CLASSES.index(station(str(value))) for value in np.unique(raw)}
            y = np.array([mapping[value] for value in raw], dtype=np.int64)
            valid = eligible & available
            boundary, coverage, reps, cut = reference_targets(t, valid, refs["sessions"].get(name), location)
            path = dest / name
            path.mkdir()
            for key, array in dict(x=x, t=t, y=y, raw=raw, valid=valid, available=available,
                                   eligible=eligible, probe=probe & available,
                                   boundary=boundary, coverage=coverage).items():
                np.save(path / f"{key}.npy", array, allow_pickle=False)
            write_json(path / "references.json", reps)
            candidates = phase_candidates(t, raw)
            write_json(path / "unverified_phase_candidates.json", candidates)
            record = {"session": name, "subject": name.split("_", 1)[1], "sensor": paths[0].name,
                      "samples": len(t), "available_samples": int(available.sum()),
                      "valid_samples": int(valid.sum()), "masked_log_samples": int((~eligible).sum()),
                      "probe_samples": int((probe & available).sum()), "exclusions": exclusions,
                      "reference_samples": int(coverage.sum()), "reference_repetitions": len(reps),
                      "removed_references": cut, "unverified_phase_candidates": len(candidates),
                      "raw_label_counts": dict(Counter(map(str, raw)))}
            manifest["sessions"].append(record)
            print(f"Prepared {name}: {valid.sum():,} eligible samples; {len(reps)} validated reps", flush=True)
        unknown = set(refs["sessions"]) - {p.name for p in source.iterdir() if p.is_dir()}
        if unknown:
            raise ValueError(f"References name unknown sessions: {sorted(unknown)}")
        manifest["split"] = subject_split(manifest["sessions"], seed)
        sums, squares, count = np.zeros(len(FEATURES)), np.zeros(len(FEATURES)), 0
        class_counts = np.zeros(len(CLASSES), np.int64)
        boundary_positives, boundary_frames = np.zeros(2), 0
        for name in manifest["split"]["train"]:
            arrays = load_arrays(dest / name)
            selected = arrays["x"][arrays["valid"]].astype(np.float64)
            sums += selected.sum(0)
            squares += (selected ** 2).sum(0)
            count += len(selected)
            class_counts += np.bincount(arrays["y"][arrays["valid"]], minlength=len(CLASSES))
            mask = arrays["coverage"]
            boundary_positives += arrays["boundary"][mask].sum(0)
            boundary_frames += int(mask.sum())
        if count == 0:
            raise ValueError("No eligible training samples")
        mean = sums / count
        std = np.sqrt(np.maximum(squares / count - mean ** 2, 1e-8))
        manifest["normalization"] = {"mean": mean.tolist(), "std": std.tolist(), "fit_samples": count}
        manifest["train_class_counts"] = class_counts.tolist()
        manifest["train_boundary_positives"] = boundary_positives.tolist()
        manifest["train_boundary_frames"] = boundary_frames
        write_json(dest / "manifest.json", manifest)
        dest.rename(output)
    return manifest


def load_arrays(path):
    return {key: np.load(Path(path) / f"{key}.npy", mmap_mode="r", allow_pickle=False)
            for key in ("x", "t", "y", "valid", "boundary", "coverage", "probe")}
