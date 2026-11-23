"""Reference-defined clips, subject assignment, and a sensor-only predictor interface."""

from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import sys
import warnings

import numpy as np

# The release is a read-only checkout. Use its actual label and score functions.
_bytecode = sys.dont_write_bytecode
try:
    sys.dont_write_bytecode = True
    with warnings.catch_warnings():
        from StrengthTraining_wrist_activity_agnostic.src.data import utils
        from StrengthTraining_wrist_activity_agnostic.src.utils import rep
finally:
    sys.dont_write_bytecode = _bytecode

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT / "StrengthTraining_wrist_activity_agnostic"
RAW = ROOT / "strength_training/strength_training_data_clean"
ACTIVITIES = ("bench_press", "dips", "pull-up", "biceps_curls", "push-up",
              "lat_pulldown", "overhead_press_(ohp)", "rowing", "deadlift",
              "crunch", "hip_thrust", "leg_raise", "lunges", "squat")
MISSING = {"20260529_DQTQDQTQ", "20260529_JCLQJCLQ"}


@dataclass
class Prediction:
    """Clip-local, half-open sample indices. reported_at is the emission frame."""
    intervals: list[dict]
    # Optional frame traces for example plots. Each key names a probability.
    traces: dict[str, np.ndarray] | None = None


def subject(session):
    return session.split("_", 1)[1]


def session_plan(raw, fold_subjects):
    """Keep saved held-out subjects; use release folds for wholly unseen subjects."""
    splits = json.loads((REFERENCE / "splits.json").read_text())
    release = {s: i for i, f in enumerate(splits["folds"], 1) for s in f["val"]}
    missing = sorted(s for s in release if not (raw / s).is_dir())
    if set(missing) - MISSING:
        raise ValueError(f"Unexpected missing reference sessions: {missing}")
    if set(fold_subjects) != set(range(1, 6)):
        raise ValueError("Expected five folds with train, validation, and test subjects")
    groups = {}
    for fold, parts in fold_subjects.items():
        train, val, test = (set(parts[k]) for k in ("train", "validation", "test"))
        if train & val or (train | val) & test:
            raise ValueError(f"Overlapping subject partitions in fold {fold}")
        for person in test:
            if person in groups:
                raise ValueError(f"Repeated held-out subject: {person}")
            groups[person] = fold
    known = set().union(*(set(ids) for p in fold_subjects.values() for ids in p.values()))
    plan = []
    for session, ref_fold in sorted(release.items()):
        if session in missing:
            continue
        person = subject(session)
        if person in known and person not in groups:
            raise ValueError(f"No held-out checkpoint for subject {person}")
        fold = groups.get(person, ref_fold)
        if person in fold_subjects[fold]["train"] or person in fold_subjects[fold]["validation"]:
            raise ValueError(f"Subject leakage: {session} in fold {fold}")
        plan.append(dict(session=session, subject=person, fold=fold, release_fold=ref_fold,
                         assignment="held_out" if person in groups else "previously_unused"))
    if len({(r["subject"], r["fold"]) for r in plan}) != len({r["subject"] for r in plan}):
        raise ValueError("A subject was assigned to multiple folds")
    return plan, missing


def clips(folder):
    """Use the release's wrist join, four-class mapping, and five-second margins."""
    label_file = folder / "label.csv"
    sides = sorted(folder.glob("*wrist.csv"))
    if not sides or not label_file.is_file():
        raise ValueError(f"Missing wrist or label data in {folder}")
    for wrist in sides:
        df, labels = utils.read_device_csv_with_time(wrist, label_file)
        device = utils.short_source_name(wrist.name)
        label = labels[device]
        timestamps = df.sync_timestamp.to_numpy(dtype=np.int64)
        sensors = df[utils.REQUIRED_COLS].to_numpy(dtype=np.float32)
        for activity in ACTIVITIES:
            first, second = activity + "_phase1", activity + "_phase2"
            if not (label == first).any():
                continue
            pos = np.flatnonzero(label.isin((first, second)))
            start = max(0, pos[0] - 5 * utils.FS)
            stop = min(len(df), pos[-1] + 5 * utils.FS + 1)
            y = utils.map_label_column_to_int(activity, label.iloc[start:stop])
            yield device, activity, timestamps[start:stop], sensors[start:stop], y


def score(prediction, y):
    n = len(y)
    intervals = []
    for event in sorted(prediction.intervals, key=lambda r: r["start"]):
        for key in ("start", "end", "reported_at"):
            if key in event and (isinstance(event[key], bool) or not isinstance(event[key], (int, np.integer))):
                raise ValueError(f"Expected integer {key} in {event}")
        if not 0 <= event["start"] < event["end"] <= n:
            raise ValueError(f"Interval outside clip of {n} samples: {event}")
        if "reported_at" in event and not event["end"] - 1 <= event["reported_at"] < n:
            raise ValueError(f"Emission outside observable clip: {event}")
        intervals.append({k: int(event[k]) for k in ("start", "end", "reported_at") if k in event})
    gt = rep.extract_reps_from_labels(y)
    metrics, _, _ = rep.compute_rep_detection_metrics(gt, intervals, y, fs=utils.FS)
    matched, _, _ = rep.match_reps_greedy(gt, intervals)
    delays = [int(p["reported_at"] - g["end"]) * 40 for g, p in matched if "reported_at" in p]
    return gt, intervals, metrics, delays


def clip_signature(t, y):
    h = hashlib.sha256(t.astype("<i8", copy=False).tobytes())
    h.update(y.astype("u1", copy=False).tobytes())
    return h.hexdigest()


def input_provenance(raw, sessions):
    """Fingerprint raw inputs and the release code separately from model weights."""
    files = []
    for session in sorted(sessions):
        folder = Path(raw) / session
        files.extend([folder / "label.csv", *sorted(folder.glob("*wrist.csv"))])
    fingerprint = hashlib.sha256()
    for path in files:
        with path.open("rb") as f:
            hashed = hashlib.file_digest(f, "sha256").hexdigest()
        fingerprint.update(f"{path.relative_to(raw)}:{hashed}\n".encode())
    ref_files = ("splits.json", "src/data/dataset.py", "src/data/utils.py",
                 "src/utils/rep.py", "src/utils/plot.py")
    return dict(raw_inputs_sha256=fingerprint.hexdigest(), reference_sha256={
        name: hashlib.sha256((REFERENCE / name).read_bytes()).hexdigest() for name in ref_files})
