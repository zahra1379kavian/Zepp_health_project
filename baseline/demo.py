"""Generate a small labelled continuous workout for integration checks only."""

from pathlib import Path

import numpy as np
import pandas as pd

from strength_training.temporal_segmentation.constants import DT_MS, FEATURES
from strength_training.temporal_segmentation.data import prepare, write_json


def synthetic_source(root, subjects=6):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    rng, logs, alignments = np.random.default_rng(19), [], []
    for subject in range(subjects):
        subject_id = f"RECODEMO{subject:03d}"
        session = "20260101_"+subject_id
        folder = root/session
        folder.mkdir()
        t = 1767225600000+np.arange(1400, dtype=np.int64)*DT_MS
        labels = np.full(len(t), "rest", dtype="<U40")
        x = rng.normal(0, .015, (len(t), len(FEATURES)))
        for name, start, length, count, bias in (("squat", 100, 50, 8, 2.),
                                                 ("push-up", 650, 40, 8, -2.)):
            for rep in range(count):
                a, b = start+rep*length, start+(rep+1)*length
                phase = np.arange(length)*2*np.pi/length
                labels[a:a+length//2] = name+"_phase1"
                labels[a+length//2:b] = name+"_phase2"
                x[a:b, 0] += np.sin(phase)
                x[a:b, 1] += .15*np.cos(phase)
                x[a:b, 2] += bias
                x[a:b, 3] += np.cos(phase)
                x[a:b, 4] += .2*np.sin(phase)
        labels[1100:1200] = "plank"
        x[1100:1200, 2] += 4
        pd.DataFrame({"sync_timestamp": t, "left_wrist": labels, "right_wrist": labels}).to_csv(folder/"label.csv", index=False)
        for side in ("left_wrist", "right_wrist"):
            frame = pd.DataFrame(x+rng.normal(0, .003, x.shape), columns=FEATURES)
            frame.insert(0, "sync_timestamp", t)
            frame["timestamp"] = t+1234
            frame["labels"] = labels
            frame.to_csv(folder/f"{side}.csv", index=False)
        logs.append({"collect_date": "2026/01/01", "masked_id": subject_id, "scene_id": "action",
                     "device_start_time": "08:00:00", "device_end_time": "08:00:00.4"})
        alignments.append({"session": session, "offset_hours": 8.,
                           "named_station_match_fraction": 1., "runner_up_fraction": 0.})
    pd.DataFrame(logs).to_csv(root/"demo-Sheet1.csv", index=False)
    alignment = root.parent/"alignment.csv"
    pd.DataFrame(alignments).to_csv(alignment, index=False)
    return alignment


def run_demo(output):
    from .pipeline import cross_validate, predict
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    alignment = synthetic_source(output/"source")
    prepare(output/"source", output/"data", alignment)
    cross_validate(output/"data", output/"cross_validation", folds=3, window_seconds=1.,
                   train_step_ms=200, max_windows_per_class=128, c_values=(.1, 1.))
    predict(output/"cross_validation"/"fold_1"/"model"/"model.joblib",
            output/"source"/"20260101_RECODEMO000"/"left_wrist.csv", output/"prediction")
    write_json(output/"SYNTHETIC_ONLY.json", {"purpose": "Integration check, not evidence of real-data accuracy"})
