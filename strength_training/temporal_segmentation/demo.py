"""Small synthetic fixture with known boundaries; never a real-data benchmark."""

from pathlib import Path

import numpy as np
import pandas as pd

from .constants import DT_MS, FEATURES
from .data import write_json


def synthetic_source(root, subjects=8):
    root = Path(root)
    root.mkdir(parents=True)
    rng, log, align, sessions = np.random.default_rng(7), [], [], {}
    for subject in range(subjects):
        name = f"20260101_DEMO{subject:03d}"
        folder = root / name
        folder.mkdir()
        t = 1767225600000 + np.arange(600) * DT_MS
        raw = np.full(len(t), "rest", dtype="<U40")
        x = rng.normal(0, .05, (len(t), len(FEATURES))).astype(np.float32)
        repetitions = []
        for a, b, station in [(100, 150, "squat"), (175, 225, "squat"),
                              (300, 375, "push-up"), (400, 475, "push-up")]:
            mid = (a+b)//2
            raw[a:mid], raw[mid:b] = station + "_phase1", station + "_phase2"
            phase = np.linspace(0, 2*np.pi, b-a)
            x[a:b, 0] += np.sin(phase)
            x[a:b, 1] += np.cos(phase)
            x[a:b, 2] += 1 if station == "squat" else -1
            repetitions.append({"station": station, "start_ms": int(t[a]), "end_ms": int(t[b])})
        raw[500:550] = "plank"
        x[500:550, 3] += 2
        pd.DataFrame({"sync_timestamp": t, "left_wrist": raw}).to_csv(folder / "label.csv", index=False)
        frame = pd.DataFrame(x, columns=FEATURES)
        frame.insert(0, "sync_timestamp", t)
        frame["timestamp"] = t + 1234
        frame["labels"] = raw  # Deliberate leakage trap: loader must ignore it.
        frame.to_csv(folder / "left_wrist.csv", index=False)
        log.append({"collect_date": "2026/01/01", "masked_id": f"DEMO{subject:03d}",
                    "scene_id": "action", "device_start_time": "08:00:00", "device_end_time": "08:00:00.4"})
        align.append({"session": name, "offset_hours": 8,
                      "named_station_match_fraction": 1., "runner_up_fraction": 0.})
        sessions[name] = {"validated": True, "location": "left_wrist", "source": "synthetic generator",
                          "convention": "Known generated cycles; end is exclusive",
                          "coverage": [{"start_ms": int(t[10]), "end_ms": int(t[-1]+DT_MS)}],
                          "repetitions": repetitions}
    pd.DataFrame(log).to_csv(root / "demo-Sheet1.csv", index=False)
    pd.DataFrame(align).to_csv(root.parent / "alignment.csv", index=False)
    write_json(root.parent / "references.json", {"version": 1, "sessions": sessions})
    return root.parent / "alignment.csv", root.parent / "references.json"


def run_demo(output, device="cpu"):
    from .__main__ import main
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    alignment, references = synthetic_source(output / "source")
    main(["prepare", "--source", str(output / "source"), "--alignment", str(alignment),
          "--reference-mode", "labels", "--output", str(output / "prepared")])
    main(["train", "--data", str(output / "prepared"), "--output", str(output / "model"),
          "--epochs", "2", "--max-steps", "4", "--stages", "2", "--layers", "3", "--channels", "16",
          "--batch-size", "4", "--window", "128", "--device", device])
    checkpoint = str(output / "model" / "best.pt")
    main(["tune", "--data", str(output / "prepared"), "--checkpoint", checkpoint,
          "--output", str(output / "decoder.json"), "--device", device])
    main(["evaluate", "--data", str(output / "prepared"), "--checkpoint", checkpoint,
          "--decoder", str(output / "decoder.json"), "--output", str(output / "evaluation"), "--device", device])
    main(["predict", "--input", str(output / "source" / "20260101_DEMO000" / "left_wrist.csv"),
          "--checkpoint", checkpoint, "--decoder", str(output / "decoder.json"),
          "--output", str(output / "prediction"), "--device", device])
    write_json(output / "SYNTHETIC_ONLY.json", {"purpose": "Pipeline smoke test, not evidence of real-world accuracy"})
    print(f"Synthetic smoke test finished: {output}")
