"""Score independent wrist clips and preserve everything needed to audit the reports."""

from collections import Counter
from pathlib import Path
import hashlib
import json
import tempfile

import numpy as np
import pandas as pd

from .protocol import REFERENCE, clips, clip_signature, score, session_plan


def dump(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def example(candidates, scope, activity, row, t, x, y, gt, pred, traces):
    if activity not in ("push-up", "deadlift", "leg_raise"):
        return
    denom = 2 * row["TP"] + row["FN"] + row["FP"]
    if not denom or not gt:
        return
    f1 = 2 * row["TP"] / denom
    for kind in ("good", "bad"):
        key = (scope, activity, kind)
        target = -f1 if kind == "good" else f1
        # A deterministic tie-break favors clips with more labeled repetitions.
        rank = (target, -len(gt), row["session"], row["device"])
        if key in candidates and candidates[key][0] <= rank:
            continue
        traces = traces or {}
        if any(np.shape(v) != (len(t),) for v in traces.values()):
            raise ValueError("Model traces must have one value per input frame")
        candidates[key] = (rank, dict(session=row["session"], device=row["device"], f1=f1,
                                      ground_truth=gt, predictions=pred),
                           dict(labels=y, acc_norm=np.linalg.norm(x[:, :3], axis=1),
                                gyro_norm=np.linalg.norm(x[:, 3:6], axis=1), **traces))


def evaluate(raw, output, model_name, fold_subjects, load_model):
    """load_model(fold) returns an object with metadata and predict(t_ms, sensors)."""
    raw, output = Path(raw), Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    plan, missing = session_plan(raw, fold_subjects)
    with tempfile.TemporaryDirectory(dir=output.parent, prefix=".rep-benchmark-") as tmp:
        stage = Path(tmp) / "results"
        stage.mkdir()
        rows, candidates, models = [], {}, {}
        coverage = Counter()
        signature = hashlib.sha256()
        with (stage / "intervals.jsonl").open("w") as events:
            for fold in sorted(fold_subjects):
                model = load_model(fold)
                models[fold] = model.metadata
                for item in (p for p in plan if p["fold"] == fold):
                    session = item["session"]
                    for side, activity, t, x, y in clips(raw / session):
                        prediction = model.predict(t, x)
                        gt, pred, metrics, delays = score(prediction, y)
                        row = dict(session=session, device=side, activity=activity, fold=fold,
                                   release_fold=item["release_fold"], **metrics)
                        rows.append(row)
                        coverage[session] += 1
                        sig = clip_signature(t, y)
                        signature.update(f"{fold}/{session}/{side}/{activity}/{sig}\n".encode())
                        record = dict(session=session, device=side, activity=activity, fold=fold,
                                      first_ms=int(t[0]), last_ms=int(t[-1]), frames=len(t),
                                      clip_sha256=sig, ground_truth=gt, predictions=pred,
                                      matched_delay_ms=delays)
                        events.write(json.dumps(record, separators=(",", ":")) + "\n")
                        for scope in (["pooled", "fold_5"] if fold == 5 else ["pooled"]):
                            example(candidates, scope, activity, row, t, x, y, gt, pred, prediction.traces)
                    print(f"Fold {fold} {session}: {coverage[session]} clips", flush=True)
        frame = pd.DataFrame(rows)
        frame.to_csv(stage / "test_metrics_detailed.csv", index=False)
        for (scope, activity, kind), (_, meta, signals) in candidates.items():
            path = stage / scope / "examples"
            path.mkdir(parents=True, exist_ok=True)
            dump(path / f"{activity}_{kind}.json", meta)
            np.savez_compressed(path / f"{activity}_{kind}.npz", **signals)
        dump(stage / "manifest.json", dict(model=model_name, sensor_hz=25, models=models,
             missing_sessions=missing, fold_subjects=fold_subjects,
             sessions=[dict(**p, clips=coverage[p["session"]]) for p in plan],
             reference_splits_sha256=hashlib.sha256((REFERENCE / "splits.json").read_bytes()).hexdigest(),
             protocol_sha256=signature.hexdigest(), raw_source=str(raw.resolve())))
        from .report import render
        render(stage)
        stage.rename(output)
    return output
