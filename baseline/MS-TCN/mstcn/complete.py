"""Endpoint-only targets and held-out rest intervals for complete-rep training."""

import json
from pathlib import Path

import numpy as np

from .constants import DT_MS, START_HISTORY
from .data import load_arrays, runs, write_json


def complete_targets(arrays, repetitions, reviews=()):
    t, valid = arrays["t"], arrays["valid"]
    covered = np.array(arrays["coverage"] & valid)
    weight = covered.astype(np.float32)
    for row in reviews:
        a, b = np.searchsorted(t, [row["start_ms"], row["end_ms"]])
        if (a >= b or a >= len(t) or t[a] != row["start_ms"] or
                t[b-1] + DT_MS != row["end_ms"] or
                not covered[a:b].all() or arrays["y"][a:b].any()):
            raise ValueError(f"Review interval must be within certified rest: {row}")
        status = row["review"]
        if status == "confirmed_rest":
            weight[a:b] = 3
        elif status in ("contains_reps", "uncertain", "unreviewed"):
            weight[a:b] = 0
        else:
            raise ValueError(f"Unknown review decision: {status}")
    end = np.zeros(len(t), np.float32)
    start_choice = np.full(len(t), -1, np.int64)
    # Use paired timestamps, including shared end/start ticks and pauses.
    pairs = [(int(np.searchsorted(t, r["start_ms"])), int(np.searchsorted(t, r["end_ms"])))
             for r in repetitions]
    for a, b in pairs:
        middle = (a + b) // 2
        if a < middle < b:
            start_choice[middle] = START_HISTORY  # false end -> no start
    # Prefer the nearer true end if two jitter neighborhoods touch.
    for shift in (-2, 2, -1, 1, 0):
        for a, b in pairs:
            i = b + shift
            if not 0 <= i < len(t) or not covered[min(i, b):max(i, b)+1].all():
                continue
            end[i] = 1 - abs(shift) / 3
            distance = i - a
            start_choice[i] = START_HISTORY - distance if 1 <= distance <= START_HISTORY else START_HISTORY
    return {"end": end, "start_choice": start_choice, "weight": weight}


def rest_candidates(arrays, repetitions, max_gap_ms=10000):
    """Consecutive predictions wholly within one continuous, certified rest interval."""
    t = arrays["t"]
    rest = arrays["coverage"] & arrays["valid"] & (arrays["y"] == 0)
    run_id = np.full(len(t), -1, np.int32)
    for i, (a, b) in enumerate(runs(rest)):
        run_id[a:b] = i
    groups, group, previous = [], [], -1
    for rep in sorted(repetitions, key=lambda r: r["end_ms"]):
        a, b = np.searchsorted(t, [rep["start_ms"], rep["end_ms"]])
        current = int(run_id[a]) if 0 <= a < b < len(t) and run_id[a] == run_id[b] else -1
        if current < 0 or current != previous or (group and rep["start_ms"] - group[-1]["end_ms"] > max_gap_ms):
            if len(group) >= 3:
                groups.append(group)
            group = []
        if current >= 0:
            group.append(rep)
        previous = current
    if len(group) >= 3:
        groups.append(group)
    return [{"start_ms": group[0]["start_ms"], "end_ms": group[-1]["end_ms"] + DT_MS,
             "predicted_reps": len(group), "review": "unreviewed"} for group in groups]


def mine_rest(args):
    root, cv, out = Path(args.data), Path(args.predictions), Path(args.output)
    if out.exists():
        raise FileExistsError(out)
    manifest = json.loads((root / "manifest.json").read_text())
    subjects = {r["session"]: r["subject"] for r in manifest["sessions"]}
    rows, seen = [], set()
    for fold in sorted(cv.glob("fold_*")):
        split = json.loads((fold / "data/manifest.json").read_text())["split"]
        fitting = {subjects[s] for s in split["train"] + split["validation"]}
        for name in split["test"]:
            if subjects[name] in fitting or name in seen:
                raise ValueError(f"Invalid held-out split for {name}")
            seen.add(name)
            predictions = json.loads((fold / "test" / f"{name}.json").read_text())
            candidates = rest_candidates(load_arrays(root / name, include_phase=False), predictions["repetitions"])
            rows.extend({"session": name, "fold": fold.name, **row} for row in candidates)
    if not seen:
        raise ValueError("No held-out fold predictions found")
    out.parent.mkdir(parents=True, exist_ok=True)
    write_json(out, {"prediction_source": str(cv.resolve()), "held_out_streams": len(seen), "intervals": rows})
    print(f"Found {len(rows)} candidate rest stretches across {len(seen)} held-out streams")
