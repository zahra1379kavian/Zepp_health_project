"""Online event pairing, with no ground-truth stations or set boundaries."""

from collections import Counter
from dataclasses import asdict, dataclass

import numpy as np

from .constants import CLASSES, DT_MS


@dataclass
class DecoderConfig:
    start_threshold: float = .5
    end_threshold: float = .5
    release_threshold: float = .25
    station_threshold: float = .5
    min_rep_ms: int = 240
    max_rep_ms: int = 30000
    refractory_ms: int = 200
    set_gap_ms: int = 10000

    def __post_init__(self):
        if not (0 < self.release_threshold < min(self.start_threshold, self.end_threshold) <= 1):
            raise ValueError("release threshold must be below both event thresholds")
        if not 0 <= self.station_threshold <= 1 or not 0 < self.min_rep_ms < self.max_rep_ms:
            raise ValueError("Invalid station threshold or repetition duration")
        if self.refractory_ms < 0 or self.set_gap_ms < 0:
            raise ValueError("Time intervals must be nonnegative")


class RepDecoder:
    def __init__(self, config=None):
        self.config = config or DecoderConfig()
        self.reset()

    def reset(self):
        self.active = None
        self.previous_time = None
        self.armed = [True, True]
        self.last_event = [-10**18, -10**18]

    def step(self, timestamp, station_prob, boundary_prob, valid=True, reported_at_ms=None):
        t, c = int(timestamp), self.config
        if not valid:
            self.reset()
            return []
        if self.previous_time is not None and t - self.previous_time != DT_MS:
            self.reset()
        self.previous_time = t
        if self.active and t - self.active["start_ms"] > c.max_rep_ms:
            self.active = None
        events = []
        fire = [False, False]
        for i, threshold in enumerate((c.start_threshold, c.end_threshold)):
            if boundary_prob[i] < c.release_threshold:
                self.armed[i] = True
            if (self.armed[i] and boundary_prob[i] >= threshold and
                    t - self.last_event[i] >= c.refractory_ms):
                fire[i], self.armed[i], self.last_event[i] = True, False, t
        # Close first; an adjacent next repetition can start at this same tick.
        if fire[1] and self.active:
            active = self.active
            if t - active["start_ms"] >= c.min_rep_ms:
                probs = active["sum"] / max(active["samples"], 1)
                label = int(np.argmax(probs))
                if label != 0 and CLASSES[label] != "plank" and probs[label] >= c.station_threshold:
                    events.append({"station": CLASSES[label], "start_ms": active["start_ms"],
                                   "end_ms": t, "confidence": float(probs[label]),
                                   "reported_at_ms": t if reported_at_ms is None else int(reported_at_ms)})
                self.active = None
        # An unmatched new start replaces an incomplete cycle; no guessed end.
        if fire[0]:
            self.active = {"start_ms": t, "sum": np.zeros(len(CLASSES)), "samples": 0}
        if self.active:
            self.active["sum"] += station_prob
            self.active["samples"] += 1
        return events


def group_sets(repetitions, gap_ms=10000):
    sets = []
    for rep in repetitions:
        if (not sets or sets[-1]["station"] != rep["station"] or
                rep["start_ms"] - sets[-1]["end_ms"] > gap_ms):
            sets.append({"station": rep["station"], "start_ms": rep["start_ms"],
                         "end_ms": rep["end_ms"], "count": 1})
        else:
            sets[-1]["end_ms"] = rep["end_ms"]
            sets[-1]["count"] += 1
    return sets


def decode_sequence(t, station_prob, boundary_prob, valid, config=None):
    decoder, reps, holds = RepDecoder(config), [], []
    hold_start = None
    previous = None
    for i, clock in enumerate(t):
        gap = previous is not None and clock - previous != DT_MS
        is_hold = bool(valid[i]) and int(station_prob[i].argmax()) == CLASSES.index("plank")
        if hold_start is not None and (not is_hold or gap):
            end = int(previous + DT_MS)
            holds.append({"station": "plank", "start_ms": hold_start,
                          "end_ms": end, "duration_s": (end - hold_start) / 1000})
            hold_start = None
        if is_hold and hold_start is None:
            hold_start = int(clock)
        reps.extend(decoder.step(clock, station_prob[i], boundary_prob[i], bool(valid[i])))
        previous = int(clock)
    if hold_start is not None:
        end = int(t[-1] + DT_MS)
        holds.append({"station": "plank", "start_ms": hold_start, "end_ms": end,
                      "duration_s": (end - hold_start) / 1000, "truncated": True})
    # Active partial cycles are deliberately not emitted at EOF.
    return {"repetitions": reps, "counts": dict(Counter(r["station"] for r in reps)),
            "holds": holds, "sets": group_sets(reps, decoder.config.set_gap_ms),
            "decoder": asdict(decoder.config)}
