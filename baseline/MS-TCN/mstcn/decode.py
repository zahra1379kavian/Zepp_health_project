"""Online event pairing, with no ground-truth stations or set boundaries."""

from collections import Counter, deque
from dataclasses import asdict, dataclass

import numpy as np

from .constants import CLASSES, DT_MS


@dataclass
class DecoderConfig:
    start_threshold: float = .5
    end_threshold: float = .5
    release_threshold: float = .25
    station_threshold: float = .5
    phase_threshold: float = .5
    lag_frames: int = 2
    min_rep_ms: int = 240
    max_rep_ms: int = 30000
    refractory_ms: int = 200
    set_gap_ms: int = 10000

    def __post_init__(self):
        if not (0 < self.release_threshold < min(self.start_threshold, self.end_threshold)
                and max(self.start_threshold, self.end_threshold) <= 1):
            raise ValueError("release threshold must be below both event thresholds")
        if not 0 <= self.station_threshold <= 1 or not 0 < self.min_rep_ms < self.max_rep_ms:
            raise ValueError("Invalid station threshold or repetition duration")
        if self.refractory_ms < 0 or self.set_gap_ms < 0:
            raise ValueError("Time intervals must be nonnegative")
        if (not isinstance(self.lag_frames, int) or not 0 <= self.lag_frames <= 3
                or not 0 < self.phase_threshold <= 1):
            raise ValueError("Use 0..3 confirmation frames and a phase threshold in (0, 1]")


class RepDecoder:
    def __init__(self, config=None):
        self.config = config or DecoderConfig()
        self.reset()

    def reset(self):
        self.active = None
        self.previous_time = None
        self.armed = [True, True]
        self.last_event = [-10**18, -10**18]
        self.frames = deque()
        self.phase = 0
        self.previous_boundary = np.zeros(2)

    def step(self, timestamp, station_prob, boundary_prob, valid=True, reported_at_ms=None):
        t, c = int(timestamp), self.config
        if not valid:
            self.reset()
            return []
        if self.previous_time is not None and t - self.previous_time != DT_MS:
            self.reset()
        self.previous_time = t
        # Delay decisions by two frames (80 ms by default), but retain the
        # original boundary timestamp. Never flush unconfirmed events at EOF.
        self.frames.append((t, np.array(station_prob, copy=True), np.array(boundary_prob, copy=True)))
        if len(self.frames) <= c.lag_frames:
            return []
        reported = t if reported_at_ms is None else max(t, int(reported_at_ms))
        t, station_prob, boundary_prob = self.frames[0]
        future = list(self.frames)[1:]
        self.frames.popleft()
        if self.active and t - self.active["start_ms"] > c.max_rep_ms:
            self.active = None
        events, fire = [], [False, False]
        if len(boundary_prob) == 3:
            phase = int(boundary_prob.argmax())
            stable = boundary_prob[phase] >= c.phase_threshold and all(
                int(q.argmax()) == phase and q[phase] >= c.phase_threshold for _, _, q in future)
            if stable and phase != self.phase:
                fire = [phase == 1, self.phase == 2]
                if self.phase == 1 and phase == 0:
                    self.active = None  # An incomplete phase1 is not a cycle.
                self.phase = phase
        elif len(boundary_prob) == 2:
            for i, threshold in enumerate((c.start_threshold, c.end_threshold)):
                if boundary_prob[i] < c.release_threshold:
                    self.armed[i] = True
                peak = (boundary_prob[i] >= self.previous_boundary[i] and
                        all(boundary_prob[i] >= q[i] for _, _, q in future))
                if (peak and self.armed[i] and boundary_prob[i] >= threshold and
                        t - self.last_event[i] >= c.refractory_ms):
                    fire[i], self.armed[i], self.last_event[i] = True, False, t
            self.previous_boundary = boundary_prob
        else:
            raise ValueError("Expected two boundary probabilities or three phase probabilities")
        # Close first; an adjacent next repetition can start at this same tick.
        if fire[1] and self.active:
            active = self.active
            if t - active["start_ms"] >= c.min_rep_ms:
                probs = active["sum"] / max(active["samples"], 1)
                label = 1 + int(np.argmax(probs[1:]))
                # Station uncertainty must not veto a complete cycle. Weak
                # known-class evidence is labelled unknown and still counted.
                name = CLASSES[label] if probs[label] >= c.station_threshold else "unknown"
                if name != "plank":
                    events.append({"station": name, "start_ms": active["start_ms"],
                                   "end_ms": t, "confidence": float(probs[label]),
                                   "activity_probability": float(1 - probs[0]),
                                   "reported_at_ms": reported})
                self.active = None
            elif len(boundary_prob) == 3:
                # A completed but implausibly short phase cycle cannot remain
                # open and later absorb an orphan phase2.
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


def decode_sequence(t, station_prob, boundary_prob, valid, config=None, lookahead_ms=0):
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
        reps.extend(decoder.step(clock, station_prob[i], boundary_prob[i], bool(valid[i]),
                                 reported_at_ms=int(clock) + lookahead_ms))
        previous = int(clock)
    if hold_start is not None:
        end = int(t[-1] + DT_MS)
        holds.append({"station": "plank", "start_ms": hold_start, "end_ms": end,
                      "duration_s": (end - hold_start) / 1000, "truncated": True})
    # Active partial cycles are deliberately not emitted at EOF.
    return {"repetitions": reps, "counts": dict(Counter(r["station"] for r in reps)),
            "holds": holds, "sets": group_sets(reps, decoder.config.set_gap_ms),
            "decoder": asdict(decoder.config)}
