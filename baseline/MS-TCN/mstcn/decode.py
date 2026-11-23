"""Online event pairing, with no ground-truth stations or set boundaries."""

from collections import Counter, deque
from dataclasses import asdict, dataclass

import numpy as np

from .constants import CLASSES, DT_MS, SET_GAP_MS, START_HISTORY, STATION_STATES


END_REFINEMENT_FRAMES = 4
END_WINDOW_OFFSETS_MS = np.arange(-END_REFINEMENT_FRAMES, END_REFINEMENT_FRAMES + 1) * DT_MS


def refined_end_time(peak_ms, scores):
    return peak_ms + round(np.average(END_WINDOW_OFFSETS_MS, weights=np.asarray(scores, dtype=np.float64)))


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
    set_gap_ms: int = SET_GAP_MS
    start_rule: str = "learned"
    refine_end: bool = False

    def __post_init__(self):
        if self.start_rule not in ("learned", "previous_end"):
            raise ValueError("Unknown start rule")
        if self.refine_end and self.start_rule != "learned":
            raise ValueError("End refinement requires the learned start rule")
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

    @property
    def confirmation_frames(self):
        return END_REFINEMENT_FRAMES if self.refine_end else self.lag_frames


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
        self.station_history = deque(maxlen=START_HISTORY + 1)
        self.last_rest = None
        self.previous_peak = None
        self.previous_rep_end = None
        self.previous_end_score = -1.
        self.recent_end_scores = deque(maxlen=END_REFINEMENT_FRAMES)

    def complete_peak(self, t, station, prediction, future, reported):
        c = self.config
        if station[1] < c.station_threshold:
            self.last_rest = t
        self.station_history.append((t, self.last_rest))
        end, distance, confidence, no_start = prediction
        if 1 - no_start < c.start_threshold:
            distance = 0
        peak = (end >= c.end_threshold and end > self.previous_end_score
                and all(end >= q[0] for _, _, q in future[:c.lag_frames]))
        self.previous_end_score = end
        end_ms = t
        if c.refine_end:
            if peak and len(self.recent_end_scores) == END_REFINEMENT_FRAMES:
                end_ms = refined_end_time(t, [*self.recent_end_scores, end, *(q[0] for _, _, q in future)])
            self.recent_end_scores.append(end)
        if not peak:
            return []
        return self.close_peak(t, float(station[1]), end, distance, confidence, reported,
                               len(self.station_history), lambda d: self.station_history[-d-1][1], end_ms)

    def close_peak(self, t, station_probability, end, distance, confidence, reported, history_length, rest_at, end_ms):
        c = self.config
        if self.previous_peak is not None and t - self.previous_peak < c.refractory_ms:
            return []
        previous, self.previous_peak = self.previous_peak, t
        if c.start_rule == "previous_end":
            if previous is None:
                return []
            distance, confidence = (t - previous) // DT_MS, end
        distance = int(distance)
        if not 1 <= distance <= START_HISTORY or distance >= history_length:
            return []
        start, rest_at_start = t - distance * DT_MS, rest_at(distance)
        if end_ms <= start or (self.previous_rep_end is not None and start < self.previous_rep_end):
            return []
        new_set = (self.previous_rep_end is None or
                   (start > self.previous_rep_end and rest_at_start is not None and rest_at_start >= self.previous_rep_end))
        self.previous_rep_end = end_ms
        event = {"station": "unknown", "start_ms": start, "end_ms": end_ms,
                 "confidence": float(end * confidence) if c.start_rule == "learned" else float(end),
                 "activity_probability": station_probability, "new_set": new_set,
                 "reported_at_ms": reported}
        if c.refine_end:
            event["end_peak_ms"] = t
        return [event]

    def step(self, timestamp, station_prob, boundary_prob, valid=True, reported_at_ms=None):
        t, c = int(timestamp), self.config
        if not valid:
            self.reset()
            return []
        if self.previous_time is not None and t - self.previous_time != DT_MS:
            self.reset()
        self.previous_time = t
        # Buffer the frames needed to confirm or refine an endpoint. Never
        # flush unconfirmed events at EOF.
        station_prob = None if station_prob is None else np.array(station_prob, copy=True)
        self.frames.append((t, station_prob, np.array(boundary_prob, copy=True)))
        if len(self.frames) <= c.confirmation_frames:
            return []
        reported = t if reported_at_ms is None else max(t, int(reported_at_ms))
        t, station_prob, boundary_prob = self.frames[0]
        future = list(self.frames)[1:]
        self.frames.popleft()
        if len(boundary_prob) == 4:
            return self.complete_peak(t, station_prob, boundary_prob, future, reported)
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
                activity = float(1 - probs[0])
                name, confidence = "unknown", activity
                if station_prob is not None and len(station_prob) != len(STATION_STATES):
                    label = 1 + int(np.argmax(probs[1:]))
                    confidence = float(probs[label])
                    name = CLASSES[label] if confidence >= c.station_threshold else "unknown"
                if name != "plank":
                    events.append({"station": name, "start_ms": active["start_ms"],
                                   "end_ms": t, "confidence": confidence,
                                   "activity_probability": activity,
                                   "reported_at_ms": reported})
                self.active = None
            elif len(boundary_prob) == 3:
                # A completed but implausibly short phase cycle cannot remain
                # open and later absorb an orphan phase2.
                self.active = None
        # An unmatched new start replaces an incomplete cycle; no guessed end.
        evidence = boundary_prob if station_prob is None else station_prob
        if fire[0]:
            self.active = {"start_ms": t, "sum": np.zeros(len(evidence)), "samples": 0}
        if self.active:
            self.active["sum"] += evidence
            self.active["samples"] += 1
        return events


def group_sets(repetitions, gap_ms=SET_GAP_MS):
    sets = []
    for rep in repetitions:
        if (not sets or rep.get("new_set") or sets[-1]["station"] != rep["station"] or
                rep["start_ms"] - sets[-1]["end_ms"] > gap_ms):
            sets.append({"station": rep["station"], "start_ms": rep["start_ms"],
                         "end_ms": rep["end_ms"], "count": 1})
        else:
            sets[-1]["end_ms"] = rep["end_ms"]
            sets[-1]["count"] += 1
    return sets


def decode_sequence(t, station_prob, boundary_prob, valid, config=None, lookahead_ms=0, *, offline=False):
    decoder, reps, holds = RepDecoder(config), [], []
    if len(boundary_prob) and len(boundary_prob[0]) == 4:
        from .data import runs
        t, station_prob, boundary_prob, valid = map(np.asarray, (t, station_prob, boundary_prob, valid))
        c = decoder.config
        # Replay the same peak rule in bulk when whole sessions are available.
        # A time jump starts a fresh run even when its two samples are valid.
        edges = np.r_[0, np.flatnonzero(np.diff(t) != DT_MS) + 1, len(t)]
        for left, right in zip(edges[:-1], edges[1:]):
            for a, b in runs(valid[left:right]):
                a, b = a + left, b + left
                decoder.reset()
                end = boundary_prob[a:b, 0]
                n = len(end) - c.confirmation_frames
                if n <= 0:
                    continue
                peaks = (end[:n] >= c.end_threshold) & (end[:n] > np.r_[-1., end[:n-1]])
                for lag in range(1, c.lag_frames + 1):
                    peaks &= end[:n] >= end[lag:lag+n]
                last_rest = np.maximum.accumulate(np.where(station_prob[a:b, 1] < c.station_threshold,
                                                           np.arange(b-a), -1))
                for i in np.flatnonzero(peaks):
                    clock = int(t[a+i])
                    end_ms = clock
                    if c.refine_end and i >= END_REFINEMENT_FRAMES:
                        end_ms = refined_end_time(clock, end[i-END_REFINEMENT_FRAMES:i+END_REFINEMENT_FRAMES+1])
                    _, distance, confidence, no_start = boundary_prob[a+i]
                    if 1 - no_start < c.start_threshold:
                        distance = 0
                    def rest_at(d):
                        j = last_rest[i-d]
                        return int(t[a+j]) if j >= 0 else None
                    reported = int(t[-1]) if offline else clock + c.confirmation_frames * DT_MS + lookahead_ms
                    reps.extend(decoder.close_peak(clock, float(station_prob[a+i, 1]), end[i], distance,
                                                   confidence, reported,
                                                   i+1, rest_at, end_ms))
        return {"repetitions": reps, "counts": dict(Counter(r["station"] for r in reps)), "holds": holds,
                "sets": group_sets(reps, c.set_gap_ms), "decoder": asdict(c)}
    hold_start = None
    previous = None
    for i, clock in enumerate(t):
        gap = previous is not None and clock - previous != DT_MS
        station = None if station_prob is None else station_prob[i]
        is_hold = (station is not None and bool(valid[i])
                   and int(station.argmax()) == CLASSES.index("plank"))
        if hold_start is not None and (not is_hold or gap):
            end = int(previous + DT_MS)
            holds.append({"station": "plank", "start_ms": hold_start,
                          "end_ms": end, "duration_s": (end - hold_start) / 1000})
            hold_start = None
        if is_hold and hold_start is None:
            hold_start = int(clock)
        reps.extend(decoder.step(clock, station, boundary_prob[i], bool(valid[i]),
                                 reported_at_ms=int(t[-1]) if offline else int(clock) + lookahead_ms))
        previous = int(clock)
    if hold_start is not None:
        end = int(t[-1] + DT_MS)
        holds.append({"station": "plank", "start_ms": hold_start, "end_ms": end,
                      "duration_s": (end - hold_start) / 1000, "truncated": True})
    # Active partial cycles are deliberately not emitted at EOF.
    return {"repetitions": reps, "counts": dict(Counter(r["station"] for r in reps)),
            "holds": holds, "sets": group_sets(reps, decoder.config.set_gap_ms),
            "decoder": asdict(decoder.config)}
