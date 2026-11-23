"""Training-only repetitions built from motion and certified rest in the training fold."""

from collections import Counter

import numpy as np
from scipy.signal import find_peaks
from torch.utils.data import DataLoader, RandomSampler, default_collate

from .complete import complete_targets
from .constants import CLASSES, DT_MS
from .data import runs
from .orientation import rep_values, rotate_window


MIN_PIECE, MAX_PIECE = 600 // DT_MS, 2000 // DT_MS
MIN_REST, MAX_REST = 5000 // DT_MS, 30000 // DT_MS
STEP_LAGS = np.arange(10, 18)  # 0.40 through 0.68 s on the 25 Hz grid


def step_rhythm(acc):
    """A local peak >= 0.65 in the overlap-normalized tri-axial autocorrelation."""
    centered = acc.astype(np.float64) - acc.mean(axis=0, dtype=np.float64)
    n = len(acc)
    # Require at least eight overlapping samples to judge a lag.
    lags = STEP_LAGS[STEP_LAGS + 1 <= n - 8]
    if not len(lags):
        return False
    tested = np.arange(lags[0] - 1, lags[-1] + 2)
    corr = sum(np.correlate(axis, axis, "full")[n-1+tested] for axis in centered.T)
    energy = np.r_[0., np.cumsum(np.einsum("ij,ij->i", centered, centered))]
    denom = np.sqrt(energy[n-tested] * np.maximum(energy[-1] - energy[tested], 0.))
    corr = np.divide(corr, denom, out=np.zeros_like(corr), where=denom > 1e-10)
    peak = corr[1:-1]
    return bool(np.any((peak >= .65) & (peak > corr[:-2]) & (peak >= corr[2:])))


def make_cycle(piece, speed, amplitude):
    length = round(len(piece) / speed)
    clock = np.linspace(0, len(piece)-1, length)
    forward = np.column_stack([np.interp(clock, np.arange(len(piece)), v) for v in piece.T]).astype(np.float32)
    forward[:, :3] = piece[0, :3] + amplitude * (forward[:, :3] - piece[0, :3])
    forward[:, 3:6] *= speed * amplitude
    reverse = forward[::-1].copy()
    reverse[:, 3:6] *= -1
    return np.concatenate([forward, reverse])


class SyntheticReps:
    def __init__(self, root, manifest, windows, seed):
        self.rng = np.random.default_rng(seed)
        self.windows = windows
        self.raw, self.pieces, self.rest = [], [], []
        self.rest_by_stream = {}
        analysis = manifest.get("analysis") or {}
        forbidden = [CLASSES.index(e) for e in (analysis.get("held_out_exercise"), analysis.get("selection_exercise")) if e]
        named = manifest.get("reference_mode") != "endpoints"
        self.counts = Counter()
        for i, (name, a) in enumerate(zip(manifest["split"]["train"], windows.arrays, strict=True)):
            # Read physical axes even when the real windows already hold gravity features.
            raw = np.load(root / name / "x.npy", mmap_mode="r", allow_pickle=False)
            raw = raw if windows.rep_inputs == "original" else raw[:, :6]
            self.raw.append(raw)
            eligible = a["valid"] & a["coverage"] & (a["weight"] > 0)
            pads = [(i, lo, hi) for lo, hi in runs(eligible & (a["y"] == 0)) if hi - lo >= MIN_REST]
            self.rest_by_stream[i] = pads
            self.rest.extend(pads)
            motion = eligible & (a["y"] != 0)
            if named:
                motion &= ~np.isin(a["y"], [CLASSES.index("plank"), *forbidden])
            # Bound the snippet and its rhythm check to one uninterrupted exercise.
            for lo, hi in runs(motion):
                edges = np.r_[lo, lo + 1 + np.flatnonzero(np.diff(a["y"][lo:hi])), hi]
                for left, right in zip(edges[:-1], edges[1:]):
                    self.index_bout(i, int(left), int(right))
        if not self.pieces or not self.rest:
            raise ValueError("Synthetic reps need eligible non-step motion and at least five seconds of continuous certified training rest")
        self.counts.update(accepted_pieces=len(self.pieces), rest_spans=len(self.rest))
        self.samples = self.iter_windows()

    def index_bout(self, stream, left, right):
        raw = self.raw[stream]
        # Short pieces need neighboring samples to distinguish a repeating step
        # from a single movement. Use at least two periods of the longest lag.
        if right - left < 2 * STEP_LAGS[-1] + 1:
            return
        gyro = np.linalg.norm(raw[left:right, 3:6], axis=1)
        minima, _ = find_peaks(-gyro, distance=MIN_PIECE)
        for index in minima:
            start = left + int(index)
            maximum = min(MAX_PIECE, right - start)
            if maximum < MIN_PIECE:
                continue
            length = int(self.rng.integers(MIN_PIECE, maximum + 1))
            if gyro[index] >= np.median(gyro[index:index+length]):
                continue
            context_start = max(left, min(start - (MAX_PIECE - length)//2, right - MAX_PIECE))
            if step_rhythm(raw[context_start:min(right, context_start+MAX_PIECE), :3]) or step_rhythm(raw[start:start+length, :3]):
                self.counts["step_rhythm_rejected"] += 1
                continue
            self.pieces.append((stream, start, start + length))

    def sample_rest(self, preferred):
        spans = self.rest_by_stream[preferred] or self.rest
        stream, lo, hi = spans[self.rng.integers(len(spans))]
        length = int(self.rng.integers(MIN_REST, min(MAX_REST, hi-lo) + 1))
        start = int(self.rng.integers(lo, hi-length+1))
        selection = slice(start, start+length)
        return self.raw[stream][selection], self.windows.arrays[stream]["weight"][selection]

    def sequence(self):
        rng = self.rng
        stream, lo, hi = self.pieces[rng.integers(len(self.pieces))]
        piece = self.raw[stream][lo:hi]
        before, before_weights = self.sample_rest(stream)
        after, after_weights = self.sample_rest(stream)
        parts, bounds = [before], []
        cursor = len(before)
        for i in range(rng.integers(3, 16)):
            if i and rng.random() < .25:
                pause = np.repeat(piece[:1], rng.integers(0, 1000//DT_MS + 1), axis=0)
                pause[:, 3:6] = 0
                parts.append(pause)
                cursor += len(pause)
            cycle = make_cycle(piece, rng.uniform(.85, 1.15), rng.uniform(.9, 1.1))
            parts.append(cycle)
            bounds.append({"start_ms": cursor * DT_MS, "end_ms": (cursor + len(cycle)) * DT_MS})
            cursor += len(cycle)
        raw = rotate_window(np.concatenate([*parts, after]), rng)
        valid = np.ones(len(raw), bool)
        station = np.zeros(len(raw), dtype=np.int64)
        station[len(before):cursor] = 1
        a = dict(t=np.arange(len(raw)) * DT_MS, valid=valid, coverage=valid, y=station,
                 x=rep_values(raw, valid, self.windows.rep_inputs))
        a.update(complete_targets(a, bounds))
        a["weight"][:len(before)], a["weight"][cursor:] = before_weights, after_weights
        a["weight"][a["end"] > 0] = 1  # Jittered end targets beside the padding are positives.
        return a, bounds

    def iter_windows(self):
        while True:
            for _ in range(100):
                a, _ = self.sequence()
                end = len(a["t"]) - self.windows.lookahead
                if end > 0:
                    break
            else:
                raise ValueError("Generated sequences have no targets left after the configured lookahead")
            for start in self.rng.permutation(np.arange(0, end, self.windows.width)):
                yield self.windows.window(a, 0, start, min(start + self.windows.width, end), len(a["t"]), rotate=False)

    def mix(self, real):
        return default_collate([*real, *(next(self.samples) for _ in range(len(real)//3))])

    def loader(self, batch_size):
        real = batch_size * 3 // 4
        # Match the control's number of updates, including its last partial batch.
        steps = (len(self.windows) + batch_size - 1) // batch_size
        sampler = RandomSampler(self.windows, num_samples=real * steps)
        return DataLoader(self.windows, batch_size=real, sampler=sampler, collate_fn=self.mix)
