"""PCA + locally autocorrelation-guided peak counting from RecoFit (2014).

Filter order/ripple, PCA polarity and AC context size are explicit implementation
choices because the paper does not specify them. The interval conversion is an
additional symmetric-period heuristic, not a result claimed by RecoFit.
"""

from dataclasses import dataclass

import numpy as np
from scipy.signal import ellip, find_peaks, sosfilt, sosfilt_zi

from .features import FS, autocorrelation, pc1


@dataclass(frozen=True)
class CounterConfig:
    spacing_fraction: float = .75
    amplitude_fraction: float = .5
    buffer_seconds: float = 30.
    confirmation_ms: int = 200

    def __post_init__(self):
        if not 0 < self.spacing_fraction <= 1 or not 0 <= self.amplitude_fraction <= 1:
            raise ValueError("Invalid peak thresholds")
        if self.buffer_seconds < 2 or self.confirmation_ms < 0:
            raise ValueError("Invalid counter buffer/confirmation")


def bandpass(x, valid):
    """Causal 0.15--11 Hz accelerometer bandpass, initialized per valid run."""
    from strength_training.temporal_segmentation.data import runs
    output = np.zeros((len(x), 3), np.float64)
    sos = ellip(4, .1, 60, [.15, 11.], btype="bandpass", fs=FS, output="sos")
    zi = sosfilt_zi(sos)[:, :, None]
    for a, b in runs(valid):
        samples = np.asarray(x[a:b, :3], np.float64)
        # Start at the first sample's steady state to avoid a gravity step.
        output[a:b], _ = sosfilt(sos, samples, axis=0, zi=zi*samples[0])
    return output


def select_peaks(values, periods, minimum_distance):
    """Amplitude-ranked nonmaximum suppression with candidate-specific distance."""
    accepted = []
    for i in sorted(range(len(values)), key=lambda k: (-values[k][1], values[k][0])):
        p = values[i][0]
        distance = minimum_distance if periods is None else minimum_distance*periods[i]
        if all(abs(p-values[j][0]) >= distance for j in accepted):
            accepted.append(i)
    return sorted(accepted, key=lambda i: values[i][0])


def count_peaks(filtered_acceleration, min_period_s, max_period_s, config=CounterConfig()):
    """Count an observed buffer; return sample index, local period and amplitude."""
    if not (0 < min_period_s < max_period_s):
        raise ValueError("Expected 0 < min_period < max_period")
    x = np.asarray(filtered_acceleration, np.float64)
    if x.ndim != 2 or x.shape[1] != 3 or not np.isfinite(x).all():
        raise ValueError("Expected finite (time,3) accelerometer data")
    minimum = max(2, int(np.ceil(min_period_s*FS)))
    maximum = max(minimum+1, int(np.floor(max_period_s*FS)))
    if len(x) <= minimum+2:
        return []
    signal = pc1(x)
    if np.ptp(signal) < 1e-8:
        return []
    candidates = [(int(p), float(signal[p])) for p in find_peaks(signal)[0]]
    initial = select_peaks(candidates, None, minimum)
    candidates = [candidates[i] for i in initial]
    periods = []
    for peak, _ in candidates:
        # Two max-periods either side, clipped to samples actually observed.
        local = signal[max(0, peak-2*maximum):min(len(signal), peak+2*maximum+1)]
        ac = autocorrelation(local)
        hi = min(maximum, len(ac)-1)
        periods.append(minimum+int(np.argmax(ac[minimum:hi+1])))
    chosen = select_peaks(candidates, periods, config.spacing_fraction)
    if not chosen:
        return []
    threshold = config.amplitude_fraction*np.quantile([candidates[i][1] for i in chosen], .4)
    return [{"index": candidates[i][0], "period_samples": int(periods[i]),
             "amplitude": candidates[i][1]} for i in chosen
            if candidates[i][1] > 0 and candidates[i][1] >= threshold]


def decode(t, acceleration, labels, valid, period_bounds, mode="causal", step=5,
           config=CounterConfig()):
    """Counter consumes predicted labels only. Causal events are never retracted.

    Labels are stabilized by classify; a committed station change ends the set.
    Boundary adaptation: peak +/- half the estimated period, rounded to 40 ms.
    Emit only when that full interval has been observed. No artificial EOF cycle.
    """
    from strength_training.temporal_segmentation.constants import CLASSES, DT_MS
    if mode not in ("causal", "offline") or step < 1:
        raise ValueError("Invalid decode mode or step")
    t, labels, valid = np.asarray(t), np.asarray(labels), np.asarray(valid, bool)
    filtered = bandpass(acceleration, valid)
    repetitions, holds, peaks = [], [], []
    boundaries = np.r_[0, np.flatnonzero((labels[1:] != labels[:-1]) |
                                        (valid[1:] != valid[:-1]))+1, len(t)]
    for a, b in zip(boundaries[:-1], boundaries[1:]):
        if not valid[a] or labels[a] == 0:
            continue
        name = CLASSES[int(labels[a])]
        if name == "plank":
            holds.append({"station": name, "start_ms": int(t[a]), "end_ms": int(t[b-1]+DT_MS),
                          "duration_ms": int(t[b-1]+DT_MS-t[a])})
            continue
        minimum, maximum = period_bounds[name]["min_s"], period_bounds[name]["max_s"]
        last_peak, last_period = -np.inf, 0
        ticks = [b-1] if mode == "offline" else range(a+step-1, b, step)
        for tick in ticks:
            left = a if mode == "offline" else max(a, tick+1-int(config.buffer_seconds*FS))
            candidates = count_peaks(filtered[left:tick+1], minimum, maximum, config)
            for row in candidates:
                peak, period = left+row["index"], row["period_samples"]
                if mode == "causal" and peak-last_peak < config.spacing_fraction*max(2, min(period, last_period)):
                    continue
                start, end = peak-period//2, peak+(period-period//2)
                if mode == "offline":
                    peaks.append({"station": name, "peak_ms": int(t[peak]),
                                  "period_ms": int(period*DT_MS)})
                if start < a or end >= b or end > tick:
                    continue
                if mode == "causal" and t[tick]-t[end] < config.confirmation_ms:
                    continue
                report = int(t[tick]) if mode == "causal" else int(t[-1])
                repetitions.append({"station": name, "start_ms": int(t[start]), "end_ms": int(t[end]),
                                    "peak_ms": int(t[peak]), "period_ms": int(period*DT_MS),
                                    "reported_at_ms": report, "boundary_method": "peak_plus_minus_half_period"})
                last_peak, last_period = peak, period
    repetitions.sort(key=lambda r: (r["reported_at_ms"], r["start_ms"]))
    from collections import Counter
    return {"repetitions": repetitions, "holds": holds,
            "counts_by_station": dict(Counter(r["station"] for r in repetitions)),
            "offline_peaks": peaks,
            "offline_peak_counts_by_station": dict(Counter(r["station"] for r in peaks))}
