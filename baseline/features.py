"""RecoFit's 224 exercise/rest and 60 recognition features, adapted to 25 Hz.

The armband's anatomical X axis is unavailable here: X means device X. Spectral
bins span 0.1 Hz to Nyquist, rather than the paper's 0.1--25 Hz at 50 Hz.
All batch rows are independent trailing windows; no future window is used.
"""

import numpy as np
from scipy.signal import butter, find_peaks, sosfilt

FS = 25


def pc1(x):
    """Deterministic PCA; accepts (..., time, axes), including constant inputs."""
    centered = x - x.mean(axis=-2, keepdims=True)
    covariance = np.swapaxes(centered, -1, -2) @ centered
    _, vectors = np.linalg.eigh(covariance)
    axis = vectors[..., -1]
    largest = np.argmax(np.abs(axis), axis=-1)
    sign = np.take_along_axis(axis, largest[..., None], axis=-1)[..., 0]
    axis = axis * np.where(sign < 0, -1., 1.)[..., None]
    return np.einsum("...tc,...c->...t", centered, axis)


def filtered_inputs(x, valid, cutoff=8.):
    """Causal low-pass, reset at every unavailable/excluded sample."""
    from strength_training.temporal_segmentation.data import runs
    result = np.zeros((len(x), 6), np.float64)
    sos = butter(4, cutoff, fs=FS, output="sos")
    for a, b in runs(valid):
        result[a:b] = sosfilt(sos, np.asarray(x[a:b, :6], np.float64), axis=0)
    return result


def autocorrelation(x):
    """Energy-normalized, mean-centered autocorrelation along the final axis."""
    x = x - x.mean(axis=-1, keepdims=True)
    n = x.shape[-1]
    fft = np.fft.rfft(x, n=1 << (2*n-1).bit_length(), axis=-1)
    ac = np.fft.irfft(fft * fft.conj(), axis=-1)[..., :n]
    return np.divide(ac, ac[..., :1], out=np.zeros_like(ac), where=ac[..., :1] > 1e-12)


def basic_statistics(x):
    return np.stack((np.sqrt(np.mean(x*x, axis=-1)), x.mean(-1), x.std(-1), x.var(-1)), -1)


def power_bands(x):
    power = np.abs(np.fft.rfft(x, axis=-1)) ** 2 / x.shape[-1] ** 2
    frequencies = np.fft.rfftfreq(x.shape[-1], 1/FS)
    bins = np.linspace(.1, FS/2, 11)
    return np.stack([power[..., (frequencies >= a) &
                                     (frequencies <= b if i == 9 else frequencies < b)].sum(-1)
                     for i, (a, b) in enumerate(zip(bins[:-1], bins[1:]))], -1)


def extract(windows):
    """Return (B,224) segmentation and (B,60) recognition feature matrices."""
    windows = np.asarray(windows, np.float64)
    if windows.ndim != 3 or windows.shape[1] < 4 or windows.shape[2] != 6:
        raise ValueError("Expected (batch, >=4 time samples, 6 acc/gyro channels)")
    if not np.isfinite(windows).all():
        raise ValueError("Nonfinite sensor features")
    signals = []
    for offset in (0, 3):
        xyz = windows[:, :, offset:offset+3]
        signals.extend((xyz[:, :, 0], np.linalg.norm(xyz, axis=-1), pc1(xyz), pc1(xyz[:, :, 1:])))
    values = np.stack(signals, axis=1)  # batch, eight signals, time
    ac = autocorrelation(values)
    stats, bands = basic_statistics(values), power_bands(values)
    ac_features = np.zeros((*values.shape[:2], 5))
    lag_start = int(np.ceil(.5*FS))
    # The paper leaves these two peak thresholds unspecified. Fixed adaptations:
    # prominence .1 of normalized AC; separation .5 seconds.
    for i in range(len(values)):
        for j in range(8):
            signal = ac[i, j]
            peaks = find_peaks(signal)[0]
            peaks = peaks[peaks >= lag_start]
            prominent = find_peaks(signal, prominence=.1, distance=lag_start)[0]
            prominent = prominent[prominent >= lag_start]
            zero = np.flatnonzero(signal <= 0)
            first = peaks[peaks > zero[0]] if len(zero) else []
            ac_features[i, j] = (len(peaks), len(prominent), max(0, len(peaks)-len(prominent)),
                                     np.max(signal[lag_start:], initial=0),
                                     signal[first[0]] if len(first) else 0)
    integrated_rms = np.sqrt(np.mean(np.cumsum(values, axis=-1)**2, axis=-1))[..., None]
    mid = values.shape[-1]//2
    segmentation = np.concatenate((ac_features, stats[..., :1], bands, stats[..., 1:],
                                    integrated_rms, basic_statistics(values[..., :mid]),
                                    basic_statistics(values[..., mid:])), axis=-1)
    recognition = []
    for j in (0, 3, 6):  # aX, aYZPC1, gPC1
        v = values[:, j]
        variance = v.var(-1)
        kurtosis = np.divide(np.mean((v-v.mean(-1, keepdims=True))**4, axis=-1),
                             variance**2, out=np.zeros(len(v)), where=variance > 1e-12) - 3
        moments = np.stack((v.mean(-1), v.std(-1), kurtosis,
                            np.quantile(v, .75, axis=-1)-np.quantile(v, .25, axis=-1)), -1)
        bins = np.stack([part.sum(-1) for part in np.array_split(ac[:, j], 5, axis=-1)], -1)
        recognition.append(np.concatenate((bins, stats[:, j, :1], bands[:, j], moments), -1))
    return segmentation.reshape(len(windows), 224), np.concatenate(recognition, axis=-1)


def window_ends(valid, window, step):
    from strength_training.temporal_segmentation.data import runs
    return np.concatenate([np.arange(a+window-1, b, step, dtype=np.int64)
                           for a, b in runs(valid) if b-a >= window]) if np.any(valid) else np.empty(0, np.int64)


def feature_batches(filtered, ends, window, batch_size=256):
    for offset in range(0, len(ends), batch_size):
        batch = ends[offset:offset+batch_size]
        windows = filtered[batch[:, None] + np.arange(1-window, 1)[None, :]]
        yield batch, *extract(windows)
