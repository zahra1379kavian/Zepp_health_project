"""Wrist accelerometer and gyroscope inputs at the prepared 25 Hz sample rate."""

import numpy as np
from scipy.signal import lfilter
from scipy.spatial.transform import Rotation

from .constants import DT_MS, FEATURES


GRAVITY_FEATURES = ["acc_magnitude", "linear_acc_along_gravity", "linear_acc_across_gravity",
                    "gyro_magnitude", "gyro_about_gravity", "gyro_across_gravity"]
INPUT_MODES = ("original", "gravity", "rotated")
RETENTION = np.exp(-2 * np.pi * .3 * DT_MS / 1000)


def gravity_features(x, previous=None):
    """Return six features and the last low-pass estimate for the next contiguous chunk."""
    acc, gyro = x[:, :3], x[:, 3:6]
    initial = acc[0] if previous is None else previous
    gravity, _ = lfilter([1 - RETENTION], [1., -RETENTION], acc, axis=0,
                         zi=(RETENTION * initial.astype(np.float64))[None])
    norm = np.linalg.norm(gravity, axis=1, keepdims=True)
    direction = np.divide(gravity, norm, out=np.zeros_like(gravity), where=norm > 1e-12)
    linear = acc - gravity

    def components(values):
        along = (values * direction).sum(axis=1)
        across = np.linalg.norm(values - along[:, None] * direction, axis=1)
        return along, across

    features = np.column_stack((np.linalg.norm(acc, axis=1), *components(linear),
                                np.linalg.norm(gyro, axis=1), *components(gyro)))
    return features.astype(np.float32), gravity[-1]


def rep_values(x, valid, mode):
    if mode == "original":
        return x
    if mode == "rotated":
        return x[:, :6]
    if mode != "gravity":
        raise ValueError(f"Unknown repetition input mode {mode!r}")
    from .data import runs
    values = np.zeros((len(x), 6), dtype=np.float32)
    for a, b in runs(valid):
        values[a:b], _ = gravity_features(x[a:b])
    return values


def rotate_window(x, rng=None):
    """One Haar-uniform proper rotation for both triads and every frame."""
    rotation = Rotation.random(random_state=rng).as_matrix().astype(np.float32)
    return (x.reshape(-1, 2, 3) @ rotation.T).reshape(-1, 6)


def model_feature_names(mode, raw):
    return GRAVITY_FEATURES if mode == "gravity" else FEATURES[:6] if mode == "rotated" else raw
