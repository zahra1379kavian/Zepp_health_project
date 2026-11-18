"""Training-only inertial augmentation, applied before normalization.

Time warp uses the same monotone clock for every sensor, target and mask.
Rotations share one matrix across sensor triads on each physical device.
"""

import numpy as np


def augment(x, targets, features, rng=None):
    rng = np.random if rng is None else rng
    x = np.array(x, dtype=np.float32, copy=True)
    n = len(x)
    # Smooth piecewise-linear warp; preserve endpoints and use nearest-neighbour
    # sampling for discrete labels/masks, linear interpolation for sensor values.
    knots = np.linspace(0, max(n - 1, 1), 5)
    steps = rng.uniform(.9, 1.1, 4)
    source_knots = np.r_[0., np.cumsum(steps)]
    source_knots *= max(n - 1, 1) / source_knots[-1]
    clock = np.interp(np.arange(n), knots, source_knots)
    nearest = np.rint(clock).astype(int).clip(0, n - 1)
    x = np.stack([np.interp(clock, np.arange(n), column) for column in x.T], axis=1).astype(np.float32)
    targets = {key: np.asarray(value)[nearest].copy() for key, value in targets.items()}
    locations = ["wrist", "waist"] if features[0].startswith("wrist_") else [""]
    for location in locations:
        prefix = location + "_" if location else ""
        axis = rng.normal(size=3)
        axis /= max(np.linalg.norm(axis), 1e-12)
        angle = rng.uniform(-10, 10) * np.pi / 180
        a, b, c = axis
        skew = np.array([[0, -c, b], [c, 0, -a], [-b, a, 0]])
        rotation = np.eye(3) + np.sin(angle) * skew + (1 - np.cos(angle)) * (skew @ skew)
        for sensor in ("acc", "gyro", "mag"):
            names = [f"{prefix}{sensor}_{axis}" for axis in "xyz"]
            if all(name in features for name in names):
                indices = [features.index(name) for name in names]
                x[:, indices] = (x[:, indices] @ rotation.T) * rng.uniform(.95, 1.05)
    return x, targets
