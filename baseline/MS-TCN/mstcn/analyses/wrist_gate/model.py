"""Causal wrist-plus-waist phase correction and wrist-first mixing."""

import numpy as np
import torch
from torch import nn

from ...constants import DT_MS
from ...data import runs
from ...decode import decode_sequence
from ...runner import probabilities


ARMS = ("wrist", "fused", "gated")
WINDOWS = (1, 5, 15, 40)


def history(values):
    """Current value and trailing means up to 1.6 s; call per continuous run."""
    prefix = np.vstack((np.zeros(values.shape[1]), np.cumsum(values, axis=0, dtype=np.float64)))
    end = np.arange(1, len(values) + 1)
    parts = []
    for width in WINDOWS:
        start = np.maximum(0, end - width)
        parts.append((prefix[end] - prefix[start]) / (end - start)[:, None])
    return np.column_stack(parts).astype(np.float32)


def motion(wrist, waist):
    parts = []
    for device in (wrist, waist):
        change = np.diff(device[:, :3], axis=0, prepend=device[:1, :3])
        parts.extend((np.linalg.norm(change, axis=1), np.linalg.norm(device[:, 3:6], axis=1)))
    return np.column_stack(parts)


class PhaseHead(nn.Module):
    def __init__(self, gating):
        super().__init__()
        self.gating = gating
        # Gate sees both phase distributions and four motion magnitudes. The
        # fused expert also sees the signed wrist and waist acc/gyro axes.
        width = (10 if gating else 19) * len(WINDOWS)
        hidden = 16 if gating else 64
        self.register_buffer("mean", torch.zeros(width))
        self.register_buffer("scale", torch.ones(width))
        self.net = nn.Sequential(nn.Linear(width, hidden), nn.ReLU(),
                                 nn.Linear(hidden, 1 if gating else 3))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, -2.2 if gating else 0.)

    def forward(self, features, wrist, fused=None):
        output = self.net(((features - self.mean) / self.scale).clamp(-12, 12))
        if self.gating:
            weight = output.sigmoid()
            return torch.lerp(wrist, fused, weight), weight
        return (wrist.clamp_min(1e-7).log() + output).softmax(dim=1), None


def expert_features(wrist, waist, wrist_phase, movement):
    return history(np.column_stack((wrist[:, :6], waist, wrist_phase, movement)))


@torch.no_grad()
def table_predict(head, features, wrist, fused=None):
    args = [torch.from_numpy(v) for v in (features, wrist, fused) if v is not None]
    scores, weight = head(*args)
    return scores.numpy(), None if weight is None else weight.numpy().ravel()


def gate_inputs(expert, wrist, waist, wrist_phase):
    movement = motion(wrist, waist)
    features = expert_features(wrist, waist, wrist_phase, movement)
    fused, _ = table_predict(expert, features, wrist_phase)
    return history(np.column_stack((wrist_phase, fused, movement))), fused


def align_waist(t, waist_t, values):
    idx = np.searchsorted(waist_t, t)
    present = idx < len(waist_t)
    present[present] &= waist_t[idx[present]] == t[present]
    aligned = np.zeros((len(t), 6), np.float32)
    aligned[present] = values[idx[present]]
    return aligned, present


def phase_streams(base, expert, gate, wrist, waist, present):
    """Replay one continuous wrist run. Waist gaps reset only the extra paths."""
    station, wrist_phase = probabilities(base.model, wrist[:, base.columns],
                                         np.ones(len(wrist), bool), base.norm, base.device)
    fused, mixed = wrist_phase.copy(), wrist_phase.copy()
    weights = np.zeros(len(wrist), np.float32)
    for a, b in runs(present):
        inputs, fused[a:b] = gate_inputs(expert, wrist[a:b], waist[a:b], wrist_phase[a:b])
        mixed[a:b], weights[a:b] = table_predict(gate, inputs, wrist_phase[a:b], fused[a:b])
    return station, dict(wrist=wrist_phase, fused=fused, gated=mixed), weights


def predict_clip(base, expert, gate, t, wrist, waist, present):
    """Sensor-only entry point. All three arms use the same station scores and decoder."""
    events = {arm: [] for arm in ARMS}
    weights = np.zeros(len(t), np.float32)
    edges = np.r_[0, np.flatnonzero(np.diff(t) != DT_MS) + 1, len(t)]
    for a, b in zip(edges[:-1], edges[1:]):
        station, scores, weights[a:b] = phase_streams(base, expert, gate, wrist[a:b],
                                                      waist[a:b], present[a:b])
        clocks = np.arange(b - a) * DT_MS
        for arm, phase in scores.items():
            decoded = decode_sequence(clocks, station, phase, np.ones(len(clocks), bool), base.config)
            for r in decoded["repetitions"]:
                events[arm].append({key: int(a + r[field] // DT_MS) for key, field in
                                    (("start", "start_ms"), ("end", "end_ms"), ("reported_at", "reported_at_ms"))})
    return events, weights
