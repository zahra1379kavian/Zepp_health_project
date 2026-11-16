"""MS-TCN-inspired causal refinement with station and dense phase heads.

Original MS-TCN uses symmetric convolutions and one action head. Here left-only
padding enables online use. The repetition head predicts NULL/phase1/phase2;
an explicit boundary-head baseline remains available for external references.
Refinement receives predicted probabilities only, never true labels.
"""

from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .constants import CLASSES, FEATURES


@dataclass
class ModelConfig:
    inputs: int = len(FEATURES)
    classes: int = len(CLASSES)
    stages: int = 3
    layers: int = 6
    channels: int = 64
    dropout: float = .3
    repetition_head: str = "phase"

    def __post_init__(self):
        if min(self.inputs, self.classes, self.stages, self.layers, self.channels) < 1:
            raise ValueError("Model dimensions must be positive")
        if self.repetition_head not in ("phase", "boundary"):
            raise ValueError("repetition_head must be phase or boundary")

    @property
    def repetition_classes(self):
        return 3 if self.repetition_head == "phase" else 2


class Residual(nn.Module):
    def __init__(self, channels, dilation, dropout):
        super().__init__()
        self.history = 2 * dilation
        self.conv = nn.Conv1d(channels, channels, 3, dilation=dilation)
        self.project = nn.Conv1d(channels, channels, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        h = self.conv(F.pad(x, (self.history, 0)))
        return (x + self.dropout(self.project(F.relu(h)))) * mask

    def stream(self, x, previous):
        if previous is None:
            previous = x.new_zeros(x.shape[0], x.shape[1], self.history)
        joined = torch.cat((previous, x), dim=-1)
        result = x + self.project(F.relu(self.conv(joined)))
        return result, joined[:, :, -self.history:].detach()


class Stage(nn.Module):
    def __init__(self, inputs, config):
        super().__init__()
        self.project = nn.Conv1d(inputs, config.channels, 1)
        self.layers = nn.ModuleList([Residual(config.channels, 2 ** i, config.dropout)
                                     for i in range(config.layers)])
        self.station = nn.Conv1d(config.channels, config.classes, 1)
        self.boundary = nn.Conv1d(config.channels, config.repetition_classes, 1)

    def forward(self, x, mask):
        h = self.project(x) * mask
        for layer in self.layers:
            h = layer(h, mask)
        return self.station(h) * mask, self.boundary(h) * mask

    def stream(self, x, state):
        h, new_state = self.project(x), []
        for i, layer in enumerate(self.layers):
            h, cache = layer.stream(h, None if state is None else state[i])
            new_state.append(cache)
        return self.station(h), self.boundary(h), new_state


class DenseMSTCN(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or ModelConfig()
        c = self.config
        self.stages = nn.ModuleList([Stage(c.inputs if i == 0 else c.classes + c.repetition_classes, c)
                                     for i in range(c.stages)])

    @property
    def history_samples(self):
        return self.config.stages * 2 * (2 ** self.config.layers - 1)

    def forward(self, x, mask=None):
        mask = x.new_ones(x.shape[0], 1, x.shape[-1]) if mask is None else mask
        results = []
        for stage in self.stages:
            station, boundary = stage(x, mask)
            results.append((station, boundary))
            x = torch.cat((station.softmax(1), self.repetition_probabilities(boundary)), dim=1) * mask
        return results

    @torch.no_grad()
    def stream(self, x, state=None):
        """Exact cached causal inference. Reset state at gaps/session changes.

        Accepts [batch, channels, time]; predictions for a prefix are independent
        of subsequent samples and of the caller's chunk boundaries.
        """
        if self.training:
            raise ValueError("Call model.eval() before streaming")
        caches = []
        for i, stage in enumerate(self.stages):
            station, boundary, cache = stage.stream(x, None if state is None else state[i])
            caches.append(cache)
            x = torch.cat((station.softmax(1), self.repetition_probabilities(boundary)), dim=1)
        return station, boundary, caches

    def repetition_probabilities(self, logits):
        return logits.softmax(1) if self.config.repetition_head == "phase" else logits.sigmoid()


def dense_loss(outputs, y, boundary, valid, reference_mask, class_weights=None,
               positive_weights=None, boundary_weight=1., smoothing_weight=.15,
               phase_weights=None):
    """Station CE + truncated smoothing + masked phase CE (or boundary BCE).

    Smoothing uses both adjacent eligibility flags and detaches the preceding
    prediction, following the authors' MS-TCN implementation. Repetition logits
    are not smoothed: boundaries and phase transitions must remain sharp.
    """
    total = outputs[0][0].sum() * 0
    for station, events in outputs:
        ce = F.cross_entropy(station, y, weight=class_weights, reduction="none")
        cls = (ce * valid).sum() / valid.sum().clamp_min(1)
        logp = station.log_softmax(1)
        pair = valid[:, 1:] & valid[:, :-1]
        smooth = (logp[:, :, 1:] - logp[:, :, :-1].detach()).square().clamp(max=16).mean(1)
        smooth = (smooth * pair).sum() / pair.sum().clamp_min(1)
        refs = reference_mask & valid
        if events.shape[1] == 3:
            phase_ce = F.cross_entropy(events, boundary, weight=phase_weights, reduction="none")
            rep_loss = (phase_ce * refs).sum() / refs.sum().clamp_min(1)
        else:
            bce = F.binary_cross_entropy_with_logits(events.transpose(1, 2), boundary,
                                                    pos_weight=positive_weights, reduction="none")
            rep_loss = (bce * refs[..., None]).sum() / (2 * refs.sum()).clamp_min(1)
        total = total + cls + smoothing_weight * smooth + boundary_weight * rep_loss
    return total / len(outputs)


def configuration(model):
    return asdict(model.config)
