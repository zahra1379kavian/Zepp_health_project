"""MS-TCN refinement with an optional station head.

Left padding supports online use and centered padding supports offline teachers.
Repetition modes include dense phases,
independent boundaries, and end detection with conditional start selection.
"""

from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .constants import CLASSES, FEATURES, START_HISTORY, STATION_STATES


@dataclass
class ModelConfig:
    inputs: int = len(FEATURES)
    classes: int | None = None
    stages: int = 3
    layers: int = 6
    channels: int = 64
    dropout: float = .3
    repetition_head: str = "phase"
    lookahead_samples: int = 0
    causal_similarity: bool = False
    acausal: bool = False

    def __post_init__(self):
        if self.classes is None:
            self.classes = len(STATION_STATES) if self.repetition_head == "complete" else len(CLASSES)
        if min(self.inputs, self.stages, self.layers, self.channels) < 1 or self.classes < 0:
            raise ValueError("Model dimensions must be positive; use zero classes for phase-only models")
        if self.repetition_head not in ("phase", "boundary", "complete"):
            raise ValueError("Unknown repetition head")
        if self.repetition_head == "complete" and self.classes != len(STATION_STATES):
            raise ValueError("Complete repetitions use a binary station head")
        if self.classes == 0 and self.repetition_head != "phase":
            raise ValueError("Models without a station head require dense phases")
        if self.causal_similarity and (self.repetition_head != "complete" or self.stages < 2):
            raise ValueError("Causal similarity requires the complete rep head and at least two stages")
        if not isinstance(self.lookahead_samples, int) or self.lookahead_samples < 0:
            raise ValueError("lookahead_samples must be a nonnegative integer")
        if self.acausal and self.lookahead_samples:
            raise ValueError("The offline teacher uses centered convolutions and requires zero lookahead shift")

    @property
    def repetition_classes(self):
        if self.repetition_head == "complete":
            return 1  # Each stage refines the end probability.
        return 3 if self.repetition_head == "phase" else 2

    @property
    def replay_columns(self):
        # End probability, best past distance, its probability, no-start probability.
        return 4 if self.repetition_head == "complete" else self.repetition_classes


class Residual(nn.Module):
    def __init__(self, channels, dilation, dropout, acausal=False):
        super().__init__()
        self.history = 2 * dilation
        self.padding = (dilation, dilation) if acausal else (self.history, 0)
        self.conv = nn.Conv1d(channels, channels, 3, dilation=dilation)
        self.project = nn.Conv1d(channels, channels, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        h = self.conv(F.pad(x, self.padding))
        return (x + self.dropout(self.project(F.relu(h)))) * mask

    def stream(self, x, previous):
        if previous is None:
            previous = x.new_zeros(x.shape[0], x.shape[1], self.history)
        joined = torch.cat((previous, x), dim=-1)
        result = x + self.project(F.relu(self.conv(joined)))
        return result, joined[:, :, -self.history:].detach()


class CausalSimilarity(nn.Module):
    """Cosine similarities in start-selector order, from t-150 through t-1."""

    def __init__(self, channels):
        super().__init__()
        self.embedding = nn.Conv1d(channels, 32, 1)
        self.projection = nn.Conv1d(START_HISTORY, 64, 1)

    def forward(self, hidden, mask=None, previous=None):
        z = F.normalize(self.embedding(hidden), dim=1)
        if mask is not None:
            z = z * mask
        if previous is None:
            previous = z.new_zeros((*z.shape[:2], START_HISTORY))
        joined = torch.cat((previous, z), dim=2)

        # Nearby queries share keys. Tile the dot products to keep memory linear
        # in sequence length. Column j for time t is always the key at t-150+j.
        frames = z.size(2)
        tile = min(64, frames)
        columns = torch.arange(tile, device=z.device)[:, None] + torch.arange(START_HISTORY, device=z.device)
        rows = []
        for left in range(0, frames, tile):
            right = min(left + tile, frames)
            queries = z[:, :, left:right].transpose(1, 2)
            keys = joined[:, :, left:right + START_HISTORY - 1]
            dots = queries @ keys
            rows.append(dots.gather(2, columns[:right-left].expand(z.size(0), -1, -1)))
        similarities = torch.cat(rows, dim=1).transpose(1, 2)
        cache = joined[:, :, -START_HISTORY:].detach().clone()
        return similarities, self.projection(similarities), cache


class StartSelector(nn.Module):
    """Score each past start using motion there and duration evidence at the end."""

    def __init__(self, channels, causal_similarity=False):
        super().__init__()
        self.start_score = nn.Conv1d(channels, 1, 1)
        self.offset_scores = nn.Conv1d(channels, START_HISTORY + 1, 1)
        self.alpha = nn.Parameter(torch.ones(())) if causal_similarity else None

    def forward(self, h, mask=None, previous=None, similarities=None):
        scores = self.start_score(h)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e4)
        if previous is None:
            previous = scores.new_full((*scores.shape[:2], START_HISTORY), -1e4)
        joined = torch.cat((previous, scores), dim=-1)
        # At t: oldest through newest candidates [t-150, t-1], then no start.
        past = joined[:, 0, :-1].unfold(1, START_HISTORY, 1).transpose(1, 2)
        if self.alpha is not None:
            past = past + self.alpha * similarities
        offsets = self.offset_scores(h)
        logits = torch.cat((past + offsets[:, :-1], offsets[:, -1:]), dim=1)
        return logits, joined[:, :, -START_HISTORY:].detach()


class Stage(nn.Module):
    def __init__(self, inputs, config, select_start=False):
        super().__init__()
        self.project = nn.Conv1d(inputs, config.channels, 1)
        self.layers = nn.ModuleList([Residual(config.channels, 2 ** i, config.dropout, config.acausal)
                                     for i in range(config.layers)])
        self.station = nn.Conv1d(config.channels, config.classes, 1) if config.classes else None
        self.boundary = nn.Conv1d(config.channels, config.repetition_classes, 1)
        self.selector = StartSelector(config.channels, config.causal_similarity) if select_start else None

    def forward(self, x, mask, similarities=None):
        h = self.project(x) * mask
        for layer in self.layers:
            h = layer(h, mask)
        station = self.station(h) * mask if self.station is not None else None
        events = self.boundary(h) * mask
        if self.selector is not None:
            starts, _ = self.selector(h, mask, similarities=similarities)
            events = torch.cat((events, starts), dim=1)
        return station, events, h

    def stream(self, x, state, similarities=None):
        h, new_state = self.project(x), []
        for i, layer in enumerate(self.layers):
            h, cache = layer.stream(h, None if state is None else state[i])
            new_state.append(cache)
        station = self.station(h) if self.station is not None else None
        events = self.boundary(h)
        if self.selector is not None:
            starts, cache = self.selector(h, previous=None if state is None else state[-1], similarities=similarities)
            events = torch.cat((events, starts), dim=1)
            new_state.append(cache)
        return station, events, h, new_state


class DenseMSTCN(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or ModelConfig()
        c = self.config
        refinement_width = (c.inputs if c.repetition_head == "complete" else c.classes) + c.repetition_classes
        self.similarity = CausalSimilarity(c.channels) if c.causal_similarity else None
        if self.similarity is not None:
            refinement_width += self.similarity.projection.out_channels
        self.stages = nn.ModuleList([Stage(c.inputs if i == 0 else refinement_width, c,
                                           select_start=c.repetition_head == "complete" and i == c.stages - 1)
                                     for i in range(c.stages)])

    @property
    def future_samples(self):
        return self.config.stages * (2 ** self.config.layers - 1) if self.config.acausal else 0

    @property
    def history_samples(self):
        history = self.config.stages * 2 * (2 ** self.config.layers - 1) - self.future_samples
        if self.similarity is not None:
            # A candidate start can itself depend on another six seconds of motion.
            history += START_HISTORY
        return history + (START_HISTORY if self.config.repetition_head == "complete" else 0)

    def forward(self, x, mask=None):
        mask = x.new_ones(x.shape[0], 1, x.shape[-1]) if mask is None else mask
        wrist = x
        results = []
        similarities = projected = None
        for i, stage in enumerate(self.stages):
            station, boundary, hidden = stage(x, mask, similarities)
            results.append((station, boundary))
            if i == 0 and self.similarity is not None:
                similarities, projected, _ = self.similarity(hidden, mask)
            if i + 1 < len(self.stages):
                x = self._refinement_input(wrist, station, boundary, projected) * mask
        # A causal output acquired at t+L predicts the target at t. Shift only
        # after all refinement stages so their lookaheads do not accumulate.
        lag = self.config.lookahead_samples
        if lag:
            results = [(F.pad(p[:, :, lag:], (0, min(lag, p.shape[-1]))) if p is not None else None,
                        F.pad(q[:, :, lag:], (0, min(lag, q.shape[-1])))) for p, q in results]
        return results

    @torch.no_grad()
    def stream(self, x, state=None):
        """Exact cached causal inference. Reset state at gaps/session changes.

        Accepts [batch, channels, time]. Returns acquisition-clock outputs:
        an output at sample i belongs to target i-lookahead_samples. Callers
        discard the initial lookahead outputs after each reset; never flush
        missing future samples at EOF. Chunk boundaries do not affect outputs.
        """
        if self.config.acausal:
            raise ValueError("Acausal teacher checkpoints require offline session evaluation")
        if self.training:
            raise ValueError("Call model.eval() before streaming")
        caches = []
        wrist = x
        similarities = projected = embedding_cache = None
        for i, stage in enumerate(self.stages):
            station, boundary, hidden, cache = stage.stream(x, None if state is None else state[i], similarities)
            caches.append(cache)
            if i == 0 and self.similarity is not None:
                previous = None if state is None else state[-1]
                similarities, projected, embedding_cache = self.similarity(hidden, previous=previous)
            if i + 1 < len(self.stages):
                x = self._refinement_input(wrist, station, boundary, projected)
        if embedding_cache is not None:
            caches.append(embedding_cache)
        return station, boundary, caches

    def _refinement_input(self, wrist, station, boundary, projected=None):
        if self.config.repetition_head == "complete":
            inputs = [wrist, boundary.sigmoid()]
            if projected is not None:
                inputs.append(projected)
            return torch.cat(inputs, dim=1)
        phases = self.repetition_probabilities(boundary)
        return phases if station is None else torch.cat((station.softmax(1), phases), dim=1)

    def repetition_probabilities(self, logits):
        if self.config.repetition_head == "complete" and logits.shape[1] > 1:
            choices = logits[:, 1:].softmax(1)
            probability, index = choices[:, :-1].max(1, keepdim=True)
            return torch.cat((logits[:, :1].sigmoid(), START_HISTORY - index,
                              probability, choices[:, -1:]), dim=1)
        return logits.softmax(1) if self.config.repetition_head == "phase" else logits.sigmoid()


def dense_loss(outputs, y, boundary, valid, reference_mask, class_weights=None,
               positive_weights=None, boundary_weight=1., smoothing_weight=.15,
               phase_weights=None, start_choice=None, rep_weights=None):
    """Masked repetition loss and, when present, station CE and smoothing.

    Smoothing uses both adjacent eligibility flags and detaches the preceding
    prediction, following the authors' MS-TCN implementation. Repetition logits
    are not smoothed: boundaries and phase transitions must remain sharp.
    """
    total = outputs[0][1].sum() * 0
    selector_loss = 0.
    for station, events in outputs:
        if station is not None:
            ce = F.cross_entropy(station, y, weight=class_weights, reduction="none")
            cls = (ce * valid).sum() / valid.sum().clamp_min(1)
            logp = station.log_softmax(1)
            pair = valid[:, 1:] & valid[:, :-1]
            smooth = (logp[:, :, 1:] - logp[:, :, :-1].detach()).square().clamp(max=16).mean(1)
            smooth = (smooth * pair).sum() / pair.sum().clamp_min(1)
            total = total + cls + smoothing_weight * smooth
        refs = reference_mask & valid
        if start_choice is not None:
            weights = refs * rep_weights
            end_logits = events[:, 0]
            bce = F.binary_cross_entropy_with_logits(end_logits, boundary,
                                                     pos_weight=positive_weights, reduction="none")
            rep_loss = (bce * weights).sum() / weights.sum().clamp_min(1)
            if events.shape[1] > 1:
                # Mine this model's local peaks in certified rest, excluding end jitter.
                p = end_logits.detach().sigmoid()
                peaks = torch.zeros_like(refs)
                peaks[:, 1:-1] = ((p[:, 1:-1] >= .5) & (p[:, 1:-1] > p[:, :-2])
                                  & (p[:, 1:-1] >= p[:, 2:]))
                false_end = peaks & (y == 0) & (boundary == 0) & (start_choice < 0)
                choice = start_choice.masked_fill(false_end, START_HISTORY)
                supervised = (choice >= 0) & (weights > 0)
                # All 150 candidate frames compete, including mid-rep and prior starts.
                logits = events[:, 1:].transpose(1, 2)[supervised]
                ce = F.cross_entropy(logits, choice[supervised], reduction="none")
                w = weights[supervised]
                selector_loss = (ce * w).sum() / w.sum().clamp_min(1)
        elif events.shape[1] == 3:
            phase_ce = F.cross_entropy(events, boundary, weight=phase_weights, reduction="none")
            rep_loss = (phase_ce * refs).sum() / refs.sum().clamp_min(1)
        else:
            bce = F.binary_cross_entropy_with_logits(events.transpose(1, 2), boundary,
                                                    pos_weight=positive_weights, reduction="none")
            rep_loss = (bce * refs[..., None]).sum() / (2 * refs.sum()).clamp_min(1)
        total = total + boundary_weight * rep_loss
    return total / len(outputs) + selector_loss


def configuration(model):
    return asdict(model.config)
