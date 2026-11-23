"""Replay frozen MS-TCN folds through the shared wrist benchmark."""

from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from baseline.rep_benchmark.protocol import Prediction, utils
from ..constants import DT_MS
from ..decode import decode_sequence
from ..runner import decoder_config, digest, load_checkpoint, probabilities, prediction_mask


class WristModel:
    def __init__(self, fold, device, start_rule=None, refine_end=False):
        path = fold / "model/best.pt"
        self.model, ckpt = load_checkpoint(path, device, fold / "data")
        self.config, _, _ = decoder_config(fold / "decoder.json", path, refine_end=refine_end)
        self.head = self.model.config.repetition_head
        if ckpt["station_only"] or any(f not in utils.REQUIRED_COLS for f in ckpt["features"]):
            raise ValueError("This benchmark requires a wrist-only repetition checkpoint")
        if (start_rule or refine_end) and self.head != "complete":
            raise ValueError("Start rule and end refinement require a complete-rep checkpoint")
        if start_rule:
            self.config = replace(self.config, start_rule=start_rule)
        self.columns = [utils.REQUIRED_COLS.index(f) for f in ckpt["features"]]
        self.norm = ckpt["normalization"]
        self.device = device
        self.lag = self.model.config.lookahead_samples
        self.metadata = dict(checkpoint=str(path), checkpoint_sha256=digest(path),
                             acausal=self.model.config.acausal, future_context_ms=self.model.future_samples * DT_MS,
                             decoder_sha256=digest(fold / "decoder.json"), head=self.head,
                             causal_similarity=self.model.config.causal_similarity,
                             synthetic_reps=ckpt.get("synthetic_reps", False),
                             features=ckpt["features"], model_features=ckpt.get("model_features", ckpt["features"]),
                             rep_inputs=self.norm.get("rep_inputs", "original"), start_rule=self.config.start_rule,
                             refine_end=self.config.refine_end,
                             model_lookahead_ms=None if self.model.config.acausal else self.lag * DT_MS,
                             decoder_confirmation_ms=self.config.confirmation_frames * DT_MS,
                             simulated_input_buffer_ms=0)

    def predict(self, t, sensors):
        # Reset both network and decoder at clip boundaries and actual sensor gaps.
        intervals = []
        traces = np.full((len(t), self.model.config.replay_columns), np.nan, np.float32)
        cuts = np.r_[0, np.flatnonzero(np.diff(t) != DT_MS) + 1, len(t)]
        for a, b in zip(cuts[:-1], cuts[1:]):
            valid = np.ones(b - a, bool)
            station, q = probabilities(self.model, sensors[a:b, self.columns], valid,
                                       self.norm, self.device)
            available = prediction_mask(valid, self.lag)
            traces[a:b][available] = q[available]
            local_ms = np.arange(b - a) * DT_MS
            reps = decode_sequence(local_ms, station, q, available, self.config,
                                   lookahead_ms=self.lag * DT_MS)["repetitions"]
            for r in reps:
                event = {key: int(a + round(r[field] / DT_MS))
                         for key, field in (("start", "start_ms"), ("end", "end_ms"), ("reported_at", "reported_at_ms"))}
                if self.model.config.acausal:
                    event["reported_at"] = len(t) - 1
                intervals.append(event)
        names = {"phase": ("background", "phase 1", "phase 2"),
                 "boundary": ("start", "end"), "complete": ("end",)}[self.head]
        return Prediction(intervals, {name: traces[:, i] for i, name in enumerate(names)})


def benchmark(args):
    from baseline.rep_benchmark.evaluate import evaluate

    root = Path(args.run_dir)
    subjects = {}
    for i in range(1, 6):
        manifest = json.loads((root / f"fold_{i}/data/manifest.json").read_text())
        people = {stream["session"]: stream["subject"] for stream in manifest["sessions"]}
        subjects[i] = {part: sorted({people[s] for s in streams})
                       for part, streams in manifest["split"].items()}

    def load(fold):
        return WristModel(root / f"fold_{fold}", args.device, args.start_rule, args.refine_end)

    evaluate(args.source, args.output, args.model_name, subjects, load)
