import json
from pathlib import Path
import tempfile
import unittest
from copy import deepcopy

import numpy as np
import torch
from torch.nn import functional as F

from baseline.mstcn.__main__ import main
from baseline.mstcn.constants import START_HISTORY
from baseline.mstcn.data import prepare
from baseline.mstcn.demo import synthetic_source
from baseline.mstcn.model import CausalSimilarity, DenseMSTCN, ModelConfig
from baseline.mstcn.runner import load_checkpoint, probabilities


class SimilarityTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        torch.set_num_threads(1)
        self.config = ModelConfig(repetition_head="complete", stages=3, layers=2,
                                  channels=8, dropout=0, causal_similarity=True)

    def test_every_lag_matches_direct_cosines_including_missing_history(self):
        layer = CausalSimilarity(8)
        x = torch.randn(2, 8, 249)
        mask = torch.ones(2, 1, x.size(2))
        mask[1, :, :17] = 0
        rows, projected, cache = layer(x, mask)
        z = F.normalize(layer.embedding(x), dim=1) * mask
        self.assertEqual(z.shape[1], 32)
        self.assertEqual(projected.shape, (2, 64, x.size(2)))
        for lag in range(1, START_HISTORY + 1):
            expected = (z[:, :, :-lag] * z[:, :, lag:]).sum(1)
            torch.testing.assert_close(rows[:, START_HISTORY-lag], F.pad(expected, (lag, 0)))
        torch.testing.assert_close(cache, z[:, :, -START_HISTORY:])

    def test_start_bias_and_gradient_paths(self):
        model = DenseMSTCN(self.config)
        x = torch.randn(1, self.config.inputs, 175)
        selector = model.stages[-1].selector
        self.assertEqual(selector.alpha.item(), 1.)
        rows = []
        handle = model.similarity.register_forward_hook(lambda module, args, output: rows.append(output[0]))
        _, events = model(x)[-1]
        handle.remove()
        # Changing alpha changes precisely the 150 matching start slots.
        with torch.no_grad():
            original = events.clone()
        (events[:, 1, -1] + events[:, 0, -1]).backward()
        self.assertNotEqual(selector.alpha.grad.item(), 0.)
        self.assertGreater(model.similarity.embedding.weight.grad.abs().sum().item(), 0.)
        self.assertGreater(model.similarity.projection.weight.grad.abs().sum().item(), 0.)
        with torch.no_grad():
            selector.alpha.zero_()
            _, unbiased = model(x)[-1]
            torch.testing.assert_close(original[:, [0, -1]], unbiased[:, [0, -1]])
            torch.testing.assert_close((original - unbiased)[:, 1:-1], rows[0], atol=2e-6, rtol=1e-5)
        for stage in model.stages[1:]:
            self.assertEqual(stage.project.in_channels, self.config.inputs + 1 + 64)

    @torch.no_grad()
    def test_chunk_window_padding_and_gap_reset_match_continuous_replay(self):
        model = DenseMSTCN(self.config).eval()
        x = torch.randn(1, self.config.inputs, 535)
        expected = model(x)[-1]
        changed = x.clone()
        changed[:, :, 173:] *= -7
        for a, b in zip(expected, model(changed)[-1]):
            torch.testing.assert_close(a[:, :, :173], b[:, :, :173])
        for chunk_size in (1, 13, 173):
            state, results = None, []
            for part in x.split(chunk_size, dim=2):
                station, events, state = model.stream(part, state)
                results.append((station, events))
            self.assertEqual(state[-1].shape, (1, 32, START_HISTORY))
            for value, chunks in zip(expected, zip(*results)):
                torch.testing.assert_close(value, torch.cat(chunks, dim=2), atol=2e-6, rtol=1e-5)
        start, history = 430, model.history_samples
        window = model(x[:, :, start-history:])[-1]
        padded = F.pad(x, (40, 0), value=23.)
        masked = model(padded, F.pad(torch.ones_like(x[:, :1]), (40, 0)))[-1]
        for value, local, prefix in zip(expected, window, masked):
            torch.testing.assert_close(value[:, :, start:], local[:, :, history:])
            torch.testing.assert_close(value, prefix[:, :, 40:])
        raw = x[0].T.numpy()
        valid = np.ones(len(raw), bool)
        valid[270:280] = False
        norm = {"mean": [0.] * raw.shape[1], "std": [1.] * raw.shape[1]}
        gap_outputs = probabilities(model, raw, valid, norm)
        fresh = probabilities(model, raw[280:], valid[280:], norm)
        for whole, suffix in zip(gap_outputs, fresh):
            np.testing.assert_allclose(whole[280:], suffix)

    def test_comparison_matches_training_protocol_and_supports_pre_similarity_reports(self):
        guards = {"seen_f1_at_400": .7, "crunch_f1_at_400": .3, "certified_rest_hours": 1,
                  "false_reps_during_rest": 2, "first_rep_recall": {"matched": 3, "total": 5},
                  "unseen_first_rep_recall": {"matched": 1, "total": 2}, "seen_exercises": ["crunch"]}
        row = {"by_tolerance": {"400": {"tp": 5, "missed": 5, "extra": 5}}, "guard_rails": guards}
        base = {"training": {"rep_inputs": "gravity", "seed": 42}, "protocol": "paired", "rep_inputs": "gravity",
                "repetition_head": "complete", "exercises": ["squat"], "subject_fold": 1,
                "split": {"test": ["subject1"], "validation": ["subject2"], "train": ["subject3"]},
                "selection_exercises": {"squat": "pull-up"}, "results": {"squat": row}}
        sim = deepcopy(base)
        sim["training"]["causal_similarity"] = True
        sim["results"]["squat"]["by_tolerance"]["400"] = {"tp": 8, "missed": 2, "extra": 2}
        with tempfile.TemporaryDirectory() as tmp:
            control, candidate, out = (Path(tmp) / f"{key}.json" for key in ("control", "candidate", "decision"))
            control.write_text(json.dumps(base))
            candidate.write_text(json.dumps(sim))
            cmd = ["compare-similarity", "--control-reports", str(control), "--similarity-reports", str(candidate)]
            main([*cmd, "--output", str(out)])
            self.assertTrue(json.loads(out.read_text())["passes_step_2_gate"])
            sim["training"]["seed"] = 41
            candidate.write_text(json.dumps(sim))
            with self.assertRaisesRegex(ValueError, "identical"):
                main([*cmd, "--output", str(out.with_suffix(".mismatch"))])

    def test_checkpoint_round_trip_and_resume_checks_architecture(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            alignment, _ = synthetic_source(root / "source", subjects=3)
            prepare(root / "source", root / "data", alignment)
            args = ["cross-validate", "--data", str(root / "data"), "--output", str(root / "cv"),
                    "--causal-similarity", "--rep-inputs", "gravity", "--repetition-head", "complete",
                    "--folds", "3", "--epochs", "1", "--max-steps", "1", "--layers", "2", "--channels", "8",
                    "--threads", "1"]
            main(args)
            fold = root / "cv/fold_1"
            model, _ = load_checkpoint(fold / "model/best.pt", data=fold / "data")
            self.assertTrue(model.config.causal_similarity)
            self.assertTrue(json.loads((fold / "test/report.json").read_text())["causal_similarity"])
            main(args + ["--resume-completed"])
            args.remove("--causal-similarity")
            with self.assertRaisesRegex(ValueError, "configuration changed"):
                main(args + ["--resume-completed"])
