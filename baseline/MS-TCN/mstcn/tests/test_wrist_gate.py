import unittest
from unittest.mock import patch

import numpy as np
import torch

from ..analyses.wrist_gate import model


class WristGateTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(17)
        self.wrist = rng.normal(size=(110, 9)).astype("f")
        self.waist = rng.normal(size=(110, 6)).astype("f")
        self.q = rng.dirichlet([1, 1, 1], size=110).astype("f")
        self.expert = model.PhaseHead(False)
        self.gate = model.PhaseHead(True)
        with torch.no_grad():
            # Nonconstant heads make future leakage and stale state observable.
            self.expert.net[-1].weight.fill_(.08)
            self.expert.net[-1].weight[1].mul_(-1)
            self.gate.net[-1].weight.fill_(.1)

    def scores(self, a, b, present):
        from types import SimpleNamespace
        base = SimpleNamespace(model=None, columns=list(range(9)), norm=None, device="cpu")
        with patch.object(model, "probabilities", return_value=(None, self.q[a:b])):
            return model.phase_streams(base, self.expert, self.gate, self.wrist[a:b],
                                       self.waist[a:b], present[a:b])[1:]

    def test_prefix_invariance_and_gap_fallback(self):
        present = np.ones(len(self.q), bool)
        present[54:62] = False
        full, weights = self.scores(0, 110, present)
        prefix, prefix_weights = self.scores(0, 47, present)
        suffix, suffix_weights = self.scores(62, 110, present)
        for arm in model.ARMS:
            np.testing.assert_allclose(full[arm][:47], prefix[arm], atol=1e-6)
            np.testing.assert_allclose(full[arm][62:], suffix[arm], atol=1e-6)
            np.testing.assert_array_equal(full[arm][~present], self.q[~present])
        np.testing.assert_array_equal(weights[~present], 0)
        np.testing.assert_allclose(weights[:47], prefix_weights, atol=1e-6)
        np.testing.assert_allclose(weights[62:], suffix_weights, atol=1e-6)
        mix = self.q + weights[:, None] * (full["fused"] - self.q)
        np.testing.assert_allclose(full["gated"], mix, atol=1e-6)

    def test_exact_waist_alignment_including_absent_device(self):
        wrist_t = np.arange(6) * 40
        for idx in ([], [1, 2, 4]):
            x = self.waist[:len(idx)]
            aligned, valid = model.align_waist(wrist_t, wrist_t[idx], x)
            np.testing.assert_array_equal(np.flatnonzero(valid), idx)
            np.testing.assert_array_equal(aligned[valid], x)
            np.testing.assert_array_equal(aligned[~valid], 0)
