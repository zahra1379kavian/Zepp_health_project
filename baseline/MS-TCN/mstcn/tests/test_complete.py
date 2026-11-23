import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from baseline.mstcn.__main__ import main
from baseline.mstcn.complete import complete_targets, rest_candidates
from baseline.mstcn.constants import DT_MS, START_HISTORY
from baseline.mstcn.data import prepare
from baseline.mstcn.decode import DecoderConfig, RepDecoder, decode_sequence
from baseline.mstcn.demo import synthetic_source
from baseline.mstcn.model import DenseMSTCN, ModelConfig, dense_loss


class CompleteRepTests(unittest.TestCase):
    def test_endpoint_targets_jitter_false_ends_reviews_and_cutoff(self):
        t = np.arange(280) * DT_MS
        valid = np.ones(len(t), bool)
        valid[91] = False
        arrays = dict(t=t, y=np.zeros(len(t)), valid=valid, coverage=valid)
        reps = [dict(start_ms=t[a], end_ms=t[b]) for a, b in [(10, 50), (50, 90), (100, 260)]]
        reviews = [dict(start_ms=t[a], end_ms=t[b], review=s) for a, b, s in
                   [(3, 8, "confirmed_rest"), (93, 98, "contains_reps")]]
        target = complete_targets(arrays, reps, reviews)
        np.testing.assert_allclose(target["end"][48:53], [1/3, 2/3, 1, 2/3, 1/3])
        np.testing.assert_equal(target["start_choice"][48:53], START_HISTORY - np.arange(38, 43))
        self.assertEqual(target["start_choice"][70], START_HISTORY)
        self.assertEqual(target["start_choice"][90], START_HISTORY - 40)
        self.assertEqual(target["start_choice"][260], START_HISTORY)
        self.assertEqual(target["start_choice"][92], -1)
        np.testing.assert_equal(target["weight"][3:8], 3)
        np.testing.assert_equal(target["weight"][93:98], 0)

    def test_hard_negatives_require_three_reps_inside_one_rest_stretch(self):
        t = np.arange(180) * DT_MS
        y, valid = np.zeros(len(t)), np.ones(len(t), bool)
        y[40], valid[80] = 1, False
        reps = [dict(start_ms=int(t[i]), end_ms=int(t[i+4])) for i in (10, 20, 45, 55, 82, 92, 102)]
        result = rest_candidates(dict(t=t, y=y, coverage=valid, valid=valid), reps)
        self.assertEqual([(r["start_ms"], r["end_ms"], r["predicted_reps"]) for r in result],
                         [(t[82], t[107], 3)])

    def test_selector_streaming_causality_and_station_independence(self):
        torch.manual_seed(7)
        config = ModelConfig(classes=2, repetition_head="complete", stages=2, layers=2, channels=6)
        model = DenseMSTCN(config).eval()
        x = torch.randn(1, config.inputs, 181)
        with torch.no_grad():
            p, q = model(x)[-1]
            self.assertTrue((q[:, 1:-1, 0] < -1000).all())
            state, chunks = None, []
            for part in x.split(17, dim=-1):
                _, events, state = model.stream(part, state)
                chunks.append(events)
            torch.testing.assert_close(torch.cat(chunks, dim=-1), q)
            torch.testing.assert_close(model(x[:, :, :70])[-1][1], q[:, :, :70])
            model.stages[0].station.bias.add_(torch.tensor([80., -80.]))
            torch.testing.assert_close(model(x)[-1][1], q)

    def test_no_start_competes_at_rest_peak_and_empty_mask_is_finite(self):
        end = torch.tensor([[[-4., 4., -4.]]], requires_grad=True)
        start = torch.zeros(1, START_HISTORY + 1, 3, requires_grad=True)
        station = torch.zeros(1, 2, 3, requires_grad=True)
        labels = torch.zeros(1, 3, dtype=torch.long)
        valid = torch.ones_like(labels, dtype=torch.bool)
        def loss(mask):
            return dense_loss([(station, torch.cat((end, start), dim=1))], labels, labels.float(),
                              valid, mask, start_choice=labels-1, rep_weights=valid.float())
        self.assertTrue(torch.isfinite(loss(~valid)))
        loss(valid).backward()
        self.assertLess(start.grad[0, -1, 1], 0)
        self.assertFalse(start.grad[0, :, 0].any())

    def test_first_rep_partial_gaps_confirmation_and_previous_end(self):
        t = np.arange(55) * DT_MS
        q = np.zeros((len(t), 4))
        for i in (12, 22, 32, 41, 54):
            q[i] = [1, 10, .8, .05]
        q[32, 3] = .95
        p = np.tile([.99, .01], (len(t), 1))
        valid = np.ones(len(t), bool)
        valid[36] = False
        c = DecoderConfig()
        for rule, expected in [("learned", [(2, 12), (12, 22)]), ("previous_end", [(12, 22), (22, 32)])]:
            config = replace(c, start_rule=rule)
            bulk = decode_sequence(t, p, q, valid, config)["repetitions"]
            online, decoder = [], RepDecoder(config)
            for i in range(len(t)):
                online.extend(decoder.step(t[i], p[i], q[i], valid[i]))
            self.assertEqual(online, bulk)
            self.assertEqual([(r["start_ms"], r["end_ms"]) for r in bulk],
                             [(a * DT_MS, b * DT_MS) for a, b in expected])
            self.assertTrue(all(r["reported_at_ms"] == r["end_ms"] + 2 * DT_MS for r in bulk))

    def test_end_refinement_matches_online_replay_and_respects_gaps(self):
        t = np.arange(36) * DT_MS
        p = np.tile([.01, .99], (len(t), 1))
        q = np.zeros((len(t), 4))
        q[6:15, 0] = [0, 0, 0, .5, 1., .75, .5, .25, .125]
        q[10, 1:] = [5, .8, .05]
        q[[20, 33]] = [1., 2, .8, .05]
        valid = np.ones(len(t), bool)
        valid[16] = False
        config = DecoderConfig(refine_end=True)

        decoder = RepDecoder(config)
        online = []
        for i in range(len(t)):
            online.extend(decoder.step(t[i], p[i], q[i], valid[i]))
        bulk = decode_sequence(t, p, q, valid, config)["repetitions"]

        self.assertEqual(online, bulk)
        self.assertEqual([(r["start_ms"], r["end_ms"], r["reported_at_ms"]) for r in bulk],
                         [(t[5], t[10] + 32, t[14]), (t[18], t[20], t[24])])

    def test_endpoint_only_cross_validation_and_sensor_replay(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            alignment, references = synthetic_source(root / "source", subjects=3)
            entries = json.loads(references.read_text())
            for name, entry in entries["sessions"].items():
                for rep in entry["repetitions"]:
                    del rep["station"]
                label_file = root / "source" / name / "label.csv"
                labels = pd.read_csv(label_file)
                labels["left_wrist"] = np.where(labels.left_wrist == "rest", "rest", "at_station")
                labels.to_csv(label_file, index=False)
            references.write_text(json.dumps(entries))
            manifest = prepare(root / "source", root / "data", alignment, references_path=references,
                               reference_mode="endpoints")
            # Fail if training, fold statistics or scoring attempts to load a phase.
            for row in manifest["sessions"]:
                for key in ("phase", "phase_mask"):
                    phase_file = root / "data" / row["session"] / f"{key}.npy"
                    self.assertFalse(phase_file.exists())
                    phase_file.write_bytes(b"unused")
            cv = root / "cv"
            args = ["cross-validate", "--data", str(root/"data"),
                    "--output", str(cv), "--folds", "3", "--epochs", "1", "--max-steps", "1",
                    "--layers", "2", "--stages", "2", "--channels", "6", "--threads", "1"]
            main(args)
            main([*args, "--resume-completed"])
            report = json.loads((cv / "report.json").read_text())
            self.assertEqual(report["subjects"], 3)
            self.assertIn("previous_end", report["by_tolerance"]["200"])
            self.assertEqual(report["station_presence"]["classes"], ["rest", "at_station"])
            fold = cv / "fold_1"
            main(["predict", "--input", str(root/"source"/"20260101_DEMO000"/"left_wrist.csv"),
                  "--checkpoint", str(fold/"model"/"best.pt"), "--decoder", str(fold/"decoder.json"),
                  "--output", str(root/"replay"), "--chunk", "7"])
            self.assertIn("no_start_probability", (root/"replay"/"frames.csv").read_text().splitlines()[0])
            main(["train", "--data", str(root/"data"), "--output", str(root/"boundary"),
                  "--repetition-head", "boundary", "--epochs", "1", "--max-steps", "1",
                  "--stages", "1", "--layers", "2", "--channels", "6"])
            main(["evaluate", "--data", str(root/"data"), "--checkpoint", str(root/"boundary"/"best.pt"),
                  "--output", str(root/"boundary_test")])
            boundary = json.loads((root/"boundary_test"/"report.json").read_text())
            self.assertEqual(boundary["station_presence"]["classes"], ["rest", "at_station"])
