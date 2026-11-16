from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch

from strength_training.temporal_segmentation.constants import CLASSES, DT_MS, EXCLUDED_SESSIONS, FEATURES
from strength_training.temporal_segmentation.data import (load_arrays, load_sensor, log_masks, phase_candidates,
                                                         prepare, reference_targets, subject_split)
from strength_training.temporal_segmentation.decode import DecoderConfig, RepDecoder, decode_sequence
from strength_training.temporal_segmentation.demo import synthetic_source
from strength_training.temporal_segmentation.metrics import (covered_repetitions, match_repetitions,
                                                            repetition_metrics)
from strength_training.temporal_segmentation.model import DenseMSTCN, ModelConfig, dense_loss
from strength_training.temporal_segmentation.runner import Windows


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(3)
        torch.set_num_threads(1)
        self.model = DenseMSTCN(ModelConfig(stages=2, layers=3, channels=8, dropout=0)).eval()
        self.x = torch.randn(1, 9, 75)

    def test_future_samples_cannot_change_prefix(self):
        changed = self.x.clone()
        changed[:, :, 35:] = torch.randn_like(changed[:, :, 35:]) * 50
        first, second = self.model(self.x), self.model(changed)
        for (a, b), (c, d) in zip(first, second):
            torch.testing.assert_close(a[:, :, :35], c[:, :, :35])
            torch.testing.assert_close(b[:, :, :35], d[:, :, :35])

    def test_cached_chunks_equal_full_causal_forward(self):
        expected = self.model(self.x)[-1]
        for chunk in (1, 7, 75):
            state, stations, boundaries = None, [], []
            for start in range(0, 75, chunk):
                p, q, state = self.model.stream(self.x[:, :, start:start+chunk], state)
                stations.append(p)
                boundaries.append(q)
            torch.testing.assert_close(torch.cat(stations, -1), expected[0], atol=1e-6, rtol=1e-5)
            torch.testing.assert_close(torch.cat(boundaries, -1), expected[1], atol=1e-6, rtol=1e-5)

    def test_left_padding_mask_matches_actual_sequence_start(self):
        padded = torch.cat((torch.zeros(1, 9, 20), self.x), -1)
        mask = torch.ones(1, 1, 95)
        mask[:, :, :20] = 0
        actual = self.model(padded, mask)[-1]
        expected = self.model(self.x)[-1]
        for p, q in zip(actual, expected):
            torch.testing.assert_close(p[:, :, 20:], q)

    def test_all_masked_loss_is_zero_and_differentiable(self):
        outputs = self.model(self.x)
        valid = torch.zeros(1, 75, dtype=torch.bool)
        loss = dense_loss(outputs, torch.zeros(1, 75, dtype=torch.long), torch.zeros(1, 75, 2), valid, valid)
        self.assertEqual(loss.item(), 0)
        loss.backward()
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in self.model.parameters()))

    def test_unreferenced_boundaries_do_not_supervise_head(self):
        output = [(torch.randn(1, 16, 4, requires_grad=True), torch.randn(1, 2, 4, requires_grad=True))]
        valid = torch.ones(1, 4, dtype=torch.bool)
        loss = dense_loss(output, torch.zeros(1, 4, dtype=torch.long), torch.ones(1, 4, 2), valid, ~valid)
        loss.backward()
        self.assertEqual(output[0][1].grad.abs().sum(), 0)


class DataTests(unittest.TestCase):
    def test_uncertain_log_alignment_is_not_rest(self):
        rows = pd.DataFrame([dict(scene_id="spull", collect_date="2026/01/01",
                                  device_start_time="08:00:00", device_end_time="08:00:01")])
        with self.assertRaisesRegex(ValueError, "unresolved"):
            log_masks(np.arange(25), rows, None)

    def test_invalid_affected_interval_requires_review(self):
        rows = pd.DataFrame([dict(scene_id="Other", collect_date="2026/01/01",
                                  device_start_time="missing", device_end_time="08:00:01")])
        evidence = dict(offset_hours=8, named_station_match_fraction=.8, runner_up_fraction=.1)
        with self.assertRaisesRegex(ValueError, "invalid affected"):
            log_masks(np.arange(25), rows, evidence)

    def test_exclusion_priority_over_spull_probe(self):
        rows = pd.DataFrame([dict(scene_id=s, collect_date="2026/01/01", device_start_time="08:00:00",
                                  device_end_time=b) for s, b in [("spull", "08:00:02"), ("Other", "08:00:01")]])
        t = 1767225600000 + np.arange(75) * 40
        evidence = dict(offset_hours=8, named_station_match_fraction=.8, runner_up_fraction=.1)
        valid, probe, _ = log_masks(t, rows, evidence)
        self.assertFalse(valid[:50].any())
        self.assertFalse(probe[:25].any())
        self.assertTrue(probe[25:50].all())
        self.assertTrue(valid[50:].all())

    def entry(self):
        return {"validated": True, "source": "test", "convention": "full cycle", "location": "waist",
                "coverage": [{"start_ms": 0, "end_ms": 1000}],
                "repetitions": [{"station": "squat", "start_ms": 80, "end_ms": 400}]}

    def test_no_implicit_phase_repetition_supervision(self):
        raw = np.array(["squat_phase1", "squat_phase2", "rest"])
        candidates = phase_candidates(np.arange(3) * 40, raw)
        self.assertEqual(len(candidates), 1)
        self.assertFalse(candidates[0]["validated"])
        b, m, r, _ = reference_targets(np.arange(3)*40, np.ones(3, bool), None, "waist")
        self.assertFalse(m.any())
        self.assertEqual(b.sum(), 0)
        self.assertEqual(r, [])

    def test_unvalidated_or_wrong_location_references_rejected(self):
        for update in ({"validated": False}, {"location": "left_wrist"}):
            with self.assertRaises(ValueError):
                reference_targets(np.arange(25)*40, np.ones(25, bool), self.entry() | update, "waist")

    def test_end_impulse_is_exclusive_end_tick(self):
        b, mask, reps, _ = reference_targets(np.arange(25)*40, np.ones(25, bool), self.entry(), "waist")
        self.assertEqual(b[2, 0], 1)
        self.assertEqual(b[10, 1], 1)
        self.assertTrue(mask.all())
        self.assertEqual(len(reps), 1)

    def test_cut_reference_does_not_become_negative(self):
        valid = np.ones(25, bool)
        valid[5] = False
        b, mask, refs, removed = reference_targets(np.arange(25)*40, valid, self.entry(), "waist")
        self.assertFalse(mask[2:11].any())
        self.assertFalse(b.any())
        self.assertEqual(len(refs), 0)
        self.assertEqual(len(removed), 1)

    def test_subjects_never_cross_splits(self):
        rows = [{"session": f"date{i}_{s}", "subject": s} for i in range(2) for s in "abcdef"]
        split = subject_split(rows)
        subjects = [{n.split("_")[1] for n in names} for names in split.values()]
        self.assertFalse(subjects[0] & subjects[1] or subjects[1] & subjects[2] or subjects[0] & subjects[2])
        self.assertEqual(split, subject_split(rows))

    def test_prepare_masks_splits_and_train_only_normalization(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            alignment, references = synthetic_source(root / "source")
            # Entirely excluded sessions must be dropped before any file loading.
            (root / "source" / sorted(EXCLUDED_SESSIONS)[0]).mkdir()
            manifest = prepare(root / "source", root / "prepared", alignment, references_path=references)
            self.assertEqual(len(manifest["sessions"]), 8)
            self.assertEqual(len(manifest["excluded"]), 1)
            selected = []
            for session in manifest["split"]["train"]:
                a = load_arrays(root / "prepared" / session)
                self.assertFalse(a["valid"][:10].any())
                self.assertEqual(a["x"].shape[1], 9)
                selected.append(a["x"][a["valid"]])
            expected = np.concatenate(selected).astype(np.float64)
            np.testing.assert_allclose(manifest["normalization"]["mean"], expected.mean(0), atol=1e-8)
            np.testing.assert_allclose(manifest["normalization"]["std"], expected.std(0), atol=1e-8)
            windows = Windows(root / "prepared", manifest["split"]["train"], manifest["normalization"], 28, 128)
            x, mask, y, boundary, valid, refs = windows[0]
            self.assertFalse(valid[:28].any())
            self.assertEqual(x.shape, (9, 156))
            self.assertTrue(refs[valid].all())

    def test_sensor_labels_cannot_change_features(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "sensor.csv"
            frame = pd.DataFrame(np.ones((5, 9)), columns=FEATURES)
            frame["sync_timestamp"], frame["labels"] = np.arange(5)*40, "squat"
            frame.to_csv(p, index=False)
            t, x = load_sensor(p)
            frame["labels"] = "rest"
            frame.to_csv(p, index=False)
            t2, x2 = load_sensor(p)
            np.testing.assert_array_equal(x, x2)
            np.testing.assert_array_equal(t, t2)


class DecoderTests(unittest.TestCase):
    def sequence(self, label="squat"):
        t = np.arange(50)*40
        p, q = np.zeros((50, 16)), np.zeros((50, 2))
        p[:, CLASSES.index(label)] = 1
        q[[2, 20], 0], q[[20, 40], 1] = 1, 1
        return t, p, q

    def test_adjacent_repetitions_shared_boundary_are_counted(self):
        t, p, q = self.sequence()
        result = decode_sequence(t, p, q, np.ones(50, bool))
        self.assertEqual(result["counts"], {"squat": 2})
        self.assertEqual(result["repetitions"][0]["end_ms"], result["repetitions"][1]["start_ms"])

    def test_event_plateau_does_not_duplicate_counts(self):
        t, p, q = self.sequence()
        q[40:, 1] = 1
        self.assertEqual(len(decode_sequence(t, p, q, np.ones(50, bool))["repetitions"]), 2)

    def test_null_and_plank_do_not_generate_reps(self):
        for label in ("NULL", "plank"):
            t, p, q = self.sequence(label)
            result = decode_sequence(t, p, q, np.ones(50, bool))
            self.assertEqual(result["repetitions"], [])
            if label == "plank":
                self.assertEqual(result["holds"][0]["duration_s"], 2.)

    def test_gap_cancels_partial_rep(self):
        t, p, q = self.sequence()
        valid = np.ones(50, bool)
        valid[10] = False
        result = decode_sequence(t, p, q, valid)
        self.assertEqual(len(result["repetitions"]), 1)
        keep = np.arange(50) != 10
        result = decode_sequence(t[keep], p[keep], q[keep], valid[keep])
        self.assertEqual(len(result["repetitions"]), 1)

    def test_partial_rep_at_end_is_not_counted(self):
        t, p, q = self.sequence()
        result = decode_sequence(t[:15], p[:15], q[:15], np.ones(15, bool))
        self.assertEqual(result["repetitions"], [])

    def test_long_incomplete_movement_times_out(self):
        t, p, q = self.sequence()
        result = decode_sequence(t, p, q, np.ones(50, bool), DecoderConfig(max_rep_ms=500))
        self.assertEqual(result["repetitions"], [])


class MetricsTests(unittest.TestCase):
    def rep(self, a, b, station="squat"):
        return {"station": station, "start_ms": a, "end_ms": b}

    def test_duplicates_match_one_reference_once(self):
        reference = [self.rep(0, 1000)]
        result = repetition_metrics([self.rep(0, 1000), self.rep(40, 1040)], reference)
        self.assertEqual((result["tp"], result["extra"], result["missed"]), (1, 1, 0))

    def test_matching_maximizes_cardinality_in_ambiguous_case(self):
        # A greedy nearest-first match could consume the only valid match for p1.
        p = [self.rep(200, 1200), self.rep(-200, 800)]
        r = [self.rep(0, 1000), self.rep(500, 1500)]
        self.assertEqual(len(match_repetitions(p, r, 400)), 2)

    def test_wrong_station_is_missed_and_extra(self):
        result = repetition_metrics([self.rep(0, 1000, "push-up")], [self.rep(0, 1000)])
        self.assertEqual((result["tp"], result["extra"], result["missed"]), (0, 1, 1))

    def test_reference_coverage_excludes_crossing_predictions(self):
        t = np.arange(50)*40
        mask = np.ones(50, bool)
        mask[20] = False
        scored, unscored = covered_repetitions([self.rep(0, 400), self.rep(400, 1200)], t, mask)
        self.assertEqual(len(scored), 1)
        self.assertEqual(len(unscored), 1)


if __name__ == "__main__":
    unittest.main()
