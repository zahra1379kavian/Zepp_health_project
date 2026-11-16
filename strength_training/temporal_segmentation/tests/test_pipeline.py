from dataclasses import replace
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch
from argparse import Namespace

import numpy as np
import pandas as pd
import torch

from strength_training.temporal_segmentation.constants import CLASSES, DT_MS, EXCLUDED_SESSIONS, FEATURES
from strength_training.temporal_segmentation.data import (load_arrays, load_sensor, log_masks,
                                                         phase_label_targets, prepare, reference_targets,
                                                         select_wrists, subject_split, subject_folds,
                                                         dense_phase_targets, training_statistics)
from strength_training.temporal_segmentation.decode import DecoderConfig, RepDecoder, decode_sequence
from strength_training.temporal_segmentation.demo import synthetic_source
from strength_training.temporal_segmentation.metrics import (covered_repetitions, match_repetitions,
                                                            repetition_metrics, score_stream,
                                                            grouped_stream_report, reporting_tolerances)
from strength_training.temporal_segmentation.model import DenseMSTCN, ModelConfig, dense_loss
from strength_training.temporal_segmentation.runner import Windows, checkpoint_key, score_cache, run_pipeline, evaluate


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
        loss = dense_loss(outputs, torch.zeros(1, 75, dtype=torch.long),
                          torch.zeros(1, 75, dtype=torch.long), valid, valid)
        self.assertEqual(loss.item(), 0)
        loss.backward()
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in self.model.parameters()))

    def test_unreferenced_boundaries_do_not_supervise_head(self):
        output = [(torch.randn(1, 16, 4, requires_grad=True), torch.randn(1, 2, 4, requires_grad=True))]
        valid = torch.ones(1, 4, dtype=torch.bool)
        loss = dense_loss(output, torch.zeros(1, 4, dtype=torch.long), torch.ones(1, 4, 2), valid, ~valid)
        loss.backward()
        self.assertEqual(output[0][1].grad.abs().sum(), 0)

    def test_dense_phase_loss_supervises_interior_frames_only_where_covered(self):
        output = [(torch.zeros(1, 16, 12, requires_grad=True), torch.zeros(1, 3, 12, requires_grad=True))]
        target = torch.tensor([[0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 0, 0]])
        valid = torch.ones(1, 12, dtype=torch.bool)
        covered = valid.clone()
        covered[:, 0] = False
        dense_loss(output, torch.zeros_like(target), target, valid, covered).backward()
        self.assertTrue((output[0][1].grad[:, :, 1:].abs().sum(1) > 0).all())
        self.assertEqual(output[0][1].grad[:, :, 0].abs().sum(), 0)

    def test_boundary_baseline_streaming_is_still_supported(self):
        model = DenseMSTCN(ModelConfig(stages=2, layers=2, channels=8, repetition_head="boundary")).eval()
        expected = model(self.x)[-1]
        p, q, _ = model.stream(self.x)
        torch.testing.assert_close(p, expected[0])
        torch.testing.assert_close(q, expected[1])


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

    def test_missing_external_reference_entry_is_not_negative_supervision(self):
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

    def test_cut_reference_preserves_preceding_shared_end(self):
        entry = self.entry()
        entry["repetitions"].append({"station": "squat", "start_ms": 400, "end_ms": 800})
        valid = np.ones(25, bool)
        valid[15] = False
        boundary, coverage, refs, removed = reference_targets(np.arange(25)*40, valid, entry, "waist")
        self.assertEqual(len(refs), 1)
        self.assertEqual(len(removed), 1)
        self.assertTrue(coverage[10])
        self.assertEqual(boundary[10, 1], 1)
        self.assertFalse(coverage[11:21].any())
        scored, _ = covered_repetitions(refs, np.arange(25)*40, coverage, refs)
        self.assertEqual(repetition_metrics(scored, refs)["tp"], 1)

    def test_subject_folds_are_disjoint_and_test_every_subject_once(self):
        rows = [{"session": f"date{i}_{s}", "subject": s} for i in range(2) for s in "abcdefghi"]
        folds = subject_folds(rows, 4, seed=7)
        self.assertEqual(folds, subject_folds(rows, 4, seed=7))
        tested = []
        for split in folds:
            groups = [{n.split("_")[1] for n in split[k]} for k in ("train", "validation", "test")]
            self.assertFalse(groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
            tested.extend(split["test"])
        self.assertCountEqual(tested, [r["session"] for r in rows])
        with self.assertRaises(ValueError):
            subject_folds(rows, 2)

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
            for split in subject_folds(manifest["sessions"], 3):
                stats = training_statistics(root / "prepared", split["train"])
                fold_x = np.concatenate([load_arrays(root / "prepared" / n)["x"][
                    load_arrays(root / "prepared" / n)["valid"]] for n in split["train"]])
                np.testing.assert_allclose(stats["normalization"]["mean"], fold_x.astype(float).mean(0))

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

    def test_wrist_selection_keeps_every_available_wrist(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "S1_waist.csv").touch()
            self.assertEqual(select_wrists(root), [])
            (root / "right_wrist.csv").touch()
            self.assertEqual(select_wrists(root), [("right_wrist", root / "right_wrist.csv")])
            (root / "left_wrist.csv").touch()
            self.assertEqual([s for s, _ in select_wrists(root)], ["left_wrist", "right_wrist"])
            self.assertEqual([s for s, _ in select_wrists(root, "right_wrist")], ["right_wrist"])
            with self.assertRaisesRegex(ValueError, "wrist signals only"):
                select_wrists(root, "waist")
            with self.assertRaisesRegex(ValueError, "waist file"):
                load_sensor(root / "S1_waist.csv")

    def test_label_targets_match_synthetic_full_references(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            alignment, references = synthetic_source(root / "source", subjects=3)
            # A right-only session must use right-wrist labels, never waist labels.
            first = root / "source" / "20260101_DEMO000"
            (first / "left_wrist.csv").rename(first / "right_wrist.csv")
            labels = pd.read_csv(first / "label.csv").rename(columns={"left_wrist": "right_wrist"})
            labels["waist"] = "rest"
            labels.to_csv(first / "label.csv", index=False)
            (first / "S1_waist.csv").write_text("invalid waist stream must never be read")
            manifest = prepare(root / "source", root / "prepared", alignment)
            expected = json.loads(references.read_text())["sessions"]
            for session in manifest["sessions"]:
                a = load_arrays(root / "prepared" / session["session"])
                reps = json.loads((root / "prepared" / session["session"] / "references.json").read_text())
                got = [{k: r[k] for k in ("station", "start_ms", "end_ms")} for r in reps]
                self.assertEqual(got, expected[session["source_session"]]["repetitions"])
                self.assertEqual(a["boundary"].sum(), 8)
                self.assertTrue(session["sensor"].endswith("wrist.csv"))
            self.assertEqual(manifest["sessions"][0]["label_column"], "right_wrist")

    def test_both_wrist_streams_keep_own_labels_gaps_and_subject_folds(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            alignment, _ = synthetic_source(root / "source", subjects=6, both_wrists=True)
            first = root / "source" / "20260101_DEMO000"
            labels = pd.read_csv(first / "label.csv")
            labels["right_wrist"] = labels.right_wrist.str.replace("squat", "biceps_curls")
            labels.to_csv(first / "label.csv", index=False)
            right = pd.read_csv(first / "right_wrist.csv").drop(index=120)
            right.to_csv(first / "right_wrist.csv", index=False)
            (root / "source" / "20260101_DEMO001" / "left_wrist.csv").unlink()
            (root / "source" / "20260101_DEMO002" / "right_wrist.csv").unlink()
            manifest = prepare(root / "source", root / "data", alignment)
            self.assertEqual(manifest["version"], 4)
            self.assertEqual((manifest["subjects"], manifest["source_sessions"]), (6, 6))
            self.assertEqual(manifest["streams_per_side"], {"left_wrist": 5, "right_wrist": 5})
            records = {r["session"]: r for r in manifest["sessions"]}
            left_name, right_name = [f"20260101_DEMO000__{side}_wrist" for side in ("left", "right")]
            left, right_arrays = [load_arrays(root / "data" / n) for n in (left_name, right_name)]
            self.assertTrue(left["valid"][120])
            self.assertFalse(right_arrays["valid"][120])
            self.assertEqual(left["y"][100], CLASSES.index("squat"))
            self.assertEqual(right_arrays["y"][100], CLASSES.index("biceps_curls"))
            self.assertEqual(records[left_name]["reference_repetitions"], 4)
            self.assertEqual(records[right_name]["reference_repetitions"], 3)
            for split in [manifest["split"], *subject_folds(manifest["sessions"], 3)]:
                ownership = {}
                for partition, names in split.items():
                    for name in names:
                        subject = records[name]["subject"]
                        self.assertEqual(ownership.setdefault(subject, partition), partition)
                self.assertEqual(len(ownership), 6)
            train_arrays = [load_arrays(root / "data" / n) for n in manifest["split"]["train"]]
            expected = np.concatenate([a["x"][a["valid"]] for a in train_arrays]).astype(float)
            np.testing.assert_allclose(manifest["normalization"]["mean"], expected.mean(0))

    def test_external_references_are_not_copied_to_other_wrist(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            alignment, refs_path = synthetic_source(root / "source", subjects=3, both_wrists=True)
            refs = json.loads(refs_path.read_text())
            name = "20260101_DEMO000"
            refs["sessions"][name] = refs["sessions"][name]["wrists"]["left_wrist"]
            refs_path.write_text(json.dumps(refs))
            prepare(root / "source", root / "data", alignment, references_path=refs_path)
            left = load_arrays(root / "data" / f"{name}__left_wrist")
            right = load_arrays(root / "data" / f"{name}__right_wrist")
            self.assertTrue(left["coverage"].any())
            self.assertFalse(right["coverage"].any())
            self.assertTrue(load_arrays(root / "data" / "20260101_DEMO001__right_wrist")["coverage"].any())

    def test_missing_right_wrist_labels_fail_instead_of_using_left(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            alignment, _ = synthetic_source(root / "source", subjects=3, both_wrists=True)
            path = root / "source" / "20260101_DEMO000" / "label.csv"
            pd.read_csv(path).drop(columns="right_wrist").to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "missing authoritative right_wrist"):
                prepare(root / "source", root / "data", alignment)


class PhaseReferenceTests(unittest.TestCase):
    def targets(self, labels, invalid=None):
        raw = np.array(labels)
        valid = np.ones(len(raw), bool)
        if invalid is not None:
            valid[invalid] = False
        return phase_label_targets(np.arange(len(raw))*40, raw, valid)

    def test_internal_transition_is_inside_rep_trailing_transition_outside(self):
        b, coverage, reps, rejected = self.targets([
            "rest", "squat_phase1", "squat_transition", "squat_phase2",
            "squat_transition", "rest"])
        self.assertEqual([(r["start_ms"], r["end_ms"]) for r in reps], [(40, 160)])
        self.assertEqual(b[1, 0], 1)
        self.assertEqual(b[4, 1], 1)
        self.assertTrue(coverage.all())
        self.assertEqual(rejected, [])

    def test_initial_phase2_is_incomplete_and_does_not_reverse_convention(self):
        _, _, reps, rejected = self.targets([
            "rest", "squat_phase2", "squat_transition", "squat_phase2",
            "squat_phase1", "rest"])
        self.assertEqual(reps, [])
        self.assertEqual(len(rejected), 2)

    def test_phase1_pause_and_dense_transition_targets(self):
        raw = np.array(["rest", "squat_phase1", "squat_transition", "squat_phase1",
                        "squat_phase2", "squat_transition", "squat_phase2", "rest"])
        t, valid = np.arange(len(raw))*40, np.ones(len(raw), bool)
        _, coverage, reps, rejected = phase_label_targets(t, raw, valid)
        self.assertEqual(len(reps), 1)
        self.assertEqual(reps[0]["phase_order"], "phase1-phase2")
        self.assertEqual(rejected, [])
        phase, mask = dense_phase_targets(t, raw, valid, coverage, reps)
        np.testing.assert_array_equal(phase, [0, 1, 1, 1, 2, 2, 2, 0])
        self.assertTrue(mask.all())

    def test_orphan_phase2_then_complete_cycle(self):
        _, _, reps, rejected = self.targets([
            "rest", "squat_phase2", "squat_phase1", "squat_phase2", "rest"])
        self.assertEqual([(r["start_ms"], r["end_ms"]) for r in reps], [(80, 160)])
        self.assertEqual(len(rejected), 1)

    def test_no_pair_across_rest_or_another_station(self):
        b, coverage, reps, rejected = self.targets([
            "rest", "squat_phase1", "rest", "squat_phase2", "push-up_phase1", "rest"])
        self.assertEqual(reps, [])
        self.assertEqual(len(rejected), 3)
        self.assertFalse(b.any())
        self.assertFalse(coverage[[1, 3, 4]].any())
        self.assertTrue(coverage[[0, 2, 5]].all())

    def test_gap_drops_cycle_without_corrupting_following_one(self):
        b, coverage, reps, rejected = self.targets([
            "rest", "squat_phase1", "squat_transition", "squat_phase2",
            "squat_phase1", "squat_phase2", "rest"], invalid=2)
        self.assertEqual(len(reps), 1)
        self.assertEqual((reps[0]["start_ms"], reps[0]["end_ms"]), (160, 240))
        self.assertFalse(coverage[1:4].any())
        self.assertEqual(b.sum(), 2)

    def test_known_end_tick_survives_unpaired_following_phase(self):
        b, coverage, reps, rejected = self.targets([
            "rest", "squat_phase1", "squat_phase2", "squat_phase1", "squat_phase1", "rest"])
        self.assertEqual(len(reps), 1)
        self.assertTrue(coverage[3])
        self.assertEqual(b[3, 1], 1)
        self.assertFalse(coverage[4])

    def test_recording_edges_and_plank_do_not_invent_repetitions(self):
        b, coverage, reps, rejected = self.targets([
            "squat_phase1", "squat_phase2", "rest", "plank", "rest",
            "squat_phase1", "squat_phase2"])
        self.assertEqual(reps, [])
        self.assertEqual(len(rejected), 2)
        self.assertFalse(b.any())
        self.assertTrue(coverage[2:5].all())


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

    def test_null_station_does_not_veto_complete_repetitions(self):
        t, p, q = self.sequence("NULL")
        result = decode_sequence(t, p, q, np.ones(50, bool))
        self.assertEqual(result["counts"], {"unknown": 2})

    def test_station_label_is_selected_from_exercise_classes(self):
        t, p, q = self.sequence("NULL")
        p[:, 0], p[:, CLASSES.index("squat")] = .6, .4
        result = decode_sequence(t, p, q, np.ones(50, bool), DecoderConfig(station_threshold=.3))
        self.assertEqual(result["counts"], {"squat": 2})

    def test_chunk_arrival_includes_confirmation_delay(self):
        t, p, q = self.sequence()
        decoder, reps = RepDecoder(), []
        for a in range(0, len(t), 7):
            b = min(a + 7, len(t))
            for i in range(a, b):
                reps.extend(decoder.step(t[i], p[i], q[i], reported_at_ms=int(t[b-1])))
        self.assertEqual([r["end_ms"] for r in reps], [800, 1600])
        self.assertEqual([r["reported_at_ms"] for r in reps], [1080, 1920])

    def test_invalid_decoder_thresholds_and_noninteger_lag_are_rejected(self):
        for kwargs in ({"end_threshold": 1.1}, {"phase_threshold": 0}, {"lag_frames": 2.5}):
            with self.assertRaises(ValueError):
                DecoderConfig(**kwargs)

    def test_plank_is_a_hold(self):
        t, p, q = self.sequence("plank")
        result = decode_sequence(t, p, q, np.ones(50, bool))
        self.assertEqual(result["repetitions"], [])
        self.assertEqual(result["holds"][0]["duration_s"], 2.)

    def test_broad_boundary_peak_uses_peak_time_and_actual_confirmation_time(self):
        t, p, q = self.sequence()
        q[:, 1] = 0
        q[18:23, 1] = [.55, .7, .95, .7, .4]
        result = decode_sequence(t, p, q, np.ones(50, bool))
        rep = result["repetitions"][0]
        self.assertEqual(rep["end_ms"], 800)
        self.assertEqual(rep["reported_at_ms"], 880)

    def test_dense_phases_count_first_and_unknown_reps(self):
        t, p, _ = self.sequence("NULL")
        labels = np.zeros(50, int)
        labels[2:10], labels[10:20], labels[20:30], labels[30:40] = 1, 2, 1, 2
        q = np.eye(3)[labels]
        result = decode_sequence(t, p, q, np.ones(50, bool))
        self.assertEqual(result["counts"], {"unknown": 2})
        self.assertEqual([(r["start_ms"], r["end_ms"], r["reported_at_ms"]) for r in result["repetitions"]],
                         [(80, 800, 880), (800, 1600, 1680)])
        # A one-frame phase glitch must not invent an early boundary.
        q[15] = [0, 1, 0]
        self.assertEqual(decode_sequence(t, p, q, np.ones(50, bool))["counts"], {"unknown": 2})

    def test_phase_decoder_gap_and_eof_do_not_complete_partial_cycle(self):
        t, p, _ = self.sequence()
        labels = np.zeros(50, int)
        labels[2:10], labels[10:20] = 1, 2
        q = np.eye(3)[labels]
        self.assertEqual(decode_sequence(t[:21], p[:21], q[:21], np.ones(21, bool))["repetitions"], [])
        valid = np.ones(50, bool)
        valid[15] = False
        self.assertEqual(decode_sequence(t, p, q, valid)["repetitions"], [])

    def test_short_phase_cycle_cannot_absorb_later_orphan_phase2(self):
        t, p, _ = self.sequence()
        labels = np.zeros(50, int)
        labels[2:5], labels[5:8], labels[15:25] = 1, 2, 2
        result = decode_sequence(t, p, np.eye(3)[labels], np.ones(50, bool), DecoderConfig(min_rep_ms=280))
        self.assertEqual(result["repetitions"], [])

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

    def test_matching_precedes_coverage_filter_at_both_edges(self):
        t, coverage = np.arange(50)*40, np.zeros(50, bool)
        coverage[10:31] = True
        refs = [self.rep(400, 1200)]
        for predicted in (self.rep(360, 1200), self.rep(400, 1240)):
            scored, unscored = covered_repetitions([predicted], t, coverage, refs)
            self.assertEqual(unscored, [])
            self.assertEqual(repetition_metrics(scored, refs)["f1"], 1)

    def test_only_matched_outside_predictions_survive_coverage_filter(self):
        t, coverage = np.arange(50)*40, np.zeros(50, bool)
        coverage[10:40] = True
        refs = [self.rep(400, 1200)]
        predictions = [self.rep(360, 1200), self.rep(360, 1240), self.rep(400, 1200, "push-up")]
        scored, unscored = covered_repetitions(predictions, t, coverage, refs)
        self.assertEqual(len(unscored), 1)
        metric = repetition_metrics(scored, refs)
        self.assertEqual((metric["tp"], metric["extra"], metric["missed"]), (1, 1, 0))

    def test_unknown_station_can_match_in_segmentation_score(self):
        predicted, refs = [self.rep(0, 1000, "unknown")], [self.rep(0, 1000)]
        self.assertEqual(repetition_metrics(predicted, refs)["tp"], 0)
        self.assertEqual(repetition_metrics(predicted, refs, require_station=False)["tp"], 1)

    def test_early_matched_report_meets_deadline_but_is_reported_separately(self):
        metric = repetition_metrics([self.rep(0, 960)], [self.rep(0, 1000)])
        self.assertEqual(metric["matched_reported_before_1s_fraction"], 1)
        self.assertEqual(metric["matched_early_report_fraction"], 1)
        self.assertEqual(metric["matched_reported_0_to_999ms_fraction"], 0)
        late = self.rep(0, 1000) | {"reported_at_ms": 2000}
        self.assertEqual(repetition_metrics([late], [self.rep(0, 1000)])["matched_reported_before_1s_fraction"], 0)

    def test_checkpoint_ranking_uses_f1_then_count_error(self):
        rows = [{"f1": .7, "count_mae_all_station_sessions": 0, "loss": .1},
                {"f1": .8, "count_mae_all_station_sessions": 2, "loss": 9},
                {"f1": .8, "count_mae_all_station_sessions": 1, "loss": 10}]
        self.assertIs(max(rows, key=checkpoint_key), rows[2])

    def test_validation_selection_retains_edge_match(self):
        t = np.arange(50)*40
        p, q, coverage = np.zeros((50, 16)), np.zeros((50, 2)), np.zeros(50, bool)
        p[:, CLASSES.index("squat")] = 1
        q[9, 0], q[30, 1], coverage[10:40] = 1, 1, True
        arrays = {"t": t, "coverage": coverage, "valid": np.ones(50, bool)}
        cache = [("subject", arrays, p, q, [self.rep(400, 1200)])]
        self.assertEqual(score_cache(cache, DecoderConfig(), 400)["f1"], 1)

    def test_multiple_tolerances_score_identical_predictions(self):
        predictions, refs = [self.rep(680, 1680)], [self.rep(400, 1400)]
        original = json.dumps(predictions)
        metadata = {"session": "date_subject__left_wrist", "source_session": "date_subject",
                    "subject": "subject", "side": "left_wrist"}
        scores = score_stream(predictions, refs, np.arange(60)*40, np.ones(60, bool), metadata, [200, 400])
        self.assertEqual(scores["200"]["segmentation"]["metrics"]["f1"], 0)
        self.assertEqual(scores["400"]["segmentation"]["metrics"]["f1"], 1)
        self.assertEqual(scores["200"]["segmentation"]["counts"], scores["400"]["segmentation"]["counts"])
        self.assertEqual(json.dumps(predictions), original)

    def test_reporting_tolerances_keep_primary_and_reject_invalid_values(self):
        self.assertEqual(reporting_tolerances([400, 200, 400], 300), [200, 300, 400])
        for values in ([], [0], [-200]):
            with self.assertRaises(ValueError):
                reporting_tolerances(values)

    def test_subject_macro_does_not_give_paired_subject_twice_the_weight(self):
        records = []
        for subject, side in (("paired", "left_wrist"), ("paired", "right_wrist"), ("single", "left_wrist")):
            metadata = {"session": f"date_{subject}__{side}", "source_session": f"date_{subject}",
                        "subject": subject, "side": side}
            predicted = self.rep(1000, 2000) if subject == "paired" else self.rep(400, 1400)
            record = {**metadata, "rest_hours": 0, "false_exercise_events": 0,
                      "confusion_matrix": np.zeros((len(CLASSES), len(CLASSES)), int).tolist()}
            record["repetition_scores"] = score_stream([predicted], [self.rep(400, 1400)],
                                                        np.arange(75)*40, np.ones(75, bool), metadata, [200, 400])
            records.append(record)
        report = grouped_stream_report(records, [200, 400])
        self.assertEqual((report["subjects"], report["streams"]), (2, 3))
        self.assertAlmostEqual(report["segmentation"]["f1"], 1/3)
        self.assertEqual(report["by_tolerance"]["400"]["subject_macro_segmentation_f1"], .5)


class ReportingTests(unittest.TestCase):
    def test_completed_fold_resume_preserves_results_and_restarts_only_missing_folds(self):
        from strength_training.temporal_segmentation.__main__ import main
        from strength_training.temporal_segmentation import runner

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            alignment, _ = synthetic_source(root / "source", subjects=3, both_wrists=True)
            prepare(root / "source", root / "data", alignment)
            command = ["cross-validate", "--data", str(root / "data"), "--output", str(root / "cv"),
                       "--folds", "3", "--epochs", "1", "--max-steps", "1", "--batch-size", "2",
                       "--window", "128", "--stages", "1", "--layers", "1", "--channels", "4", "--threads", "1"]
            main(command)
            expected = json.loads((root / "cv/report.json").read_text())
            fold_one = root / "cv/fold_1"
            saved = {p.relative_to(fold_one): p.read_bytes() for p in fold_one.rglob("*")
                     if p.is_file() and not p.is_symlink() and "data" not in p.relative_to(fold_one).parts}
            for index in (2, 3):
                shutil.rmtree(root / f"cv/fold_{index}")
            with patch.object(runner, "train", wraps=runner.train) as train:
                main([*command, "--resume-completed"])
            self.assertEqual([call.args[0].output.parent.name for call in train.call_args_list], ["fold_2", "fold_3"])
            self.assertEqual(json.loads((root / "cv/report.json").read_text()), expected)
            for name, content in saved.items():
                self.assertEqual((fold_one / name).read_bytes(), content)
            with self.assertRaises(FileExistsError):
                main(command)
            with self.assertRaisesRegex(ValueError, "Training configuration changed"):
                main([*command, "--resume-completed", "--channels", "5"])
            (root / "cv/fold_2/model/best.pt").unlink()
            with self.assertRaisesRegex(FileExistsError, "Incomplete fold"):
                main([*command, "--resume-completed"])

    def test_evaluation_decodes_once_and_reports_sides_at_each_tolerance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            alignment, _ = synthetic_source(root / "source", subjects=3, both_wrists=True)
            manifest = prepare(root / "source", root / "data", alignment)
            calls = []

            def predicted_probabilities(*args):
                offset = 7 if not calls else 0
                calls.append(offset)
                p, q = np.zeros((600, len(CLASSES))), np.zeros((600, 2))
                p[:300, CLASSES.index("squat")] = 1
                p[300:, CLASSES.index("push-up")] = 1
                q[np.array([100, 175, 300, 400]) + offset, 0] = 1
                q[np.array([150, 225, 375, 475]) + offset, 1] = 1
                return p, q

            args = Namespace(data=root / "data", output=root / "evaluation", checkpoint="unused", device="cpu",
                             decoder=None, tolerance_ms=400, report_tolerances_ms=[200, 400], split="test")
            ckpt = {"normalization": manifest["normalization"], "station_only": False}
            with patch("strength_training.temporal_segmentation.runner.load_checkpoint", return_value=(None, ckpt)), \
                    patch("strength_training.temporal_segmentation.runner.probabilities", side_effect=predicted_probabilities), \
                    patch("strength_training.temporal_segmentation.runner.decode_sequence", wraps=decode_sequence) as decode:
                report = evaluate(args)
            self.assertEqual(decode.call_count, 2)
            self.assertEqual(len(calls), 2)
            self.assertEqual((report["streams"], report["subjects"], report["source_sessions"]), (2, 1, 1))
            self.assertEqual(report["by_tolerance"]["200"]["segmentation"]["f1"], .5)
            self.assertEqual(report["by_tolerance"]["400"]["segmentation"]["f1"], 1)
            self.assertEqual(report["per_side"]["left_wrist"]["by_tolerance"]["200"]["segmentation"]["f1"], 0)
            self.assertEqual(report["per_side"]["right_wrist"]["by_tolerance"]["200"]["segmentation"]["f1"], 1)
            self.assertEqual(len(report["per_subject"]), 1)
            self.assertEqual(report["report_scope"], "single_split_diagnostic")
            self.assertTrue(all(r["subject"] == next(iter(report["per_subject"])) for r in report["per_subject_session"]))

    def test_run_routes_to_cross_validation_with_declared_tolerances(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "data"
            data.mkdir()
            (data / "manifest.json").write_text(json.dumps({"version": 4, "location": "wrist", "reference_mode": "labels"}))
            args = Namespace(data=data, output=root / "run", location="wrist", epochs=1, batch_size=4,
                             tolerance_ms=400, report_tolerances_ms=[200, 400], folds=5, device="cpu", threads=1, seed=42)
            with patch("strength_training.temporal_segmentation.__main__.main") as main:
                run_pipeline(args)
            self.assertEqual(main.call_count, 1)
            command = main.call_args.args[0]
            self.assertEqual(command[0], "cross-validate")
            self.assertEqual(command[command.index("--folds") + 1], "5")
            i = command.index("--report-tolerances-ms")
            self.assertEqual(command[i+1:i+3], ["200", "400"])

    def test_old_single_wrist_dataset_cannot_be_headline_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "manifest.json").write_text(json.dumps({"version": 3, "location": "wrist"}))
            args = Namespace(data=root, output=root / "run", location="wrist", epochs=1, batch_size=4, tolerance_ms=400)
            with self.assertRaisesRegex(ValueError, "version 4"):
                run_pipeline(args)


if __name__ == "__main__":
    unittest.main()
