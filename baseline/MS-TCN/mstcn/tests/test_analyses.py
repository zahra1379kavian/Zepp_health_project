"""Leakage, temporal alignment and end-to-end tests for the analysis pilots."""

from argparse import Namespace
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch

from baseline.mstcn.__main__ import main
from baseline.mstcn.analyses.common import make_view, exercise_predictions, waist_on_grid
from baseline.mstcn.analyses.leave_one_exercise_out.study import select, summarize
from baseline.mstcn.augmentation import augment
from baseline.mstcn.constants import CLASSES, DT_MS, FEATURES
from baseline.mstcn.data import load_arrays, prepare, subject_folds
from baseline.mstcn.decode import decode_sequence
from baseline.mstcn.demo import synthetic_source
from baseline.mstcn.model import DenseMSTCN, ModelConfig
from baseline.mstcn.orientation import gravity_features, rep_values, rotate_window
from baseline.mstcn.runner import Windows, prediction_mask, probabilities, load_checkpoint


class LookaheadTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(7)
        self.model = DenseMSTCN(ModelConfig(stages=2, layers=3, channels=4, dropout=0,
                                          lookahead_samples=13)).eval()

    def test_future_limit_and_acquisition_clock_alignment(self):
        x = torch.randn(1, 9, 100)
        changed = x.clone()
        changed[:, :, 53:] += 100
        for a, b in zip(self.model(x)[-1], self.model(changed)[-1]):
            torch.testing.assert_close(a[:, :, :40], b[:, :, :40])
        core = DenseMSTCN(replace(self.model.config, lookahead_samples=0)).eval()
        core.load_state_dict(self.model.state_dict())
        for delayed, causal in zip(self.model(x)[-1], core(x)[-1]):
            torch.testing.assert_close(delayed[:, :, :-13], causal[:, :, 13:])
        state, p, q = None, [], []
        for left in range(0, 100, 7):
            a, b, state = self.model.stream(x[:, :, left:left+7], state)
            p.append(a)
            q.append(b)
        for expected, chunks in zip(self.model(x)[-1], (p, q)):
            torch.testing.assert_close(expected[:, :, :-13], torch.cat(chunks, -1)[:, :, 13:])

    def test_chunk_invariance_and_no_future_across_gap(self):
        x = np.random.default_rng(2).normal(size=(100, 9)).astype(np.float32)
        valid = np.ones(100, bool)
        valid[45:50] = False
        norm = {"mean": [0] * 9, "std": [1] * 9}
        expected = probabilities(self.model, x, valid, norm, chunk=100)
        for chunk in (1, 7, 23):
            for a, b in zip(expected, probabilities(self.model, x, valid, norm, chunk=chunk)):
                np.testing.assert_allclose(a, b, atol=1e-6)
        inferred = prediction_mask(valid, 13)
        self.assertFalse(inferred[32:50].any())
        self.assertFalse(inferred[-13:].any())
        changed = x.copy()
        changed[50:] += 100
        for a, b in zip(expected, probabilities(self.model, changed, valid, norm)):
            np.testing.assert_array_equal(a[:45], b[:45])

    def test_report_time_includes_lookahead_and_decoder_lag(self):
        t = np.arange(80) * DT_MS
        p = np.zeros((80, len(CLASSES)))
        p[:, CLASSES.index("squat")] = 1
        phases = np.zeros(80, int)
        phases[10:30], phases[30:50] = 1, 2
        q = np.eye(3)[phases]
        valid = np.ones(80, bool)
        base = decode_sequence(t, p, q, valid)["repetitions"][0]
        delayed = decode_sequence(t, p, q, valid, lookahead_ms=520)["repetitions"][0]
        phase_only = decode_sequence(t, None, q, valid)
        self.assertEqual(phase_only["counts"], {"unknown": 1})
        self.assertEqual(phase_only["holds"], [])
        self.assertEqual(phase_only["repetitions"][0]["start_ms"], base["start_ms"])
        self.assertEqual(phase_only["repetitions"][0]["end_ms"], base["end_ms"])
        self.assertEqual(delayed["start_ms"], base["start_ms"])
        self.assertEqual(delayed["end_ms"], base["end_ms"])
        self.assertEqual(delayed["reported_at_ms"] - base["reported_at_ms"], 520)
        self.assertEqual(delayed["reported_at_ms"] - delayed["end_ms"], 600)


class AugmentationTests(unittest.TestCase):
    def test_gravity_features_are_causal_rotation_invariant_and_reset_at_gaps(self):
        rng = np.random.default_rng(3)
        x = rng.normal(size=(180, 9)).astype(np.float32)
        valid = np.ones(len(x), bool)
        valid[91:96] = False
        rotated = rotate_window(x[:, :6], rng)
        features = rep_values(x, valid, "gravity")
        np.testing.assert_allclose(rep_values(rotated, valid, "gravity"), features, atol=2e-6)
        head, state = gravity_features(x[:40])
        tail, _ = gravity_features(x[40:91], state)
        np.testing.assert_array_equal(np.concatenate([head, tail]), features[:91])
        changed = x.copy()
        changed[:96] += 200
        changed[:, 6:] -= 10
        np.testing.assert_array_equal(rep_values(changed, valid, "gravity")[96:], features[96:])
        static = np.tile([0., 0., 9.8, 2., 0., -3.], (50, 1))
        expected = np.tile([9.8, 0., 0., np.sqrt(13), -3., 2.], (50, 1))
        np.testing.assert_allclose(gravity_features(static)[0], expected, atol=1e-6)
        self.assertTrue(np.isfinite(gravity_features(np.zeros((3, 6)))[0]).all())

    def test_full_rotation_preserves_angles_and_does_not_favor_original_axis(self):
        rng = np.random.default_rng(42)
        window = np.tile([1., 0., 0., 0., 1., 0.], (10, 1))
        samples = []
        for _ in range(400):
            x = rotate_window(window, rng)
            np.testing.assert_array_equal(x[0], x[-1])
            basis = x[0].reshape(2, 3)
            np.testing.assert_allclose(basis @ basis.T, np.eye(2), atol=1e-6)
            samples.append(basis[0])
        np.testing.assert_allclose(np.mean(samples, axis=0), np.zeros(3), atol=.1)

    def test_warp_synchronizes_signals_labels_and_masks(self):
        x = np.arange(200, dtype=np.float32)[:, None]
        phase = np.repeat([0, 1, 2, 0], 50)
        targets = {"y": np.arange(200), "phase": phase, "phase_mask": phase > 0}
        a, transformed = augment(x, targets, ["clock"], np.random.default_rng(42))
        nearest = np.rint(a[:, 0]).astype(int)
        np.testing.assert_array_equal(transformed["y"], nearest)
        np.testing.assert_array_equal(transformed["phase"], phase[nearest])
        np.testing.assert_array_equal(transformed["phase_mask"], phase[nearest] > 0)
        self.assertTrue((np.diff(a[:, 0]) > 0).all())
        np.testing.assert_array_equal(a[[0, -1], 0], [0, 199])
        self.assertFalse(np.array_equal(a, x))

    def test_small_rotation_preserves_triad_geometry_and_seed(self):
        x = np.tile([1., 0, 0] * 4, (40, 1)).astype(np.float32)
        features = [f"{site}_{f}" for site in ("wrist", "waist") for f in FEATURES[:6]]
        a, _ = augment(x, {}, features, np.random.default_rng(9))
        b, _ = augment(x, {}, features, np.random.default_rng(9))
        np.testing.assert_array_equal(a, b)
        for start in (0, 6):
            acc, gyro = a[0, start:start+3], a[0, start+3:start+6]
            np.testing.assert_allclose(acc / np.linalg.norm(acc), gyro / np.linalg.norm(gyro), atol=1e-6)
            for v in (acc, gyro):
                self.assertTrue(.95 <= np.linalg.norm(v) <= 1.05)
                self.assertGreater(v[0] / np.linalg.norm(v), np.cos(np.deg2rad(10)))
        np.testing.assert_array_equal(x[0], [1., 0, 0] * 4)


class AnalysisTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        alignment, _ = synthetic_source(self.root / "source", subjects=3, both_wrists=True)
        for folder in (self.root / "source").iterdir():
            if folder.is_dir():
                frame = pd.read_csv(folder / "left_wrist.csv")
                frame["labels"] = "INVALID_WAIST_LABEL_MUST_NOT_BE_USED"
                frame.drop(index=range(110, 120)).to_csv(folder / "S1_waist.csv", index=False)
        self.data = self.root / "data"
        self.manifest = prepare(self.root / "source", self.data, alignment)
        self.split = subject_folds(self.manifest["sessions"], 3, 42)[0]

    def test_heldout_not_rest_and_excluded_from_statistics_windows_and_selection(self):
        dest = self.root / "heldout"
        m = make_view(self.data, dest, self.manifest, self.split, {}, held_out="squat")
        cls = CLASSES.index("squat")
        self.assertEqual(m["train_class_counts"][cls], 0)
        train_values = []
        for part, names in self.split.items():
            for name in names:
                original, a = load_arrays(self.data / name), load_arrays(dest / name)
                np.testing.assert_array_equal(original["y"], a["y"])
                removed = a["y"] == cls
                if part == "test":
                    np.testing.assert_array_equal(original["valid"], a["valid"])
                else:
                    self.assertFalse(a["valid"][removed].any())
                    self.assertFalse(a["coverage"][removed].any())
                    self.assertFalse(a["phase_mask"][removed].any())
                    refs = json.loads((dest / name / "references.json").read_text())
                    self.assertTrue(all(r["station"] != "squat" for r in refs))
                if part == "train":
                    train_values.append(original["x"][original["valid"] & ~removed])
        np.testing.assert_allclose(m["normalization"]["mean"], np.concatenate(train_values).astype(float).mean(0))
        windows = Windows(dest, self.split["train"], m["normalization"], 28, 64, "phase", lookahead=13,
                          augmentation=True, features=FEATURES)
        for x, mask, y, phase, valid, refs in windows:
            self.assertFalse((y[mask[0].bool()] == cls).any())
            self.assertFalse(valid[-13:].any())

    def test_study_pools_subjects_then_averages_exercises_and_applies_all_gates(self):
        def score(tp, missed):
            return {"by_tolerance": {"400": dict(tp=tp, extra=0, missed=missed)},
                    "guard_rails": dict(seen_f1_at_400=.7, crunch_f1_at_400=.2, certified_rest_hours=2,
                                        false_reps_during_rest=4, first_rep_recall=dict(matched=1, total=4),
                                        unseen_first_rep_recall=dict(matched=0, total=1))}
        reports = [dict(subject_fold=1, results={"squat": score(10, 0), "push-up": score(0, 5)}),
                   dict(subject_fold=2, results={"squat": score(0, 20), "push-up": score(0, 2)})]
        summary = summarize(reports, ["squat", "push-up"], [1, 2])
        self.assertEqual(summary["mean_unseen_f1_at_400"], .25)
        self.assertEqual(summary["first_rep_recall"], .25)
        self.assertEqual(summary["false_reps_per_rest_hour"], 2)
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            summarize(reports[:1], ["squat", "push-up"], [1, 2])
        candidate = {**summary, "mean_unseen_f1_at_400": .31, "mean_seen_f1_at_400": .69,
                     "false_reps_per_rest_hour": 2.3}
        self.assertEqual(select({"original": summary, "gravity": candidate}), "gravity")
        for k, v in (("mean_unseen_f1_at_400", .29), ("mean_seen_f1_at_400", .67), ("false_reps_per_rest_hour", 2.5)):
            self.assertEqual(select({"original": summary, "gravity": {**candidate, k: v}}), "original")

    def test_waist_pair_control_uses_identical_coverage_and_wrist_labels(self):
        waist, control = self.root / "waist", self.root / "control"
        wm = make_view(self.data, waist, self.manifest, self.split, {}, sensor_set="acc_gyro", waist=True)
        cm = make_view(self.data, control, self.manifest, self.split, {}, sensor_set="acc_gyro", paired_control=True)
        self.assertEqual(len(wm["features"]), 12)
        self.assertEqual(len(cm["features"]), 6)
        for name in self.split["test"]:
            wa, ca = load_arrays(waist / name), load_arrays(control / name)
            np.testing.assert_array_equal(wa["x"][:, :6], ca["x"])
            for key in ("valid", "y", "coverage", "phase", "phase_mask"):
                np.testing.assert_array_equal(wa[key], ca[key])
            self.assertFalse(wa["valid"][110:120].any())
            self.assertEqual((waist / name / "references.json").read_bytes(),
                             (control / name / "references.json").read_bytes())

    def test_paired_complete_inputs_checkpoint_decoder_and_raw_replay(self):
        # Make six named exercise bouts in each wrist stream, including the selector
        # and a seen crunch. Keep the original fixture for the legacy CLI tests.
        source = self.root / "source"
        for path in source.glob("*/label.csv"):
            labels = pd.read_csv(path)
            for col in ("left_wrist", "right_wrist"):
                for i, e in enumerate(["squat", "pull-up", "crunch", "push-up", "rowing", "dips"]):
                    a = 40 + i * 80
                    labels.loc[a:a+19, col] = e + "_phase1"
                    labels.loc[a+20:a+39, col] = e + "_phase2"
            labels.to_csv(path, index=False)
        data = self.root / "paired_data"
        prepare(source, data, self.root / "alignment.csv")
        for mode in ("rotated", "gravity"):
            out = self.root / mode
            main(["leave-one-exercise-out", "--data", str(data), "--output", str(out),
                  "--folds", "3", "--exercises", "squat", "--repetition-head", "complete",
                  "--rep-inputs", mode, "--epochs", "1", "--max-steps", "1", "--batch-size", "2",
                  "--window", "128", "--stages", "1", "--layers", "2", "--channels", "4", "--threads", "1"])
            fold = out / "squat"
            report = json.loads((out / "report.json").read_text())
            self.assertEqual(report["mean_unseen_f1_at_400"], report["results"]["squat"]["segmentation"]["f1"])
            self.assertIsNotNone(report["results"]["squat"]["guard_rails"]["crunch_f1_at_400"])
            decoder = json.loads((fold / "decoder.json").read_text())
            self.assertEqual(decoder["selection_exercise"], "pull-up")
            model, ckpt = load_checkpoint(fold / "model/best.pt", data=fold / "data")
            self.assertEqual(model.config.inputs, 6)
            self.assertEqual(ckpt["normalization"]["rep_inputs"], mode)
            n_validation_refs = 0
            for split, names in ckpt["split"].items():
                for name in names:
                    arrays = load_arrays(fold / "data" / name)
                    for e in ("squat", "pull-up"):
                        retained = (arrays["valid"] & (arrays["y"] == CLASSES.index(e))).any()
                        self.assertEqual(retained, split == "test" or (split == "validation" and e == "pull-up"))
                    refs = json.loads((fold / "data" / name / "references.json").read_text())
                    if split == "validation":
                        n_validation_refs += sum(r["station"] == "pull-up" for r in refs)
            history = json.loads((fold / "model/history.json").read_text())
            score = history[0]["validation_segmentation"]
            self.assertGreater(n_validation_refs, 0)
            self.assertEqual(score["tp"] + score["missed"], n_validation_refs)
            # Inference from the CSV and the evaluator's clip adapter use identical features.
            from baseline.mstcn.analyses.reference import WristModel
            path = next(source.glob("*/left_wrist.csv"))
            raw = pd.read_csv(path)
            raw.loc[40:, "sync_timestamp"] += DT_MS * 10
            raw.to_csv(self.root / "input.csv", index=False)
            main(["predict", "--checkpoint", str(fold / "model/best.pt"), "--decoder", str(fold / "decoder.json"),
                  "--input", str(self.root / "input.csv"), "--output", str(out / "csv"), "--chunk", "7"])
            frames = pd.read_csv(out / "csv/frames.csv")
            prediction = WristModel(fold, "cpu").predict(raw.sync_timestamp.to_numpy(),
                                                         raw[FEATURES].to_numpy(np.float32))
            np.testing.assert_allclose(frames.end_probability, prediction.traces["end"], atol=2e-6)

    def test_all_exercise_gravity_cross_validation_and_resume(self):
        out = self.root / "all_cv"
        command = ["cross-validate", "--data", str(self.data), "--output", str(out),
                   "--folds", "3", "--repetition-head", "complete", "--rep-inputs", "gravity",
                   "--epochs", "1", "--max-steps", "1", "--batch-size", "2", "--window", "128",
                   "--stages", "1", "--layers", "2", "--channels", "4", "--threads", "1"]
        main(command)
        main(command + ["--resume-completed"])
        for fold in out.glob("fold_*"):
            manifest = json.loads((fold / "data/manifest.json").read_text())
            self.assertTrue(all(manifest["train_class_counts"][CLASSES.index(e)] for e in ("squat", "push-up")))
            model, ckpt = load_checkpoint(fold / "model/best.pt", data=fold / "data")
            self.assertEqual(ckpt["normalization"], manifest["normalization"])
            self.assertEqual(model.config.inputs, 6)

    def test_exercise_score_ignores_station_and_counts_crossing_false_positive(self):
        refs = [dict(station="squat", start_ms=400, end_ms=800)]
        reps = [dict(station="unknown", start_ms=400, end_ms=800),
                dict(station="push-up", start_ms=200, end_ms=1100),
                dict(station="squat", start_ms=1400, end_ms=1700)]
        t = np.arange(50) * 40
        coverage = (t >= 400) & (t < 1000)
        self.assertEqual(exercise_predictions(reps, refs, t, coverage, 200), reps[:2])

    def test_multiple_waist_devices_selected_by_clock_coverage_only(self):
        row = self.manifest["sessions"][0]
        folder = self.root / "source" / row["source_session"]
        frame = pd.read_csv(folder / "left_wrist.csv")
        frame["labels"] = "not_an_exercise_label"
        frame.to_csv(folder / "S2_waist.csv", index=False)
        t = load_arrays(self.data / row["session"])["t"]
        x, available, info = waist_on_grid(self.root / "source", row, t)
        self.assertEqual(info["selected"], "S2_waist.csv")
        self.assertTrue(available.all())
        frame.to_csv(folder / "S1_waist.csv", index=False)
        _, _, info = waist_on_grid(self.root / "source", row, t)
        self.assertEqual(info["selected"], "S1_waist.csv")

    def test_both_cli_analyses_train_tune_score_and_fused_predict(self):
        common = ["--data", str(self.data), "--folds", "3", "--fold", "1", "--epochs", "1",
                  "--max-steps", "1", "--batch-size", "2", "--window", "128", "--stages", "1",
                  "--layers", "2", "--channels", "4", "--threads", "1"]
        main(["leave-one-exercise-out", *common, "--stages", "2", "--lookahead-ms", "80",
              "--exercises", "squat", "push-up",
              "--output", str(self.root / "loeo")])
        report = json.loads((self.root / "loeo/report.json").read_text())
        self.assertEqual(set(report["results"]), {"squat", "push-up"})
        self.assertEqual(report["exercise_fold_count"], 2)
        self.assertEqual(report["subject_fold"], 1)
        exercise = report["results"]["squat"]
        self.assertEqual(exercise["segmentation"]["status"], "scored")
        folder = self.root / "loeo/squat"
        seen = json.loads((folder / "seen_exercise_report.json").read_text())
        comparison = exercise["seen_exercise_comparison"]["400"]
        self.assertEqual(comparison["unseen_minus_seen_f1"],
                         exercise["segmentation"]["f1"] - seen["segmentation"]["f1"])
        self.assertEqual(exercise["subjects"], seen["subjects"])
        model, checkpoint = load_checkpoint(folder / "model/best.pt")
        self.assertEqual(checkpoint["classes"], [])
        self.assertTrue(all(stage.station is None for stage in model.stages))
        self.assertEqual(len(json.loads((folder / "decoder.json").read_text())["trials"]), 3)
        test_report = json.loads((folder / "test/report.json").read_text())
        self.assertEqual(test_report["segmentation"]["status"], "scored")
        self.assertNotIn("classification", test_report)
        source = self.root / "source/20260101_DEMO000"
        main(["predict", "--input", str(source / "left_wrist.csv"),
              "--checkpoint", str(folder / "model/best.pt"), "--decoder", str(folder / "decoder.json"),
              "--chunk", "7", "--threads", "1", "--output", str(self.root / "phase_prediction")])
        phase_frames = pd.read_csv(self.root / "phase_prediction/frames.csv")
        self.assertIn("phase1_probability", phase_frames)
        self.assertNotIn("predicted_station", phase_frames)
        main(["sensor-ablation", *common, "--output", str(self.root / "ablations")])
        report = json.loads((self.root / "ablations/report.json").read_text())
        self.assertEqual(len(report["results"]), 7)
        self.assertEqual(report["results"]["lookahead"]["model_lookahead_ms"], 520)
        folder = self.root / "ablations/combined_waist"
        model, checkpoint = load_checkpoint(folder / "model/best.pt")
        self.assertEqual(model.config.inputs, 12)
        self.assertTrue(checkpoint["augmentation"])
        main(["predict", "--input", str(source / "left_wrist.csv"),
              "--waist-input", str(source / "S1_waist.csv"), "--checkpoint", str(folder / "model/best.pt"),
              "--decoder", str(folder / "decoder.json"), "--chunk", "7", "--threads", "1",
              "--output", str(self.root / "prediction")])
        frames = pd.read_csv(self.root / "prediction/frames.csv")
        # One 13-sample tail for each of the two continuous joint-sensor runs.
        self.assertEqual(int((~frames.prediction_available).sum()), 26)
        summary = json.loads((self.root / "prediction/predictions.json").read_text())
        self.assertEqual(summary["model_lookahead_ms"], 520)


if __name__ == "__main__":
    unittest.main()
