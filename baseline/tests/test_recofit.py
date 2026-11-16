import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from baseline.counter import bandpass, count_peaks, decode
from baseline.features import extract, pc1, window_ends
from baseline.pipeline import (check_split, evaluate, fit, fit_period_bounds,
                               infer, load_model, predict)
from baseline.demo import synthetic_source
from strength_training.temporal_segmentation.constants import CLASSES, DT_MS
from strength_training.temporal_segmentation.data import prepare, subject_folds


def sine_signal(seconds=24, period=2., harmonic=0.):
    time = np.arange(int(seconds*25))/25
    values = np.sin(2*np.pi*time/period)+harmonic*np.sin(4*np.pi*time/period)
    return np.column_stack((values, values*.1, np.zeros(len(values))))


class SignalTests(unittest.TestCase):
    def test_feature_shapes_and_finite_constants(self):
        for x in (np.zeros((2, 125, 6)), np.random.default_rng(3).normal(size=(2, 125, 6))):
            seg, rec = extract(x)
            self.assertEqual(seg.shape, (2, 224))
            self.assertEqual(rec.shape, (2, 60))
            self.assertTrue(np.isfinite(seg).all() and np.isfinite(rec).all())

    def test_feature_batch_rows_are_independent(self):
        windows = np.random.default_rng(1).normal(size=(3, 125, 6))
        first = extract(windows)
        windows[1:] *= 100
        second = extract(windows)
        for a, b in zip(first, second):
            np.testing.assert_allclose(a[0], b[0])

    def test_pca_polarity_is_deterministic(self):
        x = sine_signal()
        np.testing.assert_allclose(pc1(x), pc1(x.copy()))
        self.assertGreater(np.corrcoef(pc1(x), x[:, 0])[0, 1], .99)

    def test_regular_and_double_peaked_cycles(self):
        for harmonic in (0., .55):
            peaks = count_peaks(sine_signal(harmonic=harmonic), 1.5, 2.5)
            self.assertEqual(len(peaks), 12)
            self.assertTrue(all(abs(p["period_samples"]-50) <= 1 for p in peaks[1:-1]))

    def test_flat_or_short_signal_produces_no_reps(self):
        self.assertEqual(count_peaks(np.zeros((200, 3)), .5, 4), [])
        self.assertEqual(count_peaks(np.zeros((5, 3)), .5, 4), [])

    def test_local_period_tracks_a_tempo_change(self):
        values = np.r_[np.sin(np.arange(250)*2*np.pi/50), np.sin(np.arange(250)*2*np.pi/25)]
        peaks = count_peaks(np.column_stack((values, values*.1, values*0)), .7, 2.5)
        self.assertTrue(all(p["period_samples"] >= 45 for p in peaks if p["index"] < 150))
        self.assertTrue(all(p["period_samples"] <= 28 for p in peaks if p["index"] > 350))

    def test_invalid_windows_and_periods(self):
        self.assertEqual(len(window_ends(np.ones(10, bool), 125, 5)), 0)
        self.assertEqual(len(window_ends(np.zeros(10, bool), 125, 5)), 0)
        with self.assertRaises(ValueError):
            count_peaks(sine_signal(), 2, 1)

    def test_causal_filter_ignores_future_and_resets_at_masks(self):
        x = sine_signal()
        valid = np.ones(len(x), bool)
        a = bandpass(x, valid)
        changed = x.copy()
        changed[300:] *= 30
        np.testing.assert_allclose(a[:300], bandpass(changed, valid)[:300])
        valid[200:250] = False
        reset = bandpass(x, valid)
        np.testing.assert_allclose(reset[250:], bandpass(x[250:], valid[250:]))
        self.assertFalse(reset[200:250].any())

    def test_counter_causal_events_match_prefix_and_never_cross_masks(self):
        x = sine_signal(seconds=30)
        t = np.arange(len(x))*DT_MS
        valid = np.ones(len(x), bool)
        labels = np.full(len(x), CLASSES.index("squat"))
        periods = {"squat": {"min_s": 1.5, "max_s": 2.5}}
        full = decode(t, x, labels, valid, periods)
        prefix = decode(t[:400], x[:400], labels[:400], valid[:400], periods)
        early = [r for r in full["repetitions"] if r["reported_at_ms"] <= t[399]]
        self.assertEqual(early, prefix["repetitions"])
        self.assertGreater(len(early), 3)
        self.assertTrue(all(r["end_ms"] <= r["reported_at_ms"] for r in early))
        valid[300:330] = False
        cut = decode(t, x, labels, valid, periods)
        self.assertTrue(all(r["end_ms"] <= t[300] or r["start_ms"] >= t[330] for r in cut["repetitions"]))

    def test_plank_is_hold_and_tail_does_not_become_rep(self):
        x = sine_signal(seconds=1)
        t = np.arange(len(x))*40
        valid = np.ones(len(t), bool)
        labels = np.full(len(t), CLASSES.index("plank"))
        result = decode(t, x, labels, valid, {})
        self.assertEqual(result["repetitions"], [])
        self.assertEqual(result["holds"][0]["duration_ms"], 1000)
        labels[:] = CLASSES.index("squat")
        result = decode(t, x, labels, valid, {"squat": {"min_s": 1.5, "max_s": 2.5}})
        self.assertEqual(result["repetitions"], [])

    def test_offline_peaks_and_interval_counts_are_separate(self):
        x = sine_signal()
        result = decode(np.arange(len(x))*40, x, np.full(len(x), CLASSES.index("squat")),
                        np.ones(len(x), bool), {"squat": {"min_s": 1.5, "max_s": 2.5}}, mode="offline")
        self.assertGreaterEqual(len(result["offline_peaks"]), len(result["repetitions"]))
        self.assertTrue(all(r["reported_at_ms"] == (len(x)-1)*40 for r in result["repetitions"]))


class IntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        alignment = synthetic_source(cls.root/"source", subjects=6)
        cls.manifest = prepare(cls.root/"source", cls.root/"data", alignment)
        cls.path = fit(cls.root/"data", cls.root/"model", window_seconds=1., train_step_ms=200,
                       max_windows_per_class=64, c_values=(.1, 1.))
        cls.model = load_model(cls.path)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_both_wrists_stay_in_subject_folds(self):
        for split in subject_folds(self.manifest["sessions"], 3):
            check_split(self.manifest, split)
        bad = {k: list(v) for k, v in self.manifest["split"].items()}
        bad["train"].append(bad["test"][0])
        with self.assertRaisesRegex(ValueError, "leakage"):
            check_split(self.manifest, bad)

    def test_period_bounds_never_use_heldout_references(self):
        original = fit_period_bounds(self.root/"data", self.model["split"]["train"])
        name = self.model["split"]["test"][0]
        file = self.root/"data"/name/"references.json"
        before = file.read_text()
        try:
            references = json.loads(before)
            references[0]["end_ms"] += 100000000
            file.write_text(json.dumps(references))
            self.assertEqual(original, fit_period_bounds(self.root/"data", self.model["split"]["train"]))
        finally:
            file.write_text(before)

    def test_inference_is_causal_and_invariant_to_unused_magnetometer(self):
        path = self.root/"data"/self.model["split"]["test"][0]
        t, x, valid = [np.load(path/f"{key}.npy") for key in ("t", "x", "valid")]
        labels, full = infer(self.model, t, x, valid)
        prefix_labels, prefix = infer(self.model, t[:500], x[:500], valid[:500])
        np.testing.assert_array_equal(labels[:500], prefix_labels)
        self.assertEqual([r for r in full["repetitions"] if r["reported_at_ms"] <= t[499]],
                          prefix["repetitions"])
        x[:, 6:] = 999999
        changed, result = infer(self.model, t, x, valid)
        np.testing.assert_array_equal(labels, changed)
        self.assertEqual(full["repetitions"], result["repetitions"])

    def test_raw_prediction_ignores_embedded_labels_and_handles_gaps(self):
        source = next((self.root/"source").glob("*/left_wrist.csv"))
        frame = pd.read_csv(source)
        frame["labels"] = "not_a_real_label"
        frame = frame.drop(index=range(250, 260))
        sensor = self.root/"unlabelled.csv"
        frame.to_csv(sensor, index=False)
        result = predict(self.path, sensor, self.root/"prediction")
        self.assertEqual(result["reference_scoring"], "unscored_sensor_only_prediction")
        frames = pd.read_csv(self.root/"prediction"/"frames.csv")
        self.assertEqual(int((frames.available == 0).sum()), 10)
        with self.assertRaises(FileExistsError):
            predict(self.path, sensor, self.root/"prediction")

    def test_evaluation_reports_reference_metrics_and_paired_wrists(self):
        report, records = evaluate(self.root/"data", self.path, self.root/"test")
        self.assertEqual(report["report_scope"], "single_split_diagnostic")
        self.assertEqual(report["streams"], 2)
        self.assertEqual(report["subjects"], 1)
        self.assertEqual(set(report["by_tolerance"]), {"200", "400"})
        self.assertEqual(report["repetitions"]["status"], "scored")
        self.assertEqual(set(report["per_side"]), {"left_wrist", "right_wrist"})
        self.assertEqual(len(records), 2)

    def test_offline_evaluation_has_separate_peak_count_scores(self):
        report, _ = evaluate(self.root/"data", self.path, self.root/"offline_test", mode="offline")
        self.assertEqual(report["offline_peak_counts"]["status"], "scored")
        self.assertEqual(len(report["offline_peak_counts"]["count_rows"]), 28)


if __name__ == "__main__":
    unittest.main()
