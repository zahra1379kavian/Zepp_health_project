import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from baseline.RecoFit.counter import bandpass, count_peaks, decode
from baseline.RecoFit.features import extract, pc1, window_ends
from baseline.RecoFit.pipeline import (check_split, classify, evaluate, fit, fit_period_bounds,
                                      infer, load_model, predict)
from baseline.RecoFit.demo import synthetic_source
from strength_training.temporal_segmentation.constants import CLASSES, DT_MS
from strength_training.temporal_segmentation.data import prepare, subject_folds


def sine_signal(seconds=24, period=2., harmonic=0.):
    time = np.arange(int(seconds*25))/25
    values = np.sin(2*np.pi*time/period)+harmonic*np.sin(4*np.pi*time/period)
    return np.column_stack((values, values*.1, np.zeros(len(values))))


def classify_trace(exercise, stations, *, valid=None, window=5, step=5,
                   confirm_steps=3, length=None):
    """Inject deterministic SVM decisions while retaining real window/gap handling."""
    exercise, stations = np.repeat(exercise, step), np.repeat(stations, step)
    length = len(exercise) if length is None else length
    valid = np.ones(length, bool) if valid is None else valid[:length]
    model = {"window_samples": window, "prediction_step": step, "confirm_steps": confirm_steps,
             "binary": Mock(), "recognition": Mock()}
    model["binary"].predict.side_effect = lambda rows: exercise[rows[:, 0]]
    model["recognition"].predict.side_effect = lambda rows: stations[rows[:, 0]]

    def endpoint_features(filtered, ends, window):
        # Encode endpoints as features so both classifiers replay the same trace.
        yield ends, ends[:, None], ends[:, None]

    with patch("baseline.RecoFit.pipeline.feature_batches", side_effect=endpoint_features):
        return classify(model, np.zeros((length, 9)), valid)


class ClassificationTests(unittest.TestCase):
    def test_activation_uses_majority_of_exercise_decisions(self):
        squat, other = CLASSES.index("squat"), CLASSES.index("push-up")
        exercise = [0]*10+[1]*10
        stations = [other]*10+[squat, squat, other]+[squat]*7
        labels = classify_trace(exercise, stations)
        np.testing.assert_array_equal(labels, classify_trace(exercise, [squat]*20))
        self.assertEqual(labels[64], squat)

    def test_brief_and_alternating_errors_preserve_repetition_buffer(self):
        squat, other = CLASSES.index("squat"), CLASSES.index("plank")
        stations = [squat]*150
        stations[20] = other
        stations[40:80] = [other, squat]*20
        stations[100:102] = [other]*2
        labels = classify_trace([1]*150, stations)
        clean = classify_trace([1]*150, [squat]*150)
        np.testing.assert_array_equal(labels, clean)
        x = sine_signal(seconds=30)
        t, valid = np.arange(len(x))*DT_MS, np.ones(len(x), bool)
        periods = {"squat": {"min_s": 1.5, "max_s": 2.5}}
        for mode in ("causal", "offline"):
            with self.subTest(mode=mode):
                expected = decode(t, x, clean, valid, periods, mode)
                self.assertGreater(len(expected["repetitions"]), 8)
                self.assertEqual(decode(t, x, labels, valid, periods, mode), expected)

    def test_vote_ties_keep_committed_station(self):
        squat, other = CLASSES.index("squat"), CLASSES.index("push-up")
        labels = classify_trace([1]*20, [squat, other]*10, confirm_steps=1)
        self.assertTrue(np.all(labels[4:] == squat))

    def test_new_majority_must_persist_and_tie_interrupts_confirmation(self):
        squat, other = CLASSES.index("squat"), CLASSES.index("push-up")
        stations = [squat]*3+[other]*4+[squat]+[other]*3
        labels = classify_trace([1]*len(stations), stations)
        np.testing.assert_array_equal(labels[4::5], [0, 0]+[squat]*8+[other])

    def test_rest_decision_interrupts_station_confirmation_without_voting(self):
        squat, other = CLASSES.index("squat"), CLASSES.index("push-up")
        stations = [squat]*3+[other]*4+[squat]+[other]*3
        labels = classify_trace([1]*7+[0]+[1]*3, stations)
        np.testing.assert_array_equal(labels[4::5], [0, 0]+[squat]*8+[other])

    def test_confirmed_rest_clears_vote_for_next_set(self):
        squat, other = CLASSES.index("squat"), CLASSES.index("push-up")
        labels = classify_trace([1]*15+[0]*3+[1]*5, [squat]*18+[other]*5)
        np.testing.assert_array_equal(labels[4::5], [0]*2+[squat]*15+[0]*3+[other]*3)

    def test_gap_resets_vote_and_requires_fresh_activation(self):
        squat, other = CLASSES.index("squat"), CLASSES.index("push-up")
        valid = np.ones(150, bool)
        valid[75:80] = False
        labels = classify_trace([1]*30, [squat]*16+[other]*14, valid=valid)
        self.assertTrue(np.all(labels[74:94] == 0))
        self.assertTrue(np.all(labels[94:] == other))

    def test_gap_resets_even_when_window_end_cadence_is_unchanged(self):
        squat, other = CLASSES.index("squat"), CLASSES.index("push-up")
        valid = np.ones(25, bool)
        valid[3:5] = False
        labels = classify_trace([1]*5, [squat]+[other]*4, valid=valid, window=2, confirm_steps=2)
        self.assertTrue(np.all(labels[:11] == 0))
        self.assertTrue(np.all(labels[11:] == other))

    def test_station_smoothing_never_revises_prefix(self):
        squat, other = CLASSES.index("squat"), CLASSES.index("push-up")
        stations = [squat]*3+[other]*4+[squat]+[other]*20
        exercise = [1]*len(stations)
        labels = classify_trace(exercise, stations)
        for length in (3, 14, 18, 38, 51, 58, 91):
            with self.subTest(length=length):
                np.testing.assert_array_equal(labels[:length], classify_trace(exercise, stations, length=length))


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
