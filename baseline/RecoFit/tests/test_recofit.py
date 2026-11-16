from collections import Counter
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from baseline.RecoFit.counter import CounterConfig, bandpass, count_peaks, decode
from baseline.RecoFit.features import extract, filtered_inputs, pc1, window_ends
from baseline.RecoFit.pipeline import (check_split, classify, cross_validate, evaluate, fit, fit_period_bounds,
                                      fit_boundary_offsets, infer, load_model, predict)
from baseline.RecoFit.demo import synthetic_source
from strength_training.temporal_segmentation.constants import CLASSES, DT_MS, FEATURES
from strength_training.temporal_segmentation.data import prepare, subject_folds


# Explicit boundary calibration for the positive-X sine fixtures.
SINE_OFFSETS = {"squat": {"0": {"start_periods": -.25, "end_periods": .75}}}


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
                expected = decode(t, x, clean, valid, periods, mode, boundary_offsets=SINE_OFFSETS)
                self.assertGreater(len(expected["repetitions"]), 8)
                self.assertEqual(decode(t, x, labels, valid, periods, mode, boundary_offsets=SINE_OFFSETS), expected)

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
        stations = [squat]*3+[other]*7
        labels = classify_trace([1]*5+[0]+[1]*4, stations)
        np.testing.assert_array_equal(labels[4::5], [0, 0]+[squat]*7+[other])

    def test_confirmed_rest_clears_vote_for_next_set(self):
        squat, other = CLASSES.index("squat"), CLASSES.index("push-up")
        labels = classify_trace([1]*15+[0]*3+[1]*5, [squat]*18+[other]*5)
        np.testing.assert_array_equal(labels[4::5], [0]*2+[squat]*15+[0]*3+[other]*3)

    def test_gap_resets_vote_and_requires_fresh_activation(self):
        squat, other = CLASSES.index("squat"), CLASSES.index("push-up")
        valid = np.ones(150, bool)
        valid[75:80] = False
        labels = classify_trace([1]*30, [squat]*16+[other]*14, valid=valid)
        self.assertTrue(np.all(labels[75:94] == 0))
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

    def test_constant_inputs_have_no_filter_startup_transient(self):
        x = np.tile([2., -3., np.sqrt(10.4**2-13), .1, -.2, .3], (500, 1))
        valid = np.ones(len(x), bool)
        filtered = bandpass(x, valid)
        np.testing.assert_allclose(filtered, np.tile(filtered[0], (len(x), 1)), atol=1e-10)
        self.assertEqual(count_peaks(filtered, 1.5, 2.5), [])
        np.testing.assert_allclose(filtered_inputs(x, valid), x, atol=1e-12)

    def test_constant_inputs_restart_without_transients_after_gaps(self):
        x = np.full((800, 6), np.nan)
        valid = np.zeros(len(x), bool)
        stretches = ((5, 205, [2., -3., 9.5, .1, -.2, .3]),
                     (210, 211, [-10.4, 0., 0., -.4, .5, -.6]),
                     (220, 800, [0., 10.4, 0., .7, -.8, .9]))
        for a, b, value in stretches:
            x[a:b] = value
            valid[a:b] = True
        for filter_inputs in (bandpass, filtered_inputs):
            with self.subTest(filter=filter_inputs.__name__):
                filtered = filter_inputs(x, valid)
                self.assertTrue(np.isfinite(filtered).all())
                self.assertFalse(filtered[~valid].any())
                for a, b, _ in stretches:
                    expected = filter_inputs(x[a:b], valid[a:b])
                    np.testing.assert_allclose(filtered[a:b], expected, atol=1e-10)
                    np.testing.assert_allclose(filtered[a:b],
                                               np.tile(expected[0], (b-a, 1)), atol=1e-10)
                    if filter_inputs is filtered_inputs:
                        np.testing.assert_allclose(filtered[a:b], x[a:b], atol=1e-12)

    def test_causal_filter_ignores_future_and_resets_at_masks(self):
        x = np.column_stack((sine_signal(), sine_signal(harmonic=.55)))
        valid = np.ones(len(x), bool)
        changed = x.copy()
        changed[300:] *= 30
        masked = valid.copy()
        masked[200:250] = False
        for filter_inputs in (bandpass, filtered_inputs):
            with self.subTest(filter=filter_inputs.__name__):
                a = filter_inputs(x, valid)
                np.testing.assert_allclose(a[:300], filter_inputs(changed, valid)[:300])
                reset = filter_inputs(x, masked)
                np.testing.assert_allclose(reset[250:], filter_inputs(x[250:], masked[250:]))
                self.assertFalse(reset[200:250].any())

    def test_counter_causal_events_match_prefix_and_never_cross_masks(self):
        x = sine_signal(seconds=30)
        t = np.arange(len(x))*DT_MS
        valid = np.ones(len(x), bool)
        labels = np.full(len(x), CLASSES.index("squat"))
        periods = {"squat": {"min_s": 1.5, "max_s": 2.5}}
        full = decode(t, x, labels, valid, periods, boundary_offsets=SINE_OFFSETS)
        prefix = decode(t[:400], x[:400], labels[:400], valid[:400], periods, boundary_offsets=SINE_OFFSETS)
        early = [r for r in full["repetitions"] if r["reported_at_ms"] <= t[399]]
        self.assertEqual(early, prefix["repetitions"])
        self.assertGreater(len(early), 3)
        self.assertTrue(all(r["end_ms"]+200 <= r["reported_at_ms"] for r in early))
        valid[300:330] = False
        cut = decode(t, x, labels, valid, periods, boundary_offsets=SINE_OFFSETS)
        self.assertTrue(all(r["end_ms"] <= t[300] or r["start_ms"] >= t[330] for r in cut["repetitions"]))

    def test_causal_pca_loading_crossover_preserves_every_repetition(self):
        length = 1500
        values = np.sin(np.arange(length)*2*np.pi/50)
        # Opposite-signed loadings trade dominance, flipping the stateless PCA.
        weight = np.linspace(.25, 2., length)
        x = np.column_stack(((2.25-weight)*values, -weight*values, values*0))
        self.assertGreater(pc1(x[:200], return_axis=True)[1][0], 0)
        self.assertGreater(pc1(x[-200:], return_axis=True)[1][1], 0)
        t, valid = np.arange(length)*DT_MS, np.ones(length, bool)
        labels = np.full(length, CLASSES.index("squat"))
        periods = {"squat": {"min_s": 1.5, "max_s": 2.5}}
        for buffer_seconds in (4., 30.):
            with self.subTest(buffer_seconds=buffer_seconds), patch("baseline.RecoFit.counter.bandpass",
                                                                    side_effect=lambda x, valid: x):
                config = CounterConfig(buffer_seconds=buffer_seconds)
                result = decode(t, x, labels, valid, periods, config=config, boundary_offsets=SINE_OFFSETS)
                clean = decode(t, sine_signal(seconds=60), labels, valid, periods,
                               config=config, boundary_offsets=SINE_OFFSETS)
                self.assertEqual(len(result["repetitions"]), len(clean["repetitions"]))
                np.testing.assert_allclose([r["peak_ms"] for r in result["repetitions"]],
                                           [r["peak_ms"] for r in clean["repetitions"]], atol=DT_MS, rtol=0)
                self.assertGreater(len(result["repetitions"]), 25)
                for end in (660, 799, 1100):
                    prefix = decode(t[:end], x[:end], labels[:end], valid[:end], periods,
                                    config=config, boundary_offsets=SINE_OFFSETS)
                    self.assertEqual([r for r in result["repetitions"] if r["reported_at_ms"] <= t[end-1]],
                                     prefix["repetitions"])

    def test_causal_pca_anchor_resets_at_rest_station_changes_and_gaps(self):
        length, split = 600, 300
        x = np.zeros((length, 3))
        x[:split, 0] = np.sin(np.arange(split)*2*np.pi/50)
        x[split:, 1] = np.sin(np.arange(length-split)*2*np.pi/50)
        t = np.arange(length)*DT_MS
        periods = {name: {"min_s": 1.5, "max_s": 2.5} for name in ("squat", "push-up")}
        offsets = {name: {str(axis): SINE_OFFSETS["squat"]["0"] for axis in (0, 1)}
                   for name in periods}
        for reset in ("rest", "station", "gap"):
            labels = np.full(length, CLASSES.index("squat"))
            valid = np.ones(length, bool)
            if reset == "rest":
                labels[split-5:split] = 0
            elif reset == "station":
                labels[split:] = CLASSES.index("push-up")
            else:
                valid[split-5:split] = False
            with self.subTest(reset=reset), patch("baseline.RecoFit.counter.bandpass",
                                                 side_effect=lambda x, valid: x):
                full = decode(t, x, labels, valid, periods, boundary_offsets=offsets)
                fresh = decode(t[split:], x[split:], labels[split:], valid[split:], periods,
                               boundary_offsets=offsets)
                self.assertGreater(len(fresh["repetitions"]), 3)
                self.assertEqual([r for r in full["repetitions"] if r["start_ms"] >= t[split]],
                                 fresh["repetitions"])

    def test_causal_holds_only_report_observed_closures_and_match_prefixes(self):
        length = 80
        t, x, labels = np.arange(length)*DT_MS, np.zeros((length, 3)), np.zeros(length, int)
        valid = np.ones(length, bool)
        for a, b in ((5, 25), (30, 60), (65, length)):
            labels[a:b] = CLASSES.index("plank")
        valid[40:43] = False
        full = decode(t, x, labels, valid, {})
        self.assertEqual([r["end_ms"] for r in full["holds"]], t[[25, 40, 60]].tolist())
        self.assertTrue(all(r["reported_at_ms"] == r["end_ms"] and r["complete"] for r in full["holds"]))
        for end in range(1, length+1):
            prefix = decode(t[:end], x[:end], labels[:end], valid[:end], {})
            self.assertEqual([r for r in full["holds"] if r["reported_at_ms"] <= t[end-1]], prefix["holds"])
        offline = decode(t, x, labels, valid, {}, mode="offline")
        self.assertEqual(len(offline["holds"]), 4)
        self.assertFalse(offline["holds"][-1]["complete"])
        self.assertTrue(all(r["reported_at_ms"] == t[-1] for r in offline["holds"]))

    def test_plank_is_hold_and_tail_does_not_become_rep(self):
        x = sine_signal(seconds=1)
        t = np.arange(len(x))*40
        valid = np.ones(len(t), bool)
        labels = np.full(len(t), CLASSES.index("plank"))
        result = decode(t, x, labels, valid, {})
        self.assertEqual(result["repetitions"], [])
        self.assertEqual(result["holds"], [])
        result = decode(t, x, labels, valid, {}, mode="offline")
        self.assertEqual(result["holds"][0]["duration_ms"], 1000)
        self.assertFalse(result["holds"][0]["complete"])
        labels[:] = CLASSES.index("squat")
        result = decode(t, x, labels, valid, {"squat": {"min_s": 1.5, "max_s": 2.5}},
                        boundary_offsets=SINE_OFFSETS)
        self.assertEqual(result["repetitions"], [])

    def test_offline_peaks_and_interval_counts_are_separate(self):
        x = sine_signal()
        result = decode(np.arange(len(x))*40, x, np.full(len(x), CLASSES.index("squat")),
                        np.ones(len(x), bool), {"squat": {"min_s": 1.5, "max_s": 2.5}}, mode="offline",
                        boundary_offsets=SINE_OFFSETS)
        self.assertGreaterEqual(len(result["offline_peaks"]), len(result["repetitions"]))
        self.assertTrue(all(r["reported_at_ms"] == (len(x)-1)*40 for r in result["repetitions"]))

    def test_uncalibrated_peak_counts_do_not_repeat_overlapping_window_sightings(self):
        x = sine_signal(seconds=60)
        t, valid = np.arange(len(x))*DT_MS, np.ones(len(x), bool)
        labels = np.full(len(x), CLASSES.index("squat"))
        bounds = {"squat": {"min_s": 1.5, "max_s": 2.5}}
        config = CounterConfig(buffer_seconds=4.)
        for mode, step in (("offline", 5), ("causal", 1), ("causal", 5), ("causal", 7)):
            with self.subTest(mode=mode, step=step):
                result = decode(t, x, labels, valid, bounds, mode, step, config)
                self.assertEqual(result["uncalibrated_peak_counts_by_station"], {"squat": 30})
                self.assertEqual(result["uncalibrated_stations"], ["squat"])
                self.assertEqual(result["repetitions"], [])
        calibrated = decode(t, x, labels, valid, bounds, config=config, boundary_offsets=SINE_OFFSETS)
        self.assertEqual(calibrated["uncalibrated_peak_counts_by_station"], {})
        self.assertEqual(calibrated["uncalibrated_stations"], [])


class CalibrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.sessions = []

    def stream(self, name, starts, duration, *, sign=1., axis=0, station="squat", length=1000,
               side="left_wrist"):
        folder = self.root/name
        folder.mkdir()
        t = np.arange(length)*DT_MS
        x = np.zeros((length, 9))
        x[:, axis] = sign*np.sin(np.arange(length)*2*np.pi/50)
        arrays = {"t": t, "x": x, "y": np.full(length, CLASSES.index(station)),
                  "valid": np.ones(length, bool), "coverage": np.ones(length, bool),
                  "boundary": np.zeros((length, 2)), "probe": np.zeros(length, bool)}
        for key, values in arrays.items():
            np.save(folder/f"{key}.npy", values)
        references = [{"station": station, "start_ms": start*DT_MS, "end_ms": (start+duration)*DT_MS}
                      for start in starts]
        (folder/"references.json").write_text(json.dumps(references))
        self.sessions.append({"session": name, "side": side})
        (self.root/"manifest.json").write_text(json.dumps({
            "version": 4, "features": FEATURES, "classes": CLASSES,
            "sample_period_ms": DT_MS, "sessions": self.sessions}))
        return arrays

    def test_periods_use_start_gaps_including_pauses_and_pool_gaps_for_singletons(self):
        self.stream("train", [50, 190, 330], 90)
        self.stream("singleton", [50], 10, station="push-up")
        bounds = fit_period_bounds(self.root, ["train", "singleton"])
        self.assertAlmostEqual(bounds["squat"]["min_s"], 5.6*.8)
        self.assertAlmostEqual(bounds["squat"]["max_s"], 5.6*1.25)
        self.assertEqual(bounds["squat"]["training_start_gaps"], 2)
        self.assertEqual(bounds["push-up"]["source"], "pooled_training_start_gaps")
        self.assertEqual(bounds["push-up"]["max_s"], bounds["squat"]["max_s"])
        fallback = fit_period_bounds(self.root, ["singleton"])
        self.assertEqual(fallback["push-up"]["source"], "fixed_fallback")
        self.assertEqual(fallback["push-up"]["min_s"], .4)
        self.assertEqual(fallback["push-up"]["max_s"], 10.)

    def test_period_pairs_never_cross_bouts_gaps_or_uncertified_samples(self):
        arrays = self.stream("train", [50, 100, 150, 200, 250, 300], 40)
        arrays["y"][110:120] = 0
        arrays["y"][160:170] = CLASSES.index("push-up")
        arrays["valid"][210:220] = False
        arrays["coverage"][260:270] = False
        for key in ("y", "valid", "coverage"):
            np.save(self.root/"train"/f"{key}.npy", arrays[key])
        result = fit_period_bounds(self.root, ["train"])["squat"]
        self.assertEqual(result["training_start_gaps"], 1)
        self.assertAlmostEqual(result["max_s"], 2.*1.25)

    def test_boundary_calibration_resolves_opposite_wrist_phases_and_axis_changes(self):
        starts = range(50, 900, 50)
        self.stream("left", starts, 50)
        self.stream("right", starts, 50, sign=-1., side="right_wrist")
        self.stream("left_y", starts, 50, axis=1)
        names = ["left", "right", "left_y"]
        bounds = fit_period_bounds(self.root, names)
        offsets = fit_boundary_offsets(self.root, names, bounds)
        left = offsets["left_wrist"]["squat"]["0"]
        right = offsets["right_wrist"]["squat"]["0"]
        self.assertAlmostEqual(abs(left["start_periods"]-right["start_periods"]), .5, delta=.06)
        # Different recording length/context from fitting; references are not passed to decode.
        t = np.arange(1250)*DT_MS
        labels, valid = np.full(len(t), CLASSES.index("squat")), np.ones(len(t), bool)
        for side, sign, axis in (("left_wrist", 1., 0), ("right_wrist", -1., 0), ("left_wrist", 1., 1)):
            x = np.zeros((len(t), 3))
            x[:, axis] = sign*np.sin(np.arange(len(t))*2*np.pi/50)
            for mode in ("causal", "offline"):
                with self.subTest(side=side, axis=axis, mode=mode):
                    result = decode(t, x, labels, valid, bounds, mode, boundary_offsets=offsets[side])
                    rows = [r for r in result["repetitions"] if 4000 <= r["start_ms"] < 46000]
                    self.assertGreater(len(rows), 15)
                    for r in rows:
                        expected = round(r["start_ms"]/2000)*2000
                        self.assertLessEqual(abs(r["start_ms"]-expected), 200)
                        self.assertLessEqual(abs(r["end_ms"]-(expected+2000)), 200)
                        self.assertEqual(r["boundary_method"], "training_calibrated_peak_offsets")

    def test_boundary_calibration_learns_active_duration_separately_from_period(self):
        self.stream("train", range(50, 900, 50), 35)
        bounds = fit_period_bounds(self.root, ["train"])
        offsets = fit_boundary_offsets(self.root, ["train"], bounds)["left_wrist"]["squat"]["0"]
        self.assertAlmostEqual(offsets["end_periods"]-offsets["start_periods"], .7, delta=.03)

    def test_boundary_calibration_matches_inference_pca_history_and_cadence(self):
        arrays = self.stream("train", range(50, 950, 50), 35, length=1003)
        values = arrays["x"][:, 0].copy()
        weight = np.minimum(np.arange(len(values))/250, 1.)
        # Startup is +X; a full 30-second buffer chooses +Y and the opposite phase.
        arrays["x"][:, 0] = (2.-1.75*weight)*values
        arrays["x"][:, 1] = -(.25+1.75*weight)*values
        np.save(self.root/"train"/"x.npy", arrays["x"])
        bounds = {"squat": {"min_s": 1.5, "max_s": 2.5}}
        filtered = bandpass(arrays["x"], arrays["valid"])
        self.assertEqual(np.argmax(np.abs(pc1(filtered[:50], return_axis=True)[1])), 0)
        self.assertEqual(np.argmax(np.abs(pc1(filtered[:750], return_axis=True)[1])), 1)

        for step, buffer_seconds in ((5, 30.), (7, 4.)):
            config = CounterConfig(buffer_seconds=buffer_seconds)
            history = []

            def trace(samples, minimum, maximum, config, *, state=None):
                rows = count_peaks(samples, minimum, maximum, config, state=state)
                axis = None if state.axis is None else tuple(state.axis)
                history.append((len(samples), tuple(samples[0]), state.polarity_axis, axis, rows))
                return rows

            with self.subTest(step=step, buffer_seconds=buffer_seconds), patch(
                    "baseline.RecoFit.counter.count_peaks", side_effect=trace):
                offsets = fit_boundary_offsets(self.root, ["train"], bounds, config, step=step)
                calibrated = history.copy()
                history.clear()
                result = decode(arrays["t"], arrays["x"], arrays["y"], arrays["valid"], bounds,
                                step=step, config=config, boundary_offsets=offsets["left_wrist"])
                self.assertEqual(calibrated, history)
                self.assertEqual(len(history), len(values)//step)
                self.assertEqual(history[0][0], step)
                self.assertEqual(set(offsets["left_wrist"]["squat"]), {"0"})
                self.assertLess(history[-1][3][1], 0)
                self.assertGreater(len(result["repetitions"]), 12)
                for row in result["repetitions"]:
                    expected = round(row["start_ms"]/2000)*2000
                    self.assertLessEqual(abs(row["start_ms"]-expected), 200)
                    self.assertLessEqual(abs(row["end_ms"]-(expected+1400)), 200)

    def test_calibration_replay_resets_at_rest_and_gaps_without_extra_tail_calls(self):
        arrays = self.stream("train", range(50, 950, 50), 35, length=1003)
        arrays["y"][303:311] = 0
        arrays["valid"][703:711] = False
        for key in ("y", "valid"):
            np.save(self.root/"train"/f"{key}.npy", arrays[key])
        bounds = {"squat": {"min_s": 1.5, "max_s": 2.5}}
        histories = []

        def trace(samples, minimum, maximum, config, *, state=None):
            histories.append((state, len(samples), tuple(samples[0])))
            return count_peaks(samples, minimum, maximum, config, state=state)

        with patch("baseline.RecoFit.counter.count_peaks", side_effect=trace):
            fit_boundary_offsets(self.root, ["train"], bounds, step=7)
            calibrated = histories.copy()
            histories.clear()
            decode(arrays["t"], arrays["x"], arrays["y"], arrays["valid"], bounds, step=7)
        self.assertEqual([row[1:] for row in calibrated], [row[1:] for row in histories])
        for rows in (calibrated, histories):
            states = {id(row[0]) for row in rows}
            self.assertEqual(len(states), 3)
            self.assertEqual(sum(row[1] == 7 for row in rows), 3)
            self.assertEqual(len(rows), 303//7+(703-311)//7+(1003-711)//7)

    def test_uncalibrated_polarity_has_no_symmetric_interval_fallback(self):
        x = sine_signal()
        t, valid = np.arange(len(x))*DT_MS, np.ones(len(x), bool)
        labels = np.full(len(x), CLASSES.index("squat"))
        bounds = {"squat": {"min_s": 1.5, "max_s": 2.5}}
        for mode in ("causal", "offline"):
            for offsets in ({}, {"squat": {"1": SINE_OFFSETS["squat"]["0"]}}):
                with self.subTest(mode=mode, offsets=offsets):
                    result = decode(t, x, labels, valid, bounds, mode, boundary_offsets=offsets)
                    self.assertEqual(result["repetitions"], [])
                    self.assertEqual(result["uncalibrated_stations"], ["squat"])
                    self.assertEqual(result["uncalibrated_peak_counts_by_station"], {"squat": 12})
                    if mode == "offline":
                        self.assertEqual(len(result["offline_peaks"]), 12)


class IntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        alignment = synthetic_source(cls.root/"source", subjects=6)
        cls.manifest = prepare(cls.root/"source", cls.root/"data", alignment)
        cls.path = fit(cls.root/"data", cls.root/"model", window_seconds=1., train_step_ms=200,
                       prediction_step_ms=280, max_windows_per_class=64, c_values=(.1, 1.))
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
        offsets = fit_boundary_offsets(self.root/"data", self.model["split"]["train"], original)
        name = self.model["split"]["test"][0]
        file = self.root/"data"/name/"references.json"
        before = file.read_text()
        try:
            references = json.loads(before)
            references[0]["end_ms"] += 100000000
            file.write_text(json.dumps(references))
            self.assertEqual(original, fit_period_bounds(self.root/"data", self.model["split"]["train"]))
            self.assertEqual(offsets, fit_boundary_offsets(self.root/"data", self.model["split"]["train"], original))
        finally:
            file.write_text(before)

    def test_fitted_offsets_follow_the_model_prediction_step(self):
        self.assertEqual(self.model["prediction_step"], 7)
        offsets = fit_boundary_offsets(self.root/"data", self.model["split"]["train"],
                                       self.model["period_bounds"], step=self.model["prediction_step"])
        self.assertEqual(self.model["boundary_offsets"], offsets)
        training = json.loads((self.path.parent/"training.json").read_text())
        self.assertEqual(training["prediction_step_ms"], 280)
        self.assertEqual(training["boundary_offsets"], offsets)

    def test_old_models_require_retraining_and_wrist_identity_is_explicit(self):
        import joblib
        legacy = {**self.model, "format": "recofit_wrist_v1"}
        path = self.root/"legacy.joblib"
        joblib.dump(legacy, path)
        with self.assertRaisesRegex(ValueError, "Retrain"):
            load_model(path)
        with self.assertRaisesRegex(ValueError, "Specify side"):
            infer(self.model, np.arange(10)*DT_MS, np.zeros((10, 9)), np.ones(10, bool))

    def test_inference_is_causal_and_invariant_to_unused_magnetometer(self):
        path = self.root/"data"/self.model["split"]["test"][0]
        t, x, valid = [np.load(path/f"{key}.npy") for key in ("t", "x", "valid")]
        labels, full = infer(self.model, t, x, valid, side="left_wrist")
        prefix_labels, prefix = infer(self.model, t[:500], x[:500], valid[:500], side="left_wrist")
        np.testing.assert_array_equal(labels[:500], prefix_labels)
        self.assertEqual([r for r in full["repetitions"] if r["reported_at_ms"] <= t[499]],
                          prefix["repetitions"])
        x[:, 6:] = 999999
        changed, result = infer(self.model, t, x, valid, side="left_wrist")
        np.testing.assert_array_equal(labels, changed)
        self.assertEqual(full["repetitions"], result["repetitions"])

    def test_raw_prediction_ignores_embedded_labels_and_handles_gaps(self):
        source = next((self.root/"source").glob("*/left_wrist.csv"))
        frame = pd.read_csv(source)
        frame["labels"] = "not_a_real_label"
        frame = frame.drop(index=range(250, 260))
        sensor = self.root/"unlabelled.csv"
        frame.to_csv(sensor, index=False)
        result = predict(self.path, sensor, self.root/"prediction", side="left_wrist")
        self.assertEqual(result["reference_scoring"], "unscored_sensor_only_prediction")
        frames = pd.read_csv(self.root/"prediction"/"frames.csv")
        self.assertEqual(int((frames.available == 0).sum()), 10)
        with self.assertRaises(FileExistsError):
            predict(self.path, sensor, self.root/"prediction", side="left_wrist")

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

    def test_uncalibrated_counts_reach_reports_even_without_reference_coverage(self):
        names = self.model["split"]["test"]
        labels = [np.load(self.root/"data"/name/"y.npy") for name in names]
        coverage_path = self.root/"data"/names[0]/"coverage.npy"
        coverage = np.load(coverage_path)
        np.save(coverage_path, np.zeros_like(coverage))
        try:
            for mode in ("causal", "offline"):
                output = self.root/f"uncalibrated_{mode}"
                # Exercise reporting with known stations and no fitted offsets.
                with self.subTest(mode=mode), patch("baseline.RecoFit.pipeline.classify", side_effect=labels):
                    report, records = evaluate(self.root/"data", self.path, output, mode=mode,
                                               model={**self.model, "boundary_offsets": {}})
                    counts = Counter()
                    for record in records:
                        self.assertEqual(record["uncalibrated_stations"], ["push-up", "squat"])
                        self.assertTrue(all(n > 0 for n in record["uncalibrated_peak_counts_by_station"].values()))
                        counts.update(record["uncalibrated_peak_counts_by_station"])
                    self.assertEqual(report["uncalibrated_stations"], ["push-up", "squat"])
                    self.assertEqual(report["uncalibrated_peak_counts_by_station"], dict(counts))
                    self.assertEqual(report["uncalibrated_streams_by_station"], {"push-up": 2, "squat": 2})
                    affected = report["uncalibrated_streams"]
                    self.assertEqual({r["side"] for r in affected}, {"left_wrist", "right_wrist"})
                    self.assertEqual({r["session"] for r in affected}, set(names))
                    saved = json.loads((output/"report.json").read_text())
                    for key in ("uncalibrated_peak_counts_by_station", "uncalibrated_streams"):
                        self.assertEqual(saved[key], report[key])
                    self.assertEqual(records[0]["repetition_scores"], {})
        finally:
            np.save(coverage_path, coverage)

    def test_cross_validation_pools_uncalibrated_counts_and_affected_streams(self):
        labels = np.load(self.root/"data"/self.model["split"]["test"][0]/"y.npy")
        output = self.root/"uncalibrated_cv"
        # All synthetic sessions share station timing; force missing calibration.
        with patch("baseline.RecoFit.pipeline.classify", return_value=labels), patch(
                "baseline.RecoFit.pipeline.fit_boundary_offsets", return_value={}):
            report = cross_validate(self.root/"data", output, folds=3, window_seconds=1.,
                                    train_step_ms=200, max_windows_per_class=32, c_values=(.1,))
        counts = Counter()
        for fold in range(1, 4):
            saved = json.loads((output/f"fold_{fold}"/"test"/"report.json").read_text())
            counts.update(saved["uncalibrated_peak_counts_by_station"])
        self.assertEqual(report["uncalibrated_peak_counts_by_station"], dict(counts))
        self.assertTrue(all(n > 0 for n in counts.values()))
        self.assertEqual(report["uncalibrated_streams_by_station"], {"push-up": 12, "squat": 12})
        affected = report["uncalibrated_streams"]
        self.assertEqual(len({r["session"] for r in affected}), 12)
        self.assertEqual({r["side"] for r in affected}, {"left_wrist", "right_wrist"})
        saved = json.loads((output/"report.json").read_text())
        self.assertEqual(saved["uncalibrated_streams"], affected)


if __name__ == "__main__":
    unittest.main()
