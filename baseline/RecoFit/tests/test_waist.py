"""Waist fusion, clock integrity, causal counting and held-out integration."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from baseline.RecoFit.demo import synthetic_source
from baseline.RecoFit.features import feature_batches, filtered_inputs
from baseline.RecoFit.pipeline import evaluate, fit, fit_boundary_offsets, infer, load_model, predict
from baseline.RecoFit.sensors import aligned_waist, load_inputs, prepare_waist, select_waist
from baseline.mstcn.constants import CLASSES, DT_MS, FEATURES
from baseline.mstcn.data import prepare


class AlignmentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.t = np.arange(20)*DT_MS

    def sensor(self, name, clocks):
        frame = pd.DataFrame(np.tile(np.arange(6), (len(clocks), 1)), columns=FEATURES[:6])
        frame.insert(0, "sync_timestamp", clocks)
        frame["labels"] = "invalid_annotation_never_read"
        path = self.root/name
        frame.to_csv(path, index=False)
        return path

    def test_alignment_preserves_gaps_and_ignores_outside_samples_and_labels(self):
        clocks = np.r_[-40, self.t[:7], self.t[10:], 800]
        path = self.sensor("waist.csv", clocks)
        x, available = aligned_waist(path, self.t)
        np.testing.assert_array_equal(np.flatnonzero(~available), [7, 8, 9])
        self.assertFalse(x[7:10].any())
        np.testing.assert_array_equal(x[0], np.arange(6))

    def test_rejects_shifted_clock_and_missing_waist(self):
        path = self.sensor("waist.csv", self.t+1)
        with self.assertRaisesRegex(ValueError, "exactly match"):
            aligned_waist(path, self.t)
        with self.assertRaisesRegex(ValueError, "no waist"):
            select_waist(self.root/"absent", self.t)

    def test_selects_one_device_by_clock_overlap_not_union_or_labels(self):
        self.sensor("S1_waist.csv", self.t[:10])
        self.sensor("S2_waist.csv", self.t[5:])
        self.sensor("S3_waist.csv", self.t[:15])
        _, available, info = select_waist(self.root, self.t)
        self.assertEqual(info["selected"], "S2_waist.csv")
        self.assertEqual(int(available.sum()), 15)
        self.assertFalse(available[:5].any())

    def test_fused_features_keep_both_devices_separate_and_causal(self):
        rng = np.random.default_rng(4)
        wrist, waist = rng.normal(size=(100, 6)), rng.normal(size=(100, 6))
        valid = np.ones(100, bool)
        wf, af = filtered_inputs(wrist, valid), filtered_inputs(waist, valid)
        ends = np.array([49, 59])
        _, seg, rec = next(feature_batches(wf, ends, 50, waist_filtered=af))
        _, ws, wr = next(feature_batches(wf, ends, 50))
        _, ass, ar = next(feature_batches(af, ends, 50))
        np.testing.assert_allclose(seg, np.c_[ws, ass])
        np.testing.assert_allclose(rec, np.c_[wr, ar])
        waist[60:] = 999
        _, changed, _ = next(feature_batches(wf, ends, 50, waist_filtered=filtered_inputs(waist, valid)))
        np.testing.assert_array_equal(seg, changed)


class WaistIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        alignment = synthetic_source(cls.root/"source", include_waist=True)
        cls.original = prepare(cls.root/"source", cls.root/"wrist", alignment)
        cls.manifest = prepare_waist(cls.root/"wrist", cls.root/"paired")
        cls.path = fit(cls.root/"paired", cls.root/"model", sensors="wrist-waist",
                       window_seconds=1., train_step_ms=200, max_windows_per_class=96, c_values=(.1, 1.))
        cls.model = load_model(cls.path)
        cls.name = cls.model["split"]["test"][0]
        cls.arrays = load_inputs(cls.root/"paired"/cls.name, "wrist-waist")
        cls.side = next(r["side"] for r in cls.manifest["sessions"] if r["session"] == cls.name)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def run_inference(self, *, length=None, waist_valid=None, model=None, mode="causal"):
        a = self.arrays
        return infer(model or self.model, a["t"][:length], a["x"][:length], a["valid"][:length], mode,
                     side=self.side, waist_x=a["waist_x"][:length],
                     waist_valid=(a["waist_available"] if waist_valid is None else waist_valid)[:length])

    def test_preparation_preserves_targets_masks_references_and_subject_splits(self):
        self.assertEqual(self.manifest["split"], self.original["split"])
        for key in ("x", "y", "valid", "coverage", "boundary", "t"):
            np.testing.assert_array_equal(np.load(self.root/"wrist"/self.name/f"{key}.npy"),
                                          np.load(self.root/"paired"/self.name/f"{key}.npy"))
        self.assertEqual((self.root/"wrist"/self.name/"references.json").read_bytes(),
                         (self.root/"paired"/self.name/"references.json").read_bytes())
        self.assertFalse((self.root/"wrist"/self.name/"waist_x.npy").exists())
        with self.assertRaises(FileExistsError):
            prepare_waist(self.root/"wrist", self.root/"paired")

    def test_learns_leg_raise_with_stationary_wrists_and_counts_using_waist(self):
        self.assertEqual(self.model["format"], "recofit_wrist_waist_v3")
        self.assertEqual(self.model["binary"].n_features_in_, 448)
        self.assertEqual(self.model["recognition"].n_features_in_, 120)
        self.assertEqual(self.model["waist_stations"], ["leg_raise"])
        self.assertFalse(self.arrays["x"][1300:1800, :6].any())
        for mode in ("causal", "offline"):
            with self.subTest(mode=mode):
                labels, result = self.run_inference(mode=mode)
                self.assertGreater(np.mean(labels[1375:1775] == CLASSES.index("leg_raise")), .9)
                self.assertGreaterEqual(result["counts_by_station"].get("leg_raise", 0), 6)
                reps = [r for r in result["repetitions"] if r["station"] == "leg_raise"]
                self.assertTrue(all(r["counting_sensor"] == "waist" for r in reps))
                wrist_model = {**self.model, "waist_stations": [],
                               "counting_sensors": {name: "wrist" for name in self.model["period_bounds"]}}
                _, wrist = self.run_inference(mode=mode, model=wrist_model)
                self.assertEqual(wrist["counts_by_station"].get("leg_raise", 0), 0)

    def test_waist_calibration_replays_selected_sensor_and_ignores_heldout_targets(self):
        kwargs = dict(sensors="wrist-waist", waist_stations=("leg_raise",), step=self.model["prediction_step"])
        args = (self.root/"paired", self.model["split"]["train"], self.model["period_bounds"])
        offsets = fit_boundary_offsets(*args, **kwargs)
        self.assertEqual(offsets, self.model["boundary_offsets"])
        reference = self.root/"paired"/self.name/"references.json"
        before = reference.read_text()
        try:
            reference.write_text("[]")
            self.assertEqual(offsets, fit_boundary_offsets(*args, **kwargs))
        finally:
            reference.write_text(before)

    def test_causal_prefix_and_waist_gap_reset(self):
        labels, full = self.run_inference()
        prefix_labels, prefix = self.run_inference(length=1623)
        np.testing.assert_array_equal(labels[:1623], prefix_labels)
        self.assertEqual([r for r in full["repetitions"] if r["reported_at_ms"] <= self.arrays["t"][1622]],
                         prefix["repetitions"])
        valid = self.arrays["waist_available"].copy()
        valid[1500:1510] = False
        labels, gaps = self.run_inference(waist_valid=valid)
        self.assertFalse(labels[1500:1534].any())
        self.assertEqual(gaps["waist_unavailable_samples"], 10)
        t = self.arrays["t"]
        self.assertFalse(any(r["start_ms"] < t[1510] and r["end_ms"] > t[1500]
                             for r in gaps["repetitions"]))

    def test_missing_waist_fails_explicitly(self):
        a = self.arrays
        with self.assertRaisesRegex(ValueError, "requires aligned waist"):
            infer(self.model, a["t"], a["x"], a["valid"], side=self.side)

    def test_raw_prediction_and_evaluation_preserve_dropout_penalties(self):
        source_session = self.name.split("__")[0]
        source = self.root/"source"/source_session
        frame = pd.read_csv(source/"S1_waist.csv").drop(index=range(1450, 1460))
        waist_path = self.root/"gapped_waist.csv"
        frame.to_csv(waist_path, index=False)
        result = predict(self.path, source/f"{self.side}.csv", self.root/"prediction", waist_input=waist_path)
        self.assertEqual(result["waist_unavailable_samples"], 10)
        frames = pd.read_csv(self.root/"prediction"/"frames.csv")
        self.assertEqual(int((frames.available == 0).sum()), 10)
        path = self.root/"paired"/self.name/"waist_available.npy"
        before = np.load(path).copy()
        try:
            # Keep reference coverage intact even when all waist observations fail.
            np.save(path, np.zeros_like(before))
            report, records = evaluate(self.root/"paired", self.path, self.root/"evaluation")
            record = next(r for r in records if r["session"] == self.name)
            self.assertEqual(np.sum(record["confusion_matrix"]), int(self.arrays["valid"].sum()))
            self.assertEqual(record["waist_unavailable_samples"], int(self.arrays["valid"].sum()))
            self.assertEqual(report["sensors"], "wrist-waist")
            self.assertEqual(report["counting_sensors"]["leg_raise"], "waist")
            saved = json.loads((self.root/"evaluation"/f"{self.name}.json").read_text())
            self.assertEqual(saved["repetitions"], [])
            self.assertGreater(report["repetitions"]["missed"], 0)
        finally:
            np.save(path, before)


if __name__ == "__main__":
    unittest.main()
