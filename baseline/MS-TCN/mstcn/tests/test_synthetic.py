import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from baseline.mstcn.__main__ import main
from baseline.mstcn.constants import CLASSES, DT_MS, START_HISTORY
from baseline.mstcn.data import prepare
from baseline.mstcn.demo import synthetic_source
from baseline.mstcn.runner import load_checkpoint
from baseline.mstcn.synthetic import SyntheticReps, make_cycle, step_rhythm


class SyntheticTests(unittest.TestCase):
    def test_reversal_warp_amplitude_and_step_band(self):
        rng = np.random.default_rng(1)
        raw = rng.normal(size=(30, 9)).astype(np.float32)
        cycle = make_cycle(raw, .85, 1.1)
        half = round(len(raw) / .85)
        self.assertEqual(len(cycle), 2 * half)
        np.testing.assert_allclose(cycle[half-1, :3], raw[0, :3] + 1.1 * (raw[-1, :3] - raw[0, :3]))
        np.testing.assert_allclose(cycle[half-1, 3:6], raw[-1, 3:6] * (.85 * 1.1))
        np.testing.assert_array_equal(cycle[:half, :3], cycle[:half-1:-1, :3])
        np.testing.assert_array_equal(cycle[:half, 3:6], -cycle[:half-1:-1, 3:6])
        np.testing.assert_array_equal(cycle[:half, 6:], cycle[:half-1:-1, 6:])
        for lag in (10, 13, 17):
            clock = np.arange(50) * 2 * np.pi / lag
            self.assertTrue(step_rhythm(np.stack([np.sin(clock), np.cos(clock), 9+np.sin(clock)], 1)))
        self.assertFalse(step_rhythm(np.ones((50, 3))))
        self.assertFalse(step_rhythm(rng.normal(size=(50, 3))))

    def test_source_isolation_labels_review_weights_and_determinism(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "train").mkdir()
            n = 1600
            rng = np.random.default_rng(7)
            raw = rng.normal(size=(n, 9)).astype(np.float32)
            y = np.zeros(n, np.int64)
            y[280:460] = CLASSES.index("pull-up")
            y[460:640] = CLASSES.index("squat")
            y[640:840] = CLASSES.index("rowing")
            y[840:1450] = CLASSES.index("crunch")
            # Entire rowing bout has a step rhythm.
            clock = np.arange(200) * 2*np.pi / 13
            raw[640:840, :3] = np.stack([np.sin(clock), np.cos(clock), np.sin(clock)], 1)
            np.save(root / "train/x.npy", raw)
            valid = np.ones(n, bool)
            valid[1020:1060] = False
            weights = np.ones(n, np.float32)
            weights[y == 0] = 3
            weights[1450:] = 0  # reviewed uncertain rest
            arrays = dict(y=y, valid=valid, coverage=valid.copy(), weight=weights)
            arrays["coverage"][1120:1150] = False
            windows = SimpleNamespace(arrays=[arrays], rep_inputs="original")
            manifest = {"analysis": {"held_out_exercise": "squat", "selection_exercise": "pull-up"},
                        "split": {"train": ["train"], "validation": ["absent_val"], "test": ["absent_test"]}}
            generator = SyntheticReps(root, manifest, windows, seed=3)
            self.assertGreater(generator.counts["step_rhythm_rejected"], 0)
            for stream, lo, hi in generator.pieces:
                self.assertEqual(stream, 0)
                self.assertTrue((y[lo:hi] == CLASSES.index("crunch")).all())
                self.assertTrue((valid & arrays["coverage"])[lo:hi].all())
                self.assertTrue(15 <= hi-lo <= 50)
                gyro = np.linalg.norm(raw[lo-1:lo+2, 3:6], axis=1)
                self.assertLessEqual(gyro[1], min(gyro[0], gyro[2]))
            self.assertEqual(generator.rest, [(0, 0, 280)])
            second = SyntheticReps(root, manifest, windows, seed=3)
            saw_pause = False
            for _ in range(10):
                a, refs = generator.sequence()
                other, other_refs = second.sequence()
                np.testing.assert_array_equal(a["x"], other["x"])
                self.assertEqual(refs, other_refs)
                self.assertTrue(3 <= len(refs) <= 15)
                before, last = refs[0]["start_ms"]//DT_MS, refs[-1]["end_ms"]//DT_MS
                self.assertTrue(125 <= before <= 750 and 125 <= len(a["t"]) - last <= 750)
                self.assertTrue((a["weight"][:before] == 3).all())
                self.assertTrue((a["weight"][last+3:] == 3).all())
                prior = before
                for rep in refs:
                    start, end = rep["start_ms"]//DT_MS, rep["end_ms"]//DT_MS
                    self.assertTrue(0 <= start-prior <= 25)
                    if start > prior:
                        saw_pause = True
                        np.testing.assert_array_equal(a["x"][prior:start, 3:6], 0)
                    self.assertEqual(a["end"][end], 1)
                    self.assertEqual(a["start_choice"][end], START_HISTORY-(end-start))
                    forward, reverse = np.split(a["x"][start:end], 2)
                    np.testing.assert_array_equal(forward[:, :3], reverse[::-1, :3])
                    np.testing.assert_array_equal(forward[:, 3:6], -reverse[::-1, 3:6])
                    np.testing.assert_array_equal(forward[:, 6:], reverse[::-1, 6:])
                    prior = end
            self.assertTrue(saw_pause)

    def test_paired_training_mixes_exact_quarters_and_scores_real_test(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            alignment, _ = synthetic_source(root / "source", subjects=3)
            for path in (root / "source").glob("*/label.csv"):
                frame = pd.read_csv(path)
                frame["left_wrist"] = "rest"
                for start, name in [(170, "squat"), (246, "pull-up"), (324, "crunch"), (412, "push-up")]:
                    frame.loc[start:start+31, "left_wrist"] = name + "_phase1"
                    frame.loc[start+32:start+63, "left_wrist"] = name + "_phase2"
                frame.to_csv(path, index=False)
            prepare(root / "source", root / "data", alignment)
            args = ["leave-one-exercise-out", "--data", str(root / "data"), "--repetition-head", "complete",
                    "--folds", "3", "--protocol", "paired", "--exercises", "squat", "--rep-inputs", "gravity",
                    "--causal-similarity", "--batch-size", "4", "--window", "64", "--stages", "2",
                    "--channels", "4", "--layers", "1", "--threads", "1", "--epochs", "1"]
            for mode in ("control", "synthetic"):
                main([*args, "--output", str(root / mode), *(["--synthetic-reps"] if mode == "synthetic" else [])])
            folder = root / "synthetic/squat/model"
            audit = json.loads((folder / "synthetic_sources.json").read_text())
            history, = json.loads((folder / "history.json").read_text())
            self.assertEqual((audit["real_per_batch"], audit["synthetic_per_batch"]), (3, 1))
            self.assertEqual(history["real_training_windows"], 3 * history["synthetic_training_windows"])
            control, = json.loads((root / "control/squat/model/history.json").read_text())
            self.assertEqual(control["train_batches"], history["train_batches"])
            self.assertEqual(control["validation_batches"], history["validation_batches"])
            _, ckpt = load_checkpoint(folder / "best.pt")
            self.assertEqual(audit["training_streams"], ckpt["split"]["train"])
            self.assertTrue(ckpt["synthetic_reps"])
            main(["compare-synthetic", "--control-reports", str(root / "control/report.json"),
                  "--synthetic-reports", str(root / "synthetic/report.json"), "--output", str(root / "decision.json")])
            decision = json.loads((root / "decision.json").read_text())
            self.assertIn("false_reps_per_rest_hour", decision["synthetic"])
            self.assertEqual(decision["exercises"], ["squat"])
