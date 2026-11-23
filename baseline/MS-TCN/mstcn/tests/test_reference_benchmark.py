import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from baseline.rep_benchmark import protocol
from baseline.rep_benchmark.report import summaries


class WristBenchmarkTests(unittest.TestCase):
    def test_clip_uses_wrist_labels_release_margin_and_sensor_frame_clock(self):
        with TemporaryDirectory() as tmp:
            folder = Path(tmp)
            t = np.arange(400) * 40
            sensors = {k: t.copy() for k in protocol.utils.REQUIRED_COLS}
            pd.DataFrame(dict(sync_timestamp=t, **sensors)).to_csv(folder / "left_wrist.csv", index=False)
            labels = np.full(400, "rest", dtype=object)
            labels[160:170] = "push-up_phase1"
            labels[170:174] = "push-up_transition"
            labels[174:180] = "push-up_phase2"
            pd.DataFrame(dict(sync_timestamp=t + 4, left_wrist=labels, right_wrist="squat_phase1")).to_csv(
                folder / "label.csv", index=False)
            [(side, act, clocks, x, y)] = list(protocol.clips(folder))
            self.assertEqual((side, act), ("left_wrist", "push-up"))
            np.testing.assert_array_equal(clocks, t[35:305])
            np.testing.assert_array_equal(x[:, 0], clocks)
            # A single shared frame suffices. An extra prediction in the transition
            # stays an activity FP and the extra in the margin is a rest FP.
            result = protocol.Prediction([dict(start=125, end=126, reported_at=128),
                                          dict(start=136, end=137), dict(start=20, end=22)])
            gt, _, metrics, delay = protocol.score(result, y)
            self.assertEqual((gt[0]["start"], gt[0]["end"]), (125, 145))
            self.assertEqual([metrics[k] for k in ("TP", "FN", "FP_rest", "FP_activity")], [1, 0, 1, 1])
            self.assertEqual(delay, [(128 - 145) * 40])

    def test_count_error_is_absolute_per_clip_and_rates_are_rep_weighted(self):
        rows = []
        for y, reps in [(np.array([1, 2, 1, 2, 3]), [dict(start=0, end=1)]),
                        (np.array([1, 2, 3, 3, 3]), [dict(start=0, end=2), dict(start=2, end=4)])]:
            _, _, values, _ = protocol.score(protocol.Prediction(reps), y)
            rows.append(dict(activity="push-up", **values))
        row = summaries(pd.DataFrame(rows)).iloc[0]
        self.assertEqual((row.TP, row.FN, row.FP, row.cnt_error), (2, 1, 1, 2))
        self.assertAlmostEqual(row.FP_pct, 100 / 3)
        self.assertAlmostEqual(row.cnt_error_pct, 200 / 3)

    def test_unused_subjects_use_release_fold_and_trained_subjects_require_holdout(self):
        with TemporaryDirectory() as tmp, patch.object(protocol, "REFERENCE", Path(tmp)):
            raw = Path(tmp)
            folds = {i: dict(train=[], validation=[], test=[]) for i in range(1, 6)}
            folds[2]["train"].append("A")
            folds[4]["test"].append("A")
            sessions = ["20260601_A", "20260601_B"]
            for s in sessions:
                (raw / s).mkdir()
            releases = [dict(val=sessions)] + [dict(val=[]) for _ in range(4)]
            (raw / "splits.json").write_text(json.dumps(dict(folds=releases)))
            rows, missing = protocol.session_plan(raw, folds)
            self.assertFalse(missing)
            self.assertEqual([(r["fold"], r["assignment"]) for r in rows],
                             [(4, "held_out"), (1, "previously_unused")])
            folds[4]["test"].clear()
            with self.assertRaisesRegex(ValueError, "No held-out checkpoint"):
                protocol.session_plan(raw, folds)
