import json
import unittest
from argparse import Namespace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from baseline.mstcn.__main__ import main
from baseline.mstcn.analyses import lookahead
from baseline.mstcn.constants import DT_MS
from baseline.mstcn.data import prepare
from baseline.mstcn.decode import decode_sequence
from baseline.mstcn.demo import synthetic_source
from baseline.mstcn.metrics import repetition_metrics, tally


def observations(first, last, delays, total=20):
    return [{"detection_by_position": {"first": tally(first, total), "last": tally(last, total)},
             "report_delays_ms": delays}]


class LookaheadSweepTests(unittest.TestCase):
    def test_fit_before_teacher_then_screen_the_same_models(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            alignment, _ = synthetic_source(root / "source", subjects=5)
            prepare(root / "source", root / "data", alignment)
            cv = ["cross-validate", "--data", str(root / "data"), "--repetition-head", "complete", "--rep-inputs", "gravity",
                  "--causal-similarity", "--stages", "2", "--layers", "1", "--channels", "4",
                  "--epochs", "1", "--max-steps", "1", "--threads", "1"]
            main([*cv, "--output", str(root / "student")])
            sweep = ["lookahead-sweep", "--student-cv", str(root / "student"), "--output", str(root / "sweep"), "--threads", "1"]
            main([*sweep, "--stage", "fit"])
            checkpoints = list((root / "sweep").glob("*/fold_1/model/best.pt"))
            self.assertEqual(len(checkpoints), 4)
            self.assertFalse((root / "sweep/screen.json").exists())
            main([*cv, "--acausal", "--output", str(root / "teacher")])
            with patch.object(lookahead, "train", side_effect=AssertionError("screen retrained a completed fit")):
                main([*sweep, "--teacher-cv", str(root / "teacher"), "--stage", "screen"])
            screen = json.loads((root / "sweep/screen.json").read_text())
            self.assertEqual(screen["split"], "validation")
            self.assertEqual([r["lookahead_ms"] for r in screen["candidates"]], list(lookahead.LOOKAHEAD_MS))
            self.assertFalse((root / "sweep/confirmation.json").exists())

    def test_complete_rep_delay_uses_true_end_and_includes_decoder_wait(self):
        clocks = np.arange(60) * DT_MS
        p = np.tile([0., 1.], (60, 1))
        q = np.zeros((60, 4))
        q[30] = [.9, 20, .9, .1]
        # The predicted end (frame 30) is 280 ms later than the label (frame 23).
        refs = [{"start_ms": 10 * DT_MS, "end_ms": 23 * DT_MS, "station": "squat"}]
        for ms in lookahead.LOOKAHEAD_MS:
            reps = decode_sequence(clocks, p, q, np.ones(60, bool), lookahead_ms=ms)["repetitions"]
            scored = repetition_metrics(reps, refs, tolerance_ms=400, require_station=False)
            self.assertEqual(scored["report_delay_p95_ms"], 280 + 80 + ms)

    def test_screen_uses_smallest_wait_meeting_both_gains_and_strict_deadline(self):
        baseline, teacher = Path("baseline"), Path("teacher")
        scores = {baseline: observations(5, 10, [80]), teacher: observations(15, 15, [30000]),
                  160: observations(13, 13, [600]), 320: observations(13, 14, [1000]),
                  480: observations(13, 14, [999]), 640: observations(15, 15, [960])}
        args = Namespace(device="cpu")
        with patch.object(lookahead, "fit_student", side_effect=lambda args, fold, ms, data, identity: ms) as fit, \
                patch.object(lookahead, "score", side_effect=lambda folder, data, split, device: scores[folder]) as score:
            result = lookahead.screen(args, baseline, teacher, Path("data"), {})
            self.assertEqual(result["selected_lookahead_ms"], 480)
            self.assertEqual([row["lookahead_ms"] for row in result["candidates"]], list(lookahead.LOOKAHEAD_MS))
            self.assertTrue(all(call.args[2] == "validation" for call in score.call_args_list))
            self.assertEqual([call.args[2] for call in fit.call_args_list], [160, 320, 480, 640])
            scores[480] = observations(13, 14, [])
            scores[640] = observations(12, 14, [800])
            self.assertIsNone(lookahead.screen(args, baseline, teacher, Path("data"), {})["selected_lookahead_ms"])

    def test_no_positive_teacher_gain_and_no_reference_coverage(self):
        base = lookahead.summarize(observations(16, 8, [40]))
        teacher = lookahead.summarize(observations(14, 8, [2000]))
        self.assertTrue(lookahead.meets_target(base, teacher, base)["qualifies"])
        self.assertEqual(lookahead.meets_target(base, teacher, base)["fraction_of_teacher_gain"],
                         {"first": None, "last": None})
        no_refs = lookahead.summarize([])
        self.assertFalse(lookahead.meets_target(no_refs, no_refs, no_refs)["qualifies"])
        with self.assertRaisesRegex(ValueError, "identical references"):
            lookahead.meets_target(base, teacher, no_refs)

    def test_confirmation_locks_wait_and_pools_matched_rep_delays(self):
        identity = {"fit_settings": {}, "rest_reviews_sha256": None}
        selection = {"sources": identity, "selected_lookahead_ms": 320}
        args = Namespace(device="cpu")

        def score(folder, fold, split, device):
            self.assertEqual(split, "test")
            if folder == "selected":
                return observations(13, 14, [1000] if fold == 1 else [600] * 10)
            return observations(5, 10, [80]) if folder == "student" else observations(15, 15, [30000])

        with patch.object(lookahead, "source_fits", side_effect=lambda a, f: ("student", "teacher", f, identity)), \
                patch.object(lookahead, "fit_student", return_value="selected") as fit, patch.object(lookahead, "score", side_effect=score):
            result = lookahead.confirm(args, selection)
            self.assertEqual([call.args[1:3] for call in fit.call_args_list], [(i, 320) for i in range(1, 6)])
            self.assertFalse(result["folds"][0]["qualifies"])
            self.assertEqual(result["pooled"]["selected"]["report_delay_p95_ms"], 600)
            self.assertTrue(result["confirmed"])
