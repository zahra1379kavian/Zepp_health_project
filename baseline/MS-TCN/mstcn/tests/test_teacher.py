import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from baseline.mstcn.__main__ import main
from baseline.mstcn.analyses.leave_one_exercise_out import PAIRS
from baseline.mstcn.analyses.teacher import compare_teacher
from baseline.mstcn.constants import DT_MS
from baseline.mstcn.data import prepare, runs
from baseline.mstcn.demo import synthetic_source
from baseline.mstcn.model import DenseMSTCN, ModelConfig
from baseline.mstcn.runner import Windows, probabilities


class TeacherTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(13)

    @torch.no_grad()
    def test_offline_centers_match_whole_runs_and_see_future_motion(self):
        model = DenseMSTCN(ModelConfig(acausal=True, causal_similarity=True, repetition_head="complete",
                                       channels=5, stages=3, layers=2)).eval()
        x = torch.randn(1, 9, 890)
        valid = np.ones(x.size(2), bool)
        valid[415:430] = False
        norm = {"mean": [0.] * 9, "std": [1.] * 9}
        p, q = probabilities(model, x[0].T.numpy(), valid, norm, chunk=37)
        self.assertEqual(model.future_samples, 9)
        for a, b in runs(valid):
            station, events = model(x[:, :, a:b])[-1]
            np.testing.assert_allclose(p[a:b], station.softmax(1)[0].T, atol=1e-6)
            np.testing.assert_allclose(q[a:b], model.repetition_probabilities(events)[0].T, atol=1e-6)
        self.assertFalse(q[~valid].any())
        changed = x.clone()
        changed[:, :, 130:] += 8
        altered, _ = probabilities(model, changed[0].T.numpy(), valid, norm, chunk=37)
        np.testing.assert_array_equal(p[:130-model.future_samples], altered[:130-model.future_samples])
        self.assertFalse(np.allclose(p[129], altered[129]))
        with self.assertRaisesRegex(ValueError, "offline"):
            model.stream(x)

    @torch.no_grad()
    def test_training_windows_match_full_run_targets_including_edges(self):
        model = DenseMSTCN(ModelConfig(acausal=True, channels=4, stages=2, layers=3)).eval()
        x = torch.randn(109, 9)
        valid = np.ones(len(x), bool)
        valid[40:48] = False
        arrays = dict(x=x.numpy(), valid=valid, y=np.zeros(len(x), int), coverage=valid, phase_mask=valid,
                      phase=np.zeros(len(x), int), boundary=np.zeros((len(x), 2), np.float32))
        with patch("baseline.mstcn.runner.load_arrays", return_value=arrays):
            windows = Windows(".", ["s"], {"mean": [0.] * 9, "std": [1.] * 9}, model.history_samples,
                              17, "phase", future_context=model.future_samples)
        expected = torch.cat([model(x[a:b].T[None])[-1][1][0].T for a, b in runs(valid)])
        outputs = []
        for features, mask, _, _, targets, _ in windows:
            events = model(features[None], mask[None])[-1][1]
            outputs.append(events[0].T[targets])
        torch.testing.assert_close(torch.cat(outputs), expected)

    def test_five_fold_teacher_and_paired_holdout_decision(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            alignment, _ = synthetic_source(root / "source", subjects=5)
            for path in (root / "source").glob("*/label.csv"):
                frame = pd.read_csv(path)
                frame["left_wrist"] = "rest"
                for i, name in enumerate([*PAIRS, *PAIRS.values(), "crunch"]):
                    start = 20 + 40 * i
                    frame.loc[start:start+14, "left_wrist"] = name + "_phase1"
                    frame.loc[start+15:start+29, "left_wrist"] = name + "_phase2"
                frame.to_csv(path, index=False)
            prepare(root / "source", root / "data", alignment)
            fit = ["--data", str(root / "data"), "--epochs", "1", "--max-steps", "1", "--rep-inputs", "gravity",
                   "--stages", "2", "--layers", "1", "--channels", "3", "--threads", "1",
                   "--repetition-head", "complete", "--causal-similarity"]
            compare = ["compare-teacher"]
            for role in ("student", "teacher"):
                options = [*fit, "--acausal"] if role == "teacher" else fit
                cv = root / role
                main(["cross-validate", *options, "--output", str(cv)])
                compare.extend([f"--{role}-cv", str(cv), f"--{role}-holdouts"])
                for fold in (1, 2):
                    out = root / f"{role}_holdouts_{fold}"
                    main(["leave-one-exercise-out", *options, "--exercises", "confirm",
                          "--subject-fold", str(fold), "--output", str(out)])
                    compare.append(str(out / "report.json"))
            fold_report = json.loads((root / "teacher/fold_1/test/report.json").read_text())
            self.assertTrue(fold_report["acausal"])
            self.assertEqual(fold_report["future_context_ms"], 2 * DT_MS)
            main([*compare, "--output", str(root / "decision.json")])
            decision = json.loads((root / "decision.json").read_text())
            self.assertEqual(decision["holdout_subject_folds"], [1, 2])
            self.assertEqual(set(decision["unseen_f1_by_exercise"]["teacher"]), set(PAIRS))
            for key in ("first_rep_recall", "last_rep_recall", "f1_at_400", "mean_unseen_f1_at_400"):
                self.assertAlmostEqual(decision["teacher_minus_student"][key],
                                       decision["teacher"][key] - decision["student"][key])
            with self.assertRaisesRegex(ValueError, "configuration changed"):
                main(["cross-validate", *fit, "--output", str(root / "teacher"), "--resume-completed"])

    def test_either_gain_at_the_threshold_qualifies(self):
        student = dict(first_rep_recall=.20, last_rep_recall=.60, f1_at_400=.50, mean_unseen_f1_at_400=.30)
        with tempfile.TemporaryDirectory() as tmp:
            for first, unseen, passes in ((.35, .30, True), (.20, .35, True), (.349, .349, False)):
                teacher = {**student, "first_rep_recall": first, "mean_unseen_f1_at_400": unseen}
                signature = {"recipe": {}, "folds": {}, "manifests": {}}
                scores = [(s, {}, [1, 2], signature) for s in (student, teacher)]
                args = SimpleNamespace(student_cv=None, student_holdouts=None, teacher_cv=None, teacher_holdouts=None,
                                       output=Path(tmp) / f"{first}-{unseen}.json")
                with patch("baseline.mstcn.analyses.teacher.ceiling_scores", side_effect=scores):
                    self.assertEqual(compare_teacher(args)["proceed_to_step_3_3"], passes)
