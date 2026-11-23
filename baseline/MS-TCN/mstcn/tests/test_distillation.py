import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from baseline.mstcn.__main__ import main
from baseline.mstcn.constants import DT_MS, START_HISTORY
from baseline.mstcn.data import prepare, runs
from baseline.mstcn.decode import DecoderConfig
from baseline.mstcn.demo import synthetic_source
from baseline.mstcn.distillation import FrozenTeacher, distillation_losses, masked_reps
from baseline.mstcn.model import DenseMSTCN, ModelConfig
from baseline.mstcn.runner import digest, load_checkpoint


class DistillationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(12)
        torch.set_num_threads(1)

    def test_temperature_direction_and_gradient_masks(self):
        events = torch.randn(1, START_HISTORY + 2, 5, requires_grad=True)
        end = torch.tensor([[.27, .8, -1., .5, -1.]])
        starts = torch.zeros(1, 5, START_HISTORY + 1)
        starts[0, 1] = torch.randn(START_HISTORY + 1).softmax(0)
        a, b = distillation_losses(events, end, starts)
        p = end[end >= 0]
        q = (events[:, 0][end >= 0] / 2).sigmoid()
        torch.testing.assert_close(a, 4 * (p * (p / q).log() + (1-p) * ((1-p)/(1-q)).log()).mean())
        choice = starts[0, 1]
        torch.testing.assert_close(b, (choice * (choice.log() - events[0, 1:, 1].log_softmax(0))).sum())
        (a + b).backward()
        torch.testing.assert_close(events.grad[:, 0] != 0, end >= 0)
        torch.testing.assert_close(events.grad[:, 1:].abs().sum(1) > 0, starts.sum(-1) > 0)
        zero = distillation_losses(events, torch.full_like(end, -1.), torch.zeros_like(starts))
        self.assertEqual(sum(zero).item(), 0.)

    def test_only_whole_confident_masked_intervals_are_hard_labels(self):
        valid = np.ones(100, bool)
        valid[55] = False
        weight = np.zeros(100, np.float32)
        weight[:10] = 3.
        arrays = dict(t=np.arange(100)*DT_MS, valid=valid, y=np.zeros(100), weight=weight)
        refs = [dict(start_ms=70*DT_MS, end_ms=80*DT_MS)]
        detected = [dict(start_ms=a*DT_MS, end_ms=b*DT_MS, confidence=c) for a, b, c in
                    [(5, 12, .99), (21, 32, .9001), (35, 47, .9), (50, 61, .98), (68, 84, .99)]]
        accepted, mask, end, choice = masked_reps(arrays, refs, detected)
        self.assertEqual(accepted, [{"start_ms": 21*DT_MS, "end_ms": 32*DT_MS}])
        np.testing.assert_array_equal(np.flatnonzero(mask), np.arange(21, 35))
        self.assertEqual((end[32], choice[32]), (1., START_HISTORY-11))
        self.assertFalse(mask[55])

    def test_soft_targets_align_across_chunks_gaps_and_training_windows(self):
        model = DenseMSTCN(ModelConfig(acausal=True, repetition_head="complete", stages=1, layers=2, channels=4))
        teacher = FrozenTeacher(model, {"mean": [0.]*9, "std": [1.]*9}, DecoderConfig(), "cpu", {})
        n = 550
        valid = np.ones(n, bool)
        valid[29:33] = False
        x = np.random.default_rng(12).normal(size=(n, 9)).astype(np.float32)
        data = dict(t=np.arange(n)*DT_MS, x=x, valid=valid, y=np.zeros(n), weight=np.zeros(n))
        refs = [{"start_ms": 405*DT_MS, "end_ms": 515*DT_MS}]
        with patch("baseline.mstcn.distillation.decode_sequence", return_value={"repetitions": []}):
            target = teacher.targets(data, refs, "train")
        self.assertIs(target, teacher.targets(data, refs, "train"))
        np.testing.assert_array_equal(target.end_indices, [515])
        self.assertTrue(np.all(target.soft_end[~valid] == -1))
        for a, b in runs(valid):
            with torch.no_grad():
                _, scores = model(torch.from_numpy(x[a:b].T[None]))[-1]
            np.testing.assert_allclose(target.soft_end[a:b], (scores[0, 0] / 2).sigmoid(), atol=1e-6)
            if a <= 515 < b:
                np.testing.assert_allclose(target.choices[0], scores[0, 1:, 515-a].softmax(0), atol=1e-6)
        soft, choices, _ = target.window(510, 520, 7, 25, np.zeros(25), np.full(25, -1), np.zeros(25))
        np.testing.assert_array_equal(soft[7:17], target.soft_end[510:520])
        self.assertTrue(np.all(soft[:7] == -1) and np.all(soft[17:] == -1))
        np.testing.assert_array_equal(choices[12], target.choices[0])

    def test_decoded_end_adds_start_kl_and_a_hard_pseudo_rep(self):
        model = DenseMSTCN(ModelConfig(acausal=True, repetition_head="complete", stages=1, layers=1, channels=4))
        teacher = FrozenTeacher(model, {}, DecoderConfig(), "cpu", {})
        n = 65
        arrays = dict(t=np.arange(n)*DT_MS, x=np.zeros((n, 9)), y=np.zeros(n), valid=np.ones(n, bool),
                      weight=np.zeros(n, np.float32))
        arrays["weight"][40:] = 1
        refs = [{"start_ms": 40*DT_MS, "end_ms": 55*DT_MS}]

        def replay(model, x, valid, norm, device, *, prepared, capture_logits):
            logits = torch.zeros(n, START_HISTORY + 2)
            logits[:, 0] = -12
            logits[30, 0] = 12
            logits[30, 1 + START_HISTORY - 10] = 20
            capture_logits(slice(0, n), logits)
            return np.tile([0., 1.], (n, 1)), model.repetition_probabilities(logits.T[None])[0].T.numpy()

        with patch("baseline.mstcn.distillation.probabilities", side_effect=replay):
            targets = teacher.targets(arrays, refs)
        np.testing.assert_array_equal(targets.end_indices, [30, 55])
        hard, choice, weights = np.zeros(n), np.full(n, -1), arrays["weight"].copy()
        soft, starts, pseudo = targets.window(0, n, 0, n, hard, choice, weights)
        self.assertEqual((hard[30], choice[30], weights[30]), (1., START_HISTORY-10, 1.))
        self.assertTrue(pseudo[30] and not pseudo[10] and not pseudo[55])
        self.assertTrue((soft >= 0).all())
        self.assertEqual(starts[30].argmax(), START_HISTORY-10)
        self.assertEqual(weights[10], 0.)

    def test_future_cap_is_shared_across_all_stages(self):
        model = DenseMSTCN(ModelConfig(acausal=True, future_limit_samples=50, repetition_head="complete",
                                       stages=3, layers=5, channels=4, causal_similarity=True)).eval()
        self.assertEqual(sum(l.padding[1] for s in model.stages for l in s.layers), 50)
        x = torch.randn(1, 9, 200)
        changed = x.clone()
        changed[:, :, 151:] += 100
        with torch.no_grad():
            _, first = model(x)[-1]
            _, second = model(changed)[-1]
        torch.testing.assert_close(first[:, :, :101], second[:, :, :101])

    def test_fold_grid_then_curriculum_with_deployable_checkpoint(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            alignment, _ = synthetic_source(root / "source", subjects=3)
            prepare(root / "source", root / "data", alignment)
            settings = ["cross-validate", "--data", str(root / "data"), "--repetition-head", "complete",
                        "--rep-inputs", "gravity", "--causal-similarity", "--folds", "3", "--epochs", "1",
                        "--max-steps", "1", "--stages", "2", "--layers", "2", "--channels", "4", "--threads", "1"]
            gate = root / "gate.json"
            gate.write_text(json.dumps({"proceed_to_step_3_3": False}))
            distill = ["distill", "--ceiling-decision", str(gate), "--threads", "1", "--lookahead-ms", "160"]
            with self.assertRaisesRegex(ValueError, "Step 3.1"):
                main([*distill, "--teacher-fold", str(root/"missing"), "--output", str(root/"blocked")])
            warm = []
            for stage in ("limited", "full"):
                cv, out = root / stage, root / f"{stage}_students"
                main([*settings, "--output", str(cv), "--acausal", *(["--two-second-teacher"] if not warm else [])])
                teacher = cv / "fold_1"
                teacher_hash = digest(teacher / "model/best.pt")
                manifest = json.loads((teacher / "data/manifest.json").read_text())
                decision = {"proceed_to_step_3_3": True, "teacher_recipe": {"rep_inputs": "gravity"},
                            "fold_splits": {"1": manifest["split"]},
                            "fold_manifest_and_review_hashes": {"1": [digest(teacher / "data/manifest.json"), None]}}
                gate.write_text(json.dumps({**decision, "teacher_recipe": {"rep_inputs": "rotated"}}))
                command = [*distill, "--teacher-fold", str(teacher), "--output", str(out), *warm]
                with self.assertRaisesRegex(ValueError, "different training"):
                    main(command)
                gate.write_text(json.dumps(decision))
                main([*command, "--validation-only"])
                self.assertFalse(list(out.glob("*/test/report.json")))
                selection = json.loads((out / "selection.json").read_text())
                self.assertEqual(len(selection["trials"]), 9)
                model, ckpt = load_checkpoint(out / selection["checkpoint"], data=teacher / "data")
                self.assertFalse(model.config.acausal)
                self.assertEqual(model.config.lookahead_samples, 4)
                self.assertEqual(ckpt["distillation"]["teacher_sha256"], teacher_hash)
                audit = json.loads((out / selection["selected"]["folder"] / "pseudo_reps.json").read_text())
                self.assertEqual(set(audit), set(ckpt["split"]["train"]))
                self.assertEqual(digest(teacher / "model/best.pt"), teacher_hash)
                self.assertEqual(model.stream(torch.zeros(1, 6, 20))[1].shape[1], 152)
                with patch("baseline.mstcn.distillation.train", side_effect=AssertionError("unexpected retraining")):
                    main(command)
                self.assertEqual(len(list(out.glob("*/test/report.json"))), 1)
                warm = ["--initial-student", str(out / selection["checkpoint"])]

    def test_exercise_holdout_with_synthetic_training(self):
        with TemporaryDirectory() as tmp:
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
            main(["leave-one-exercise-out", "--data", str(root/"data"), "--output", str(root/"teacher"),
                  "--repetition-head", "complete", "--folds", "3", "--protocol", "paired", "--exercises", "squat",
                  "--rep-inputs", "rotated", "--causal-similarity", "--synthetic-reps", "--batch-size", "4",
                  "--window", "64", "--stages", "2", "--channels", "4", "--layers", "1", "--epochs", "1",
                  "--max-steps", "1", "--threads", "1", "--acausal"])
            gate = root / "fixture_decision.json"
            gate.write_text(json.dumps({"proceed_to_step_3_3": True}))
            out = root / "distilled"
            main(["distill", "--teacher-fold", str(root/"teacher/squat"), "--output", str(out),
                  "--ceiling-decision", str(gate), "--threads", "1"])
            selection = json.loads((out / "selection.json").read_text())
            winner = out / selection["selected"]["folder"]
            report = json.loads((winner / "test/held_out_report.json").read_text())
            self.assertEqual(report["exercise"], "squat")
            self.assertEqual(json.loads((winner / "decoder.json").read_text())["selection_exercise"], "pull-up")
            epoch, = json.loads((winner / "model/history.json").read_text())
            self.assertEqual((epoch["synthetic_training_windows"], epoch["real_training_windows"]), (1, 3))
            self.assertGreater(epoch["teacher_frames"], 0)
