"""Fold-local complete-rep distillation from a frozen offline teacher."""

from argparse import Namespace
from dataclasses import asdict, dataclass
from itertools import product
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
from torch.nn import functional as F

from .complete import complete_targets
from .constants import DT_MS, START_HISTORY
from .data import write_json
from .decode import decode_sequence
from .runner import decoder_config, digest, evaluate, fit_settings, load_checkpoint, probabilities, train, tune


TEMPERATURE = 2.
LAMBDA_GRID = (.3, 1., 3.)


def distillation_losses(events, teacher_end, teacher_start):
    """Final-stage Bernoulli KL at T=2 and categorical KL at actual end frames."""
    valid = teacher_end >= 0
    logits = events[:, 0][valid] / TEMPERATURE
    p = teacher_end[valid]
    end_kl = (F.kl_div(F.logsigmoid(logits), p, reduction="sum")
              + F.kl_div(F.logsigmoid(-logits), 1 - p, reduction="sum"))
    ends = teacher_start.sum(-1) > 0
    log_choices = events[:, 1:].transpose(1, 2)[ends].log_softmax(-1)
    start_kl = F.kl_div(log_choices, teacher_start[ends], reduction="sum")
    return (TEMPERATURE ** 2 * end_kl / valid.sum().clamp_min(1),
            start_kl / ends.sum().clamp_min(1))


def masked_reps(arrays, references, detected):
    """Accept a complete interval only where every sample allows a pseudo-label."""
    t = arrays["t"]
    eligible = arrays["valid"] & (arrays["weight"] == 0)
    for r in references:
        a, b = np.searchsorted(t, [r["start_ms"], r["end_ms"]])
        eligible[a:b+1] = False
    accepted = []
    mask = np.zeros(len(t), bool)
    for r in detected:
        end = r.get("end_peak_ms", r["end_ms"])
        a, b = np.searchsorted(t, [r["start_ms"], end])
        if r["confidence"] > .9 and 0 <= a < b < len(t) and eligible[a:b+1].all():
            accepted.append({"start_ms": int(t[a]), "end_ms": int(t[b])})
            mask[a:b+1] = True
    hard = complete_targets({**arrays, "coverage": eligible}, accepted)
    mask |= (hard["end"] > 0) & eligible
    return accepted, mask, hard["end"], hard["start_choice"]


@dataclass
class Targets:
    soft_end: np.ndarray
    end_indices: np.ndarray
    choices: np.ndarray
    pseudo_mask: np.ndarray
    pseudo_end: np.ndarray
    pseudo_choice: np.ndarray
    pseudo_reps: list

    def window(self, start, end, offset, length, hard_end, hard_choice, weight):
        # Only the central target span belongs to this training window. Left
        # history, lookahead context and padding may occur in other windows.
        source = slice(start, end)
        center = slice(offset, offset + end - start)
        soft_end = np.full(length, -1., np.float32)
        soft_end[center] = self.soft_end[source]
        choices = np.zeros((length, START_HISTORY + 1), np.float32)
        left, right = np.searchsorted(self.end_indices, [start, end])
        choices[offset + self.end_indices[left:right] - start] = self.choices[left:right]
        pseudo = np.zeros(length, bool)
        pseudo[center] = self.pseudo_mask[source]
        hard_end[pseudo] = self.pseudo_end[source][self.pseudo_mask[source]]
        hard_choice[pseudo] = self.pseudo_choice[source][self.pseudo_mask[source]]
        weight[pseudo] = 1.
        return soft_end, choices, pseudo


class FrozenTeacher:
    def __init__(self, model, normalization, decoder, device, provenance, *, pseudo_labels=True):
        self.model = model.eval().requires_grad_(False)
        self.normalization = normalization
        self.decoder = decoder
        self.device = device
        self.provenance = provenance
        self.pseudo_labels = pseudo_labels
        self.cache = {}

    @torch.no_grad()
    def targets(self, arrays, references, name=None):
        if name in self.cache:
            return self.cache[name]
        t = arrays["t"]
        labeled = np.zeros(len(t), bool)
        indices = np.searchsorted(t, [r["end_ms"] for r in references])
        labeled[indices[indices < len(t)]] = True
        soft_end = np.full(len(t), -1., np.float32)
        indices, choices = [], []

        def capture(span, logits):
            soft_end[span] = (logits[:, 0] / TEMPERATURE).sigmoid().cpu().numpy()
            # Every reported peak is a rising end above threshold. Include the
            # first sample of each chunk in case its predecessor was outside it.
            p = logits[:, 0].sigmoid().cpu().numpy()
            possible = (p >= self.decoder.end_threshold) & np.r_[True, p[1:] > p[:-1]]
            local = np.flatnonzero(possible | labeled[span])
            indices.append(span.start + local)
            choices.append(logits[torch.as_tensor(local, device=logits.device), 1:].softmax(-1).cpu().numpy())

        station, reps = probabilities(self.model, arrays["x"], arrays["valid"], self.normalization,
                                      self.device, prepared=True, capture_logits=capture)
        detected = decode_sequence(t, station, reps, arrays["valid"], self.decoder, offline=True)["repetitions"]
        labeled[np.searchsorted(t, [r.get("end_peak_ms", r["end_ms"]) for r in detected])] = True
        rows = np.concatenate(indices) if indices else np.empty(0, int)
        scores = np.concatenate(choices) if choices else np.empty((0, START_HISTORY + 1), np.float32)
        use = labeled[rows]
        accepted, mask, end, choice = masked_reps(arrays, references, detected if self.pseudo_labels else ())
        target = Targets(soft_end, rows[use], scores[use], mask, end, choice, accepted)
        if name is not None:
            self.cache[name] = target
        return target


def run(args):
    gate = json.loads(args.ceiling_decision.read_text())
    if gate.get("proceed_to_step_3_3") is not True:
        raise ValueError("Step 3.1 has not qualified this experiment for distillation")
    fold = args.teacher_fold
    data, path = fold / "data", fold / "model/best.pt"
    manifest = json.loads((data / "manifest.json").read_text())
    model, ckpt = load_checkpoint(path, args.device, data)
    config = model.config
    decoder, tolerance, _ = decoder_config(fold / "decoder.json", path)
    selected = json.loads((fold / "decoder.json").read_text())
    if (not config.acausal or config.repetition_head != "complete" or ckpt["station_only"]
            or selected["sessions"] != manifest["split"]["validation"] or ckpt["split"] != manifest["split"]
            or decoder.start_rule != "learned" or tolerance != 400):
        raise ValueError("Use the matching fold's offline complete-rep teacher and validation-selected decoder")
    if ckpt.get("rest_reviews_sha256") != (digest(args.rest_reviews) if args.rest_reviews else None):
        raise ValueError("Use the reviewed rest file from the teacher fit")
    if args.lookahead_ms < 0 or args.lookahead_ms % DT_MS:
        raise ValueError("Choose a nonnegative student lookahead in 40 ms samples")
    schedule = ckpt.get("fit_settings")
    if schedule is None:
        report = json.loads((fold.parent / "report.json").read_text())
        schedule = report.get("fit_settings") or fit_settings(Namespace(**report["training"]))
    recipe = {**schedule, "data": data, "device": args.device, "station_only": False, "repetition_head": "complete",
              "stages": config.stages, "layers": config.layers, "channels": config.channels,
              "rep_inputs": ckpt["normalization"].get("rep_inputs", "original"), "seed": ckpt["seed"],
              "causal_similarity": config.causal_similarity, "synthetic_reps": ckpt.get("synthetic_reps", False),
              "lookahead_ms": args.lookahead_ms, "rest_reviews": args.rest_reviews,
              "max_steps": ckpt["partial_epoch_max_steps"], "tolerance_ms": tolerance}
    # Older ceiling reports predate fold provenance. New ones also bind the
    # approval to the original student recipe, subject split and review file.
    if "teacher_recipe" in gate:
        analysis = manifest.get("analysis") or {}
        split_id = str(analysis.get("subject_fold") or manifest.get("cross_validation", {}).get("fold"))
        hashes = gate["fold_manifest_and_review_hashes"].get(split_id)
        if (any(recipe.get(k) != v for k, v in gate["teacher_recipe"].items())
                or manifest["split"] != gate["fold_splits"].get(split_id) or not hashes
                or hashes[1] != ckpt.get("rest_reviews_sha256")
                or (not analysis and hashes[0] != ckpt["data_manifest_sha256"])
                or (analysis and (int(split_id) not in gate["holdout_subject_folds"]
                                  or analysis.get("held_out_exercise") not in gate["unseen_f1_by_exercise"]["teacher"]))):
            raise ValueError("The step-3.1 decision is for different training settings or data")
    provenance = {"teacher_sha256": digest(path), "teacher_decoder_sha256": digest(fold / "decoder.json"),
                  "future_limit_samples": config.future_limit_samples, "temperature": TEMPERATURE,
                  "ceiling_decision_sha256": digest(args.ceiling_decision)}
    initial_state = None
    if args.initial_student:
        prior, prior_ckpt = load_checkpoint(args.initial_student, "cpu", data)
        expected = {**asdict(config), "acausal": False, "future_limit_samples": None,
                    "lookahead_samples": args.lookahead_ms // DT_MS}
        if (config.future_limit_samples is not None or asdict(prior.config) != expected
                or prior_ckpt.get("distillation", {}).get("future_limit_samples") != 2000 // DT_MS
                or prior_ckpt["normalization"] != ckpt["normalization"]):
            raise ValueError("Initialize the full-teacher pass from a matching two-second-teacher student")
        initial_state = prior_ckpt["model"]
        provenance["initial_student_sha256"] = digest(args.initial_student)
    soft_target_pilot = getattr(args, "soft_target_pilot", False)
    if soft_target_pilot:
        provenance["pseudo_labels"] = False
    teacher = FrozenTeacher(model, ckpt["normalization"], decoder, args.device, provenance,
                            pseudo_labels=not soft_target_pilot)
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    # An interrupted grid can reuse completed candidates for exactly this fit.
    identity = {**provenance, "lookahead_ms": args.lookahead_ms, "manifest_sha256": ckpt["data_manifest_sha256"]}
    # The pilot and the grid share the unit-weight folder. Check it before
    # doing any work so changing modes cannot leave a mixed run directory.
    unit_trial = out / "end_1_start_1/trial.json"
    if unit_trial.is_file() and json.loads(unit_trial.read_text())["sources"] != identity:
        raise ValueError(f"Saved unit-weight trial uses different settings: {unit_trial.parent}")
    weights = [(1., 1.)] if soft_target_pilot else product(LAMBDA_GRID, repeat=2)
    trials = []
    for end_weight, start_weight in weights:
        folder = out / f"end_{end_weight:g}_start_{start_weight:g}"
        if not folder.exists():
            with TemporaryDirectory(dir=out) as tmp:
                work = Path(tmp) / "fit"
                params = Namespace(**recipe, lambda_end=end_weight, lambda_start=start_weight, output=work / "model")
                print(f"Training student with end weight {end_weight:g} and start weight {start_weight:g}", flush=True)
                params.checkpoint = train(params, teacher=teacher, initial_state=initial_state)
                params.output = work / "decoder.json"
                selection = tune(params)
                best = max(selection["trials"], key=lambda r: (r["f1"], -r["count_mae"]))
                write_json(work / "trial.json", {"sources": identity, "lambda_end": end_weight,
                           "lambda_start": start_weight, "validation_f1_at_400": best["f1"],
                           "validation_count_mae": best["count_mae"], "folder": folder.name})
                write_json(work / "pseudo_reps.json", {name: target.pseudo_reps for name, target in teacher.cache.items()})
                work.rename(folder)
        trial = json.loads((folder / "trial.json").read_text())
        if trial["sources"] != identity or (trial["lambda_end"], trial["lambda_start"]) != (end_weight, start_weight):
            raise ValueError(f"Saved distillation trial uses different settings: {folder}")
        trials.append(trial)
    winner = max(trials, key=lambda r: (r["validation_f1_at_400"], -r["validation_count_mae"]))
    folder = out / winner["folder"]
    result = {"selected_on": "validation", "tolerance_ms": tolerance, "trials": trials, "selected": winner,
              "checkpoint": str(folder.relative_to(out) / "model/best.pt"), "decoder": str(folder.relative_to(out) / "decoder.json")}
    with TemporaryDirectory(dir=out) as tmp:
        pending = Path(tmp) / "selection.json"
        write_json(pending, result)
        pending.replace(out / "selection.json")
    if args.validation_only:
        return result
    # Publish the held-out evaluation only after both weights and decoder are frozen.
    test = folder / "test"
    if not test.exists():
        with TemporaryDirectory(dir=folder) as tmp:
            pending = Path(tmp) / "test"
            evaluate(Namespace(**recipe, checkpoint=folder / "model/best.pt", decoder=folder / "decoder.json",
                               output=pending, split="test"))
            exercise = (manifest.get("analysis") or {}).get("held_out_exercise")
            if exercise:
                from .analyses.common import score_exercise
                held_out = score_exercise(data, pending, exercise, [200, 400], tolerance)
                write_json(pending / "held_out_report.json", held_out)
            pending.rename(test)
    return result
