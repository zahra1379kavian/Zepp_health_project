"""Run the five subject folds of Step 2 and benchmark the selected recipe."""

import argparse
import csv
from dataclasses import dataclass, replace
import fcntl
from itertools import product
import json
from pathlib import Path
import subprocess
import sys
import time

import torch

from . import PAIRS, SCREEN
from .study import select, summarize
from ...data import subject_folds, write_json
from ...orientation import INPUT_MODES
from ...runner import digest, fit_settings


RUNS = Path("baseline/MS-TCN/experiment_runs")
FOLDS = range(1, 6)
MINIMUM_F1_GAIN = .05
SCHEDULE = argparse.Namespace(epochs=30, batch_size=8, window=256, learning_rate=.001, patience=6)


@dataclass(frozen=True)
class Variant:
    inputs: str
    causal_similarity: bool = False
    synthetic_reps: bool = False
    exercise_adversarial: bool = False

    @property
    def name(self):
        return self.inputs + ("_similarity" if self.causal_similarity else "") + (
            "_synthetic" if self.synthetic_reps else "") + ("_adversarial" if self.exercise_adversarial else "")

    def options(self):
        return ["--rep-inputs", self.inputs, *(
            "--" + option.replace("_", "-") for option in
            ("causal_similarity", "synthetic_reps", "exercise_adversarial") if getattr(self, option))]


def atomic_json(path, value):
    pending = path.with_suffix(".pending")
    write_json(pending, value)
    pending.replace(path)


def read_json(path):
    return json.loads(path.read_text())


def preserve_partial(folder):
    if folder.exists():
        archive = folder.with_name(f"{folder.name}.incomplete-{time.time_ns()}")
        folder.rename(archive)
        print(f"Preserved incomplete output at {archive}", flush=True)


class Study:
    def __init__(self, args):
        self.args = args
        self.out = args.output
        self.manifest = read_json(args.data / "manifest.json")
        self.splits = subject_folds(self.manifest["sessions"], len(FOLDS), 42)
        self.data_hash = digest(args.data / "manifest.json")
        self.review_hash = digest(args.rest_reviews)
        self.phase_manifest = read_json(args.phase_benchmark / "manifest.json")
        people = {row["session"]: row["subject"] for row in self.manifest["sessions"]}
        subjects = {str(fold): {split: sorted({people[name] for name in streams}) for split, streams in part.items()}
                    for fold, part in enumerate(self.splits, 1)}
        if self.phase_manifest["fold_subjects"] != subjects:
            raise ValueError("The frozen phase benchmark uses different subjects")
        self.records = {}
        self.signatures = {}
        self.common = ["--data", str(args.data), "--rest-reviews", str(args.rest_reviews),
                       "--repetition-head", "complete", "--device", args.device, "--threads", str(args.threads)]

    def folder(self, variant, fold, exercise):
        return self.out / "holdouts" / variant.name / f"fold_{fold}" / exercise

    def existing(self, variant, fold, exercise):
        root = self.args.runs
        if fold == 1 and exercise in SCREEN:
            if variant == Variant("gravity", True, True):
                return root / "step_2_3" / exercise / "report.json"
            if not (variant.synthetic_reps or variant.exercise_adversarial):
                step = "step_2_2" if variant.causal_similarity else "step_2"
                return root / step / f"screen_{variant.inputs}" / "report.json"
        if fold == 2 and variant in (Variant("original"), Variant("original", True)):
            branch = "causal_similarity" if variant.causal_similarity else "control"
            return root / "step_2_2/fold_2_confirmation" / branch / "report.json"
        return None

    def validate(self, path, variant, fold, exercise):
        if path is None or not path.is_file():
            return None
        try:
            report = read_json(path)
        except json.JSONDecodeError:
            # A live fold-2 process may be publishing its next exercise.
            return None
        if exercise not in report["results"]:
            return None
        settings = report["training"]
        lane = report.get("execution_by_exercise", {}).get(exercise, {})
        if (lane.get("device", settings["device"]) != self.args.device
                or lane.get("torch_version", torch.__version__) != torch.__version__):
            return None
        expected = {**fit_settings(SCHEDULE), "seed": 42, "folds": len(FOLDS), "tolerance_ms": 400,
                    "report_tolerances_ms": [200, 400], "max_steps": 0, "lookahead_ms": 0,
                    "stages": 3, "layers": 6, "channels": 64, "station_only": False,
                    "acausal": False, "causal_similarity": variant.causal_similarity,
                    "synthetic_reps": variant.synthetic_reps, "exercise_adversarial": variant.exercise_adversarial}
        if (report["subject_fold"] != fold or report["split"] != self.splits[fold - 1]
                or report["protocol"] != "paired" or report["repetition_head"] != "complete"
                or report["rep_inputs"] != variant.inputs or report["selection_exercises"][exercise] != PAIRS[exercise]
                or any(settings.get(key, False) != val for key, val in expected.items())):
            return None
        fit = path.parent / exercise
        if not all((fit / name).is_file() for name in ("model/best.pt", "decoder.json", "data/manifest.json",
                                                       "held_out_report.json", "test/report.json")):
            return None
        score = report["results"][exercise]
        data = read_json(fit / "data/manifest.json")
        checkpoint_path = fit / "model/best.pt"
        checkpoint = torch.load(checkpoint_path, weights_only=True, map_location="cpu")
        decoder = read_json(fit / "decoder.json")
        if (data.get("parent_manifest_sha256") != self.data_hash or checkpoint["split"] != self.splits[fold - 1]
                or checkpoint.get("rest_reviews_sha256") != self.review_hash
                or checkpoint["data_manifest_sha256"] != digest(fit / "data/manifest.json")
                or decoder["checkpoint_sha256"] != digest(checkpoint_path)
                or decoder["sessions"] != self.splits[fold - 1]["validation"]
                or read_json(fit / "held_out_report.json") != score
                or read_json(fit / "test/report.json")["report_scope"] != "exercise_fold_test"):
            return None
        return score

    def command(self, name, *options, log):
        with log.open("a", buffering=1) as handle:
            print(f"{time.strftime('%FT%TZ', time.gmtime())} {name}", file=handle)
            subprocess.run([sys.executable, "-B", "-u", "-m", "baseline.mstcn", name, *options],
                           stdout=handle, stderr=subprocess.STDOUT, check=True)

    def fit(self, variant, fold, exercise):
        dest = self.folder(variant, fold, exercise)
        local = dest / "report.json"
        reused = self.existing(variant, fold, exercise)
        for path in (local, reused):
            result = self.validate(path, variant, fold, exercise)
            if result is not None:
                return path, result
        # Let the earlier matched fold-2 session finish before filling its gaps.
        while reused is not None and fold == 2 and subprocess.run(
                ["tmux", "has-session", "-t", "=mstcn_step22_fold2"], capture_output=True).returncode == 0:
            result = self.validate(reused, variant, fold, exercise)
            if result is not None:
                return reused, result
            print(f"Waiting for the existing fold-2 fit for {variant.name}, {exercise}", flush=True)
            time.sleep(30)
        result = self.validate(reused, variant, fold, exercise)
        if result is not None:
            return reused, result
        dest.parent.mkdir(parents=True, exist_ok=True)
        log = dest.with_suffix(".log")
        for attempt in range(2):
            preserve_partial(dest)
            print(f"Training {variant.name} | fold {fold} | {exercise} | attempt {attempt + 1}", flush=True)
            try:
                self.command("leave-one-exercise-out", *self.common, *variant.options(), "--protocol", "paired",
                             "--subject-fold", str(fold), "--exercises", exercise, "--output", str(dest), log=log)
            except subprocess.CalledProcessError:
                if attempt == 1:
                    raise
                continue
            result = self.validate(local, variant, fold, exercise)
            if result is not None:
                return local, result
        raise ValueError(f"Could not verify the completed training and test artifacts at {dest}")

    def fit_matrix(self, variants):
        for fold, variant, exercise in product(FOLDS, variants, PAIRS):
            key = variant.name, fold, exercise
            if key in self.records:
                continue
            path, score = self.fit(variant, fold, exercise)
            m, guards = score["by_tolerance"]["400"], score["guard_rails"]
            signature = (m["tp"] + m["missed"], guards["certified_rest_hours"], guards["seen_exercises"])
            if self.signatures.setdefault((fold, exercise), signature) != signature:
                raise ValueError(f"Evaluation exposure changed at {path}")
            self.records[key] = path, score
            progress = [{"variant": mode, "fold": f, "exercise": e, "report": str(p)}
                        for (mode, f, e), (p, _) in self.records.items()]
            atomic_json(self.out / "progress.json", {"complete_holdout_fits": len(progress), "fits": progress})
            print(f"Verified {len(progress)} fits | {variant.name} | fold {fold} | {exercise}", flush=True)

    def summary(self, variant):
        reports = [{"subject_fold": fold, "results": {e: self.records[variant.name, fold, e][1] for e in PAIRS}}
                   for fold in FOLDS]
        return summarize(reports, PAIRS, FOLDS)

    def compare(self, base, candidate):
        scores = {"original": self.summary(base), "treatment": self.summary(candidate)}
        passed = select(scores) == "treatment"
        return (candidate if passed else base), {"control": base.name, "candidate": candidate.name, "passed": passed}

    def benchmark(self, selected):
        cv = self.out / "all_exercise_cv"
        log = self.out / "all_exercise_cv.log"
        options = [*self.common, *selected.options(), "--folds", "5", "--resume-completed", "--output", str(cv)]
        print(f"Training the five all-exercise folds for {selected.name}", flush=True)
        for fold in FOLDS:
            folder = cv / f"fold_{fold}"
            if not (folder / "test/report.json").is_file():
                preserve_partial(folder)
            self.command("cross-validate", *options, "--subject-fold", str(fold), log=log)
        self.command("cross-validate", *options, log=log)
        reference = self.out / "selected_benchmark"
        if not reference.is_dir():
            self.command("reference-benchmark", "--run-dir", str(cv), "--output", str(reference),
                         "--source", self.manifest["source"], "--model-name", f"Step 2 five-fold {selected.name}",
                         "--device", self.args.device, "--threads", str(self.args.threads),
                         log=self.out / "selected_benchmark.log")
        output = self.out / "phase_comparison"
        if not output.is_dir():
            from baseline.rep_benchmark.compare import compare

            compare([self.args.phase_benchmark, reference], output)
        comparison = output / "model_comparison.csv"
        if not comparison.is_file():
            raise ValueError(f"Benchmark did not publish {comparison}")
        return comparison

    def run(self):
        plain = {mode: Variant(mode) for mode in INPUT_MODES}
        with_similarity = {mode: replace(variant, causal_similarity=True) for mode, variant in plain.items()}
        screen_synthetic = replace(with_similarity["gravity"], synthetic_reps=True)
        initial = [*plain.values(), *with_similarity.values(), screen_synthetic]
        plan = {"folds": list(FOLDS), "exercise_pairs": PAIRS, "initial_variants": [v.name for v in initial],
                "initial_holdout_fits": len(FOLDS) * len(PAIRS) * len(initial), "device": self.args.device,
                "torch_version": torch.__version__, "data_manifest_sha256": self.data_hash,
                "rest_reviews_sha256": self.review_hash, "phase_protocol_sha256": self.phase_manifest["protocol_sha256"]}
        path = self.out / "plan.json"
        if path.is_file() and read_json(path) != plan:
            raise ValueError("Use a fresh output directory for a changed study plan")
        atomic_json(path, plan)
        self.fit_matrix(initial)
        summary = {v.name: self.summary(v) for v in initial}
        orientation = select({mode: summary[mode] for mode in plain})
        similarity = {mode: self.compare(plain[mode], with_similarity[mode]) for mode in plain}
        selected, similarity_decision = similarity[orientation]
        synthetic = replace(selected, synthetic_reps=True)
        self.fit_matrix([synthetic])
        summary[synthetic.name] = self.summary(synthetic)
        selected, synthetic_decision = self.compare(selected, synthetic)
        # Use the same minimum F1 difference as the acceptance rule to define
        # a material remaining gap for the conditional exercise head.
        row = summary[selected.name]
        gap = row["mean_seen_f1_at_400"] - row["mean_unseen_f1_at_400"]
        adversarial = {"remaining_f1_gap": gap, "minimum_gap": MINIMUM_F1_GAIN, "ran": gap >= MINIMUM_F1_GAIN}
        if adversarial["ran"]:
            candidate = replace(selected, exercise_adversarial=True)
            self.fit_matrix([candidate])
            summary[candidate.name] = self.summary(candidate)
            selected, decision = self.compare(selected, candidate)
            adversarial.update(decision)
        atomic_json(self.out / "selection.json", {"folds": list(FOLDS), "variants": summary,
                    "orientation": orientation, "causal_similarity": similarity_decision,
                    "similarity_by_input": {mode: decision for mode, (_, decision) in similarity.items()},
                    "gravity_synthetic_screen": self.compare(with_similarity["gravity"], screen_synthetic)[1],
                    "synthetic": synthetic_decision,
                    "exercise_adversarial": adversarial, "selected": selected.name})
        with (self.out / "holdout_results.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["variant", "fold", "exercise", "tp", "missed", "extra", "unseen_f1_at_400", "report"])
            for (mode, fold, exercise), (path, score) in self.records.items():
                m = score["by_tolerance"]["400"]
                writer.writerow([mode, fold, exercise, m["tp"], m["missed"], m["extra"], m["f1"], path])
        comparison = self.benchmark(selected)
        atomic_json(self.out / "completed.json", {"selected": selected.name, "holdout_fits": len(self.records),
                    "selection": str(self.out / "selection.json"), "phase_comparison": str(comparison)})
        print(f"Five-fold Step 2 complete. Phase comparison: {comparison}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--runs", default=RUNS, type=Path)
    parser.add_argument("--data", default=Path("baseline/MS-TCN/prepared_datasets/separate_wrist_streams"), type=Path)
    parser.add_argument("--rest-reviews", default=RUNS / "complete_rep_review/reviewed.json", type=Path)
    parser.add_argument("--phase-benchmark", default=RUNS / "feishu_wrist_rep_comparison/phase_sequence_model", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threads", default=2, type=int)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        Study(args).run()


if __name__ == "__main__":
    main()
