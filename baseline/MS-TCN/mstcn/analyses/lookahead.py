"""Select a streaming lookahead using fold-1 validation, then test the fixed choice."""

from argparse import Namespace
from collections import defaultdict
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from ..data import subject_folds, write_json
from ..metrics import report_delay_percentiles, tally
from ..runner import digest, evaluate_streams, load_checkpoint, train, tune


LOOKAHEAD_MS = (0, 160, 320, 480, 640)
POSITIONS = ("first", "last")


def summarize(streams):
    delays = [d for s in streams for d in s["report_delays_ms"]]
    result = {"matched_reps": len(delays), "report_delay_p95_ms": report_delay_percentiles(delays)["report_delay_p95_ms"]}
    for position in POSITIONS:
        rows = [s["detection_by_position"][position] for s in streams]
        result[position] = tally(sum(r["matched"] for r in rows), sum(r["total"] for r in rows))
    return result


def meets_target(student, teacher, candidate):
    fractions = {}
    qualifies = candidate["report_delay_p95_ms"] is not None and candidate["report_delay_p95_ms"] < 1000
    for position in POSITIONS:
        base, ceiling, row = (s[position] for s in (student, teacher, candidate))
        if not base["total"] == ceiling["total"] == row["total"]:
            raise ValueError("Compare first and last reps on identical references")
        teacher_gain = ceiling["matched"] - base["matched"]
        candidate_gain = row["matched"] - base["matched"]
        fractions[position] = candidate_gain / teacher_gain if teacher_gain > 0 else None
        # Equal denominators allow an exact inclusive 80% check in integer counts.
        qualifies &= base["total"] > 0 and (teacher_gain <= 0 or 5 * candidate_gain >= 4 * teacher_gain)
    return {"fraction_of_teacher_gain": fractions, "qualifies": qualifies}


def source_fits(args, fold):
    student = args.student_cv / f"fold_{fold}"
    teacher = args.teacher_cv / f"fold_{fold}" if args.teacher_cv else None
    data = student / "data"
    manifest = json.loads((data / "manifest.json").read_text())
    if (manifest.get("cross_validation") != {"fold": fold, "folds": 5} or manifest.get("analysis")
            or manifest["split"] != subject_folds(manifest["sessions"], 5, manifest["seed"])[fold - 1]):
        raise ValueError(f"Expected all-exercise subject fold {fold} of five in {data}")
    review_hash = digest(args.rest_reviews) if args.rest_reviews else None
    recipes, schedules, decoders = [], [], {}
    sources = [(student, False)]
    if teacher is not None and args.stage != "fit":
        sources.append((teacher, True))
    for folder, offline in sources:
        path = folder / "model/best.pt"
        model, ckpt = load_checkpoint(path, data=data)
        config = model.config
        decoder = json.loads((folder / "decoder.json").read_text())
        if (config.acausal != offline or config.lookahead_samples or config.repetition_head != "complete"
                or config.future_limit_samples is not None
                or ckpt["station_only"] or ckpt["split"] != manifest["split"] or ckpt["seed"] != manifest["seed"]
                or ckpt.get("rest_reviews_sha256") != review_hash or ckpt["tolerance_ms"] != 400
                or decoder["tolerance_ms"] != 400 or decoder["selected_on"] != "validation"
                or decoder["sessions"] != ckpt["split"]["validation"] or decoder["checkpoint_sha256"] != digest(path)):
            raise ValueError(f"Expected a matched {'offline teacher' if offline else 'zero-lookahead student'} "
                             f"with the same rest reviews and validation decoder in {folder}")
        settings = ckpt.get("fit_settings")
        if settings is None and (folder.parent / "report.json").is_file():
            # Earlier runs stored these settings in the CV summary.
            settings = json.loads((folder.parent / "report.json").read_text()).get("fit_settings")
        if settings is not None:
            schedules.append(settings)
        recipes.append(dict(seed=ckpt["seed"], max_steps=ckpt["partial_epoch_max_steps"],
                            stages=config.stages, layers=config.layers, channels=config.channels, station_only=False,
                            repetition_head=config.repetition_head, rep_inputs=ckpt["normalization"].get("rep_inputs", "original"),
                            causal_similarity=config.causal_similarity, synthetic_reps=ckpt.get("synthetic_reps", False),
                            tolerance_ms=400))
        decoders["teacher_decoder_sha256" if offline else "student_decoder_sha256"] = digest(folder / "decoder.json")
    if not schedules:
        raise ValueError("The source fits need a saved training schedule in a checkpoint or CV report")
    if recipes[0] != recipes[-1] or schedules[0] != schedules[-1]:
        raise ValueError("Student and teacher must use the same training settings")
    identity = {"fit_settings": {**recipes[0], **schedules[0]}, "manifest_sha256": digest(data / "manifest.json"),
                "rest_reviews_sha256": review_hash, **decoders}
    return student, teacher, data, identity


def fit_student(args, fold, ms, data, identity):
    folder = args.output / f"lookahead_{ms}ms" / f"fold_{fold}"
    settings = {**identity["fit_settings"], "lookahead_ms": ms}
    provenance = {"settings": settings, "manifest_sha256": identity["manifest_sha256"],
                  "rest_reviews_sha256": identity["rest_reviews_sha256"]}
    if folder.exists():
        if json.loads((folder / "fit.json").read_text()) != provenance:
            raise ValueError(f"Saved lookahead fit uses different data or settings: {folder}")
        return folder
    folder.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=folder.parent) as tmp:
        work = Path(tmp) / "fit"
        params = Namespace(**settings, data=data, rest_reviews=args.rest_reviews, output=work / "model", device=args.device)
        params.checkpoint = train(params)
        params.output = work / "decoder.json"
        tune(params)
        write_json(work / "fit.json", provenance)
        work.rename(folder)
    return folder


def score(folder, data, split, device):
    """Keep individual matched delays so the five-fold percentile pools repetitions."""
    _, records = evaluate_streams(Namespace(checkpoint=folder / "model/best.pt", decoder=folder / "decoder.json",
                                            data=data, split=split, device=device, tolerance_ms=400, report_tolerances_ms=[400]))
    streams = []
    for record in records:
        scores = record["repetition_scores"]
        if "400" in scores:
            metrics = scores["400"]["segmentation"]["metrics"]
            streams.append({k: metrics[k] for k in ("detection_by_position", "report_delays_ms")})
    return streams


def screen(args, student, teacher, data, identity):
    baseline = summarize(score(student, data, "validation", args.device))
    ceiling = summarize(score(teacher, data, "validation", args.device))
    rows = []
    for ms in LOOKAHEAD_MS:
        candidate = baseline
        if ms:
            folder = fit_student(args, 1, ms, data, identity)
            candidate = summarize(score(folder, data, "validation", args.device))
        rows.append({"lookahead_ms": ms, **candidate, **meets_target(baseline, ceiling, candidate)})
    return {"subject_fold": 1, "split": "validation", "tolerance_ms": 400, "sources": identity,
            "teacher": ceiling, "candidates": rows,
            "selected_lookahead_ms": next((r["lookahead_ms"] for r in rows if r["qualifies"]), None)}


def confirm(args, selection):
    ms = selection["selected_lookahead_ms"]
    folds, streams = [], defaultdict(list)
    for fold in range(1, 6):
        student, teacher, data, identity = source_fits(args, fold)
        if any(identity[k] != selection["sources"][k] for k in ("fit_settings", "rest_reviews_sha256")):
            raise ValueError(f"Training settings differ from the screened student in fold {fold}")
        evaluated = {"baseline": score(student, data, "test", args.device), "teacher": score(teacher, data, "test", args.device)}
        evaluated["selected"] = (score(fit_student(args, fold, ms, data, identity), data, "test", args.device)
                                 if ms else evaluated["baseline"])
        for name, rows in evaluated.items():
            streams[name].extend(rows)
        summary = {name: summarize(rows) for name, rows in evaluated.items()}
        folds.append({"fold": fold, **summary, **meets_target(summary["baseline"], summary["teacher"], summary["selected"])})
    pooled = {name: summarize(rows) for name, rows in streams.items()}
    decision = meets_target(pooled["baseline"], pooled["teacher"], pooled["selected"])
    return {"lookahead_ms": ms, "split": "test", "tolerance_ms": 400, "folds": folds, "pooled": pooled,
            **decision, "confirmed": decision["qualifies"]}


def save_report(path, value):
    with TemporaryDirectory(dir=path.parent) as tmp:
        pending = Path(tmp) / path.name
        write_json(pending, value)
        pending.replace(path)


def run(args):
    if args.stage != "fit" and args.teacher_cv is None:
        raise ValueError("Selection needs --teacher-cv; use --stage fit to train the fold-1 students first")
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "screen.json"
    sources = source_fits(args, 1)
    if args.stage == "fit":
        _, _, data, identity = sources
        for ms in LOOKAHEAD_MS[1:]:
            fit_student(args, 1, ms, data, identity)
        print("The four delayed students are ready for the fold-1 validation screen.")
        return
    if path.is_file():
        selection = json.loads(path.read_text())
        if selection["sources"] != sources[-1]:
            raise ValueError("The saved screen belongs to a different student or teacher")
    elif args.stage == "confirm":
        raise ValueError("Run the fold-1 validation screen before confirmation")
    else:
        selection = screen(args, *sources)
        save_report(path, selection)
    ms = selection["selected_lookahead_ms"]
    if ms is None:
        print("No lookahead met both the recall and reporting-delay criteria on validation.")
    elif args.stage != "screen":
        result = confirm(args, selection)
        save_report(args.output / "confirmation.json", result)
        print(f"Selected {ms} ms. Five-fold confirmation: {result['confirmed']}")
    else:
        print(f"Fold-1 validation selected {ms} ms.")
