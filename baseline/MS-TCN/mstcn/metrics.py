"""Classification and one-to-one repetition scoring on certified coverage only."""

from collections import Counter

import numpy as np
from scipy.optimize import linear_sum_assignment

from .constants import CLASSES, DT_MS
from .data import runs


def classification_metrics(confusion):
    cm = np.asarray(confusion, dtype=np.int64)
    tp, support, predicted = np.diag(cm), cm.sum(1), cm.sum(0)
    precision = np.divide(tp, predicted, out=np.zeros(len(cm)), where=predicted > 0)
    recall = np.divide(tp, support, out=np.zeros(len(cm)), where=support > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(len(cm)),
                   where=precision + recall > 0)
    exercise_tp, exercise_fp, exercise_fn = cm[1:, 1:].sum(), cm[0, 1:].sum(), cm[1:, 0].sum()
    return {"confusion_matrix": cm.tolist(), "macro_f1_all_16": float(f1.mean()),
            "macro_f1_present_targets": float(f1[support > 0].mean()) if support.any() else None,
            "exercise_rest_f1": float(2 * exercise_tp / max(2 * exercise_tp + exercise_fp + exercise_fn, 1)),
            "per_class": {name: {"precision": float(precision[i]), "recall": float(recall[i]),
                                  "f1": float(f1[i]), "support": int(support[i])}
                          for i, name in enumerate(CLASSES)}}


def confusion_matrix(target, predicted, valid):
    return np.bincount(target[valid] * len(CLASSES) + predicted[valid],
                       minlength=len(CLASSES) ** 2).reshape(len(CLASSES), len(CLASSES))


def false_exercise_events(target, predicted, valid):
    rest_samples = int(np.count_nonzero(valid & (target == 0)))
    count = sum(bool(np.all(target[a:b] == 0)) for a, b in runs(valid & (predicted != 0)))
    return {"false_exercise_events": count, "rest_hours": rest_samples * DT_MS / 3600000,
            "false_exercise_events_per_rest_hour": count / (rest_samples * DT_MS / 3600000)
            if rest_samples else None}


def match_repetitions(predictions, references, tolerance_ms=400, require_station=True):
    """Maximum-cardinality matching, then minimum total start/end error.

    Both endpoints must be within tolerance. Station agreement is optional so
    repetition segmentation can be assessed independently of station recognition.
    No reference station is ever used in inference or decoding.
    """
    if tolerance_ms <= 0:
        raise ValueError("Matching tolerance must be positive")
    matches = []
    names = sorted({r["station"] for r in references}) if require_station else [None]
    for name in names:
        pi = [i for i, row in enumerate(predictions) if name is None or row["station"] == name]
        ri = [i for i, row in enumerate(references) if name is None or row["station"] == name]
        if not pi or not ri:
            continue
        pa = np.array([[predictions[i]["start_ms"], predictions[i]["end_ms"]] for i in pi])
        ra = np.array([[references[i]["start_ms"], references[i]["end_ms"]] for i in ri])
        errors = np.abs(pa[:, None, :] - ra[None, :, :])
        allowed = (errors <= tolerance_ms).all(2)
        penalty = (max(len(pi), len(ri)) + 1) * (2 * tolerance_ms + 1)
        cost = np.where(allowed, errors.sum(2), penalty)
        rows, cols = linear_sum_assignment(cost)
        matches.extend((pi[p], ri[r]) for p, r in zip(rows, cols) if allowed[p, r])
    return matches


def repetition_metrics(predictions, references, tolerance_ms=400, require_station=True):
    pairs = match_repetitions(predictions, references, tolerance_ms, require_station)
    tp, fp, fn = len(pairs), len(predictions) - len(pairs), len(references) - len(pairs)
    start_errors = [abs(predictions[p]["start_ms"] - references[r]["start_ms"]) for p, r in pairs]
    end_errors = [abs(predictions[p]["end_ms"] - references[r]["end_ms"]) for p, r in pairs]
    delays = [predictions[p].get("reported_at_ms", predictions[p]["end_ms"]) - references[r]["end_ms"]
              for p, r in pairs]
    return {"tp": tp, "extra": fp, "missed": fn,
            "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1),
            "f1": 2 * tp / max(2 * tp + fp + fn, 1),
            "start_mae_ms": float(np.mean(start_errors)) if pairs else None,
            "end_mae_ms": float(np.mean(end_errors)) if pairs else None,
            "report_delay_mean_ms": float(np.mean(delays)) if pairs else None,
            "matched_reported_before_1s_fraction": float(np.mean(np.array(delays) < 1000)) if pairs else None,
            "matched_early_report_fraction": float(np.mean(np.array(delays) < 0)) if pairs else None,
            "matched_reported_0_to_999ms_fraction": float(np.mean([(0 <= d < 1000) for d in delays])) if pairs else None,
            "start_errors_ms": start_errors, "end_errors_ms": end_errors, "report_delays_ms": delays}


def covered_repetitions(predictions, t, coverage, references=(), tolerance_ms=400,
                        require_station=True):
    """Match first; coverage can exclude only unmatched predictions.

    Retain matches across coverage edges within the declared timing tolerance,
    including accepted references whose shared endpoint abuts an exclusion.
    """
    matched = {p for p, _ in match_repetitions(predictions, references, tolerance_ms, require_station)}
    intervals = [(int(t[a]), int(t[b-1] + DT_MS)) for a, b in runs(coverage)]
    scored, unscored = [], []
    for i, rep in enumerate(predictions):
        inside = any(a <= rep["start_ms"] < rep["end_ms"] < b for a, b in intervals)
        (scored if i in matched or inside else unscored).append(rep)
    return scored, unscored


def count_rows(predictions, references, session, require_station=True):
    p, r = Counter(x["station"] for x in predictions), Counter(x["station"] for x in references)
    if not require_station:
        p, r = Counter({"all": len(predictions)}), Counter({"all": len(references)})
    names = [n for n in CLASSES[1:] if n != "plank"] if require_station else ["all"]
    names += sorted((set(p) | set(r)) - set(names))
    return [{"session": session, "station": name, "predicted": p[name], "reference": r[name],
             "absolute_error": abs(p[name] - r[name]), "exact": p[name] == r[name]}
            for name in names]


def aggregate_repetitions(results, counts):
    if not results:
        return {"status": "not_scored_no_validated_reference_coverage"}
    tp = sum(r["tp"] for r in results)
    fp, fn = sum(r["extra"] for r in results), sum(r["missed"] for r in results)
    starts = [v for r in results for v in r["start_errors_ms"]]
    ends = [v for r in results for v in r["end_errors_ms"]]
    delays = [v for r in results for v in r["report_delays_ms"]]
    active = [r for r in counts if r["predicted"] or r["reference"]]
    return {"status": "scored", "tp": tp, "extra": fp, "missed": fn,
            "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1),
            "f1": 2 * tp / max(2 * tp + fp + fn, 1),
            "start_mae_ms": float(np.mean(starts)) if starts else None,
            "end_mae_ms": float(np.mean(ends)) if ends else None,
            "report_delay_mean_ms": float(np.mean(delays)) if delays else None,
            "matched_reported_before_1s_fraction": float(np.mean(np.array(delays) < 1000)) if delays else None,
            "matched_early_report_fraction": float(np.mean(np.array(delays) < 0)) if delays else None,
            "matched_reported_0_to_999ms_fraction": float(np.mean([(0 <= d < 1000) for d in delays])) if delays else None,
            "count_mae_all_station_sessions": float(np.mean([r["absolute_error"] for r in counts])) if counts else None,
            "exact_count_accuracy_all_station_sessions": float(np.mean([r["exact"] for r in counts])) if counts else None,
            "count_mae_active_station_sessions": float(np.mean([r["absolute_error"] for r in active])) if active else None,
            "exact_count_accuracy_active_station_sessions": float(np.mean([r["exact"] for r in active])) if active else None,
            "count_rows": counts}


def reporting_tolerances(values=None, primary=400):
    """Keep the selection tolerance and all declared reporting tolerances."""
    values = [200, 400] if values is None else values
    if primary <= 0 or not values or any(value <= 0 for value in values):
        raise ValueError("Reporting tolerances must be positive and nonempty")
    return sorted(set([primary, *values]))


def score_stream(predictions, references, t, coverage, metadata, tolerances):
    """Score an unchanged decoded stream at several endpoint tolerances."""
    scores = {}
    for tolerance in tolerances:
        result = {}
        for require_station, key in ((True, "repetitions"), (False, "segmentation")):
            scored, _ = covered_repetitions(predictions, t, coverage, references, tolerance, require_station)
            rows = count_rows(scored, references, metadata["session"], require_station)
            for row in rows:
                row.update({k: metadata[k] for k in ("source_session", "subject", "side")})
            result[key] = {"metrics": repetition_metrics(scored, references, tolerance, require_station),
                           "counts": rows}
            if require_station:
                result["per_exercise"] = {
                    name: repetition_metrics([r for r in scored if r["station"] == name],
                                             [r for r in references if r["station"] == name], tolerance)
                    for name in CLASSES[1:] if name != "plank"}
        scores[str(tolerance)] = result
    return scores


def aggregate_streams(records, tolerances, primary=400):
    """Pool stream observations without treating paired wrists as extra subjects."""
    by_tolerance = {}
    for tolerance in tolerances:
        scores = [r["repetition_scores"][str(tolerance)] for r in records
                  if str(tolerance) in r["repetition_scores"]]
        summary = {key: aggregate_repetitions([s[key]["metrics"] for s in scores],
                                               [row for s in scores for row in s[key]["counts"]])
                   for key in ("repetitions", "segmentation")}
        summary["per_exercise"] = {
            name: aggregate_repetitions([s["per_exercise"][name] for s in scores],
                                         [row for s in scores for row in s["repetitions"]["counts"]
                                          if row["station"] == name])
            for name in CLASSES[1:] if name != "plank"}
        by_tolerance[str(tolerance)] = summary
    cm = sum((np.array(r["confusion_matrix"], dtype=np.int64) for r in records),
             np.zeros((len(CLASSES), len(CLASSES)), np.int64))
    hours = sum(r["rest_hours"] for r in records)
    events = sum(r["false_exercise_events"] for r in records)
    return {"streams": len(records), "subjects": len({r["subject"] for r in records}),
            "source_sessions": len({r["source_session"] for r in records}),
            "classification": classification_metrics(cm),
            "false_exercise_events": events, "rest_hours": hours,
            "false_exercise_events_per_rest_hour": events / hours if hours else None,
            "by_tolerance": by_tolerance, **by_tolerance[str(primary)]}


def grouped_stream_report(records, tolerances, primary=400):
    result = aggregate_streams(records, tolerances, primary)
    result["per_side"] = {side: aggregate_streams([r for r in records if r["side"] == side], tolerances, primary)
                          for side in ("left_wrist", "right_wrist")}
    result["per_subject"] = {subject: aggregate_streams([r for r in records if r["subject"] == subject],
                                                         tolerances, primary)
                             for subject in sorted({r["subject"] for r in records})}
    for tolerance in tolerances:
        f1 = [r["by_tolerance"][str(tolerance)]["segmentation"]["f1"]
              for r in result["per_subject"].values()
              if r["by_tolerance"][str(tolerance)]["segmentation"]["status"] == "scored"]
        result["by_tolerance"][str(tolerance)]["subject_macro_segmentation_f1"] = float(np.mean(f1)) if f1 else None
    return result
