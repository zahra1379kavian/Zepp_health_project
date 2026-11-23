"""Classification and one-to-one repetition scoring on certified coverage only."""

from bisect import bisect_right
from collections import Counter

import numpy as np
from scipy.optimize import linear_sum_assignment

from .constants import CLASSES, DT_MS, SET_GAP_MS
from .data import runs


POSITIONS = ("first", "second", "middle", "last")
DURATION_BINS_MS = (1000, 2000, 3000)
DURATION_BUCKETS = ("under_1s", "1_to_under_2s", "2_to_under_3s", "3s_and_over")
EXTRA_TYPES = ("inside_real_rep", "during_rest", "between_reps", "other_labeled_activity")
MISS_TYPES = ("overlapped_prediction", "no_overlapping_prediction")


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


class ReferenceTimeline:
    """Set and rest context from certified reference data on a single wrist."""

    def __init__(self, references, t, coverage, station_targets=None):
        ordered = sorted(references, key=lambda row: row["start_ms"])
        self.starts = [rep["start_ms"] for rep in ordered]
        self.ends = [rep["end_ms"] for rep in ordered]
        self.t = t
        self.station_targets = station_targets
        self.sets = []
        for rep in ordered:
            if self.sets:
                prior = self.sets[-1][-1]
                lo, hi = np.searchsorted(t, [prior["end_ms"], rep["start_ms"]])
                same_bout = coverage[lo:hi].all()
                if station_targets is not None:
                    same_bout &= np.all(station_targets[lo:hi] == station_targets[lo - 1])
                if (rep["station"] == prior["station"] and same_bout
                        and rep["start_ms"] - prior["end_ms"] <= SET_GAP_MS):
                    self.sets[-1].append(rep)
                    continue
            self.sets.append([rep])

    def extra_type(self, rep):
        midpoint = (rep["start_ms"] + rep["end_ms"]) // 2
        index = bisect_right(self.starts, midpoint) - 1
        if index >= 0 and midpoint < self.ends[index]:
            return "inside_real_rep"
        if any(group[0]["start_ms"] <= midpoint < group[-1]["end_ms"] for group in self.sets):
            return "between_reps"
        if self.station_targets is not None:
            frame = self.t.searchsorted(midpoint, side="right") - 1
            if self.station_targets[frame] != 0:
                return "other_labeled_activity"
        return "during_rest"


def has_overlap(rows, intervals):
    """Return whether each half-open interval overlaps any interval in the other list."""
    if not rows or not intervals:
        return np.zeros(len(rows), dtype=bool)
    starts, ends = np.array(sorted((r["start_ms"], r["end_ms"]) for r in intervals)).T
    first, last = np.array([(r["start_ms"], r["end_ms"]) for r in rows]).T
    preceding = np.searchsorted(starts, last, side="left") - 1
    return (preceding >= 0) & (np.maximum.accumulate(ends)[preceding] > first)


def tally(matched, total, count="matched", rate="recall"):
    return {count: matched, "total": total, rate: matched / total if total else None}


def repetition_diagnostics(predictions, references, pairs, timeline, observed_predictions):
    matched_predictions = {p for p, _ in pairs}
    matched_references = {r for _, r in pairs}
    extras = [row for i, row in enumerate(predictions) if i not in matched_predictions]
    misses = [row for i, row in enumerate(references) if i not in matched_references]
    extra_counts = Counter(timeline.extra_type(rep) for rep in extras)
    overlap_count = int(has_overlap(misses, observed_predictions).sum())

    detected = {rep["start_ms"]: i in matched_references for i, rep in enumerate(references)}
    positions = {key: [] for key in POSITIONS}
    durations = {key: [] for key in DURATION_BUCKETS}
    sets = {}
    for group in timeline.sets:
        # All members of a reference set share the exercise used for filtering.
        if group[0]["start_ms"] not in detected:
            continue
        hits = [detected[rep["start_ms"]] for rep in group]
        positions["first"].append(hits[0])
        positions["second"].extend(hits[1:2])
        positions["middle"].extend(hits[2:-1])
        positions["last"].append(hits[-1])
        sets.setdefault(group[0]["station"], []).append(not any(hits))
        for rep, hit in zip(group, hits):
            duration = rep["end_ms"] - rep["start_ms"]
            bucket = np.searchsorted(DURATION_BINS_MS, duration, side="right")
            durations[DURATION_BUCKETS[bucket]].append(hit)

    return {
        "extra_breakdown": {key: extra_counts[key] for key in EXTRA_TYPES},
        "missed_breakdown": dict(zip(MISS_TYPES, (overlap_count, len(misses) - overlap_count))),
        "detection_by_position": {key: tally(sum(hits), len(hits)) for key, hits in positions.items()},
        "detection_by_duration": {key: tally(sum(hits), len(hits)) for key, hits in durations.items()},
        "entire_sets_missed_by_exercise": {key: tally(sum(missed), len(missed), "missed", "rate")
                                          for key, missed in sorted(sets.items())},
    }


def report_delay_percentiles(delays):
    return {
        "report_delay_median_ms": float(np.median(delays)) if delays else None,
        "report_delay_p95_ms": float(np.percentile(delays, 95, method="inverted_cdf")) if delays else None,
        "report_delay_max_ms": float(max(delays)) if delays else None,
    }


def repetition_metrics(predictions, references, tolerance_ms=400, require_station=True, *,
                       timeline=None, observed_predictions=None):
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
            **report_delay_percentiles(delays),
            "matched_reported_before_1s_fraction": float(np.mean(np.array(delays) < 1000)) if pairs else None,
            "matched_early_report_fraction": float(np.mean(np.array(delays) < 0)) if pairs else None,
            "matched_reported_0_to_999ms_fraction": float(np.mean([(0 <= d < 1000) for d in delays])) if pairs else None,
            "start_errors_ms": start_errors, "end_errors_ms": end_errors, "report_delays_ms": delays,
            **(repetition_diagnostics(predictions, references, pairs, timeline,
                                      predictions if observed_predictions is None else observed_predictions)
               if timeline is not None else {})}


def exercise_predictions(predictions, references, t, coverage, tolerance):
    """Score matches plus unmatched predictions overlapping a selected exercise bout."""
    matched = {p for p, _ in match_repetitions(predictions, references, tolerance, require_station=False)}
    intervals = [(int(t[a]), int(t[b-1] + DT_MS)) for a, b in runs(coverage)]
    return [r for i, r in enumerate(predictions) if i in matched or
            any(r["start_ms"] < b and r["end_ms"] > a for a, b in intervals)]


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


def aggregate_diagnostics(results):
    if not all("extra_breakdown" in r for r in results):
        return {}
    summary = {
        field: {key: sum(r[field][key] for r in results) for key in keys}
        for field, keys in (("extra_breakdown", EXTRA_TYPES), ("missed_breakdown", MISS_TYPES))
    }
    for field, count, rate in (("detection_by_position", "matched", "recall"),
                               ("detection_by_duration", "matched", "recall"),
                               ("entire_sets_missed_by_exercise", "missed", "rate")):
        buckets = {}
        for result in results:
            for key, row in result[field].items():
                buckets.setdefault(key, []).append(row)
        summary[field] = {key: tally(sum(row[count] for row in rows), sum(row["total"] for row in rows),
                                     count, rate) for key, rows in buckets.items()}
    return summary


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
            **(report_delay_percentiles(delays) if all("report_delay_p95_ms" in r for r in results) else {}),
            "matched_reported_before_1s_fraction": float(np.mean(np.array(delays) < 1000)) if delays else None,
            "matched_early_report_fraction": float(np.mean(np.array(delays) < 0)) if delays else None,
            "matched_reported_0_to_999ms_fraction": float(np.mean([(0 <= d < 1000) for d in delays])) if delays else None,
            "count_mae_all_station_sessions": float(np.mean([r["absolute_error"] for r in counts])) if counts else None,
            "exact_count_accuracy_all_station_sessions": float(np.mean([r["exact"] for r in counts])) if counts else None,
            "count_mae_active_station_sessions": float(np.mean([r["absolute_error"] for r in active])) if active else None,
            "exact_count_accuracy_active_station_sessions": float(np.mean([r["exact"] for r in active])) if active else None,
            "count_rows": counts, **aggregate_diagnostics(results)}


def reporting_tolerances(values=None, primary=400):
    """Keep the selection tolerance and all declared reporting tolerances."""
    values = [200, 400] if values is None else values
    if primary <= 0 or not values or any(value <= 0 for value in values):
        raise ValueError("Reporting tolerances must be positive and nonempty")
    return sorted(set([primary, *values]))


def score_stream(predictions, references, t, coverage, metadata, tolerances, station_metrics=True, *,
                 station_targets=None):
    """Score an unchanged decoded stream at several endpoint tolerances."""
    scores = {}
    timeline = ReferenceTimeline(references, t, coverage, station_targets)
    for tolerance in tolerances:
        result = {}
        modes = ((True, "repetitions"), (False, "segmentation")) if station_metrics else ((False, "segmentation"),)
        for require_station, key in modes:
            scored, _ = covered_repetitions(predictions, t, coverage, references, tolerance, require_station)
            rows = count_rows(scored, references, metadata["session"], require_station)
            for row in rows:
                row.update({k: metadata[k] for k in ("source_session", "subject", "side")})
            result[key] = {"metrics": repetition_metrics(scored, references, tolerance, require_station,
                                                         timeline=timeline, observed_predictions=predictions),
                           "counts": rows}
            if require_station:
                result["per_exercise"] = {
                    name: repetition_metrics([r for r in scored if r["station"] == name],
                                             [r for r in references if r["station"] == name], tolerance,
                                             timeline=timeline, observed_predictions=predictions)
                    for name in CLASSES[1:] if name != "plank"}
        scores[str(tolerance)] = result
    return scores


def aggregate_streams(records, tolerances, primary=400, station_metrics=True):
    """Pool stream observations without treating paired wrists as extra subjects."""
    by_tolerance = {}
    keys = ("repetitions", "segmentation") if station_metrics else ("segmentation",)
    for tolerance in tolerances:
        scores = [r["repetition_scores"][str(tolerance)] for r in records
                  if str(tolerance) in r["repetition_scores"]]
        summary = {key: aggregate_repetitions([s[key]["metrics"] for s in scores],
                                               [row for s in scores for row in s[key]["counts"]])
                   for key in keys}
        if station_metrics:
            summary["per_exercise"] = {
                name: aggregate_repetitions([s["per_exercise"][name] for s in scores],
                                             [row for s in scores for row in s["repetitions"]["counts"]
                                              if row["station"] == name])
                for name in CLASSES[1:] if name != "plank"}
        baseline = [r["previous_end_scores"][str(tolerance)]["segmentation"] for r in records
                    if str(tolerance) in r.get("previous_end_scores", {})]
        if baseline:
            summary["previous_end"] = aggregate_repetitions(
                [s["metrics"] for s in baseline], [row for s in baseline for row in s["counts"]])
        by_tolerance[str(tolerance)] = summary
    result = {"streams": len(records), "subjects": len({r["subject"] for r in records}),
              "source_sessions": len({r["source_session"] for r in records}),
              "by_tolerance": by_tolerance, **by_tolerance[str(primary)]}
    if not station_metrics:
        matrices = [r["station_presence_confusion"] for r in records if "station_presence_confusion" in r]
        if matrices:
            cm = np.sum(matrices, axis=0)
            result["station_presence"] = {"classes": ["rest", "at_station"], "confusion_matrix": cm.tolist(),
                                          "recall": float(cm[1, 1] / max(cm[1].sum(), 1))}
        return result
    cm = sum((np.array(r["confusion_matrix"], dtype=np.int64) for r in records),
             np.zeros((len(CLASSES), len(CLASSES)), np.int64))
    hours = sum(r["rest_hours"] for r in records)
    events = sum(r["false_exercise_events"] for r in records)
    return {**result, "classification": classification_metrics(cm),
            "false_exercise_events": events, "rest_hours": hours,
            "false_exercise_events_per_rest_hour": events / hours if hours else None}


def grouped_stream_report(records, tolerances, primary=400, station_metrics=True):
    result = aggregate_streams(records, tolerances, primary, station_metrics)
    result["per_side"] = {side: aggregate_streams([r for r in records if r["side"] == side],
                                                  tolerances, primary, station_metrics)
                          for side in ("left_wrist", "right_wrist")}
    result["per_subject"] = {subject: aggregate_streams([r for r in records if r["subject"] == subject],
                                                         tolerances, primary, station_metrics)
                             for subject in sorted({r["subject"] for r in records})}
    for tolerance in tolerances:
        f1 = [r["by_tolerance"][str(tolerance)]["segmentation"]["f1"]
              for r in result["per_subject"].values()
              if r["by_tolerance"][str(tolerance)]["segmentation"]["status"] == "scored"]
        result["by_tolerance"][str(tolerance)]["subject_macro_segmentation_f1"] = float(np.mean(f1)) if f1 else None
    return result
