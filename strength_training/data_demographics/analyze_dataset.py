"""Audit the cleaned dataset and export reproducible tables and static figures.

Run from any directory after installing requirements.txt:
    python3 strength_training/analysis/analyze_dataset.py

Input CSVs are read only. Durations use sync_timestamp; sensor clocks are not
assumed interchangeable. Labels are authoritative in each session's label.csv.
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


OUT = Path(__file__).resolve().parent
ROOT = OUT.parent / "strength_training_data_clean"
FIGURES = OUT / "figures"
TABLES = OUT / "tables"
DT_MS = 40
FEATURES = [f"{sensor}_{axis}" for sensor in ["acc", "gyro", "mag"] for axis in "xyz"]
DISPLAY = {
    "overhead_press_(ohp)": "Overhead press (OHP)",
    "pull-up": "Pull-up", "push-up": "Push-up",
}
PALETTE = {"rest": "#CBD5E1", "exercise": "#147D92", "transition": "#E7A33E"}


def base_label(label):
    return re.sub(r"_(phase[12]|transition)$", "", str(label))


def pretty(label):
    return DISPLAY.get(label, label.replace("_", " ").capitalize())


def save_table(name, rows):
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    frame.to_csv(TABLES / f"{name}.csv", index=False, float_format="%.6f")
    return frame


def runs(values):
    starts = np.r_[0, np.flatnonzero(values[1:] != values[:-1]) + 1]
    ends = np.r_[starts[1:], len(values)]
    return starts, ends, values[starts]


def save_figure(fig, name):
    fig.savefig(FIGURES / f"{name}.png", dpi=170, bbox_inches="tight", facecolor="white")
    fig.savefig(FIGURES / f"{name}.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def describe(values):
    a = np.array(values, dtype=float)
    missing = int((~np.isfinite(a)).sum())
    a = a[np.isfinite(a)]
    return {"n_valid": len(a), "n_missing": missing,
            "min": float(a.min()), "q25": float(np.percentile(a, 25)),
            "median": float(np.median(a)), "q75": float(np.percentile(a, 75)),
            "mean": float(a.mean()), "max": float(a.max())}


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    folders = sorted(p for p in ROOT.iterdir() if p.is_dir() and (p / "label.csv").exists())
    cleanup = pd.read_csv(ROOT / "cleanup_summary.csv").set_index("session")
    log = pd.read_csv(next(ROOT.glob("*Sheet1.csv")))
    log["session"] = pd.to_datetime(log.collect_date, format="%Y/%m/%d").dt.strftime("%Y%m%d") + "_" + log.masked_id
    kept_log = log[log.session.isin([p.name for p in folders])].copy()
    excluded_log = log[~log.session.isin([p.name for p in folders])].copy()
    demographics = kept_log.groupby("session", sort=True)[["masked_id", "gender", "age", "height", "weight", "bmi"]].first().reset_index()
    demographics = demographics.rename(columns={"masked_id": "subject_id", "height": "height_cm", "weight": "weight_kg"})
    demographic_conflicts = []
    for session, group in kept_log.groupby("session"):
        for col in ["gender", "age", "height", "weight", "bmi", "birth"]:
            if group[col].dropna().nunique() > 1:
                demographic_conflicts.append({"session": session, "field": col, "values": " | ".join(map(str, group[col].dropna().unique()))})
    save_table("subjects", demographics)

    session_rows, sensor_rows, interval_rows, station_rows, exact_rows, location_rows = [], [], [], [], [], []
    label_counts = Counter()
    phase_counts = Counter()
    coverage = {}
    timelines = {}
    quality_issues = []
    total_class_counts = Counter()
    log_alignment_rows = []
    inferred_offsets = []

    for number, folder in enumerate(folders, 1):
        print(f"[{number}/{len(folders)}] {folder.name}", flush=True)
        labels = pd.read_csv(folder / "label.csv", keep_default_na=False)
        t = labels.sync_timestamp.to_numpy(dtype=np.int64)
        n = len(t)
        differences = np.diff(t)
        if not np.all(differences == DT_MS):
            raise ValueError(f"Label grid is not strictly regular at 40 ms: {folder.name}")
        location_cols = [c for c in labels if c != "sync_timestamp"]
        base = {}
        phases = {}
        for col in location_cols:
            category = labels[col].astype("category")
            names = category.cat.categories.to_numpy()
            codes = category.cat.codes.to_numpy()
            base[col] = np.array([base_label(x) for x in names], dtype=object)[codes]
            phases[col] = np.array(["rest" if x == "rest" else "transition" if x.endswith("_transition") else "exercise" for x in names], dtype=object)[codes]
            for label, count in labels[col].value_counts().items():
                label_counts[label] += int(count)
                exact_rows.append({"session": folder.name, "location": col, "label": label, "samples": int(count), "seconds": int(count) * .04})
            counts = Counter(phases[col])
            location_rows.append({"session": folder.name, "location": col, "samples": n,
                                  "rest_samples": counts["rest"], "exercise_samples": counts["exercise"], "transition_samples": counts["transition"],
                                  "nonrest_pct": 100 * (n - counts["rest"]) / n})
        matrix = np.stack([base[c] for c in location_cols], axis=1)
        phase_matrix = np.stack([phases[c] for c in location_cols], axis=1)
        union = np.full(n, "rest", dtype=object)
        conflict = np.zeros(n, dtype=bool)
        for col in location_cols:
            current = base[col]
            conflict |= (union != "rest") & (current != "rest") & (union != current)
            union = np.where(current != "rest", current, union)
        union[conflict] = "conflict"
        state = np.where(np.any(phase_matrix == "exercise", axis=1), "exercise",
                         np.where(np.any(phase_matrix == "transition", axis=1), "transition", "rest"))
        class_counts = Counter(state)
        total_class_counts.update(class_counts)
        phase_counts.update({"conflict": int(conflict.sum())})
        label_disagreement = np.any(labels[location_cols].to_numpy() != labels[location_cols[0]].to_numpy()[:, None], axis=1)
        binary_matrix = matrix != "rest"
        binary_disagreement = binary_matrix.any(axis=1) != binary_matrix.all(axis=1)
        station_names = sorted(set(matrix.ravel()) - {"rest"})
        if not station_names:
            quality_issues.append({"session": folder.name, "file": "label.csv", "issue": "all-rest labels",
                                   "detail": "Every available location contains rest only; review against workout log before using as a negative session."})
        coverage[folder.name] = {}
        for station in station_names:
            present = np.any(matrix == station, axis=1)
            coverage[folder.name][station] = int(present.sum()) * .04 / 60
            station_rows.append({"session": folder.name, "station": station, "samples_any_location": int(present.sum()),
                                 "minutes_any_location": int(present.sum()) * .04 / 60})
        start, end, names = runs(union)
        for a, b, name in zip(start, end, names):
            interval_rows.append({"session": folder.name, "station": name, "start_ms": int(t[a]),
                                  "end_exclusive_ms": int(t[b - 1] + DT_MS), "start_min": a * .04 / 60,
                                  "duration_s": (b - a) * .04})
        timelines[folder.name] = (t, union, state)

        # Infer the log's hour offset from agreement with named exercise labels.
        # Store all scores; accept an offset only when >50% of valid named-workout
        # interval samples match their station and the winner exceeds runner-up by .1.
        session_log = kept_log[kept_log.session == folder.name]
        valid_intervals = []
        for _, row in session_log.iterrows():
            station = str(row.scene_id).lower().replace(" ", "_")
            try:
                date = pd.Timestamp(row.collect_date)
                a = (date + pd.to_timedelta(row.device_start_time)).value // 1_000_000
                b = (date + pd.to_timedelta(row.device_end_time)).value // 1_000_000
            except (ValueError, TypeError):
                continue
            if b <= a:
                continue
            valid_intervals.append((station, a, b))
        scores = []
        for hours in range(-12, 15):
            same, total = 0, 0
            for station, a, b in valid_intervals:
                if station not in station_names:
                    continue
                x, y = np.searchsorted(t, [a - hours * 3600000, b - hours * 3600000])
                total += max(1, int(round((b-a) / DT_MS)))
                same += int(np.count_nonzero(union[x:y] == station))
            scores.append((same / total if total else 0, hours))
        scores.sort(reverse=True)
        best_score, best_offset = scores[0]
        accepted = best_score > .5 and best_score - scores[1][0] > .1
        inferred_offsets.append({"session": folder.name, "offset_hours": best_offset if accepted else None,
                                 "named_station_match_fraction": best_score, "runner_up_fraction": scores[1][0]})
        if accepted:
            for station, a, b in valid_intervals:
                x, y = np.searchsorted(t, [a - best_offset * 3600000, b - best_offset * 3600000])
                count = y - x
                log_alignment_rows.append({"session": folder.name, "scene": station, "log_duration_s": (b-a)/1000,
                                           "overlap_samples": count, "rest_samples": int(np.count_nonzero(state[x:y] == "rest")),
                                           "matching_station_samples": int(np.count_nonzero(union[x:y] == station))})

        idx = pd.Index(t)
        selected_waist = str(cleanup.loc[folder.name, "selected_waist"]) + ".csv"
        session_sensor_rows = []
        for path in sorted(folder.glob("*.csv")):
            if path.name == "label.csv":
                continue
            with path.open() as handle:
                header = next(csv.reader(handle))
            duplicate_headers = [key for key, count in Counter(header).items() if count > 1]
            sensor = pd.read_csv(path)
            st = sensor.sync_timestamp.to_numpy(dtype=np.int64)
            delta = np.diff(st)
            aligned = idx.get_indexer(st)
            valid = aligned >= 0
            loc = "waist" if "waist" in path.name else path.stem
            values = sensor[FEATURES].to_numpy(dtype=float)
            record = {
                "session": folder.name, "file": path.name, "location": loc,
                "is_selected_waist": path.name == selected_waist, "rows": len(sensor),
                "start_ms": int(st[0]), "end_ms": int(st[-1]), "duration_min": (st[-1]-st[0])/60000,
                "median_step_ms": float(np.median(delta)), "max_step_ms": int(delta.max()),
                "duplicate_timestamps": int(np.count_nonzero(delta == 0)), "backward_steps": int(np.count_nonzero(delta < 0)),
                "gaps_over_40ms": int(np.count_nonzero(delta > DT_MS)),
                "missing_grid_slots_internal": int(np.maximum(np.rint(delta / DT_MS).astype(int)-1, 0).sum()),
                "off_label_grid_rows": int((~valid).sum()),
                "timeline_coverage_pct": 100 * len(np.unique(aligned[valid])) / n,
                "missing_or_nonfinite_feature_cells": int((~np.isfinite(values)).sum()),
                "nonfinite_feature_rows": int((~np.isfinite(values)).any(axis=1).sum()),
                "duplicate_header_names": " | ".join(duplicate_headers),
                "timestamp_offset_median_ms": float(np.median(sensor.timestamp.to_numpy() - st)),
            }
            active = state != "rest"
            strict = state == "exercise"
            covered = np.zeros(n, dtype=bool)
            covered[aligned[valid]] = True
            record["union_nonrest_coverage_pct"] = 100 * np.count_nonzero(covered & active) / np.count_nonzero(active) if active.any() else np.nan
            record["strict_exercise_coverage_pct"] = 100 * np.count_nonzero(covered & strict) / np.count_nonzero(strict) if strict.any() else np.nan
            label_columns = [c for c in sensor if c == "labels" or c.startswith("labels.")]
            record["embedded_label_columns"] = len(label_columns)
            record["embedded_label_mismatches"] = 0
            for col in label_columns:
                record["embedded_label_mismatches"] += int(np.count_nonzero(sensor[col].to_numpy()[valid] != labels[loc].to_numpy()[aligned[valid]]))
            sensor_rows.append(record)
            session_sensor_rows.append(record)
            if duplicate_headers:
                quality_issues.append({"session": folder.name, "file": path.name, "issue": "duplicate CSV header", "detail": " | ".join(duplicate_headers)})
            if record["missing_or_nonfinite_feature_cells"]:
                quality_issues.append({"session": folder.name, "file": path.name, "issue": "missing/nonfinite sensor features", "detail": record["missing_or_nonfinite_feature_cells"]})

        row = {"session": folder.name, "subject_id": folder.name.split("_",1)[1], "date": folder.name[:8],
               "label_rows": n, "duration_min": (t[-1]-t[0])/60000, "sample_duration_min": n*.04/60,
               "n_stations": len(station_names), "rest_min": class_counts["rest"]*.04/60,
               "exercise_min": class_counts["exercise"]*.04/60, "transition_min": class_counts["transition"]*.04/60,
               "nonrest_pct": 100*(n-class_counts["rest"])/n,
               "label_disagreement_samples": int(label_disagreement.sum()),
               "binary_disagreement_samples": int(binary_disagreement.sum()),
               "conflicting_station_samples": int(conflict.sum()), "n_sensor_files": len(session_sensor_rows),
               "n_waist_files": sum(r["location"]=="waist" for r in session_sensor_rows),
               "has_left_wrist": "left_wrist" in location_cols, "has_right_wrist": "right_wrist" in location_cols,
               "log_rows": len(session_log), "selected_waist": selected_waist,
               "minimum_sensor_timeline_coverage_pct": min(r["timeline_coverage_pct"] for r in session_sensor_rows),
               "minimum_sensor_union_nonrest_coverage_pct": min(r["union_nonrest_coverage_pct"] for r in session_sensor_rows)}
        session_rows.append(row)
        if abs(row["duration_min"]*60-float(cleanup.loc[folder.name,"label_duration_s"])) > .001:
            raise ValueError(f"Cleanup duration mismatch: {folder.name}")

    sessions = save_table("sessions", session_rows)
    save_table("sessions_requiring_label_review", sessions[sessions.n_stations == 0])
    sensors = save_table("sensor_audit", sensor_rows)
    intervals = save_table("label_intervals", interval_rows)
    station_session = save_table("station_by_session", station_rows)
    save_table("label_counts_by_location", exact_rows)
    locations = save_table("location_balance", location_rows)
    save_table("quality_issues", quality_issues)
    save_table("log_clock_alignment", inferred_offsets)
    log_alignment = save_table("log_interval_label_agreement", log_alignment_rows)
    stations = station_session.groupby("station").agg(sessions=("session","nunique"), minutes=("minutes_any_location","sum"), samples=("samples_any_location","sum")).reset_index()
    stations["coverage_pct"] = 100 * stations.sessions / len(sessions)
    stations = stations.sort_values("minutes", ascending=False)
    save_table("stations", stations)
    pd.DataFrame({"label": list(label_counts), "samples_pooled_locations": list(label_counts.values())}).sort_values("label").to_csv(TABLES / "label_vocabulary.csv",index=False)
    scenes = kept_log.groupby("scene_id").size().rename("log_rows").reset_index()
    scenes["present_in_label_vocabulary"] = scenes.scene_id.map(lambda x: x.lower().replace(" ","_") in set(stations.station))
    save_table("workout_log_scenes", scenes)
    save_table("workout_log_excluded_sessions", excluded_log.groupby(["session","masked_id"]).size().rename("log_rows").reset_index())

    # Aggregate demographic summaries at participant level, never at log-row level.
    summary = {
        "dataset": str(ROOT.relative_to(OUT.parent.parent)),
        "subjects": int(sessions.subject_id.nunique()), "sessions": len(sessions),
        "sessions_per_subject": sessions.groupby("subject_id").size().value_counts().to_dict(),
        "collection_dates": sorted(sessions.date.unique().tolist()),
        "total_hours_endpoint": float(sessions.duration_min.sum()/60),
        "total_hours_sample_support": float(sessions.sample_duration_min.sum()/60),
        "session_duration_min": describe(sessions.duration_min),
        "label_rows": int(sessions.label_rows.sum()), "label_rate_hz": 1000/DT_MS,
        "sensor_files": len(sensors), "sensor_rows": int(sensors.rows.sum()),
        "total_csv_files": len(list(ROOT.rglob("*.csv"))),
        "total_bytes": sum(p.stat().st_size for p in ROOT.rglob("*") if p.is_file()),
        "station_count": len(stations), "exact_label_count_including_rest": len(label_counts),
        "stations_per_session": describe(sessions.n_stations),
        "sessions_with_exercise_labels": int((sessions.n_stations > 0).sum()),
        "all_rest_sessions": sessions.loc[sessions.n_stations == 0, ["session", "duration_min", "log_rows"]].to_dict(orient="records"),
        "all_rest_hours": float(sessions.loc[sessions.n_stations == 0, "duration_min"].sum()/60),
        "classes_any_location": {k:{"samples":int(v),"hours":v*.04/3600,"pct":100*v/sessions.label_rows.sum()} for k,v in total_class_counts.items()},
        "demographics": {"gender": demographics.gender.value_counts().to_dict(), **{key:describe(demographics[key]) for key in ["age","height_cm","weight_kg","bmi"]}},
        "demographic_conflicts": demographic_conflicts,
        "workout_log_rows_all": len(log), "workout_log_rows_matched": len(kept_log),
        "workout_log_subjects_all": int(log.masked_id.nunique()), "workout_log_rows_excluded": len(excluded_log),
        "label_disagreement_samples": int(sessions.label_disagreement_samples.sum()),
        "binary_disagreement_samples": int(sessions.binary_disagreement_samples.sum()),
        "conflicting_station_samples": int(sessions.conflicting_station_samples.sum()),
        "duplicate_header_files": int(sensors.duplicate_header_names.astype(bool).sum()),
        "missing_nonfinite_feature_cells": int(sensors.missing_or_nonfinite_feature_cells.sum()),
        "sensor_duplicate_timestamps": int(sensors.duplicate_timestamps.sum()),
        "sensor_backward_timestamp_steps": int(sensors.backward_steps.sum()),
        "sensor_internal_missing_grid_slots": int(sensors.missing_grid_slots_internal.sum()),
        "sensor_off_label_grid_rows": int(sensors.off_label_grid_rows.sum()),
        "embedded_label_mismatches": int(sensors.embedded_label_mismatches.sum()),
        "sensor_timeline_coverage_pct": describe(sensors.timeline_coverage_pct),
        "sensor_union_nonrest_coverage_pct": describe(sensors.union_nonrest_coverage_pct),
        "sensor_strict_exercise_coverage_pct": describe(sensors.strict_exercise_coverage_pct),
        "log_clock_offset_counts": dict(Counter("unresolved" if r["offset_hours"] is None else str(r["offset_hours"]) for r in inferred_offsets)),
    }
    rest_intervals = intervals[intervals.station == "rest"].duration_s
    active_intervals = intervals[intervals.station != "rest"].duration_s
    summary["contiguous_nonrest_station_runs"] = {"count":len(active_intervals), "duration_s":describe(active_intervals)}
    summary["contiguous_rest_runs"] = {"count":len(rest_intervals), "duration_s":describe(rest_intervals)}
    if len(log_alignment):
        other = log_alignment.groupby("scene").agg(intervals=("scene","size"), log_duration_s=("log_duration_s","sum"), overlap_samples=("overlap_samples","sum"), rest_samples=("rest_samples","sum"), matching_station_samples=("matching_station_samples","sum"))
        other["rest_pct_of_overlapping_samples"] = 100*other.rest_samples/other.overlap_samples.replace(0,np.nan)
        save_table("log_scene_label_agreement",other.reset_index())
        summary["unrepresented_log_scenes"] = other.loc[other.index.isin(["action","spull","other"])].reset_index().to_dict(orient="records")
    with (OUT / "summary.json").open("w") as handle:
        json.dump(summary,handle,indent=2,allow_nan=False,default=lambda x: x.item() if isinstance(x,np.generic) else str(x))

    make_figures(sessions, sensors, stations, station_session, demographics, intervals, timelines, summary)
    print(json.dumps(summary, indent=2, default=str),flush=True)


def make_figures(sessions, sensors, stations, station_session, people, intervals, timelines, summary):
    plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":10,
                         "axes.spines.top":False,"axes.spines.right":False,
                         "axes.titleweight":"bold","figure.dpi":130,
                         "axes.labelcolor":"#334155","text.color":"#172B40"})
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9), layout="constrained")
    fig.suptitle(f"Strength training dataset  |  {summary['subjects']} participants · {summary['sessions']} sessions · {summary['total_hours_endpoint']:.1f} hours",fontsize=18,weight="bold")
    ax = axes[0,0]
    ax.hist(sessions.duration_min,bins=np.arange(0,181,15),color="#147D92",edgecolor="white")
    median = sessions.duration_min.median()
    ax.axvline(median,color="#C7771E",ls="--",label=f"Median {median:.1f} min")
    ax.set(xlabel="Session duration (minutes)",ylabel="Sessions",title="Most sessions last around 1–2 hours")
    ax.legend(frameon=False)
    ax = axes[0,1]
    daily = sessions.groupby("date").size()
    ax.bar(np.arange(len(daily)),daily.values,color="#45799D")
    ax.set_xticks(np.arange(len(daily)),[f"{str(d)[4:6]}/{str(d)[6:]}" for d in daily.index],rotation=45)
    ax.set(xlabel="Collection date (2026)",ylabel="Sessions",title="Collection spans June 2–17")
    ax = axes[1,0]
    classes = ["rest","exercise","transition"]
    values = [summary["classes_any_location"][c]["hours"] for c in classes]
    bars = ax.barh(classes[::-1],values[::-1],color=[PALETTE[c] for c in classes[::-1]])
    for bar,c in zip(bars,classes[::-1]):
        v=summary["classes_any_location"][c]
        ax.text(bar.get_width()+.5,bar.get_y()+bar.get_height()/2,f"{v['hours']:.2f} h  ({v['pct']:.1f}%)",va="center",fontsize=10)
    ax.set(xlabel="Sample-supported hours (each time point counted once)",title="Annotated rest dominates the timeline",xlim=(0,max(values)*1.4))
    ax = axes[1,1]
    sensor_config = sessions.groupby(["n_waist_files","has_left_wrist","has_right_wrist"]).size()
    names=[]
    for waist,left,right in sensor_config.index:
        wrist = "both wrists" if left and right else "left wrist" if left else "right wrist"
        names.append(f"{waist} waist + {wrist}")
    bars=ax.barh(names,sensor_config.values,color="#527A69")
    for bar in bars: ax.text(bar.get_width()+.3,bar.get_y()+bar.get_height()/2,str(int(bar.get_width())),va="center")
    ax.set(xlabel="Sessions",title="Sensor availability varies",xlim=(0,sensor_config.max()+5))
    fig.supxlabel("Exercise = phase labels / plank at any location; transition = transition-only samples; rest = all available locations say rest.",fontsize=9)
    save_figure(fig,"01_dataset_overview")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), layout="constrained")
    fig.suptitle("Participant profile  |  one record per cleaned-data participant",fontsize=17,weight="bold")
    genders=people.gender.value_counts()
    bars=axes[0].bar(genders.index,genders.values,color=["#45799D","#BD6E78"][:len(genders)])
    for b in bars: axes[0].text(b.get_x()+b.get_width()/2,b.get_height()+.5,f"{int(b.get_height())}",ha="center")
    axes[0].set(ylabel="Participants",title="Gender recorded in workout log",ylim=(0,genders.max()+5))
    axes[1].hist(people.age,bins=np.arange(15,61,5),color="#147D92",edgecolor="white")
    axes[1].set(xlabel="Age (years)",ylabel="Participants",title=f"Age: {people.age.min():.0f}–{people.age.max():.0f}; median {people.age.median():.0f}")
    axes[2].hist(people.bmi,bins=np.arange(15,41,2.5),color="#C38E48",edgecolor="white")
    axes[2].set(xlabel="BMI (kg/m², as recorded)",ylabel="Participants",title=f"BMI: median {people.bmi.median():.1f}")
    save_figure(fig,"02_participant_profile")

    fig, axes=plt.subplots(1,2,figsize=(14,7),layout="constrained",sharey=True)
    fig.suptitle(f"{len(stations)} annotated exercise types  |  coverage and labeled time",fontsize=17,weight="bold")
    order=stations.sort_values("minutes")
    y=np.arange(len(order))
    axes[0].barh(y,order.sessions,color="#45799D")
    axes[0].set_yticks(y,[pretty(s) for s in order.station])
    for i,v in enumerate(order.sessions): axes[0].text(v+.4,i,str(v),va="center",fontsize=9)
    axes[0].set(xlabel="Sessions containing this exercise",xlim=(0,56),title="Not every session contains every exercise")
    axes[1].barh(y,order.minutes,color="#147D92")
    for i,v in enumerate(order.minutes): axes[1].text(v+.7,i,f"{v:.1f}",va="center",fontsize=9)
    axes[1].set(xlabel="Annotated minutes, including transition labels",xlim=(0,order.minutes.max()*1.17),title="Each time point counted once across locations")
    save_figure(fig,"03_exercise_coverage")

    matrix=station_session.pivot(index="session",columns="station",values="minutes_any_location").fillna(0)
    matrix=matrix.reindex(index=sessions.session,columns=stations.station.tolist()).fillna(0)
    fig,ax=plt.subplots(figsize=(14,14),layout="constrained")
    im=ax.imshow(matrix.to_numpy(),aspect="auto",cmap="YlGnBu",interpolation="nearest")
    ax.set_xticks(np.arange(len(matrix.columns)),[pretty(x) for x in matrix.columns],rotation=50,ha="right")
    ax.set_yticks(np.arange(len(matrix)),[s.split('_',1)[1] for s in matrix.index],fontsize=8)
    ax.set(title="Exercise coverage by participant / session",ylabel="Participant ID (one session each)")
    fig.colorbar(im,ax=ax,label="Annotated minutes (0 = absent)",shrink=.55)
    save_figure(fig,"04_subject_exercise_matrix")

    fig,axes=plt.subplots(2,1,figsize=(14,6),layout="constrained")
    median=sessions.duration_min.median()
    selected=[sessions.iloc[(sessions.duration_min-median).abs().argmin()].session,
              sessions.iloc[sessions.duration_min.argmin()].session]
    for ax,key in zip(axes,selected):
        t,union,state=timelines[key]
        binary=np.where(state=="rest",0,np.where(state=="transition",2,1))
        ax.imshow(binary[None,:],aspect="auto",extent=[0,len(t)*.04/60,0,1],interpolation="nearest",cmap=ListedColormap([PALETTE["rest"],PALETTE["exercise"],PALETTE["transition"]]),vmin=0,vmax=2)
        ax.set(yticks=[],xlabel="Minutes from recording start")
        ax.set_title(f"{key}  |  {(t[-1]-t[0])/60000:.1f} min",loc="left",pad=8)
        for station,group in intervals[(intervals.session==key)&(intervals.station!="rest")].groupby("station"):
            midpoint=(group.start_min*group.duration_s).sum()/group.duration_s.sum()
            ax.text(midpoint,1.15,pretty(station),rotation=55,ha="left",va="bottom",fontsize=8)
        if not np.any(state != "rest"):
            ax.text(len(t)*.04/120,1.35,"All locations contain rest only — review against the workout log",ha="center",fontsize=10,color="#8E5B23")
        ax.set_ylim(0,3)
    fig.suptitle("Annotation timelines  |  median-length and shortest sessions",fontsize=17,weight="bold")
    fig.legend(handles=[Patch(facecolor=PALETTE[x],label=x.capitalize()) for x in ["rest","exercise","transition"]],loc="outside lower center",ncols=3,frameon=False)
    save_figure(fig,"05_example_timelines")

    fig,axes=plt.subplots(1,2,figsize=(14,5.5),layout="constrained")
    fig.suptitle("Label definitions and sensor coverage affect the baseline",fontsize=17,weight="bold")
    labels=["Waist", "Left wrist", "Right wrist", "Any location"]
    values=[]
    common=sessions[sessions.has_left_wrist & sessions.has_right_wrist]
    frame=pd.read_csv(TABLES / "location_balance.csv")
    frame=frame[frame.session.isin(common.session)]
    for loc in ["waist","left_wrist","right_wrist"]:
        local=frame[frame.location==loc]
        values.append(100*(local.exercise_samples+local.transition_samples).sum()/local.samples.sum())
    values.append(100*(common.exercise_min+common.transition_min).sum()/common.sample_duration_min.sum())
    axes[0].bar(labels,values,color=["#45799D"]*3+["#147D92"])
    for i,v in enumerate(values): axes[0].text(i,v+.2,f"{v:.2f}%",ha="center")
    axes[0].set(ylabel="Non-rest label share (%)",title=f"Same {len(common)} sessions with both wrists",ylim=(0,max(values)*1.2))
    axes[1].scatter(sensors.timeline_coverage_pct,sensors.union_nonrest_coverage_pct,c=["#45799D" if x=="waist" else "#C38E48" for x in sensors.location],s=28,alpha=.65)
    axes[1].set(xlabel="Coverage of full label timeline (%)",ylabel="Coverage of union non-rest samples (%)",title=f"{len(sensors)} sensor files audited")
    axes[1].legend(handles=[Patch(color="#45799D",label="Waist"),Patch(color="#C38E48",label="Wrist")],frameon=False)
    save_figure(fig,"06_label_and_sensor_quality")


if __name__ == "__main__":
    main()
