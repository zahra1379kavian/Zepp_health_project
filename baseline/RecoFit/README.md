# RecoFit-style baseline for the strength-training dataset

This implements the classical counter in Morris et al., [RecoFit: Using a
Wearable Sensor to Find, Recognize, and Count Repetitive Exercises (CHI 2014)](
https://dmorris.net/publications/Morris_Workout_CHI_2014.pdf), with a two-stage
exercise/rest and exercise-type SVM pipeline adapted to this project's **25 Hz
wrist data**, with an optional synchronized **wrist + waist** extension for
exercises with little wrist motion, such as leg raises. It needs no station ID
or session exercise list at inference.

The implementation lives under `baseline/RecoFit/`. It reuses the project's data
preparation and metrics modules, which require NumPy, pandas and SciPy, but
**does not import or require PyTorch**. Source recordings are preserved.

## Run on your existing prepared dataset

From the repository root:

```bash
python3 -m venv baseline/RecoFit/.venv
source baseline/RecoFit/.venv/bin/activate
python -m pip install -r baseline/requirements.txt

python -m baseline.RecoFit run \
  --data baseline/MS-TCN/prepared_datasets/separate_wrist_streams \
  --output baseline/RecoFit/runs/recofit_cv \
  --folds 5
```

The main report is `baseline/RecoFit/runs/recofit_cv/cross_validation/report.json`.
Each `fold_*/model/` contains `model.joblib` and `training.json`; each
`fold_*/test/` contains predictions, metrics and a fold report. Every subject
is tested once. Both wrists from the same subject stay together in training,
validation or testing. Paired wrists are separate sensor observations of the
same physical workout, so inspect `per_side` and `per_subject` reports too.

All output destinations must be new. To repeat an experiment, use another path.
`run` prepares data under its output directory if the supplied prepared dataset
does not exist. A directly invoked `cross-validate` writes `report.json` at its
output root. The commands run on CPU. Default fitting caps each endpoint class
and each pure recognition class at 3,000 sampled windows; evaluation uses every
eligible timestamp, without subsampling the held-out recordings.

`prepare` defaults to `baseline/RecoFit/data`; `demo` defaults to
`baseline/RecoFit/runs/demo`. Generated data, run outputs, models, and the local
virtual environment under `baseline/RecoFit/` are ignored by Git.

## Waist support for leg raises

Run the extension on the same subjects and wrist references:

```bash
python -m baseline.RecoFit run \
  --data baseline/MS-TCN/prepared_datasets/separate_wrist_streams \
  --output baseline/RecoFit/runs/recofit_waist_cv \
  --sensors wrist-waist --waist-stations leg_raise --folds 5
```

`run` automatically creates a paired dataset under its output's `data/` directory.
It copies the prepared wrist data and adds `waist_x.npy` and
`waist_available.npy` per stream. Original recordings and prepared inputs are
preserved. Waist acceleration and gyro are joined by exact `sync_timestamp` on
the 40 ms grid. No waist labels or magnetometer channels are read. Where multiple
`*waist*.csv` devices exist, the device with the most overlapping timestamps is
selected, with filename as a tie-breaker; devices are never stitched together.
The chosen filename and availability are recorded in the manifest. Missing or
non-overlapping devices and off-grid clocks raise an error. Recording tails
outside the wrist grid are ignored; missing internal samples are never interpolated.

Both classifiers concatenate independent wrist and waist features: **448
exercise/rest features and 120 recognition features**. This lets the detector
observe activity even when the wrist is still. For predicted `leg_raise` bouts,
the counter uses waist acceleration; other exercises use wrist acceleration.
`--waist-stations leg_raise crunch hip_thrust`, for example, selects additional
dynamic exercises for waist counting. This is a fixed configuration, not routing
from reference labels at inference. Devices are filtered separately to avoid
switching artifacts. Boundary offsets are fitted using the same configured
sensor and causal replay cadence used at inference, separately for each exercise,
reference wrist and PCA polarity axis. Training subjects alone supply calibration
and scaling; validation selects SVM C. A fresh **v3 paired model** is required.
Existing v2 wrist-only models and the default `--sensors wrist` remain supported.

A gap in either required sensor resets classification and counting. Paired models
require both sensors and have no wrist-only fallback. Evaluation keeps the
original wrist validity and reference coverage, so missing waist samples do not
remove difficult intervals from scoring. Reports include `sensors`,
`counting_sensors`, and `waist_unavailable_samples`; repetition and offline-peak
events include `counting_sensor`. Paired wrist streams still represent separate
observations of the same workout, even when they share a waist device.

For separate preparation and fitting:

```bash
python -m baseline.RecoFit prepare-waist \
  --data baseline/MS-TCN/prepared_datasets/separate_wrist_streams \
  --output baseline/RecoFit/runs/paired_data
python -m baseline.RecoFit train \
  --data baseline/RecoFit/runs/paired_data --sensors wrist-waist \
  --output baseline/RecoFit/runs/paired_model
```

`prepare-waist --source PATH` overrides the original recordings path from the
manifest. `train` and `cross-validate` require an already paired dataset when
`--sensors wrist-waist` is selected. For raw prediction, add
`--waist-input path/to/S1_waist.csv` to the existing `predict` command; Python
callers provide aligned `(time, 6)` `waist_x` and `waist_valid` to `infer`.
Waist inputs are mandatory for paired models and rejected by wrist-only models.

Compare the new held-out report with the wrist-only run using the same folds,
seed and training settings, especially `leg_raise` recognition, repetition recall,
count MAE and delay. Synthetic tests verify recognition and counting with flat
wrist signals, but do not establish an improvement on real subjects. Waist motion
can also be weak, and the existing classifier cold start and calibration limits
still apply.

## Prepare from the cleaned recordings

If the version-4 prepared dataset is unavailable, use:

```bash
python -m baseline.RecoFit prepare \
  --source strength_training/strength_training_data_clean \
  --output baseline/RecoFit/data

python -m baseline.RecoFit run --data baseline/RecoFit/data --output baseline/RecoFit/runs/recofit_cv
```

Preparation reuses `baseline.mstcn.data.prepare`:

- Join by `sync_timestamp`; retain missing-sample and exclusion masks.
- Read targets from each wrist's corresponding `label.csv` column; ignore
  embedded sensor labels, including duplicate headers.
- Apply the four whole-session exclusions, calibration/`Other` exclusions,
  and training/evaluation exclusion of `spull` from `plan.md`.
- Exclude additional sessions whose affected log intervals cannot be aligned
  safely, using the existing preparation policy. Retain all reasons in the
  manifest; this matches the temporal baseline's eligible cohort.
- Keep every available wrist as a separate stream. `--location left_wrist`
  and `--location right_wrist` are available for separate preparations.
- Reuse the repository's documented, previously validated **phase1 then
  phase2** repetition convention. Reject incomplete or gap-crossing cycles;
  treat plank as a duration. Reference conversion supplies training/evaluation
  targets only, never inference boundaries.

`prepare --references FILE` supports the existing side-specific external
reference schema. `--reference-mode none` is an explicit unscored diagnostic;
period limits then use fixed fallbacks when no training references exist.
Without reference calibration, offline peaks can still be counted, but the
counter does not produce repetition interval estimates.

## What follows the paper, and what is adapted

| Component | Implementation |
| --- | --- |
| Exercise/rest features | 28 statistics for each of 8 acceleration/gyro signals: 224 features |
| Exercise recognition features | 20 statistics for aX, aYZPC1 and gPC1: 60 features |
| Optional waist extension | Concatenate independently extracted wrist/waist features (448 / 120); waist counter for configured exercises, default `leg_raise` |
| Classifiers | StandardScaler + balanced L2 linear SVMs; a binary detector and a multiclass recognizer |
| Counting signal | 0.15–11 Hz elliptic bandpass of raw acceleration, mean subtraction, first PCA component |
| Candidate peaks | Amplitude-ranked rejection inside the minimum period |
| Period refinement | Local autocorrelation, then rejection inside 0.75 times the local period |
| Amplitude rejection | Reject peaks below half the 40th-percentile retained-peak amplitude |

Explicit adaptations and assumptions:

1. Data are 25 Hz, compared with 50 Hz in RecoFit. Classification uses a causal
   fourth-order 8 Hz Butterworth low-pass, and ten spectral bands from 0.1 Hz to
   12.5 Hz. It does not upsample or apply an impossible 20 Hz filter at 25 Hz.
   The counting filter is fourth-order elliptic, 0.1 dB ripple and 60 dB
   stopband attenuation. Both filters initialize each valid stretch at the
   first sample's steady state to avoid a gravity startup transient. Excluded
   or unavailable samples reset the state.
2. Device X replaces anatomical arm X. Wrist orientation is not guaranteed to
   match the original armband. Feature PCA signs use the largest loading's
   positive direction. Causal counting initializes the same way, then aligns
   each new PCA direction with the previous observed direction by their dot
   product. The initial loading axis remains the calibration anchor, even if
   another loading later becomes largest. Rest, station changes and sensor gaps
   reset this state. Offline counting uses one PCA projection per bout.
   Autocorrelations are mean-centered and normalized at lag zero.
   The unspecified feature-peak thresholds are fixed at 0.1 prominence and
   0.5 seconds separation. Counter AC context extends two maximum periods
   on either side of a peak, clipped to observations available so far.
3. Default classification uses **5-second trailing windows, updated every
   200 ms**. Training uses 1-second strides; binary targets are the current
   window-end class. Recognition trains only on windows fully within one
   exercise. A saturating accumulator requires 3 decisions (600 ms), replacing
   the paper's six seconds. Exercise-positive decisions vote for a single station
   across the active segment, including the decisions confirming its start.
   The initial station is the vote leader; ties retain the committed label.
   A different leader must persist for `--confirm-steps` consecutive
   exercise-positive decisions (3 by default) before the station changes.
   Brief recognition errors therefore preserve the set and counting buffer.
   Rest decisions add no votes and interrupt station-change confirmation;
   confirmed rest and sensor gaps clear all votes. A sustained exercise change
   without rest must first outvote the segment's accumulated history, so its
   recognition delay can exceed 600 ms. Predictions already emitted are never
   relabeled. All learned station types remain candidates; no workout circuit
   list is provided.
4. Feature scaling fits training subjects only. A shared SVM C is selected from
   0.1, 1 and 10 by macro F1 on capped validation windows. This selection is
   a window-level diagnostic; the final reports score continuous recordings.
   Period limits use the 1st/99th percentiles of **consecutive reference-start
   gaps**, expanded by 0.8/1.25, with a minimum of 240 ms. This includes pauses
   between repetitions. Pairs must belong to the same exercise bout and wrist
   stream, with continuous valid, reference-covered samples between starts;
   rest, another exercise, exclusions and sensor gaps break the pairing. Classes
   without usable pairs use pooled training start gaps, then fixed **0.4–10 second
   bounds** if no training pairs exist (0.5–8 seconds expanded by 0.8/1.25).
   `training_start_gaps` and `source` in `training.json` record the evidence
   behind each limit.
5. The paper counts peaks; it does not define individual repetition start/end
   times. Here **start and end offsets from the peak are calibrated separately
   on training references**, as fractions of the local period, then rounded to
   the 40 ms grid. Calibration is separate for each exercise, wrist and initial
   dominant PCA loading axis, whose positive sign anchors the peak phase. Opposite
   wrist phases are therefore not averaged together, and active duration need not equal
   start-to-start spacing. Calibration replays each labelled training bout with
   the same prediction cadence as causal inference (200 ms by default), starting
   with short prefixes and growing to buffers of at most 30 seconds. Both paths
   share the replay code, so their initial axis anchor and subsequent PCA sign
   alignment use the same observed samples; there is no extra call at bout end.
   Each reference contributes at most once per axis, and cycles containing
   multiple retained peaks are excluded from calibration. Median offsets and
   calibration counts are saved in `training.json` and the model. Refit existing
   models to replace offsets learned with the previous full-buffer schedule.
   A missing calibration has no symmetric-period fallback: its peaks have no estimated
   interval. Both counting modes expose `uncalibrated_stations` and
   `uncalibrated_peak_counts_by_station` in stream results and `report.json`.
   Reports also include `uncalibrated_streams_by_station` and `uncalibrated_streams`
   with each affected stream's station counts and wrist/subject metadata, including
   in pooled cross-validation reports. Causal counts apply peak spacing rejection
   so overlapping windows do not repeatedly count a peak. These diagnostics cover
   all valid predicted exercise samples, including samples without reference
   coverage; they count detected peaks, not verified complete intervals or extra
   scored misses. Learned offsets correct systematic phase shifts, but cannot track every change in
   wrist orientation, cadence or pause duration. Timing still needs held-out
   measurement; these are not the temporal model's phase-boundary predictions.

These corrections require newly fitted **v2 models**. Loading a v1 model asks
for retraining because its duration-based period limits and symmetric interval
estimates cannot be corrected from the saved classifiers alone.

This is a documented reimplementation and adaptation, not the authors' original
source or an exact reproduction of their reported experiment.

## Causal replay and offline counting

The default `--mode causal` recomputes the counter every 200 ms from at most
30 seconds of the current predicted exercise. It uses only samples already
observed, waits until a full estimated interval plus 200 ms has been observed,
and freezes emitted events. Nearby subsequent candidates cannot revise a
committed peak. Confirmed rest or station changes, missing samples, and exclusions
end a bout; incomplete tails do not generate artificial repetitions. Plank
produces holds only after an observed label change or gap closes the set.
Each hold includes `reported_at_ms`, the timestamp of that closing observation,
and `complete: true`. An active hold at the end of a causal prefix is not
reported. Offline holds summarize observed plank stretches, use the final
session timestamp as `reported_at_ms`, and mark a recording-truncated tail with
`complete: false`. Holds are stored separately from repetitions and are not scored.

`reported_at_ms` records when the event became available in sample-time replay.
The method has a five-second classifier cold start after a reset and needs
enough motion to estimate periodicity; initial repetitions can be missed.
**The implementation does not guarantee the less-than-one-second reporting
requirement.** Check measured delay relative to reference ends, together with
recall. Runtime includes total compute separately; timestamps exclude transport
delay and processing backlog. Causal replay is a CSV experiment, not a live
sensor driver.

For an offline counter comparison using the same trained classifiers:

```bash
python -m baseline.RecoFit evaluate \
  --data baseline/MS-TCN/prepared_datasets/separate_wrist_streams \
  --model baseline/RecoFit/runs/recofit_cv/cross_validation/fold_1/model/model.joblib \
  --mode offline \
  --output baseline/RecoFit/runs/recofit_fold1_offline
```

This counts each full predicted exercise bout using all its samples. It saves
`offline_peaks` and `offline_peak_counts_by_station` separately from estimated
complete intervals. The report's `offline_peak_counts` scores peaks whose
timestamps lie inside certified reference coverage and reports count MAE,
exact-count accuracy, and within-one accuracy. Those are count-only diagnostics;
the `repetitions` and `segmentation` sections always evaluate estimated intervals.
Offline events are marked available at the final session sample, so their delay
cannot be mistaken for online performance. Train a separate cross-validation
run with `--mode offline` to pool this diagnostic across all held-out subjects.

Both modes evaluate predicted station/rest labels. There is no inference mode
that receives reference station labels or repetition boundaries.

## Train once, evaluate and predict

For a diagnostic single subject split:

```bash
python -m baseline.RecoFit train \
  --data baseline/MS-TCN/prepared_datasets/separate_wrist_streams \
  --output baseline/RecoFit/runs/recofit_single/model

python -m baseline.RecoFit evaluate \
  --data baseline/MS-TCN/prepared_datasets/separate_wrist_streams \
  --model baseline/RecoFit/runs/recofit_single/model/model.joblib \
  --output baseline/RecoFit/runs/recofit_single/test

python -m baseline.RecoFit predict \
  --model baseline/RecoFit/runs/recofit_single/model/model.joblib \
  --input strength_training/strength_training_data_clean/20260602_LRFVFZEIVGAM/left_wrist.csv \
  --output baseline/RecoFit/runs/recofit_prediction
```

The prediction example illustrates the format; headline results must come from
held-out subject folds. Prediction reads only `sync_timestamp` and sensor
channels. Accelerometer and gyro channels are used; magnetometer channels are
ignored. Wrist CSV gaps are restored as unavailable grid samples and reset
processing. Raw prediction reads no labels or workout logs and is unscored.
It writes `predictions.json`, timestamped `events.jsonl`, and `frames.csv`.
The sensor wrist selects its training calibration. Prediction infers it from
`left_wrist.csv` / `right_wrist.csv` or a model trained on only one wrist; for
other filenames with a two-wrist model, provide `--side left_wrist` or
`--side right_wrist`. Python callers pass `side=` to `infer` or `predict`.

Each event contains `station`, `start_ms`, `end_ms`, `peak_ms`, `period_ms`,
`reported_at_ms`, and `boundary_method`. Intervals use `[start_ms, end_ms)`.
Calibrated events use `boundary_method: training_calibrated_peak_offsets`.
`counts_by_station` counts these estimated complete intervals; edge peaks can
therefore appear in offline peak counts without becoming complete intervals.
The method predicts known training station types and NULL. NULL is the project's
category-relative rest class; this is not an unknown-exercise detector.

## Reports and verification

The scoring routines are shared with the temporal baseline: station/rest macro
F1 and confusion matrices, false alarms during annotated rest, one-to-one
repetition detection, start/end errors, count MAE and exact-count accuracy,
per-exercise/subject/side metrics, and delay fractions at 200/400 ms endpoint
tolerances. `segmentation` ignores station agreement; `repetitions` requires it.
Missing and extra detections accompany timing errors. Reference-uncovered
predictions are not silently treated as incorrect reference repetitions.
The overview reports the percentage of annotated rest frames predicted as exercise.

Run the regression and integration checks:

```bash
python -m unittest discover -s baseline/RecoFit/tests -v
python -m baseline.RecoFit demo --output baseline/RecoFit/runs/demo
```

The tests check start-gap limits with pauses and bout/gap/coverage resets,
wrist/axis-specific boundary calibration, matching calibration/inference replay
at default and custom cadences, PCA sign continuity and bout resets,
causal hold closure and prefix equivalence, separate active durations, missing
calibration and dropped-peak report aggregation, legacy-model rejection,
regular/double peaks, local tempo changes,
constant inputs,
future independence, station-vote stability under brief/alternating errors,
sustained station changes, vote resets at rest/gaps, preserved repetition counts,
exclusion/gap resets, absent/incomplete cycles, phase-label
preparation, subject separation, training-only period fitting, raw label leakage,
multiple timing tolerances, and separate offline count metrics. The demo runs
three-fold cross-validation on synthetic data with paired wrists and then raw
sensor-only prediction. Synthetic results do not establish real accuracy.
The waist tests additionally cover exact-clock alignment, device selection without
stitching, quiet-wrist leg raises, matching training/inference sensor calibration,
held-out-target independence, causal prefixes, sensor gaps, raw paired prediction,
and retention of dropout penalties in scoring.

A capped fit was also checked on the existing real version-4 data, followed by
180 seconds of held-out replay in both modes. Generated smoke-test artifacts
are under `baseline/RecoFit/runs/real_smoke/` locally and excluded from version control.
The v2 period/boundary corrections were also checked with a new capped fit on
the 57 training streams and 180-second excerpts from two held-out streams in
both modes. This checks integration, not timing accuracy; some polarity axes
absent from calibration correctly produced no interval estimates.

## Five-fold results

The saved wrist-only and wrist + waist runs completed all five held-out folds
on the same 43 subjects and 77 wrist streams. The comparison bundle includes
the overview, recognition confusion matrices, exercise-level repetition scores,
and counting and reporting-time diagnostics. See the local
[figure index](figures/folds_1-5/README.md) and [PDF report](figures/folds_1-5/recofit_report.pdf).

From the repository root, regenerate the figures from the saved predictions.

```bash
.venv/bin/python -m baseline.RecoFit.plot_results \
  --compare-run baseline/RecoFit/runs/recofit_waist_cv/cross_validation \
  --compare-data baseline/RecoFit/runs/recofit_waist_cv/data \
  --folds 1 2 3 4 5 --output baseline/RecoFit/figures/folds_1-5
```
