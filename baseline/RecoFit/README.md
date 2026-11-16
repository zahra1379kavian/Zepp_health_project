# RecoFit-style baseline for the strength-training dataset

This implements the classical counter in Morris et al., [RecoFit: Using a
Wearable Sensor to Find, Recognize, and Count Repetitive Exercises (CHI 2014)](
https://dmorris.net/publications/Morris_Workout_CHI_2014.pdf), with a two-stage
exercise/rest and exercise-type SVM pipeline adapted to this project's **25 Hz
wrist data**. It needs no station ID or session exercise list at inference.

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
  --data strength_training/tas_data_both_wrists \
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

## Prepare from the cleaned recordings

If the version-4 prepared dataset is unavailable, use:

```bash
python -m baseline.RecoFit prepare \
  --source strength_training/strength_training_data_clean \
  --output baseline/RecoFit/data

python -m baseline.RecoFit run --data baseline/RecoFit/data --output baseline/RecoFit/runs/recofit_cv
```

Preparation reuses `strength_training.temporal_segmentation.data.prepare`:

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

## What follows the paper, and what is adapted

| Component | Implementation |
| --- | --- |
| Exercise/rest features | 28 statistics for each of 8 acceleration/gyro signals: 224 features |
| Exercise recognition features | 20 statistics for aX, aYZPC1 and gPC1: 60 features |
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
   match the original armband. PCA signs use the largest loading's positive
   direction. Autocorrelations are mean-centered and normalized at lag zero.
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
   list is provided. Existing model files use this smoothing without retraining.
4. Feature scaling fits training subjects only. A shared SVM C is selected from
   0.1, 1 and 10 by macro F1 on capped validation windows. This selection is
   a window-level diagnostic; the final reports score continuous recordings.
   Period limits use training reference-duration 1st/99th percentiles, expanded
   by 0.8/1.25, with a minimum of 240 ms. Missing classes use pooled training
   durations, then fixed 0.5–8 second defaults if there are no references.
5. The paper counts peaks; it does not define individual repetition start/end
   times. Here **start/end are estimated as peak ± half the local period**,
   rounded to the 40 ms grid. These estimates need measurement against your
   phase-derived references. They cannot establish completion or reliably
   resolve pauses. They are not the phase-boundary output of the temporal model.

This is a documented reimplementation and adaptation, not the authors' original
source or an exact reproduction of their reported experiment.

## Causal replay and offline counting

The default `--mode causal` recomputes the counter every 200 ms from at most
30 seconds of the current predicted exercise. It uses only samples already
observed, waits until a full estimated interval plus 200 ms has been observed,
and freezes emitted events. Nearby subsequent candidates cannot revise a
committed peak. Confirmed rest or station changes, missing samples, and exclusions
end a bout; incomplete tails do not generate artificial repetitions. Plank
produces holds.

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
  --data strength_training/tas_data_both_wrists \
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
  --data strength_training/tas_data_both_wrists \
  --output baseline/RecoFit/runs/recofit_single/model

python -m baseline.RecoFit evaluate \
  --data strength_training/tas_data_both_wrists \
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

Each event contains `station`, `start_ms`, `end_ms`, `peak_ms`, `period_ms`,
`reported_at_ms`, and `boundary_method`. Intervals use `[start_ms, end_ms)`.
`counts_by_station` counts these estimated complete intervals; edge peaks can
therefore appear in offline peak counts without becoming complete intervals.
The method predicts known training station types and NULL. NULL is the project's
category-relative rest class; this is not an unknown-exercise detector.

## Reports and verification

The scoring routines are shared with the temporal baseline: station/rest macro
F1 and confusion matrices, false exercise events per rest hour, one-to-one
repetition detection, start/end errors, count MAE and exact-count accuracy,
per-exercise/subject/side metrics, and delay fractions at 200/400 ms endpoint
tolerances. `segmentation` ignores station agreement; `repetitions` requires it.
Missing and extra detections accompany timing errors. Reference-uncovered
predictions are not silently treated as incorrect reference repetitions.

Run the regression and integration checks:

```bash
python -m unittest baseline.RecoFit.tests.test_recofit -v
python -m baseline.RecoFit demo --output baseline/RecoFit/runs/demo
```

The tests check regular/double peaks, local tempo changes, constant inputs,
future independence, station-vote stability under brief/alternating errors,
sustained station changes, vote resets at rest/gaps, preserved repetition counts,
exclusion/gap resets, absent/incomplete cycles, phase-label
preparation, subject separation, training-only period fitting, raw label leakage,
multiple timing tolerances, and separate offline count metrics. The demo runs
three-fold cross-validation on synthetic data with paired wrists and then raw
sensor-only prediction. Synthetic results do not establish real accuracy.

A capped fit was also checked on the existing real version-4 data, followed by
180 seconds of held-out replay in both modes. Generated smoke-test artifacts
are under `baseline/RecoFit/runs/real_smoke/` locally and excluded from version control.
The full five-fold real-data experiment has not been run as part of implementation.
