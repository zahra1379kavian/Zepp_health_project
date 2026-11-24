# Leave-one-exercise-out MS-TCN

## Step 2 paired exercise evaluation

The test exercise and its paired selection exercise are both excluded from
training, including normalization and input history. Validation subjects supply
the selection exercise for checkpoint and decoder selection at ±400 ms. The
test exercise is masked throughout validation. At test time the model replays
each wrist session continuously, then the scorer uses labels to attribute reps.

| Test exercise | Validation exercise |
| --- | --- |
| biceps_curls | bench_press |
| lat_pulldown | lunges |
| push-up | rowing |
| overhead_press_(ohp) | deadlift |
| squat | pull-up |
| hip_thrust | dips |

Use `--exercises screen` for biceps curls, push-up and squat on subject fold 1.
Use `--exercises confirm` for all six pairs. The paired report includes the
mean unseen F1@400 and the following guard rails for each fitted model.

- Pooled F1@400 on exercises seen in training, and a separate crunch F1@400.
- Extra reps classified as rest by the full-session scorer, divided by
  certified rest hours across the test wrist streams.
- First-rep recall across all sets and separately on the test exercise.

`--rep-inputs gravity` uses six channels. A causal 0.3 Hz first-order filter
estimates gravity from acceleration. Each continuous wrist stream starts the
filter from its first sample. Linear acceleration is measured relative to
this estimate. The inputs are total acceleration magnitude, signed linear
acceleration along gravity, perpendicular linear acceleration magnitude,
total gyroscope magnitude, signed angular velocity about gravity, and
perpendicular angular velocity magnitude. `--rep-inputs rotated` uses raw
accelerometer and gyroscope axes with one uniform 3-D rotation shared by
both sensors across each training window and its history. Normalization
uses training subjects and one RMS scale per rotated triad. Both modes
accept raw wrist CSVs through `predict` and the shared reference benchmark.
The existing multi-class station classifier continues to use the magnetometer.

From the workspace root, these three commands run independently in separate
shells. Use the same new output directory for all three.

```bash
study=baseline.mstcn.analyses.leave_one_exercise_out.study
out=baseline/MS-TCN/experiment_runs/step_2
.venv/bin/python -m "$study" original --output "$out"
.venv/bin/python -m "$study" gravity --output "$out"
.venv/bin/python -m "$study" rotated --output "$out"
```

After the three screen jobs complete, the following command applies the
criteria in `methods.md`. It compares both orientation variants with the
matched nine-channel complete-rep model. The eligible variant with the
highest mean unseen F1 advances. The original inputs advance if neither
candidate qualifies. Confirmation reuses the three existing fold-1 fits,
adds the other three pairs on fold 1, and evaluates all six on fold 2.
It applies the same criteria to the pooled confirmation estimates before
training the selected inputs on all exercises in five subject folds.

```bash
.venv/bin/python -m "$study" finish --output "$out"
```

`screen_decision.json` and `confirmed_decision.json` give the decisions and
scores. Each exercise's unseen F1 pools rep counts over subjects and folds
before taking the unweighted mean across exercises. Seen and crunch F1 are
averaged over holdout fits. Rest false counts and first-rep hits are pooled
with their exposures. The final `phase_comparison/model_comparison.csv`
compares the pooled release TP%, FN% and FP% with the frozen phase baseline.

The separate `biceps-diagnostic` command replays a frozen phase checkpoint
over nine thresholds. It records missed reps with and without overlap and
endpoint errors for overlapping misses. Its test sweep is for inspection
only. The study reads only the paired holdout reports.

```bash
.venv/bin/python -m baseline.mstcn biceps-diagnostic \
  --run-dir baseline/MS-TCN/experiment_runs/leave_one_exercise_out_fold_1/biceps_curls \
  --output "$out/diagnostics/fold_1_biceps_thresholds.json"
```

### Full five-fold study

The full runner evaluates all six exercise pairs across all five subject folds.
It compares the original, gravity, and rotated inputs with and without causal
similarity, plus the gravity-similarity synthetic screen. It also tests synthetic
training on the configuration selected after the similarity comparison. Use the
CUDA interpreter that trained the existing CUDA fits to reuse them.

```bash
runs=baseline/MS-TCN/experiment_runs
"$runs/step_2_mac_split/cuda-env/bin/python" -m \
  baseline.mstcn.analyses.leave_one_exercise_out.full_study --output "$runs/step_2/five_fold"
```

After synthetic training is evaluated, the runner tests the optional
exercise-adversarial head if mean seen F1 still exceeds mean unseen F1 by
at least 0.05. This uses the same 0.05 F1 difference as the acceptance rule.
The training-only head predicts exercise from stage-1 features using
certified real exercise frames. Its reversed loss rises from 0 to 0.1 by
epoch 5. The same acceptance criteria apply to this comparison.

`progress.json` lists the verified fits and report paths. `selection.json`
records the pooled scores and stage decisions. The final release-protocol
comparison is in `phase_comparison/model_comparison.csv`. Repeating the same
command reuses verified fits and preserves partial outputs before retrying.

## Step 2.2 causal motion similarity

`--causal-similarity` enables both changes to the complete-rep model. Each
stage-1 frame produces a normalized 32-dimensional motion vector. Its dot
products with the previous 150 vectors supply 64 projected features to each
refinement stage. The start selector also adds the similarity at each candidate
start, scaled by a learned weight initialized to one. Its separate no-start
choice retains its own score. Streaming replay resets the history at sensor
gaps and session boundaries. The model can score the first rep of a set.

The three fold-1 screens can train independently while step 2.1 is running.
For each `mode` in `original`, `gravity`, and `rotated`, run the following
from the workspace root in its own session.

```bash
runs=baseline/MS-TCN/experiment_runs
mode=gravity
.venv/bin/python -m baseline.mstcn leave-one-exercise-out \
  --data baseline/MS-TCN/prepared_datasets/separate_wrist_streams \
  --output "$runs/step_2_2/screen_$mode" --subject-fold 1 --protocol paired \
  --exercises screen --repetition-head complete --rep-inputs "$mode" \
  --rest-reviews "$runs/complete_rep_review/reviewed.json" --causal-similarity
```

Once both reports for an input choice are complete, compare them with the
same exercise, subject and training settings. The decision uses the thresholds
in `methods.md` for unseen F1, seen F1 and false reps per rest hour. It also
reports crunch F1 and first-rep recall. Use the input selected by step 2.1 for
the step 2.2 decision. For confirmation, the same command accepts multiple
reports on each side to pool the six exercise pairs across subject folds.

```bash
.venv/bin/python -m baseline.mstcn compare-similarity \
  --control-reports "$runs/step_2/screen_$mode/report.json" \
  --similarity-reports "$runs/step_2_2/screen_$mode/report.json" \
  --output "$runs/step_2_2/comparison_$mode.json"
```

## Step 2.3 synthetic repetitions

Enable `--synthetic-reps` with the complete-rep head. Each training batch
contains 75% real windows and 25% generated windows. Use a batch size
divisible by four. Each epoch has the control's number of optimizer updates.
Normalization and loss weights are fit on real training data. Validation,
checkpoint and decoder selection, and testing all use real recordings.

The generator draws 15–50 frames from one continuous, certified training
exercise, beginning at a local minimum of gyroscope speed below the piece's
median. It excludes held-out exercises, plank, gaps and masked frames.
It checks both the piece and up to two seconds of surrounding motion for a
step rhythm. A local peak of at least 0.65 in acceleration autocorrelation
at lags of 10–17 frames rejects the candidate. At least 35 contiguous
frames of the same exercise are needed for the surrounding check.

Each synthetic set uses 3–15 cycles. The return stroke reverses acceleration
and reverses and negates angular velocity. Speed and amplitude vary per cycle.
Acceleration is scaled about its initial value to preserve the starting pose.
Between cycles there is a 25% chance of a pause up to one second. Each side
gets 5–30 seconds of certified training rest, preferentially from the same
wrist stream. Reviewed rest weights carry through to these negatives.
One uniform 3-D rotation applies to every raw sensor triad in the full set
before computing model features. Each full set is then split into the same
causal windows as the real data. The model folder records its source pool
and batch composition in `synthetic_sources.json`.

The initial screen holds the gravity inputs and causal-similarity model fixed.
Use a separate output directory for each exercise. For example, from the
workspace root:

```bash
runs=baseline/MS-TCN/experiment_runs
exercise=biceps_curls
.venv/bin/python -m baseline.mstcn leave-one-exercise-out \
  --data baseline/MS-TCN/prepared_datasets/separate_wrist_streams \
  --output "$runs/step_2_3/$exercise" --subject-fold 1 --protocol paired \
  --exercises "$exercise" --repetition-head complete --rep-inputs gravity \
  --rest-reviews "$runs/complete_rep_review/reviewed.json" --causal-similarity --synthetic-reps
```

After all three real test reports and the matched control finish, the following
command checks the prespecified unseen F1, seen F1 and rest false-count
criteria. It also reports crunch F1 and first-rep recall.

```bash
.venv/bin/python -m baseline.mstcn compare-synthetic \
  --control-reports "$runs/step_2_2/screen_gravity/report.json" \
  --synthetic-reports "$runs/step_2_3/biceps_curls/report.json" \
                     "$runs/step_2_3/push-up/report.json" "$runs/step_2_3/squat/report.json" \
  --output "$runs/step_2_3/comparison_gravity.json"
```

## Legacy phase-only evaluation

Each exercise fold holds out one exercise identity. Every fold uses the same
participant-disjoint training, validation and test sets. Both wrists of each
participant stay together.

```bash
.venv/bin/python -m baseline.mstcn leave-one-exercise-out \
  --data baseline/MS-TCN/prepared_datasets/separate_wrist_streams \
  --output baseline/MS-TCN/experiment_runs/leave_one_exercise_out_phase_only \
  --subject-folds 5 --subject-fold 1 --epochs 30 --device cpu \
  --tolerance-ms 400 --report-tolerances-ms 200 400
```

The default holds out leg raise, squat, push-up, biceps curls and crunch in five
independent fits. Use `--exercises all` for all 14 dynamic exercises or list the
exercise names explicitly. Plank is a static hold. The subject options choose
the fixed participant split used for every exercise fold. The legacy flags
`--folds` and `--fold` are aliases. Use a fresh output directory for each run.

MS-TCN predicts three generic states from wrist signals: outside a repetition,
phase 1 and phase 2. Refinement uses those same probabilities. A complete cycle
produces an unlabelled repetition with start and end times. Use
`python -m baseline.mstcn predict` with a saved checkpoint and its decoder to
replay the model on raw sensor CSVs.

1. Mask every frame of the held-out exercise in training and validation.
   Its phases and transition frames become gaps for the loss, context,
   lookahead and normalization. Refit training statistics and rebuild
   repetition targets around those gaps.
2. Select each model's checkpoint and phase threshold using validation
   participants and the exercises included in its fit. Train one additional
   phase-only control with all exercises visible in training and validation.
3. Decode complete test sessions. Then use the exercise labels to select the
   held-out references and score repetition F1 and count MAE for both models on
   the same test streams. Count unmatched predictions that overlap certified
   frames for the selected exercise, including predictions that cross a bout
   boundary. The full-session report covers repetitions outside those bouts.

Each exercise folder includes an audit of the masked frames in
`data/mask_audit.json`, the model, the selected decoder, full-session test
outputs and reports for the held-out model and the seen-exercise control.
The parent `report.json` records each exercise's seen and unseen F1, absolute
count error and F1 difference at each endpoint tolerance. The shared control's
model and predictions are in `seen_exercise_control/`. Data views link to
immutable prepared arrays, so keep the source dataset.

For seed 42 and subject fold 1, the default exercise folds have 504 leg-raise,
575 squat, 534 push-up, 572 biceps-curl and 434 crunch test references. These are
wrist-stream observations. The generalization gap is measured by comparing
the held-out and seen-exercise reports after training.
