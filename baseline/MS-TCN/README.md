# MS-TCN for strength training

- `mstcn/` contains the causal segmentation model, command-line entry point, tests, and analysis code.
- `prepared_datasets/` contains `separate_wrist_streams/` (version 4), `single_wrist_per_session/` (version 2), and the historical `waist_sensor/` dataset (version 1).
- `experiment_runs/` contains `wrist_cross_validation/`, `sensor_ablation_fold_1/`, `leave_one_exercise_out_fold_1/`, and `job_launches/`.

The `baseline/mstcn` symlink provides the Python package name for `python -m baseline.mstcn`.

The default method uses a **station/NULL head + dense NULL/phase1/phase2 head**.
The user has validated **phase1 → phase2** as one complete repetition. Boundaries
are derived from the predicted phase sequence. An explicit start/end-head
baseline remains available with `train --repetition-head boundary`.
It uses wrist signals only. The authoritative targets are the matching wrist
column of each reliable `label.csv` in `strength_training_data_clean`.
No separate repetition-reference file is required for the default run.

## Additional analyses

Two independent analyses are implemented in separate folders:

- [Sensor and model ablations](mstcn/analyses/sensor_ablation/README.md): baseline,
  520 ms lookahead, accelerometer + gyroscope, training augmentation, their
  combination, and paired wrist + waist fusion with a matched wrist control.
- [Leave-one-exercise-out](mstcn/analyses/leave_one_exercise_out/README.md) runs five
  exercise holdouts by default on a fixed participant split. A phase-only model
  counts repetitions, and a model trained with all exercises measures the
  effect of withholding each exercise.

Both commands use the existing prepared version-4 dataset, create separate
output/data views, and keep every subject's wrists together. The descriptions
below describe the default wrist-only experiment.

The default uses **every available wrist as its own stream**. Both wrists from
the same subject always stay in the same training, validation, or test fold.
Each stream uses its own wrist labels and sensor-availability mask, preserving
side-specific annotation differences. Inputs remain nine channels from one wrist;
either wrist can be used at inference. No waist signal or waist label is used.
Explicit `--location left_wrist` and `--location right_wrist` runs remain available.

## Exactly what to run

From the repository root, set up Python once:

```bash
cd /home/arjun/zahra
python3 -m venv .venv
source .venv/bin/activate
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r baseline/MS-TCN/mstcn/requirements.txt
```

This workspace has the version-4 dataset at
`baseline/MS-TCN/prepared_datasets/separate_wrist_streams`. To prepare it on a
fresh checkout, run:

```bash
python -m baseline.mstcn prepare \
  --output baseline/MS-TCN/prepared_datasets/separate_wrist_streams
```

Run the full five-fold experiment:

```bash
python -m baseline.mstcn run \
  --data baseline/MS-TCN/prepared_datasets/separate_wrist_streams \
  --output baseline/MS-TCN/experiment_runs/wrist_cross_validation \
  --folds 5 --epochs 30 --device cpu \
  --tolerance-ms 400 --report-tolerances-ms 200 400
```

Version-2 and version-3 datasets must be prepared again for the main experiment:
they retained one wrist per session. Version 4 records stream IDs as
`<source_session>__<side>`, with the original session, subject, and wrist side
stored separately in the manifest. Old checkpoints remain usable for replay;
learning from both wrists requires retraining.

`run` now calls **cross-validate**, training both heads separately in each fold.
It selects checkpoints by validation repetition F1 (independent of station labels),
breaking ties with wrist-stream count MAE. It uses a fixed decoder and the declared
selection tolerance during checkpoint selection, then tunes thresholds on
validation subjects. All held-out test predictions are pooled for the main report.
There is no station-only fallback. Training prints progress every
100 batches. CPU training can take substantial time; the command runs up to
30 epochs with early stopping. Use `--device cuda` only if your environment has
compatible GPU PyTorch installed. `--max-steps` caps loss batches only; checkpoint
selection always decodes the complete validation sessions.

To invoke cross-validation directly on prepared data, use:

```bash
python -m baseline.mstcn cross-validate \
  --data baseline/MS-TCN/prepared_datasets/separate_wrist_streams \
  --output baseline/MS-TCN/experiment_runs/wrist_cross_validation_direct \
  --folds 5 --epochs 30 --tolerance-ms 400 --report-tolerances-ms 200 400
```

Each subject appears in exactly one test fold. The next fold supplies validation
subjects; the remaining folds train. Normalization and loss weights are refitted
for every fold. Checkpoints and decoder thresholds use only that fold's validation
subjects. `report.json` pools held-out predictions and reports fold F1 mean and
standard deviation, per-side scores, and subject-macro F1 at every reporting
tolerance. `fold_*/` retains individual models, splits, and reports.
Five folds with 43 subjects gives 8–9 test subjects per fold and tests all 43.

On a fresh checkout without prepared data, omit `--data`:

```bash
python -m baseline.mstcn run \
  --output baseline/MS-TCN/experiment_runs/wrist_cross_validation_fresh \
  --epochs 30 --device cpu
```

That also prepares the wrist inputs and targets automatically, under the run's
`data/` directory. Output directories are never overwritten. To repeat an
experiment, use a new output directory. `run` writes its main report under
`cross_validation/report.json`; direct `cross-validate` writes `report.json` in
its output directory. Use these cross-validation results for headline numbers.

To continue after an interruption while retaining completed folds, stop the old
process, move the incomplete `fold_N/` directory to a backup outside
`cross_validation/`, and invoke `cross-validate` with the original settings and
`--resume-completed --output baseline/MS-TCN/experiment_runs/wrist_cross_validation/cross_validation`.
Completed folds are checked against the dataset, splits, model configuration,
decoder and saved test scores before reuse. Missing folds train from epoch 1;
incomplete folds are refused rather than overwritten. This resumes between folds,
not within a fold: `best.pt` does not contain the optimizer state needed for an
exact continuation of interrupted epochs.

The entry point is `__main__.py`; run it as the module above so its imports work.
You do not need to run `prepare_data.py`, the demographics analysis, or the
synthetic demo before this experiment.

## Outputs

| File under `baseline/MS-TCN/experiment_runs/wrist_cross_validation/cross_validation/` | Contents |
| --- | --- |
| `report.json` | Main pooled cross-validation report, per side and at 200/400 ms |
| `folds.json` | Fold membership and fold metrics |
| `fold_*/model/best.pt` | Each fold's best two-head checkpoint |
| `fold_*/model/history.json` | Losses and validation repetition/count metrics |
| `fold_*/decoder.json` | Validation-selected thresholds and selection tolerance |
| `fold_*/test/report.json` | That fold's held-out classification, count and timing metrics |
| `fold_*/test/<stream>.json` | Predicted repetitions, holds, sets, and scores at each tolerance |

Every predicted repetition contains its station, `start_ms`, `end_ms`, confidence,
and sample-time `reported_at_ms`. A session's counts are aggregated by predicted
station, including `unknown` when station evidence is weak. Station uncertainty
does not discard a completed cycle. Reference count scores cover eligible intervals with complete phase
cycles; excluded or incomplete reference regions are reported separately.
Pooled counts and hours are **wrist-stream observations**, not distinct workout
totals: paired wrists observe the same physical repetitions. Reports count unique
subjects separately and include per-side results plus subject-macro F1.

## How label.csv becomes repetition targets

You confirmed that the `label.csv` files are reliable and that a full repetition
is phase1 followed by phase2. The code uses those files directly:

1. A bout is a continuous run of the same normalized exercise label. Rest or a
   different exercise ends it. This is only a reference-conversion step; bout
   boundaries are never supplied to the inference model.
2. Pair `phase1 → phase2` only. An initial `phase2` is incomplete and cannot
   reverse the repetition convention. A later complete pair remains eligible.
3. Include same-exercise transitions between those two phases in the repetition.
   End the repetition immediately after the second phase. A trailing transition
   is not part of the completed repetition and never becomes an extra repetition.
4. Repeated runs of the same phase separated by same-exercise transitions are
   treated as a paused continuation of that phase.
5. Discard unpaired phases and recording-edge-truncated cycles from boundary
   training/scoring. Never accept a cycle crossing rest, another exercise, a
   missing sensor sample, or an excluded interval. Those intervals can still
   train the station head when its inputs and labels are eligible.
6. Plank is a static hold and has no repetition count.

The dense targets retain phase labels throughout each accepted cycle. Internal
transition pauses continue the preceding phase; transitions outside cycles,
rest, and plank use the phase head's NULL target. Incomplete or excluded phase
regions are masked. These transition/pause rules are operational conventions;
the user's validation specifically confirmed the phase order.

All intervals use `[start_ms, end_ms)` on the session's 40 ms grid. The start
impulse is at the first phase's start; the end impulse is at the first sample
after the second phase ends. A shared end/start tick can have both targets set.
A completed cycle's known end remains supervised if the next phase is incomplete.

For example:

```text
rest | phase1 | transition | phase2 | transition | phase1 | phase2 | rest
       <-------- rep 1 ----------->              <---- rep 2 ---->
```

Label reliability and phase order are user-confirmed; no independent video audit
was performed by this code review. The convention and validation source are saved
in the dataset manifest; the convention is copied into checkpoints and reports.
Each session also has `reference_audit.json`, listing the chosen label column,
accepted cycles, and rejected intervals. This makes the reported counts traceable
to the original labels without changing the source files.

## Verified wrist data and checks

Version-4 preparation retained **77 streams from 43 subjects/source sessions**,
including **34 sessions with both wrists**, after the existing exclusions:

| Side | Streams | Eligible stream-hours | Repetition-reference observations |
| --- | ---: | ---: | ---: |
| Left | 39 | 67.1624 | 19,849 |
| Right | 38 | 65.6840 | 18,637 |

The five test folds contain **9, 9, 9, 8, and 8 subjects**. Every subject's
available wrists stay together. The previous 21,277-reference preparation used
only one wrist per source session and is a historical artifact.

The 72-test regression suite passed. A three-fold synthetic experiment with
16 streams from eight subjects verified training, tuning, pooled evaluation,
per-side metrics, both timing tolerances, and sensor-only prediction. Synthetic
training was deliberately short and does not establish model accuracy. Prior real
training checks in `mstcn/verification.json` describe the historical boundary model.
Full held-out repetition performance and the <1 second true reporting-delay goal
still need to be measured by the full experiment.

The original plan's exclusions remain in force. Four complete sessions are
excluded as specified in `plan.md`. Three more have unresolved log offsets:
`20260602_SSCDLRFV`, `20260603_MGTNMGTNOWZB`, and
`20260605_DQTQVVZPOGPA`. `20260617_WTTALRFVDQTQ` has an `Other` entry with `/`
in both time fields, so its affected interval cannot be located. These are
exclusion-timing issues, separate from your confirmation of label reliability.
All eight exclusions and their reasons are in `manifest.json`.

## Individual commands, if needed

These commands operate on the diagnostic single split. Use them for development;
publish numbers from `run` or `cross-validate` above. A standalone `evaluate`
report is explicitly marked `single_split_diagnostic`:

```bash
python -m baseline.mstcn prepare \
  --location wrist --reference-mode labels \
  --output baseline/MS-TCN/prepared_datasets/wrist_manual

python -m baseline.mstcn train \
  --data baseline/MS-TCN/prepared_datasets/wrist_manual \
  --output baseline/MS-TCN/experiment_runs/wrist_manual/model --epochs 30

python -m baseline.mstcn tune \
  --data baseline/MS-TCN/prepared_datasets/wrist_manual \
  --checkpoint baseline/MS-TCN/experiment_runs/wrist_manual/model/best.pt \
  --tolerance-ms 400 \
  --output baseline/MS-TCN/experiment_runs/wrist_manual/decoder.json

python -m baseline.mstcn evaluate \
  --data baseline/MS-TCN/prepared_datasets/wrist_manual \
  --checkpoint baseline/MS-TCN/experiment_runs/wrist_manual/model/best.pt \
  --decoder baseline/MS-TCN/experiment_runs/wrist_manual/decoder.json \
  --split test --output baseline/MS-TCN/experiment_runs/wrist_manual/test
```

The default **selection tolerance** is 400 ms at each endpoint. Declare it before
examining test results; `tune` stores it and selects thresholds only on validation
subjects. Reporting defaults to **200 and 400 ms**, using the exact same decoded
predictions and thresholds for both. No extra inference or tolerance-specific
tuning is performed. `--report-tolerances-ms` accepts additional tolerances; the
selection tolerance is always included. Read `by_tolerance["200"]` and
`by_tolerance["400"]`, and the corresponding entries under `per_side`. Top-level
`segmentation` and `repetitions` retain the selection-tolerance results for
compatibility. The `segmentation` report matches boundaries independently of station;
the `repetitions` report additionally requires station agreement. Matching is
one-to-one, maximizing match count before minimizing timing error. Predictions
are matched against all retained references **before** coverage filtering; only
unmatched predictions outside certified coverage are unscored. Both metrics and
their count tables use this rule, including threshold and checkpoint selection.
Reports include missed/extra reps, start/end MAE, count MAE and exact-count
accuracy per session/station, per-subject and per-exercise results, macro F1,
per-class precision/recall, confusion matrix and false exercise events per rest hour.

`--reference-mode none` and `train --station-only` remain available as explicit
diagnostics. They are not used by the full wrist experiment. Independently
specified reference JSONs can still override automatic conversion using
`prepare --references FILE`; entries must identify the actual selected
`left_wrist` or `right_wrist` location. A session can provide both using
`{"wrists": {"left_wrist": <entry>, "right_wrist": <entry>}}`. A legacy single-side
entry certifies only that side; the other side gets no repetition supervision or
coverage. External references default to the boundary
head because they need not certify phase labels. Their schema is implemented in
`data.reference_targets()` and shown by the synthetic demo's `references.json`.

## Prediction from sensors only

After the full run, replay a wrist CSV without supplying any labels:

```bash
python -m baseline.mstcn predict \
  --checkpoint baseline/MS-TCN/experiment_runs/wrist_cross_validation/cross_validation/fold_1/model/best.pt \
  --decoder baseline/MS-TCN/experiment_runs/wrist_cross_validation/cross_validation/fold_1/decoder.json \
  --input strength_training/strength_training_data_clean/20260602_LRFVFZEIVGAM/left_wrist.csv \
  --chunk 1 --output baseline/MS-TCN/experiment_runs/wrist_prediction
```

This writes `events.jsonl`, `frames.csv`, and `predictions.json`. The example
shows the input format; headline held-out scores come from cross-validation.
Prediction reads only `sync_timestamp` and the nine wrist sensor channels.
Embedded label columns are ignored. The timestamp is alignment metadata, not a
model feature. Raw prediction is unscored and does not read workout-log exclusions.
Named waist files and old waist checkpoints are rejected.

`DenseMSTCN.stream()` and `RepDecoder.step()` can be used in a live sensor loop.
The CLI replays a CSV, not a hardware driver. At 25 Hz, `--chunk 1` adds no
intentional input buffering. The decoder confirms phase changes over three
frames (80 ms after the boundary), preserving the original boundary timestamp.
The boundary-head baseline likewise uses a local peak with two future frames,
rather than the first threshold crossing. `reported_at_ms` includes this delay;
CSV runtime reports list decoder delay separately from chunk buffering.
Sensor gaps reset the network and decoder. Incomplete
cycles at gaps, timeouts, or EOF are not emitted. Plank holds are reported as
durations, and sets are a secondary grouping using a 10 second gap.

Runtime timings include compute and buffering relative to **predicted** ends.
True reporting delay also includes error relative to the reference end, transport,
and processing backlog. `matched_reported_before_1s_fraction` accepts an early
report only after endpoint-tolerant matching; early reports are also counted in
`matched_early_report_fraction`. The older `matched_reported_0_to_999ms_fraction`
remains as a stricter diagnostic. Exactly 1,000 ms fails the deadline. These
fractions cover matched reps only; recall accounts for missed reps. Causality
alone does not prove the <1 second requirement.

## Architecture and data handling

This is a causal adaptation of
[MS-TCN (Abu Farha and Gall, CVPR 2019)](https://openaccess.thecvf.com/content_CVPR_2019/html/Abu_Farha_MS-TCN_Multi-Stage_Temporal_Convolutional_Network_for_Action_Segmentation_CVPR_2019_paper.html),
with continuous inertial localization motivated by
[WEAR (Bock et al., IMWUT 2024)](https://mariusbock.github.io/wear/).
The original MS-TCN uses symmetric convolutions and one action head; causal
padding and the repetition heads are additions here. This does not
reproduce the WEAR benchmark architectures.

The default model has 3 stages, 6 dilated residual layers per stage, and 64
channels. Its history is 378 past samples (15.12 seconds), with no future
lookahead in the network. Each stage has 16 station logits and 3 phase logits.
Refinement consumes predicted probabilities; no true station IDs or boundaries
are inference inputs. The loss is the mean over stages of weighted station
cross-entropy, 0.15 times truncated station log-probability smoothing, and masked
weighted phase cross-entropy. Phase logits are not temporally smoothed. The
boundary baseline uses masked BCE instead. All weights come from training subjects.

Sensors join exactly to `label.csv` on `sync_timestamp`. No interpolation or
forward fill occurs. Only each stream's matching wrist label column supplies
targets. The nine features are `acc_x/y/z`, `gyro_x/y/z`, and `mag_x/y/z`.
Normalization is fitted only on eligible training samples. Windows include
causal context but do not cross exclusion or sensor gaps. Cached streaming is
tested against full causal inference.

Calibration, `Other`, and sled-pull intervals are excluded from standard training
and evaluation. The separate sled-pull mask is `probe.npy`; existing rest labels
cannot score unseen-exercise detection. NULL means category-relative rest, not
an unknown-exercise detector. A weak station prediction is reported as `unknown`
without vetoing the phase-derived count. This removes the known-class gate but
does not demonstrate generalization to unseen exercises; that requires the
separate probe with suitable repetition references. Confident plank predictions
remain duration-only.

## Tests and source files

```bash
python -m unittest discover -s baseline/MS-TCN/mstcn/tests -v
python -m baseline.mstcn demo \
  --output baseline/MS-TCN/experiment_runs/my_wrist_demo
```

| File under `mstcn/` | Responsibility |
| --- | --- |
| `__main__.py` | CLI, including the single `run` entry point |
| `data.py` | Wrist selection, joins, masks, phase conversion, splits, normalization |
| `model.py` | Causal two-head model, losses, cached inference |
| `decode.py` | Rep pairing, plank holds, secondary sets |
| `metrics.py` | One-to-one matching, classification, count and timing metrics |
| `runner.py` | Full pipeline, training, tuning, evaluation, inference |
| `tests/test_pipeline.py` | Causality, exclusion, phase-pairing, wrist-only and metric checks |
| `verification.json` | Recorded execution checks and package versions |

Previous waist datasets/checkpoints are historical artifacts and cannot be used
with this wrist-only method.
