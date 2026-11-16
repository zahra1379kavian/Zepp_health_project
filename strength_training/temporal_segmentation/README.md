# Wrist-only temporal action segmentation and repetition counting

This is the confirmed **station/NULL head + repetition start/end head** method.
It uses wrist signals only. The authoritative targets are the matching wrist
column of each reliable `label.csv` in `strength_training_data_clean`.
No separate repetition-reference file is required for the default run.

The default uses **one wrist per session: left when present, otherwise right**.
It does not fuse both wrists or switch sensors inside a session. No waist signal
or waist label is used. Explicit `--location left_wrist` and `--location
right_wrist` runs are also supported; sessions missing that wrist are excluded.

## Exactly what to run

From the repository root, set up Python once:

```bash
cd /home/arjun/zahra
python3 -m venv .venv
source .venv/bin/activate
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r strength_training/temporal_segmentation/requirements.txt
```

The real wrist dataset is already prepared in this workspace. Run the full
experiment with:

```bash
python -m strength_training.temporal_segmentation run \
  --data strength_training/tas_data_wrist \
  --output strength_training/tas_runs/wrist_full \
  --epochs 30 --device cpu
```

This trains **both heads**, selects the best checkpoint by validation loss,
selects repetition thresholds on validation subjects, and evaluates the held-out
test subjects. There is no station-only fallback. Training prints progress every
100 batches. CPU training can take substantial time; the command runs up to
30 epochs with early stopping. Use `--device cuda` only if your environment has
compatible GPU PyTorch installed.

On a fresh checkout without prepared data, omit `--data`:

```bash
python -m strength_training.temporal_segmentation run \
  --output strength_training/tas_runs/wrist_full \
  --epochs 30 --device cpu
```

That also prepares the wrist inputs and targets automatically, under the run's
`data/` directory. Output directories are never overwritten. To repeat an
experiment, use a new name such as `--output strength_training/tas_runs/wrist_full_v2`.

The entry point is `__main__.py`; run it as the module above so its imports work.
You do not need to run `prepare_data.py`, the demographics analysis, or the
synthetic demo before this experiment.

## Outputs

| File under `strength_training/tas_runs/wrist_full/` | Contents |
| --- | --- |
| `model/best.pt` | Best two-head checkpoint |
| `model/history.json` | Training/validation losses |
| `decoder.json` | Validation-selected thresholds and declared timing tolerance |
| `test/report.json` | Held-out classification, count, detection and timing metrics |
| `test/<session>.json` | Predicted repetitions, counts, plank holds, sets and session metrics |

Every predicted repetition contains its station, `start_ms`, `end_ms`, confidence,
and sample-time `reported_at_ms`. A session's counts are aggregated by predicted
station. Reference count scores cover eligible intervals with complete phase
cycles; excluded or incomplete reference regions are reported separately.

## How label.csv becomes repetition targets

You confirmed that the `label.csv` files are reliable. The code therefore uses
those files directly, with the following **declared conversion convention**:

1. A bout is a continuous run of the same normalized exercise label. Rest or a
   different exercise ends it. This is only a reference-conversion step; bout
   boundaries are never supplied to the inference model.
2. Pair opposite phases of that exercise, anchored to the first observed phase
   in each bout. Usually the sequence is `phase1 → phase2`; a bout beginning with
   `phase2` uses `phase2 → phase1`.
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

All intervals use `[start_ms, end_ms)` on the session's 40 ms grid. The start
impulse is at the first phase's start; the end impulse is at the first sample
after the second phase ends. A shared end/start tick can have both targets set.
A completed cycle's known end remains supervised if the next phase is incomplete.

For example:

```text
rest | phase1 | transition | phase2 | transition | phase1 | phase2 | rest
       <-------- rep 1 ----------->              <---- rep 2 ---->
```

Label reliability is user-confirmed. The pairing rule is an explicit operational
convention, not a claim that a separate video audit established these boundaries.
It is saved in the dataset manifest, model checkpoint, and evaluation report.
Each session also has `reference_audit.json`, listing the chosen label column,
accepted cycles, and rejected intervals. This makes the reported counts traceable
to the original labels without changing the source files.

## Verified wrist data and checks

The updated preparation produced:

- **43 sessions:** 39 using left wrist, 4 using right wrist.
- **31 training / 6 validation / 6 test subjects**, split before creating windows.
- **21,277 complete phase-derived repetition references** and 12 unpaired phases
  excluded from boundary supervision/scoring.
- **74.1357 eligible wrist-sensor hours**, including ordinary inter-set rest.
- **60.65 minutes** of available sled-pull data kept separately as an unscored probe.

The 43-test suite passed. The updated wrist-only synthetic pipeline and the new
`run` command both completed through held-out evaluation. A short real-wrist
training/validation check verifies both heads can train on the derived targets.
These checks establish executable code, not a fully trained model's accuracy.
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

The `run` command already performs these steps. Use them separately only when
working on one stage:

```bash
python -m strength_training.temporal_segmentation prepare \
  --location wrist --reference-mode labels \
  --output strength_training/tas_data_wrist_new

python -m strength_training.temporal_segmentation train \
  --data strength_training/tas_data_wrist_new \
  --output strength_training/tas_runs/wrist_manual/model --epochs 30

python -m strength_training.temporal_segmentation tune \
  --data strength_training/tas_data_wrist_new \
  --checkpoint strength_training/tas_runs/wrist_manual/model/best.pt \
  --tolerance-ms 400 \
  --output strength_training/tas_runs/wrist_manual/decoder.json

python -m strength_training.temporal_segmentation evaluate \
  --data strength_training/tas_data_wrist_new \
  --checkpoint strength_training/tas_runs/wrist_manual/model/best.pt \
  --decoder strength_training/tas_runs/wrist_manual/decoder.json \
  --split test --output strength_training/tas_runs/wrist_manual/test
```

The default matching tolerance is 400 ms at **each** endpoint. Declare it before
examining test results; `tune` stores it and selects thresholds only on validation
subjects. Prediction stations must match reference stations for a repetition
match. Matching is one-to-one, maximizing match count before minimizing timing
error. Reports include missed/extra reps, start/end MAE, count MAE and exact-count
accuracy per session/station, per-subject and per-exercise results, macro F1,
per-class precision/recall, confusion matrix and false exercise events per rest hour.

`--reference-mode none` and `train --station-only` remain available as explicit
diagnostics. They are not used by the full wrist experiment. Independently
specified reference JSONs can still override automatic conversion using
`prepare --references FILE`; entries must identify the actual selected
`left_wrist` or `right_wrist` location. Their schema is implemented in
`data.reference_targets()` and shown by the synthetic demo's `references.json`.

## Prediction from sensors only

After the full run, replay a wrist CSV without supplying any labels:

```bash
python -m strength_training.temporal_segmentation predict \
  --checkpoint strength_training/tas_runs/wrist_full/model/best.pt \
  --decoder strength_training/tas_runs/wrist_full/decoder.json \
  --input strength_training/strength_training_data_clean/20260602_LRFVFZEIVGAM/left_wrist.csv \
  --chunk 1 --output strength_training/tas_runs/wrist_prediction
```

This writes `events.jsonl`, `frames.csv`, and `predictions.json`. The example
shows the input format; use `evaluate --split test` for held-out scoring.
Prediction reads only `sync_timestamp` and the nine wrist sensor channels.
Embedded label columns are ignored. The timestamp is alignment metadata, not a
model feature. Raw prediction is unscored and does not read workout-log exclusions.
Named waist files and old waist checkpoints are rejected.

`DenseMSTCN.stream()` and `RepDecoder.step()` can be used in a live sensor loop.
The CLI replays a CSV, not a hardware driver. At 25 Hz, `--chunk 1` adds no
intentional buffering. Sensor gaps reset the network and decoder. Incomplete
cycles at gaps, timeouts, or EOF are not emitted. Plank holds are reported as
durations, and sets are a secondary grouping using a 10 second gap.

Runtime timings include compute and buffering relative to **predicted** ends.
True reporting delay also includes error relative to the reference end, transport,
and processing backlog. Causality alone does not prove the <1 second requirement.

## Architecture and data handling

This is a causal adaptation of
[MS-TCN (Abu Farha and Gall, CVPR 2019)](https://openaccess.thecvf.com/content_CVPR_2019/html/Abu_Farha_MS-TCN_Multi-Stage_Temporal_Convolutional_Network_for_Action_Segmentation_CVPR_2019_paper.html),
with continuous inertial localization motivated by
[WEAR (Bock et al., IMWUT 2024)](https://mariusbock.github.io/wear/).
The original MS-TCN uses symmetric convolutions and one action head; causal
padding and the repetition boundary head are additions here. This does not
reproduce the WEAR benchmark architectures.

The default model has 3 stages, 6 dilated residual layers per stage, and 64
channels. Its history is 378 past samples (15.12 seconds), with no future
lookahead. Each stage has 16 station logits and independent start/end logits.
Refinement consumes predicted probabilities; no true station IDs or boundaries
are inference inputs. The loss is the mean over stages of weighted station
cross-entropy, 0.15 times truncated log-probability smoothing, and masked boundary
BCE. Class and boundary weights come from training subjects only.

Sensors join exactly to `label.csv` on `sync_timestamp`. No interpolation or
forward fill occurs. Only the matching selected wrist label column supplies
targets. The nine features are `acc_x/y/z`, `gyro_x/y/z`, and `mag_x/y/z`.
Normalization is fitted only on eligible training samples. Windows include
causal context but do not cross exclusion or sensor gaps. Cached streaming is
tested against full causal inference.

Calibration, `Other`, and sled-pull intervals are excluded from standard training
and evaluation. The separate sled-pull mask is `probe.npy`; existing rest labels
cannot score unseen-exercise detection. NULL means category-relative rest, not
an unknown-exercise detector. Recognition of genuinely new station types remains
phase 2. The current model infers which of the 15 stations is being used.

## Tests and source files

```bash
python -m unittest discover -s strength_training -p 'test_*.py' -v
python -m strength_training.temporal_segmentation demo \
  --output strength_training/tas_runs/my_wrist_demo
```

| File | Responsibility |
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
with this wrist-only method. The earlier `figures/workflow-v2.png` also depicts
an older status before automatic label conversion; its pending-reference note
and test count are superseded by this document.
