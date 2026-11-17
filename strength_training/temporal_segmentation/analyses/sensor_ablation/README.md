# MS-TCN sensor and model ablations

This analysis keeps the current station/NULL + NULL/phase1/phase2 model and
tests each proposed change separately, followed by their combination. It runs
one subject fold initially, with identical subject membership and seed for all
variants. Each variant selects its own checkpoint and decoder on validation
subjects. Test performance does not select a variant automatically.

From the repository root:

```bash
.venv/bin/python -m strength_training.temporal_segmentation sensor-ablation \
  --data strength_training/tas_data_both_wrists \
  --output strength_training/tas_runs/sensor_ablation_fold1 \
  --folds 5 --fold 1 --epochs 30 --device cpu \
  --tolerance-ms 400 --report-tolerances-ms 200 400
```

| Separate output folder | Inputs | Lookahead | Augmentation |
| --- | --- | --- | --- |
| `baseline/` | Wrist acc + gyro + mag, 9 channels | 0 | No |
| `lookahead/` | Wrist acc + gyro + mag | 520 ms | No |
| `acc_gyro/` | Wrist acc + gyro, 6 channels | 0 | No |
| `augmentation/` | Wrist acc + gyro + mag | 0 | Yes |
| `combined/` | Wrist acc + gyro | 520 ms | Yes |
| `combined_paired_wrist/` | Wrist acc + gyro, joint wrist/waist availability | 520 ms | Yes |
| `combined_waist/` | Wrist + waist acc + gyro, 12 channels | 520 ms | Yes |

`--variants` can select any subset, for example `--variants baseline lookahead`.
Use a new output directory for each invocation; existing results are refused.
Choose a different pilot fold with `--fold 2` and a separate output. `--folds 5`
defines the split partition; it does not run all five folds in this command.
For a quick plumbing check, add `--epochs 1 --max-steps 2`; this still evaluates
complete validation/test streams and is not an accuracy benchmark.

The first four variants isolate individual changes. Compare `combined` with
`baseline` for their joint effect. Compare `combined_waist` specifically with
`combined_paired_wrist` to measure the value of the waist signal on identical
frames and reference repetitions. Waist is supplied throughout inference,
including leg raises; no true exercise label switches it on or off. Labels
always come from the matching wrist, never from the waist CSV. A separate
station-ignored score for leg raises is highlighted in the summary, and all
dynamic exercises appear in `exercise_scores.json`.

Waist CSVs are found as `*waist*.csv` in each source session. When there are
multiple devices, select the one with the most aligned clock samples; filename
order breaks ties. Selection uses availability only, never exercise labels or
performance, and does not stitch devices together. The selected filename and
all candidates' sample counts are recorded in `data/mask_audit.json`.
`--source PATH` overrides the prepared manifest's raw-data root. Joins use exact
`sync_timestamp` matches. Missing waist frames
create gaps in both paired variants; there is no interpolation across gaps.
Six-channel variants omit magnetometer inputs and fit six-channel statistics.
They inherit the original prepared wrist availability/exclusion policy.

Lookahead is implemented as delayed prediction: the causal network output
acquired at t+13 predicts the target at t. Refinement stages do not multiply
the delay. 500 ms rounds up to 13 samples, or 520 ms, on the 25 Hz grid. The
decoder's default 80 ms confirmation is additional, giving 600 ms intentional
sample-time delay before timing error, compute, transport and chunk buffering.
The last 13 targets of each continuous run have no prediction; no fake future
samples are padded at gaps or EOF. Reference repetitions are retained for
scoring, so unavailable tail predictions can cause misses. Frame classification
excludes these unavailable predictions. `train` and `cross-validate` also expose
`--lookahead-ms 500` for custom runs.

Augmentation occurs before normalization, only in training windows:

- A random axis rotation within ±10 degrees, shared by all triads on a device.
  Wrist and waist use independent rotations.
- A monotone four-segment time warp with sampled segment weights in [0.9, 1.1],
  normalized to preserve window endpoints. All signals use the same warp;
  discrete labels and masks use nearest-neighbor sampling.
- A scale factor in [0.95, 1.05] for each sensor triad.

Validation and test samples are unchanged. Augmentation is seeded and requires
the dense phase head; the sparse boundary baseline is rejected. `train` and
`cross-validate` expose the same augmentation through `--augment`.

Each variant saves `data/manifest.json`, `data/mask_audit.json`, `model/best.pt`,
`model/history.json`, `decoder.json`, `test/report.json`, decoded stream JSONs,
and `exercise_scores.json`. The parent saves `plan.json` and `report.json`.
Data views share immutable source arrays using symlinks, so keep the original
prepared dataset available. Normalization and class/phase weights are fitted
again using only that variant's training frames.

For a fused checkpoint, sensor-only replay requires both files:

```bash
.venv/bin/python -m strength_training.temporal_segmentation predict \
  --checkpoint strength_training/tas_runs/sensor_ablation_fold1/combined_waist/model/best.pt \
  --decoder strength_training/tas_runs/sensor_ablation_fold1/combined_waist/decoder.json \
  --input strength_training/strength_training_data_clean/20260602_LRFVFZEIVGAM/left_wrist.csv \
  --waist-input strength_training/strength_training_data_clean/20260602_LRFVFZEIVGAM/S1_waist.csv \
  --output strength_training/tas_runs/fused_prediction
```

These are one-fold exploratory comparisons. Pooled repetitions are wrist-stream
observations; paired wrists observe the same physical workout. Review per-side
and subject-macro scores. Real-data performance has not yet been established by
this implementation's synthetic smoke tests.
