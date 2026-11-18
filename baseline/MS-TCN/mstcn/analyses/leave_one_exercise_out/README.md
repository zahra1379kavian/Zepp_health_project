# Leave-one-exercise-out MS-TCN pilot

Train one independent phase-head model per removed exercise, then measure
station-ignored repetition F1 for that exercise in held-out subjects. The
initial pilot removes `leg_raise`, `squat`, `push-up`, and `biceps_curls`, one at a
time, using fold 1 of the existing five-part subject partition.

```bash
.venv/bin/python -m baseline.mstcn leave-one-exercise-out \
  --data baseline/MS-TCN/prepared_datasets/separate_wrist_streams \
  --output baseline/MS-TCN/experiment_runs/leave_one_exercise_out_fold_1 \
  --exercises leg_raise squat push-up biceps_curls \
  --folds 5 --fold 1 --epochs 30 --device cpu \
  --tolerance-ms 400 --report-tolerances-ms 200 400
```

`--exercises` is the new removal option. This command runs only the selected
`--fold`; `--folds` defines the subject partition. The default architecture is
the unchanged causal, nine-channel wrist MS-TCN with the phase head. Optional
`--lookahead-ms 500 --augment` allows a separately declared follow-up comparison.
Use a new output directory for each invocation. The test score must not choose
training settings or decoder thresholds.

For each removed exercise:

1. Keep the existing subject-disjoint training, validation and test sets. Both
   wrists of each subject stay together.
2. Mask all its training frames, including phase1, phase2 and transition labels.
   Preserve the original `y.npy` and `raw.npy`: these frames are never relabeled
   rest. They are gaps for window context and lookahead, and are excluded from
   both heads' losses, temporal smoothing, normalization and class/phase weights.
   Rebuild repetition targets to prevent cycles crossing the masked gaps.
3. Apply the same removal to validation. Early stopping and decoder tuning see
   only known exercises in separate validation subjects. This prevents the
   held-out exercise from selecting the model indirectly.
4. Decode complete, unmodified test sessions. The decoder never sees reference
   exercise labels or exercise-specific crop boundaries.
5. After decoding, select the removed exercise's references and match endpoints
   without requiring station agreement. A predicted known class or `unknown`
   can match. Count unmatched predictions that overlap any certified frame of
   the held-out exercise, including predictions crossing its bout boundaries.

This is an **exercise-conditioned** repetition score: false positives entirely
inside other exercises or rest are outside its scope. The full-session report
is retained alongside it. The decoder is unchanged, including its existing
confident-plank hold behavior. This experiment tests rep segmentation transfer;
it does not establish open-set exercise classification or unknown detection.
Plank cannot be selected as a repetition probe because it is a static hold.

Each exercise gets a separate folder containing:

- `data/manifest.json` and `data/mask_audit.json`: split, removal policy, refitted
  statistics and per-stream masked-frame counts.
- `model/best.pt`, `model/history.json`, and `decoder.json`: independently trained
  model and known-exercise validation selection.
- `test/`: full-session predictions and diagnostic metrics.
- `held_out_report.json`: the primary station-ignored held-out-exercise score,
  endpoint errors, count errors, timing, per-side and per-subject scores at
  200/400 ms using the same decoded predictions.

The parent `report.json` collects the four exercise reports and protocol.
`plan.json` records the requested settings before fitting. Data views use
symlinks to immutable prepared arrays; retain the original dataset. The tool
checks that each requested exercise has training and test references before
starting. No learned model or decoder is shared between removals.

For the current data, seed 42 and fold 1 give 46 training, 16 validation and
15 test wrist streams. Before removal, the test set has 504 leg-raise, 575 squat,
534 push-up and 572 biceps-curl references. These counts describe coverage, not
accuracy. Full pilot training has not been run as part of the implementation;
the automated smoke test uses synthetic subjects and short training.
