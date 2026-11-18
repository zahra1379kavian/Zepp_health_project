# Leave-one-exercise-out MS-TCN

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
