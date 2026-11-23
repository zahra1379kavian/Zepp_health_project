---
name: compare-wrist-reps
description: Integrate a wrist repetition model or compare its KPIs, tables, and figures with the HT Project Feishu activity-agnostic wrist release in the zahra workspace. Applies to MS-TCN and future wrist model adapters.
---

# Compare wrist repetition models

Use the August 31 activity-agnostic wrist release, pages 7–9 of the local Feishu PDF. Work in the zahra project, with `StrengthTraining_wrist_activity_agnostic` as the read-only reference. The shared implementation and adapter contract live in `baseline/rep_benchmark` and `baseline/MS-TCN/README.md`.

- Start with the saved session assignments and input coverage in the phase benchmark's `manifest.json`. Keep both wrists of a subject in the same fold. Check training and validation subjects against the evaluation subjects. Existing fold assignments are the agreed exception to release parity. The only unavailable release session IDs in this workspace are `20260529_DQTQDQTQ` and `20260529_JCLQJCLQ`. Include every available session in the audit, including ones with zero eligible clips.
- Give the predictor a fresh clip with timestamps and the nine raw channels from one wrist at 25 Hz. Derive model features inside the adapter. The evaluator alone sees activity identity and annotations. Declare any lookahead, confirmation delay and extra buffering. Report matched emission delay on the replay clock when available.
- Have the shared evaluator build the reference clips. It joins the matching wrist label, crops the first through last target phase with 125 samples on either side where available, and applies the reference four-class mapping. Use the reference extractor for phase 1 followed by phase 2 across transitions. Retain all 14 dynamic exercises and all negatives inside those clips.
- Implement `load_model(fold)` with checkpoint and timing metadata and `predict(t_ms, sensors) -> Prediction`. Return sorted half-open sample intervals relative to the clip and the actual emission frame for online events. Replay predictions from the frozen fold checkpoint and decoder. Ground truth, clip identity and scoring stay outside the adapter.
- Use the release's greedy, one-to-one greatest-positive-overlap scorer. Split extras by the label at the prediction midpoint. Sum counts across clips before dividing TP, FN and FP by labeled reps. Take absolute count error within each session/activity/wrist clip before pooling. Preserve duration error as the release's mean of clip-level means.
- Generate the per-exercise rows, overall, and overall without leg raise and crunch for the five-fold pool and local fifth fold. Include the outcome and outlier plots, session count-error plots, and representative good/bad predictions. Use the published five-fold and final-fold columns with their respective labels and denominators. Compare local runs only after checking their common evaluation signature. Keep full-session and endpoint-tolerance F1 as separately named analyses.

Run MS-TCN via `python -m baseline.mstcn reference-benchmark --help`. For a new architecture, reuse `baseline.rep_benchmark.evaluate.evaluate` and the saved phase evaluation plan. Use `python -m baseline.rep_benchmark.compare --help` for a combined report.
