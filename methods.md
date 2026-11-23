# Activity-agnostic rep detection from wrist motion

## Reference wrist benchmark

The August 31 Feishu activity-agnostic wrist release is the comparison for the 14-exercise benchmark. Run each model through the [shared evaluation API](baseline/MS-TCN/README.md#compare-with-the-feishu-wrist-release). Reuse the sensor-only test clips, frozen subject-fold checkpoints, release rep extractor, overlap matcher and KPI denominators. The existing endpoint-tolerance F1 experiments below answer a separate timing question.

| Five folds pooled | Labeled reps | TP % | FN % | FP % |
| --- | ---: | ---: | ---: | ---: |
| Published reference, p. 7 | 41,363 | 92.7 | 7.3 | 4.4 |
| MS-TCN phase | 41,345 | 83.6 | 16.4 | 5.6 |
| MS-TCN boundary | 41,345 | 78.4 | 21.6 | 10.4 |
| MS-TCN complete rep | 41,345 | 69.4 | 30.6 | 1.5 |
| MS-TCN previous end | 41,345 | 85.1 | 14.9 | 82.1 |

All 51 available sessions were checked. The release crop rule yielded 1,086 clips from 47 sessions and 83 wrist streams. Four available sessions contained no eligible wrist exercise clips. The local runs share identical clips and fold assignments. The released weights were unavailable, so the reference row uses the published figures. The local fifth fold and the published final fold are identified separately in the report. [Combined report](baseline/MS-TCN/experiment_runs/feishu_wrist_rep_comparison/all_models_summary/wrist_model_comparison.pdf) · [Per-exercise results](baseline/MS-TCN/experiment_runs/feishu_wrist_rep_comparison/all_models_summary/model_comparison.csv) · [Loader and scorer verification](baseline/MS-TCN/experiment_runs/feishu_wrist_rep_comparison/verification.json)

## Step 1 — Finish the complete-rep model and break down its errors

The complete-rep and boundary runs finished all five folds with seed 42 and identical subject splits.
Add the error breakdown to metrics.py so every report includes it:
extra reps, split into inside a real rep / in rest / between reps;
missed reps, split into overlapped by a prediction / not detected;
recall by position in the set (1st, 2nd, middle, last);
recall by rep duration;
share of sets missed entirely, per exercise;
reporting delay at the median, 95th percentile and maximum, instead of only the mean.
Compare four versions fold by fold: the phase model (your existing run), the boundary model, the complete-rep model, and the complete-rep model with the start-equals-previous-end rule.
Adopt the complete-rep model as the default if it wins F1@400 in at least 4 of 5 folds and cuts extras inside real reps (7,968 in the phase report) by at least 30%.
If split reps are still the biggest error group, add this decoder rule. It needs no retraining; tune it on validation:
Within a set, reject a rep shorter than f × the median duration of the last 3 reported reps, trying f = 0.5, 0.6 and 0.7. Reset the rule after 10 s with no reps.
It adapts to each person's own pace, so it doesn't depend on the exercise. It is a streaming version of the duration and confidence post-processing in OnlineTAS (NeurIPS 2024). That paper shows models that see only past input break one action into fragments more often than offline models.
If start/end error stays above about 100 ms, which drags down the ±200 ms score, refine the end time as a weighted average over ±4 frames around each peak. This is the boundary-as-distribution idea from TriDet (CVPR 2023); your start selector already works this way.
Fallback: Bock et al. (IMWUT 2024) found that action-localization models of the ActionFormer type, trained on raw IMU signals, beat standard activity-recognition models on 4 of 6 datasets. They are offline models and would need to be made causal.

### Step 1 results

All four variants have the requested error breakdown. The end-time experiment reused the five complete-rep checkpoints and their validation-selected thresholds. It weighted the nine timestamps around each end peak by end probability and used the refined end in the decoder's existing overlap check. [Phase report](baseline/MS-TCN/experiment_runs/wrist_cross_validation/report_with_rep_diagnostics.json) · [Boundary report](baseline/MS-TCN/experiment_runs/boundary_rep_cv/cross_validation/report.json) · [Complete-rep and previous-end report](baseline/MS-TCN/experiment_runs/complete_rep_cv/previous_end_evaluation/report.json) · [Refinement report](baseline/MS-TCN/experiment_runs/complete_rep_cv/end_time_refinement/report.json)

| Fold | Phase F1@400 | Boundary F1@400 | Complete-rep F1@400 | Previous-end F1@400 | Complete-rep F1@200 | Refined F1@200 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.640 | 0.528 | 0.591 | 0.349 | 0.443 | 0.402 |
| 2 | 0.638 | 0.534 | 0.620 | 0.362 | 0.465 | 0.425 |
| 3 | 0.715 | 0.575 | 0.662 | 0.470 | 0.520 | 0.485 |
| 4 | 0.658 | 0.538 | 0.643 | 0.340 | 0.470 | 0.439 |
| 5 | 0.654 | 0.396 | 0.626 | 0.298 | 0.483 | 0.445 |
| Pooled | 0.662 | 0.518 | 0.629 | 0.364 | 0.477 | 0.440 |

- **Default model.** Keep the phase model. Complete rep cut extras inside real reps by 33.9% (7,968 to 5,270), and phase had the highest F1@400 in all five folds.
- **Duration filter.** The two missed-repetition categories (9,304 with overlap and 8,189 without overlap) each exceeded the split-repetition count (5,270). The filter's condition was not met across all error categories.
- **End-time refinement.** The original complete-rep decoder's average end error was 108.0 ms at ±400 ms. The refinement reduced it to 107.7 ms, lowered F1@200 in all five folds, and increased 95th-percentile reporting delay from 200 to 280 ms at ±200 ms. Use the original peak timestamp for subsequent runs. The refinement remains available through `--refine-end`.

## Step 2 — Make the rep detector work for any exercise
2.0 Set up the evaluation first (no new model yet)
Let the leave-one-exercise-out analysis accept --repetition-head complete. It currently has phase_only=True hard-coded in analyses/leave_one_exercise_out/__init__.py.
Test with exercises that score well when seen, one or more per movement group. Use a second held-out exercise (E′) from a different group to select the checkpoint and decoder on validation subjects.
Test exercise	Selection exercise E′ (also removed from training)
biceps_curls	bench_press
lat_pulldown	lunges
push-up	rowing
overhead_press_(ohp)	deadlift
squat	pull-up
hip_thrust	dips
Average F1@400 over the unseen exercises. For the main result, train the selected variant on all exercises within each training fold and compare its pooled TP%, FN% and FP% with the phase baseline. [Reference wrist benchmark](#reference-wrist-benchmark)
Guard rails: F1 on seen exercises, crunch F1 (seen), false reps per rest hour, and first-rep recall.
Screen variants cheaply: use subject fold 1 and biceps / push-up / squat, which already have baselines (0.01 / 0.49 / 0.40). Confirm the winner on all 6 exercises and at least 2 subject folds.
Free diagnostic first: re-score the existing biceps-curl leave-one-out model across a range of decoder thresholds. This is a diagnostic only; never report it as a result.
If F1 rises, check whether a threshold selected on E′ transfers. If it stays near zero, inspect the missed reps and endpoint errors.

2.1 Make the rep inputs independent of wrist orientation
Estimate the gravity direction causally: low-pass the accelerometer at about 0.3 Hz and normalize it.
Feed the rep model 6 channels:
overall accelerometer magnitude;
movement acceleration along gravity, and its magnitude across gravity;
overall gyroscope magnitude;
gyroscope rotation about gravity, and its magnitude across gravity.
Optionally add a 7th: the projection onto the main movement axis of the last 5 s. This is the classic exercise-independent counting signal from RecoFit (CHI 2014).
Drop the magnetometer from the rep inputs. It encodes compass heading and the magnetic field near steel machines, which identifies the station. Keep it in the 15-class station classifier: in fold 1, dropping it raised false exercise events from 159 to 262 per rest hour.
Compare against raw accelerometer + gyroscope with fully random 3-D rotation of each training window (Um et al., ICMI 2017; see also TRI-HAR, ISWC 2026). Your current augmentation rotates by only ±10°, which leaves the orientation cue intact. So your earlier augmentation ablation didn't actually test this.

**Implementation and runs.** The paired evaluation supports the complete-rep head. The fold-1 screen compares the original nine inputs, six gravity-based inputs and six randomly rotated raw inputs on biceps curls, push-up and squat. Each fit uses the same reviewed rest intervals. The optional seventh channel is omitted. The biceps threshold sweep is saved separately for diagnosis. [Run instructions and scoring](baseline/MS-TCN/mstcn/analyses/leave_one_exercise_out/README.md) · [Experiment directory](baseline/MS-TCN/experiment_runs/step_2/)

2.2 Add a causal check for repeated motion (RepNet, CVPR 2020; TransRAC, CVPR 2022)
After stage 1, compute a 32-dimensional normalized embedding z_t for each frame.
At each frame, compute s_t(k) = z_t · z_{t−k} for k = 1…150 (the last 6 s). This is one row of RepNet's self-similarity matrix, computed using only the past.
Use it in two places:
(a) Pass it through a 1×1 conv (150→64) into stages 2–3.
(b) Add α·s_t(k) to the start selector's score for "start = t−k", with α learned and starting at 1. A rep ends when the wrist returns to its starting pose, which holds for nearly every strength exercise and also works for the first rep.
In RepNet's ablation, this similarity layer was what let training on synthetic videos carry over to real ones: error on the QUVA benchmark fell from 1.29 to 0.10.
Later option: use the first detected rep of a set as a template and add the current window's similarity to it as an input (ESCounts, ACCV 2024; CaRaCount, IEEE TPAMI 2025, which is wearable-specific).

**Implementation.** Use `--repetition-head complete --causal-similarity` to enable both uses of the six-second motion history. The fold-1 screen uses the same subjects, exercise pairs, and reviewed rest intervals as step 2.1. Each input choice has its own matched control. Use the comparison for the input selected in step 2.1 to judge the additional benefit of similarity. [Model and experiment instructions](baseline/MS-TCN/mstcn/analyses/leave_one_exercise_out/README.md#step-22-causal-motion-similarity)

2.3 Train on synthetic repetitions (RepNet; CaRaCount)
In CaRaCount's AAAI 2024 version, pretraining on synthetic data was the largest single gain (counting error 10.05 → 7.66).
Take a 0.6–2 s piece of real training signal that starts in a still pose (a local minimum of gyroscope magnitude). Never take it from E or E′.
Append its time reversal to make one cycle that returns to the start pose:
accelerometer: reversed in time;
gyroscope: reversed in time and sign-flipped, which is what playing the motion backward produces.
Repeat the cycle 3–15 times, varying each cycle's speed by ×0.85–1.15 and amplitude by ×0.9–1.1. Add occasional 0–1 s pauses.
Apply one random 3-D rotation to the whole sequence, and pad each side with 5–30 s of real rest.
Label a start and end at every cycle boundary. Make 25% of each batch synthetic.
Risk: walking is also periodic. Keep real rest as negatives, with your reviewed hard-negative weights. Don't take source pieces whose accelerometer repeats at 0.4–0.7 s (step rhythm), and watch false reps per rest hour.

**Validation.** The fold-1 screen compares synthetic training with the step 2.2 gravity-input model on biceps curls, push-up and squat. Both use the same real validation subjects, decoder selection exercise and test recordings. With a batch size of eight, synthetic training uses six real windows and two generated windows per update. [Generation and scoring details](baseline/MS-TCN/mstcn/analyses/leave_one_exercise_out/README.md#step-23-synthetic-repetitions) · [Screen logs](baseline/MS-TCN/experiment_runs/step_2_3/)

2.4 Only if a large gap remains: add a head that tries to predict the exercise from stage-1 features, trained with reversed gradients (DANN, JMLR 2016), ramping its weight from 0 to 0.1.
Add these one at a time (2.1 → 2.2 → 2.3). Keep a change only if:
unseen F1 rises by at least 0.05,
seen F1 drops by no more than 0.02, and
false reps per rest hour rise by no more than 20%.

## Step 3 — Check the teacher's best case, then lookahead, then distillation
3.1 Teacher ceiling
Add an --acausal flag: in Residual, pad (d, d) instead of (2d, 0). Use the step-2 inputs.
Train one teacher per fold, on that fold's training subjects only.
Run it on whole sessions offline and measure first-rep recall, last-rep recall, F1, and unseen-exercise F1.
Continue to 3.3 only if the teacher's first-rep recall beats the student's by at least 0.15, or its unseen F1 by at least 0.05. If not, the future signal doesn't contain the missing information, and distillation can't add it.

3.2 Lookahead sweep (existing --lookahead-ms)
Run 0 / 160 / 320 / 480 / 640 ms on fold 1. Score first- and last-rep recall and the 95th-percentile delay from true end to report, which must stay under 1000 ms.
Pick the smallest lookahead that captures at least 80% of the teacher's first/last-rep gain, then confirm it on 5 folds.

3.3 Distillation
Loss:
the usual label loss;
λ₁ × a term matching the student's end probability to the teacher's (temperature 2), applied on all valid frames, including masked-reference frames where only the teacher gives a signal;
λ₂ × the KL divergence between the teacher's and student's 151-way start choice at labeled ends and at teacher-detected ends.
Try λ ∈ {0.3, 1, 3} and pick on validation.
From OnPoint (ECCV 2026): also turn the teacher's confident reps (>0.9) inside currently masked intervals into extra training reps.
If the student can't keep up with a teacher that sees the whole session, first use a teacher limited to 2 s of future. This follows the curriculum of increasingly future-aware teachers in Privileged Knowledge Distillation (Zhao et al., arXiv 2020).
Later option: Dual-mode ASR (ICLR 2021) trains one network, with shared weights, in both streaming and full-context modes, with distillation inside the same model. It improved both accuracy and latency of the streaming mode.

## Step 4 — Report leg raise separately as sensor-limited
Put the motion table above in the paper. State the rule as data-based, not F1-based: both wrist motion ratios below about 0.3× rest. Only leg raise qualifies.
Headline results: the other 12 dynamic exercises. Then a separate leg-raise row, then all 13 pooled for transparency. Report plank as hold duration. Crunch stays in the headline.
When sensor_ablation_five_folds finishes, check leg-raise F1 with the waist sensor across all 5 folds. Offer wrist+waist as a secondary setup only if the gain holds in at least 4 folds. In fold 1, waist also dropped pull-up from 0.66 to 0.43.
Optional: CaRaCount evaluated on the public RecoFit and MM-Fit datasets. These could serve as external unseen-exercise tests if their labels fit your scoring; count-only labels would give count error but not boundary F1.
###########################
<!-- ## 2. Train MS-TCN to find complete reps

Feed the continuous wrist signal into the causal MS-TCN. Every 40 ms, train two outputs from its motion features: the probability that a rep just ended and the probability that the person is at a station. Use only the labeled start and end of each rep. Never use phase labels in training or testing, so the same model works for reps with one movement or with movements in either order.

- **End output.** Rep ends are rare, about 1 frame in 300. Spread each end target over ±2 frames (±80 ms). Report an end only at a local peak above a threshold.
- **Start output.** At each end, train a third output to choose the start from the previous 6 s, which covers 99.9% of reps, or to choose "no valid start". Include competing starts halfway through the rep and at the start of the previous rep. Train it at labeled ends, at ends shifted by up to ±2 frames, and at false ends. False ends are the midpoint of each labeled rep and end peaks that the current model reports in rest. At a false end, the correct choice is "no valid start".
- **Hard negatives.** In the current model's held-out cross-validation predictions, find rest stretches with three or more reps in a row. Check a sample by hand. Mask stretches that contain real reps, and give the others extra weight in the loss.
- **Station output.** Use it only to group reps into sets. Never let it block a rep, because an unseen exercise may look like rest.

During replay, report a rep once, when its end peak is confirmed and a valid start is chosen. This rule can count the first rep without an exercise name or a previous example.

**Baselines.** Compare with the existing boundary head (`--repetition-head boundary`), which also uses only start and end labels, and with the rule "start = previous end". That rule holds for 77% of consecutive reps in a set. -->

<!-- ## 3. Test whether XRFMamba makes better use of history

Replace the MS-TCN encoder with forward-only XRFMamba blocks. Keep the same labels, outputs, and counting rule. Retain one feature vector every 40 ms for locating boundaries. Give the station output both the current vector and a summary of earlier motion from the encoder. This lets it use context such as walking between stations, stopping to set up, and beginning a set.

## 4. Add OnPoint’s teacher–student training

Train a bidirectional XRFMamba **teacher** on the labeled training sessions. It can look ahead and use later reps to interpret an ambiguous first rep. Freeze the teacher and save its station and end probabilities at each timestamp and its start scores at each labeled end. Train a fresh copy of the streaming model from step 3 to match both the human labels and these teacher scores. Deploy only the streaming model. Comparing steps 3 and 4 isolates the effect of this extra training signal.

## 5. Compare all three models on unseen exercises

- **Replay.** Hold out entire exercises and participants from training and tuning. Replay full test sessions at 0, 160, and 320 ms of lookahead. For a decision about time `t`, the encoder and decoder together may use at most 0, 4, or 8 additional 40 ms samples. Log when both final timestamps are emitted.
- **Score.** Match predictions to labeled reps one-to-one. Report rep F1 twice, requiring both endpoints to fall within ±200 ms and then ±400 ms. Also report station recall on unseen exercises, first-rep recall, false reps per hour of walking and warm-up, and delay from the true rep end to the final report. Compare first-rep recall at a similar false-count rate, with thresholds chosen on validation sessions.

## References

- [MS-TCN: Multi-Stage Temporal Convolutional Network for Action Segmentation](https://arxiv.org/abs/1903.01945). CVPR, 2019.
- [XRF V2: A Dataset for Action Summarization with Wi-Fi Signals, and IMUs in Phones, Watches, Earbuds, and Glasses](https://doi.org/10.1145/3749521). ACM IMWUT, 2025. Introduces XRFMamba.
- [OnPoint: Offline-to-Online Multi-Level Distillation for Point-Supervised Online Temporal Action Localization](https://arxiv.org/abs/2607.00289). ECCV, 2026. -->
