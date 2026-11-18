# Activity-agnostic rep detection from wrist motion
## 2. Train MS-TCN to find complete reps

Feed the continuous wrist signal into the causal MS-TCN. Every 40 ms, train two outputs from its motion features: the probability that a rep just ended and the probability that the person is at a station.

- **End output.** Rep ends are rare, about 1 frame in 300. Spread each end target over ±2 frames (±80 ms). Report an end only at a local peak above a threshold.
- **Start output.** At each end, train a third output to choose the start from the previous 6 s, which covers 99.9% of reps, or to choose "no valid start". Include competing starts halfway through the rep and at the start of the previous rep. Train it at labeled ends, at ends shifted by up to ±2 frames, and at false ends. False ends are mid-rep turnarounds (`phase1` → `phase2`) and end peaks that the current model reports in rest. At a false end, the correct choice is "no valid start".
- **Hard negatives.** In the current model's held-out cross-validation predictions, find rest stretches with three or more reps in a row. Check a sample by hand. Mask stretches that contain real reps, and give the others extra weight in the loss.
- **Station output.** Use it only to group reps into sets. Never let it block a rep, because an unseen exercise may look like rest.

During replay, report a rep once, when its end peak is confirmed and a valid start is chosen. This rule can count the first rep without an exercise name or a previous example.

**Baselines.** Compare with the current phase model, with the existing boundary head (`--repetition-head boundary`), and with the rule "start = previous end". That rule holds for 77% of consecutive reps in a set.

## 3. Test whether XRFMamba makes better use of history

Replace the MS-TCN encoder with forward-only XRFMamba blocks. Keep the same labels, outputs, and counting rule. Retain one feature vector every 40 ms for locating boundaries. Give the station output both the current vector and a summary of earlier motion from the encoder. This lets it use context such as walking between stations, stopping to set up, and beginning a set.

## 4. Add OnPoint’s teacher–student training

Train a bidirectional XRFMamba **teacher** on the labeled training sessions. It can look ahead and use later reps to interpret an ambiguous first rep. Freeze the teacher and save its station and end probabilities at each timestamp and its start scores at each labeled end. Train a fresh copy of the streaming model from step 3 to match both the human labels and these teacher scores. Deploy only the streaming model. Comparing steps 3 and 4 isolates the effect of this extra training signal.

## 5. Compare all three models on unseen exercises

- **Replay.** Hold out entire exercises and participants from training and tuning. Replay full test sessions at 0, 160, and 320 ms of lookahead. For a decision about time `t`, the encoder and decoder together may use at most 0, 4, or 8 additional 40 ms samples. Log when both final timestamps are emitted.
- **Score.** Match predictions to labeled reps one-to-one. Report rep F1 twice, requiring both endpoints to fall within ±200 ms and then ±400 ms. Also report station recall on unseen exercises, first-rep recall, false reps per hour of walking and warm-up, and delay from the true rep end to the final report. Compare first-rep recall at a similar false-count rate, with thresholds chosen on validation sessions.

## References

- [MS-TCN: Multi-Stage Temporal Convolutional Network for Action Segmentation](https://arxiv.org/abs/1903.01945). CVPR, 2019.
- [XRF V2: A Dataset for Action Summarization with Wi-Fi Signals, and IMUs in Phones, Watches, Earbuds, and Glasses](https://doi.org/10.1145/3749521). ACM IMWUT, 2025. Introduces XRFMamba.
- [OnPoint: Offline-to-Online Multi-Level Distillation for Point-Supervised Online Temporal Action Localization](https://arxiv.org/abs/2607.00289). ECCV, 2026.
