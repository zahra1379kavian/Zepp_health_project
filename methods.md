classical counter as a baseline:
temporal action segmentation with a NULL class, as in MS-TCN (Farha and Gall, 2019) and the WEAR inertial work by Bock et al..and the dense two-head design above is a direct fit. 
Classical rep counting from RecoFit (Morris et al., 2014) .
TS2Vec is a sensible way to pretrain the encoder on all 88 hours before fine-tuning, but with 36,000 labeled reps I would try plain supervised training first and add pretraining only if cross-subject results plateau. The open-set papers stay in phase 2.


Step	Goal	Method
2	Rep segmentation (main model)	Current causal MS-TCN with the phase1/phase2 head, plus three changes: add about 0.5 s of lookahead, test accelerometer + gyroscope only (no magnetometer), and add augmentation (small rotations, time-warp, scaling).
Leave-one-exercise-out test. Train with one exercise removed, then score station-ignored rep F1 on it in held-out subjects.
Mask the held-out exercise's frames during training; do not label them rest, or the model learns to ignore it.
Start with three or four exercises on one fold, not all 15.
No flag for this exists yet in temporal_segmentation/.

Make RecoFit usable as the unknown-station fallback. Its boundary offsets are calibrated per exercise, so for an unknown station it emits no intervals. It needs a pooled, exercise-independent calibration.
Decide whether a waist sensor is allowed at deployment. In leg raises and crunches the wrist barely moves, so a wrist-only model is unlikely to fix them. If the product is wrist-only, report these two as a sensor limit.

How to enhance RecoFit model to detect start and end of a repitition? Learn the edges from training data with a small regressor, or by matching a typical repetition shape for each exercise. This is likely the most accurate, but it is no longer RecoFit.


Where the MS-TCN loses reps
Of the 3,066 missed reps:
No prediction at all (47%):
Exercises: leg raise has 72% of sets missed entirely, crunch 41%, pull-up 21%.
Set edges: the first rep of a set is missed 52% of the time and the last 39%, against 24% for interior reps.
Detected but wrong boundaries (53%): one rep split into two (26%), an endpoint off by more than 400 ms (15%), or two reps merged (12%).
Phantom reps in rest: 605 over 21.6 stream-hours of rest, about 28 per hour.
Generalisation gap: training loss is 0.23 against 0.51 on validation, with only 25 training subjects per fold.
