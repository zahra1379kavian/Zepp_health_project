classical counter as a baseline:
temporal action segmentation with a NULL class, as in MS-TCN (Farha and Gall, 2019) and the WEAR inertial work by Bock et al..and the dense two-head design above is a direct fit. 
Classical rep counting from RecoFit (Morris et al., 2014) .
TS2Vec is a sensible way to pretrain the encoder on all 88 hours before fine-tuning, but with 36,000 labeled reps I would try plain supervised training first and add pretraining only if cross-subject results plateau. The open-set papers stay in phase 2.


Step	Goal	Method
1	Baseline	RecoFit-style counter: band-pass → PCA → autocorrelation peaks. It needs no station label.
2	Rep segmentation (main model)	Current causal MS-TCN with the phase1/phase2 head, plus three changes: add about 0.5 s of lookahead, test accelerometer + gyroscope only (no magnetometer), and add augmentation (small rotations, time-warp, scaling).
3	Check unknown stations	Leave-one-exercise-out: train on 14 exercises, test rep segmentation on the 15th. If it fails, fall back to the step 1 counter whenever the station is "unknown".
4	Only if step 2 plateaus	TS2Vec pretraining on all 88 hours, optionally adding public gym data (RecoFit, MM-Fit).
5	Station recognition (later)	Contrastive embedding (Lee & Kim mixup triplets, or OOD-aware SupCon), averaged over each detected set, then nearest prototype with a distance threshold for "unknown".

