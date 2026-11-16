classical counter as a baseline:
temporal action segmentation with a NULL class, as in MS-TCN (Farha and Gall, 2019) and the WEAR inertial work by Bock et al..and the dense two-head design above is a direct fit. 
Classical rep counting from RecoFit (Morris et al., 2014) and the CrossFit work by Soro et al. (2019). 
TS2Vec is a sensible way to pretrain the encoder on all 88 hours before fine-tuning, but with 36,000 labeled reps I would try plain supervised training first and add pretraining only if cross-subject results plateau. The open-set papers stay in phase 2.
