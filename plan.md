# Study Plan: Repetition Segmentation and Strength-Training Station Recognition

## Objectives and priorities

The project has two objectives, in this order:

1. **Immediate objective — repetition segmentation and counting.** Use continuous wearable sensor data to count repetitions at each station and identify the **start and end time of every individual repetition**. A set contains multiple repetitions; detecting set starts and ends is a by-product of repetition segmentation.
2. **Type of strength training station is not know**. So you need to handle unknow stations and segment the reps.**Do not assume we can input station ID to the model**.
 The reporting delay of the rep counting should be **less than 1s** after the end of the rep.  **we don't know which exercise a subject complete during a recording** for our new data, so don't use which stations have done for a session.
3. **Later objective — recognition of strength-training station types.** Develop broader station-type recognition after successful development of the repetition-segmentation model.

The immediate pipeline is **classify the current stations and rest → segment individual repetitions → count repetitions per station**. Classification of the currently labeled stations is a supporting step in the immediate objective; broader station-type recognition is the later research objective. At inference, the pipeline must predict station/rest labels from sensors rather than receive true station labels, set boundaries, or repetition boundaries as inputs. New manual labels for inter-set transitions are not a prerequisite.

The primary output is a sequence of repetitions, each with a predicted station label, start timestamp, and end timestamp, together with a repetition count per station within the session. Set boundaries can be derived by grouping repetitions and intervening rest, but set detection is not the primary deliverable.

## Dataset

Start with `strength_training/strength_training_data_clean`. The current annotations contain **15 exercise types**. Use these current categories and rest for the initial classification stage supporting repetition segmentation. Growth to approximately 100 station types, including previously unseen exercises, belongs to the later station-recognition objective.

The overview below was calculated directly from all 51 `label.csv` files, all 178 sensor CSVs, and the matched workout-log entries on November 16, 2026. Original data files were left unchanged.

### Size, subjects, and session duration

| Item | Verified result |
| --- | --- |
| Subjects | **51 distinct `masked_id` values** in the cleaned recordings |
| Sessions | **51**, with one recorded session per subject in this cleaned subset |
| Sessions containing exercise annotations | **47**; the other **4 contain only `rest` labels** and require review |
| Collection period | June 2–17, 2026, across 12 collection dates, using folder/log dates |
| Total recording duration | **88.06 hours**, counting each session once |
| Mean / median session duration | **103.60 / 102.91 minutes** |
| Middle 50% of session durations | **84.69–124.50 minutes** |
| Shortest / longest session | **10.18 / 165.64 minutes** |
| Exercise types / exact label strings | **15 / 44**, including phase variants, transitions, plank, and rest |
| Label timeline | **7,925,698 time points**, regularly spaced at **40 ms (25 Hz)** |
| Sensor data | **178 files**, containing **27,042,909 rows across devices** |
| CSV inventory | **232 files**: 178 sensor files, 51 label files, and 3 root-level metadata/cleanup files |
| On-disk size | **5.17 GB** including metadata and hidden files |

The 47 sessions with exercise labels account for **83.71 hours**; their median duration is **106.41 minutes**, with a range of **55.41–165.64 minutes**. This is a description of annotation availability, not a guarantee that every activity in those sessions is correctly labeled.

Session duration is `(last sync_timestamp - first sync_timestamp) / 1000`. Label-duration totals below use `sample_count × 0.04 s`, so they include one additional sample period per session: 2.04 seconds across the entire dataset. Sensor rows from multiple devices measure overlapping time and must not be added together to estimate workout hours.

The full workout log contains **2,410 rows, 52 distinct IDs, and 53 date–ID sessions**. Only **2,329 rows** match the cleaned folders. Exclude the 81 rows from `20260529_DQTQDQTQ` and `20260529_JCLQJCLQ` when describing this cleaned subset. The latter ID has a different, retained session on June 4. Count complete `masked_id` strings as subjects; `S1`, `S2`, etc. identify waist devices, and four-letter pieces of a masked ID are not separate people.

![Dataset size, session durations, collection dates, label balance, and sensor configurations](strength_training/data_demographics/figures/01_dataset_overview.png)

### Participant profile

Demographics are counted once per matched subject, rather than once per workout-log row. These are recorded metadata values; ages were not recalculated from birth dates.

| Attribute | Summary |
| --- | --- |
| Recorded gender | 43 male (84.3%); 8 female (15.7%) |
| Age | Median 29 years; mean 30.9; range 19–51 |
| Height | Median 175 cm; range 163–192 cm |
| Weight | Median 75 kg; range 49–115 kg |
| BMI | Median 24.49; range 16.96–36.71 |

Height and weight are interpreted as cm and kg from the recorded values. No conflicting demographic values were found after normalizing numeric formats. There are no repeat sessions per subject in the cleaned subset, so it cannot directly measure performance across different visits by the same person. The gender imbalance also limits subgroup estimates.

![Participant gender, age, and BMI distributions](strength_training/data_demographics/figures/02_participant_profile.png)

### Exercise coverage

Normalize suffixes `_phase1`, `_phase2`, and `_transition` to obtain exercise types. `plank` is already an exercise label without those suffixes. A session contains an exercise if any available label location names it. Minutes below include transitions and count each timestamp once across locations; the audit found no simultaneous conflicts between different exercise types.

| Exercise | Sessions with labels, out of 51 | Annotated minutes |
| --- | ---: | ---: |
| Plank | 42 | 148.82 |
| Lunges | 45 | 105.93 |
| Squat | 46 | 71.20 |
| Deadlift | 45 | 66.65 |
| Lat pulldown | 47 | 60.35 |
| Rowing | 47 | 58.10 |
| Biceps curls | 47 | 56.49 |
| Bench press | 47 | 56.42 |
| Overhead press (OHP) | 47 | 52.46 |
| Push-up | 46 | 46.38 |
| Leg raise | 40 | 44.88 |
| Hip thrust | 43 | 43.10 |
| Dips | 45 | 41.92 |
| Pull-up | 42 | 41.90 |
| Crunch | 26 | 31.90 |

Only **24 of 51 sessions contain all 15 exercise types**; the median is 14 types per session. Absence of a label is not proof that the participant did not perform the exercise, especially in the four sessions flagged below. Crunch has substantially less participant coverage than the other types, while plank supplies the most labeled time.

![Number of sessions and annotated minutes per exercise](strength_training/data_demographics/figures/03_exercise_coverage.png)

[View the full participant–exercise heatmap](strength_training/data_demographics/figures/04_subject_exercise_matrix.png).

### Station/rest targets, repetition references, and annotation balance

`label.csv` is the source of labels. Its columns are `sync_timestamp`, `waist`, and whichever of `left_wrist` and `right_wrist` are available. For this overview, each timestamp is assigned to:

1. **Exercise:** at least one location contains a phase label or `plank`.
2. **Transition:** at least one location contains an exercise transition, with no phase/plank at that timestamp.
3. **Rest:** every available location says `rest`.

| Annotation group | Hours | Share of timeline |
| --- | ---: | ---: |
| Rest at all available locations | 72.62 | 82.47% |
| Exercise phases / plank at any location | 14.08 | 15.99% |
| Transition only | 1.36 | 1.54% |
| Exercise plus transitions | **15.44** | **17.53%** |

For the initial station/rest classifier, normalize the exercise suffixes to the corresponding current station class and retain `rest` as its own class. Preserve the original phase and transition labels separately for developing repetition segmentation; collapsing them for classification must not discard their timing information. Exercise-specific transitions can retain their station association for classification, but they must not automatically count as repetitions. A binary exercise–rest baseline can provide a diagnostic comparison, but it does not fulfill the repetition-segmentation objective. The combined-location rule above is a summary convention, not an established definition of station occupancy. A model using one body location may instead need the corresponding label column.

**Validate repetition references before training or scoring repetition boundaries.** Check what `_phase1`, `_phase2`, and `_transition` mean and whether their ordered sequences identify a complete repetition with reliable start and end times. Define a consistent full-repetition convention for each applicable exercise, including how to handle pauses and incomplete cycles. Phase changes alone are not verified repetition boundaries. Use existing annotations or other available references only where their repetition interpretation is supported, and document gaps in reference coverage. Plank is a static hold: report its duration separately rather than assigning artificial repetition counts.

**Rest is defined relative to the project's exercise categories, not physical inactivity.** An activity outside the current exercise categories can correctly have a `rest` label even when it involves substantial movement or effort. In particular, `spull` is intentionally rest under the current category definition; its treatment as a held-out new exercise is specified below. The descriptive counts above precede these exclusions and are not the final training/evaluation balance.

Available location labels disagree on the exact label at **21,409 timestamps (0.27%)**, including **9,851 timestamps (0.12%)** with binary rest/non-rest disagreement. Do not silently substitute wrist labels for waist labels. A classifier that always predicts rest would achieve **82.47% sample accuracy**, so accuracy alone would be misleading.

Collapsing phase suffixes produces **2,030 contiguous non-rest exercise runs**, with median duration **22.56 seconds** and range **1.08–300 seconds**. These are annotation runs, not verified set or repetition counts. There are also 19 rest runs shorter than one second. The model must resolve individual repetitions and their boundaries within the sensor sequence, including multiple repetitions within an exercise run. Do not assume every annotation run is one repetition or one complete set. Existing exercise-specific `_transition` labels must not automatically be interpreted as repetition boundaries or transitions between sets.

![Example annotations for a median-length session and the shortest session](strength_training/data_demographics/figures/05_example_timelines.png)

Use this picture guide to match the exercise names in the timeline to their movements. Each card pairs the English name with a short memory cue. Gray shows the starting position, teal shows the finishing position, and arrows show the movement; plank is a hold.

![Picture guide to all 15 strength-training exercises, with names, simple poses, and movement arrows](strength_training/data_demographics/figures/05b_exercise_picture_guide.png)

[Open the full-size exercise picture guide](strength_training/data_demographics/figures/05b_exercise_picture_guide.png). Created with the built-in imagegen tool; [generation prompt](strength_training/data_demographics/figures/05b_exercise_picture_guide.prompt.txt).


### Sensor layout and alignment

Each sensor CSV has `sync_timestamp`, `timestamp`, and nine sensor channels: `acc_x/y/z`, `gyro_x/y/z`, and `mag_x/y/z`. Sensor units are not declared in the headers and should be verified before applying physical thresholds. Waist files have no embedded labels; wrist files contain `labels`, with one duplicate-column exception noted below.

| Files available per session | Sessions |
| --- | ---: |
| Two waist devices + both wrists | 36 |
| One waist device + both wrists | 4 |
| One waist device + left wrist | 6 |
| One waist device + right wrist | 5 |

Every session has at least one waist stream and one wrist stream. Across sessions there are **87 waist**, **46 left-wrist**, and **45 right-wrist** files. Multiple waist files are devices within one session, not additional sessions or participants.

Join sensors to `label.csv` using **`sync_timestamp`**. All sensor timestamps fall on their session's 40 ms label grid, but streams can cover only part of that grid. The per-file median difference between `timestamp` and `sync_timestamp` ranges from −2.70 to +30.24 seconds; the clocks must not be treated as interchangeable.

The [alignment script](strength_training/prepare_data.py) implements this exact-clock join and writes `strength_training/strength_training_data_aligned`, with separate sensor-input and per-sensor target CSVs, full-grid sensor-availability flags, and a count manifest. All embedded wrist label columns, including duplicate headers, are omitted from inputs; targets preserve the corresponding location's authoritative labels. Source files remain unchanged. See the [usage instructions](strength_training/README.md). This stage does not apply the training/evaluation exclusions below.

### Findings to resolve before model evaluation

**Four sessions contain no exercise labels at any location**, totaling **4.35 hours**:

| Session | Duration, minutes | Matched workout-log rows | Review reason |
| --- | ---: | ---: | --- |
| `20260604_JCLQJCLQ` | 10.18 | 5 | Log contains `action` and four `spull` entries |
| `20260604_JCLQVGFFNHJI` | 91.17 | 54 | Log lists many of the 15 named exercises, but labels are entirely rest |
| `20260604_SSCDDQTQHOPK` | 10.74 | 5 | Log contains `action` and four `spull` entries |
| `20260604_WTTANHJIOBWA` | 148.93 | 52 | Log lists many of the 15 named exercises, but labels are entirely rest |

**The four “all-rest” sessions are not trustworthy examples of continuous rest.** The two short sessions are sled-only recordings apart from calibration; their `rest` labels are consistent with `spull` being outside the current exercise categories. The two long sessions contain logged workouts without corresponding exercise annotations.

**Exclude all four complete sessions from training and evaluation:**

- `20260604_JCLQJCLQ`
- `20260604_JCLQVGFFNHJI`
- `20260604_SSCDDQTQHOPK`
- `20260604_WTTANHJIOBWA`

Apply this exclusion to every sensor stream, timestamp, and derived window in these sessions, including rest-negative sampling. The whole-session exclusion also takes precedence over the `spull` probe: use only recordings from other eligible sessions for that probe. Preserve the source files for inspection. This coverage issue is separate from the intended task of classifying stations/rest and segmenting repetitions; it does not create a requirement to manually label inter-set transitions.

**Some logged activities are absent from the label vocabulary.** The matched workout log contains 84 `spull`, 51 `action`, and 11 `Other` entries. Named-exercise timing agreement supports an inferred log-to-epoch offset of **UTC+8 in 44 sessions**; seven sessions could not be aligned confidently. This offset is inferred from the data, not documented timezone metadata. In those 44 sessions, all overlapping samples in **60 valid `spull` intervals, totaling 60.65 minutes**, are labeled rest at every available location. Log remarks for `spull` include weighted work such as `100kg`. The same alignment also places overlapping `action` and `Other` entries in rest. Apply the following agreed handling; these entries must not silently enter the training data as rest negatives.

- **`spull` / 拉雪橇 — loaded sled pulling:** this is outside the current exercise categories, so its existing `rest` label is intentional. For the planned new-exercise experiment, reserve it as a previously unseen exercise and exclude its intervals from training, including rest-negative sampling. Keep it outside the standard scored evaluation and retain it as a separate unseen-exercise probe. Do not relabel the source data or score new-exercise detection against its existing rest labels.
- **`action` / 校准 — calibration:** the synchronization-jump routine occurs once per participant. Exclude calibration intervals from both training and evaluation.
- **`Other` — sensor-fell-off or video-failure notes:** do not automatically convert these entries into an exercise class or a rest label. Exclude the affected intervals from both training and evaluation because their measurements or labels cannot be trusted.

Use exclusion masks while preserving the source recordings and labels. Resolve uncertain log-to-sensor alignment before constructing interval masks; a missing or uncertain alignment must not silently make an affected interval eligible as rest.

**Sensor coverage is variable despite all cleanup-summary statuses being `OK`.** Coverage of the complete label timeline ranges from **44.01% to 100% per sensor file**, with median **99.68%**. In `20260604_HOPKVVZPWTTA`, the two waist files cover only 44.01% and 46.80% of the timeline. Coverage of any-location non-rest samples ranges from **53.73% to 100%** among files from the 47 sessions with exercise labels. Use sensor-availability masks and evaluate only timestamps with the required inputs. Coverage of exercise samples is undefined for the four all-rest sessions.

**The sensor audit found 224 internal timestamp gaps across 57 files**, corresponding to 3,582 missing 40 ms grid slots across devices; the largest consecutive timestamp jump is 4.08 seconds. It found no duplicate or backward sensor timestamps, no off-grid sensor rows, and no missing/nonfinite values in the nine sensor channels. These checks do not establish calibration quality, original sampling rates, or whether earlier cleaning interpolated samples.

**One wrist file has two identical `labels` headers:** `20260602_LRFVFZEIVGAM/left_wrist.csv`. Both columns match the authoritative left-wrist labels. Across all wrist files, embedded labels match `label.csv` at every sensor timestamp. Use `label.csv` during loading and remove redundant embedded label columns from model inputs to prevent target leakage.

![Label-location comparison and actual sensor coverage](strength_training/data_demographics/figures/06_label_and_sensor_quality.png)

### Analysis artifacts and reproduction

| Artifact | Contents |
| --- | --- |
| [Session table](strength_training/data_demographics/tables/sessions.csv) | All 51 durations, subject IDs, station counts, annotation balance, and sensor availability |
| [Subject table](strength_training/data_demographics/tables/subjects.csv) | One demographic record per retained subject |
| [Exercise table](strength_training/data_demographics/tables/stations.csv) | Coverage and annotated minutes for all 15 exercise types |
| [Sensor audit](strength_training/data_demographics/tables/sensor_audit.csv) | File-level coverage, gaps, schema issues, and embedded-label checks |
| [Sessions requiring label review](strength_training/data_demographics/tables/sessions_requiring_label_review.csv) | The four all-rest sessions |
| [Label intervals](strength_training/data_demographics/tables/label_intervals.csv) | Contiguous normalized exercise/rest annotation runs |
| [Workout-log agreement](strength_training/data_demographics/tables/log_scene_label_agreement.csv) | Agreement for valid intervals in confidently aligned sessions |
| [Machine-readable summary](strength_training/data_demographics/summary.json) | Aggregate counts and statistics |
| [Figure directory](strength_training/data_demographics/figures) | Six dataset charts and an illustrated exercise-name guide (PNG) |

Reproduce from the repository root using Python 3.12 or a compatible environment:

```bash
python3 -m pip install -r strength_training/data_demographics/requirements.txt
python3 strength_training/data_demographics/analyze_dataset.py
```

## Key Challenge

Rest periods may include walking, warm-ups, stretching, and other movements that resemble exercise. The first stage must distinguish the current station classes from rest under the project's category definition. The repetition stage must then identify complete movement cycles and their start and end times despite differences in speed, range of motion, pauses, and sensor placement. Correct exercise–rest classification alone does not establish correct repetition segmentation or counts.

**The target boundaries are the start and end of each repetition.** A set contains multiple repetitions, so predicting only the first and last movement of a set does not meet the immediate objective. Infer repetitions from continuous data after station/rest classification, without requiring true station identity, pre-segmented sets, or manually labeled inter-set transitions at inference. Ordinary inter-set rest and walking must remain in otherwise valid recordings; uncertain boundaries alone are not a reason to mask them. Grouping detected repetitions into sets is a secondary output.

The separate data-quality issues are incomplete annotation coverage in the flagged sessions, calibration, failed recordings, and partial sensor coverage. Apply the agreed exclusions without treating all movement outside the exercise categories as a labeling error. Repetition counts and boundary timing can be scored only where validated references support them. Document missing repetition references as an evaluation limitation; do not invent counts or boundaries from contiguous exercise runs or require a new manual inter-set transition-labeling task.

## Study Plan

### Phase 1 — immediate: repetition segmentation and counting

1. Exclude the complete recordings of all four sessions named above from training and evaluation before splitting subjects or generating windows. Apply the agreed interval exclusions for `spull`, calibration, and `Other` in the remaining sessions, and reserve eligible `spull` recordings for the later new-exercise probe. Do not require new manual labels for ordinary inter-set transitions or set boundaries.
2. Load labels from `label.csv`, join on `sync_timestamp`, remove label columns from inputs, and retain masks for missing sensor samples. Keep both normalized station/rest labels and the original phase/transition annotations. Validate the repetition-reference convention and identify which exercises and intervals support count and boundary evaluation; handle static holds separately.
3. Split by subject before creating windows, keeping all devices and windows from one session in the same partition. Fit normalization and select thresholds on training/validation subjects only. Compare waist and wrist baselines on clearly defined availability cohorts.
4. **Classify the current stations and rest.** Build the initial classifier using the 15 current exercise types plus rest. Report station/rest macro F1, per-class precision and recall, and a confusion matrix. Also report exercise–rest F1 and false exercise events per verified rest hour as diagnostic measures. These results support the repetition task; they are not its completion criterion.
5. **Segment and count individual repetitions.** Use the predicted station/rest sequence together with sensor data to identify each complete repetition and its start and end timestamp, handling pauses and incomplete movements explicitly. Return a station label and timestamps for every detected repetition, and aggregate the repetition count per station within the session. Derive set boundaries from groups of repetitions only as a by-product.
6. **Evaluate the complete repetition pipeline.** On intervals with validated references, report repetition-count mean absolute error and exact-count accuracy per station/session, repetition-detection precision/recall/F1 with one-to-one matching and a declared timing tolerance, and start-time and end-time absolute errors for matched repetitions. Report missed and extra repetitions alongside timing error. Select matching rules and tolerances on validation data. Evaluate using predicted station/rest labels so classification errors are reflected in the final repetition results; any comparison using reference station labels must be marked as a diagnostic experiment.
7. Report per-subject and per-exercise results alongside pooled metrics, document exclusion counts and repetition-reference coverage, and keep predictions without references separate from scored results. Document that this subset has one session per subject and uneven demographic/exercise coverage. The phase-1 deliverable is repetition counts and individual-repetition start/end times on held-out subjects; station/rest classification and set boundaries alone do not satisfy it.

"Masks" are flags that say "ignore this stretch". They cover the calibration jumps, the sled pulls, the sensor-fell-off intervals, and the four excluded sessions, along with unavailable required sensor samples. Masked data is neither positive nor negative. It is simply skipped.

### Phase 2 — later: broader station-type recognition

Begin after successful development and evaluation of the repetition-segmentation model, tentatively in 2027. Extend station-type recognition beyond the current categories and study growth toward approximately 100 station types, including previously unseen exercises.

Evaluate unseen exercises separately from unseen participants. For held-out exercises from the 15 existing categories, remove their examples from training while preserving their positive exercise labels in testing for exercise detection; assess station identification under an explicit unknown-class policy. Treat `spull` separately: it is rest under the current categories, excluded from training, and reserved as a new-exercise probe; its existing rest labels cannot score detection under an expanded exercise definition. Use the participant–exercise coverage table when designing the split, and keep unscored probes separate from standard evaluation.

## Suggested papers for the later station-recognition objective

Lee, Minjung, and Seoung Bum Kim. "Sensor-based open-set human activity recognition using representation learning with mixup triplets." IEEE Access 10 (2022): 119333-119344.
Seifi, Soroush, et al. "OOD aware supervised contrastive learning." 2024 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV). IEEE, 2024.
Lunardi, Willian T., Samridha Shrestha, and Martin Andreoni. "Learning Hyperspherical Time-Frequency Representations for Time-Series Out-of-Distribution Detection." arXiv preprint arXiv:2605.31155 (2026).
Zhang, Qi, et al. "An unsupervised open-set recognition method for user-independent human activity recognition." Neurocomputing (2026): 133172.
Yue, Zhihan, et al. "Ts2vec: Towards universal representation of time series." Proceedings of the AAAI conference on artificial intelligence. Vol. 36. No. 8. 2022.


