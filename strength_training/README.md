# Clock-aligned sensor data

For causal MS-TCN station/NULL classification and individual-repetition
segmentation, see the [implementation and run instructions](temporal_segmentation/README.md).
That pipeline includes exclusions, subject splits, both prediction heads,
streaming decoding, evaluation, and a synthetic end-to-end demo.

Run from the repository root (Python 3.10+, no additional dependencies):

```bash
python3 strength_training/prepare_data.py
```

The script reads `strength_training_data_clean` and creates
`strength_training_data_aligned`. It preserves the source files and refuses to
overwrite an existing output directory. Use `--source PATH --output PATH` for
another dataset or destination.

Each session contains:

- `inputs/<sensor>.csv`: `sync_timestamp`, the original device `timestamp`, and
  the nine accelerometer, gyroscope, and magnetometer channels. All embedded
  label columns are removed, including duplicated `labels` headers.
- `targets/<sensor>.csv`: `sync_timestamp,label`, with exactly one target per
  input row, in the same order. Targets come from the corresponding `waist`,
  `left_wrist`, or `right_wrist` column of the source `label.csv`. Both waist
  devices use `waist`; differing location labels are preserved.
- `availability.csv`: the full label clock with a 0/1 availability flag for
  each sensor. Gaps and partial coverage remain missing; no samples or targets
  are interpolated or forward filled.

The dataset-level `manifest.json` records row counts, label columns removed,
coverage, and the feature names. Alignment requires an exact `sync_timestamp`
match. Unmatched rows, duplicate/backward clocks, missing location labels, and
conflicting embedded labels stop processing before the final dataset is published.

The loader exposes features and targets separately. Its `features` field contains
only the nine sensor channels; both clocks are metadata:

```python
from itertools import islice
from pathlib import Path
from strength_training.prepare_data import iter_samples

session = Path("strength_training/strength_training_data_aligned/20260602_LRFVFZEIVGAM")
rows = list(islice(iter_samples(session, "left_wrist.csv"), 100))
X = [row.features for row in rows]
y = [row.label for row in rows]
```

This is the alignment stage. Apply the session and interval exclusions in
[`plan.md`](../plan.md) before training or evaluation. Availability flags describe
sensor coverage and do not establish training/evaluation eligibility.

Run the alignment regression tests:

```bash
python3 -m unittest discover -s strength_training -p 'test_*.py'
```
