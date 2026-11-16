"""Small regression fixtures for alignment and label-leakage prevention."""

import csv
import json
from pathlib import Path
import tempfile
import unittest

from prepare_data import FEATURE_COLUMNS, INPUT_COLUMNS, iter_samples, prepare_dataset


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        self.session = self.source / "session"
        self.session.mkdir(parents=True)
        self.output = self.root / "aligned"
        self.write("label.csv", ["sync_timestamp", "waist", "left_wrist"], [
            [1000, "rest", "rest"],
            [1040, "squat_phase1", "squat_phase2"],
            [1080, "rest", "rest"],
            [1120, "squat_transition", "rest"],
        ])

    def write(self, name, header, rows):
        with (self.session / name).open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            writer.writerows(rows)

    def sensor(self, name="left_wrist.csv", clocks=(1000, 1040, 1120), embedded=True):
        # The raw device clock is deliberately shifted: joining on it must fail.
        header = [*INPUT_COLUMNS, *(["labels", "labels"] if embedded else [])]
        labels = {1000: "rest", 1040: "squat_phase2", 1120: "rest"}
        rows = [[t, t + 8000, *range(9), *([labels.get(t, "rest")] * 2 if embedded else [])]
                for t in clocks]
        self.write(name, header, rows)

    def test_exact_clock_join_separate_targets_and_missing_sample_mask(self):
        self.sensor()
        self.sensor("S1_waist.csv", (1040, 1120), embedded=False)
        self.sensor("S2_waist.csv", (1000, 1040, 1120), embedded=False)
        before = {p.name: p.read_bytes() for p in self.session.iterdir()}
        result = prepare_dataset(self.source, self.output)
        dest = self.output / "session"
        wrist = list(iter_samples(dest, "left_wrist.csv"))
        self.assertEqual([r.sync_timestamp for r in wrist], [1000, 1040, 1120])
        self.assertEqual([r.timestamp for r in wrist], [9000, 9040, 9120])
        self.assertEqual([r.label for r in wrist], ["rest", "squat_phase2", "rest"])
        self.assertEqual(wrist[1].features, tuple(float(v) for v in range(9)))
        for device in ("S1_waist.csv", "S2_waist.csv"):
            rows = list(iter_samples(dest, device))
            self.assertEqual(rows[-1].label, "squat_transition")
            self.assertEqual(next(r.label for r in rows if r.sync_timestamp == 1040), "squat_phase1")
        with (dest / "inputs" / "left_wrist.csv").open() as handle:
            self.assertEqual(tuple(next(csv.reader(handle))), INPUT_COLUMNS)
        with (dest / "availability.csv").open() as handle:
            availability = list(csv.DictReader(handle))
        self.assertEqual(availability[0]["S1_waist"], "0")
        self.assertEqual(availability[2], {
            "sync_timestamp": "1080", "S1_waist": "0", "S2_waist": "0", "left_wrist": "0",
        })
        self.assertEqual(result["totals"]["sensor_rows"], 8)
        self.assertEqual(result["totals"]["unmatched_rows"], 0)
        self.assertEqual(result["totals"]["embedded_label_columns_removed"], 2)
        self.assertEqual(result["feature_columns"], list(FEATURE_COLUMNS))
        self.assertFalse(result["eligibility_masks_applied"])
        self.assertEqual(json.loads((self.output / "manifest.json").read_text()), result)
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.session.iterdir()})

    def assert_rejected(self, message):
        with self.assertRaisesRegex(ValueError, message):
            prepare_dataset(self.source, self.output)
        self.assertFalse(self.output.exists())
        self.assertEqual(list(self.root.glob(".aligned-*")), [])

    def test_unmatched_sensor_row_is_not_dropped_or_filled(self):
        self.sensor(clocks=(1000, 1060))
        self.assert_rejected("no exact label")

    def test_duplicate_sensor_clock_is_rejected(self):
        self.sensor(clocks=(1000, 1000))
        self.assert_rejected("duplicate or backward")

    def test_backward_sensor_clock_is_rejected(self):
        self.sensor(clocks=(1040, 1000))
        self.assert_rejected("duplicate or backward")

    def test_duplicate_label_clock_cannot_multiply_sensor_rows(self):
        self.sensor()
        with (self.session / "label.csv").open("a") as handle:
            handle.write("1120,rest,rest\n")
        self.assert_rejected("unique and spaced")

    def test_missing_location_is_not_replaced_by_other_location(self):
        self.sensor("right_wrist.csv", embedded=False)
        self.assert_rejected("right_wrist is missing")

    def test_conflicting_embedded_labels_are_reported(self):
        self.sensor()
        path = self.session / "left_wrist.csv"
        path.write_text(path.read_text().replace("squat_phase2,squat_phase2", "rest,squat_phase2"))
        self.assert_rejected("embedded label disagrees")

    def test_missing_label_file_does_not_silently_skip_session(self):
        self.sensor()
        (self.source / "missing_labels").mkdir()
        self.assert_rejected("Missing label.csv")

    def test_existing_output_and_source_overlap_are_rejected(self):
        self.sensor()
        self.output.mkdir()
        sentinel = self.output / "keep.txt"
        sentinel.write_text("keep")
        with self.assertRaises(FileExistsError):
            prepare_dataset(self.source, self.output)
        self.assertEqual(sentinel.read_text(), "keep")
        with self.assertRaisesRegex(ValueError, "no nesting"):
            prepare_dataset(self.source, self.source / "aligned")

    def test_loader_rejects_tampered_target_clock(self):
        self.sensor()
        prepare_dataset(self.source, self.output)
        path = self.output / "session" / "targets" / "left_wrist.csv"
        path.write_text(path.read_text().replace("1040,squat_phase2", "1080,squat_phase2"))
        with self.assertRaisesRegex(ValueError, "clocks/fields differ"):
            list(iter_samples(self.output / "session", "left_wrist.csv"))


if __name__ == "__main__":
    unittest.main()
