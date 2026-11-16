"""Align sensor rows to authoritative labels without modifying source CSVs.

Run: python3 strength_training/prepare_data.py
Only the Python standard library is required. This prepares alignment artifacts;
the study plan's session/interval exclusions must still be applied before use.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from itertools import zip_longest
import json
from pathlib import Path
import re
import tempfile


CLOCK = "sync_timestamp"
FEATURE_COLUMNS = tuple(f"{sensor}_{axis}" for sensor in ("acc", "gyro", "mag") for axis in "xyz")
INPUT_COLUMNS = (CLOCK, "timestamp", *FEATURE_COLUMNS)
TARGET_COLUMNS = (CLOCK, "label")
LOCATIONS = {"waist", "left_wrist", "right_wrist"}
EMBEDDED_LABEL = re.compile(r"(?:labels?|waist|left_wrist|right_wrist)(?:\.\d+)?$")
BASE = Path(__file__).resolve().parent


def load_labels(path: Path):
    """Read a unique, ordered 25 Hz clock and the original location labels."""
    timestamps, values = [], []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        header = next(reader, [])
        if (len(header) != len(set(header)) or CLOCK not in header
                or not set(header) - {CLOCK} or set(header) - {CLOCK} - LOCATIONS):
            raise ValueError(f"{path}: invalid label header {header}")
        clock_index = header.index(CLOCK)
        locations = {name: header.index(name) for name in header if name != CLOCK}
        for line, row in enumerate(reader, 2):
            if len(row) != len(header):
                raise ValueError(f"{path}:{line}: incorrect number of fields")
            clock = int(row[clock_index])
            if timestamps and clock - timestamps[-1] != 40:
                raise ValueError(f"{path}:{line}: label clock must be unique and spaced at 40 ms")
            if any(not row[index].strip() for index in locations.values()):
                raise ValueError(f"{path}:{line}: missing location label")
            timestamps.append(clock)
            values.append(tuple(row))
    if not timestamps:
        raise ValueError(f"{path}: empty label timeline")
    return timestamps, dict(zip(timestamps, values)), locations


def sensor_location(path: Path) -> str:
    if path.stem in {"left_wrist", "right_wrist"}:
        return path.stem
    if path.stem.endswith("_waist"):
        return "waist"
    raise ValueError(f"Unknown sensor location: {path}")


def prepare_sensor(path: Path, output: Path, timestamps, labels, locations):
    location = sensor_location(path)
    if location not in locations:
        raise ValueError(f"{path}: {location} is missing from label.csv")
    available = bytearray(len(timestamps))
    first_clock, last_clock, rows, internal_missing = None, None, 0, 0
    with (path.open(newline="", encoding="utf-8-sig") as source,
          (output / "inputs" / path.name).open("w", newline="") as inputs,
          (output / "targets" / path.name).open("w", newline="") as targets):
        reader = csv.reader(source)
        header = next(reader, [])
        if any(header.count(name) != 1 for name in INPUT_COLUMNS):
            raise ValueError(f"{path}: missing or duplicate clock/sensor columns")
        embedded = [i for i, name in enumerate(header) if EMBEDDED_LABEL.fullmatch(name)]
        if set(header) - set(INPUT_COLUMNS) - {header[i] for i in embedded}:
            raise ValueError(f"{path}: unexpected sensor columns: {header}")
        indices = [header.index(name) for name in INPUT_COLUMNS]
        input_writer, target_writer = csv.writer(inputs), csv.writer(targets)
        input_writer.writerow(INPUT_COLUMNS)
        target_writer.writerow(TARGET_COLUMNS)
        for line, row in enumerate(reader, 2):
            if len(row) != len(header):
                raise ValueError(f"{path}:{line}: incorrect number of fields")
            clock = int(row[indices[0]])
            if last_clock is not None and clock <= last_clock:
                raise ValueError(f"{path}:{line}: duplicate or backward sensor clock")
            if clock not in labels:
                raise ValueError(f"{path}:{line}: no exact label for {CLOCK}={clock}")
            target = labels[clock][locations[location]]
            if any(row[i] != target for i in embedded):
                raise ValueError(f"{path}:{line}: embedded label disagrees with label.csv")
            input_writer.writerow([row[i] for i in indices])
            target_writer.writerow((clock, target))
            available[(clock - timestamps[0]) // 40] = 1
            if last_clock is None:
                first_clock = clock
            else:
                internal_missing += (clock - last_clock) // 40 - 1
            last_clock = clock
            rows += 1
    if not rows:
        raise ValueError(f"{path}: empty sensor file")
    return {
        "sensor": path.name, "location": location, "rows": rows,
        "matched_rows": rows, "unmatched_rows": 0,
        "embedded_label_columns_removed": len(embedded),
        "embedded_label_mismatches": 0,
        "first_sync_timestamp": first_clock, "last_sync_timestamp": last_clock,
        "missing_label_grid_samples": len(timestamps) - rows,
        "internal_missing_grid_samples": internal_missing,
    }, available


def prepare_dataset(source: Path, output: Path) -> dict:
    source, output = source.resolve(), output.resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("Source and output directories must be separate, with no nesting")
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}; choose a new output directory")
    folders = sorted(p for p in source.iterdir() if p.is_dir())
    if not folders:
        raise ValueError(f"No sessions found in {source}")
    # Check every directory, so a missing label.csv cannot silently drop a session.
    for folder in folders:
        if not (folder / "label.csv").is_file():
            raise ValueError(f"Missing label.csv: {folder}")
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "version": 1, "source": str(source), "stage": "alignment_only",
        "join_clock": CLOCK, "join_policy": "exact; preserve every sensor row in order",
        "feature_columns": list(FEATURE_COLUMNS),
        "input_clock_columns": [CLOCK, "timestamp"],
        "target_source": "label.csv, corresponding body location",
        "eligibility_masks_applied": False,
        "note": "Apply the study plan's session and interval exclusions before training/evaluation.",
        "sessions": [],
    }
    # Publish only a complete dataset; a bad clock/label leaves no partial output.
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temp:
        prepared = Path(temp) / "dataset"
        prepared.mkdir()
        for number, folder in enumerate(folders, 1):
            timestamps, labels, locations = load_labels(folder / "label.csv")
            dest = prepared / folder.name
            (dest / "inputs").mkdir(parents=True)
            (dest / "targets").mkdir()
            sensors = sorted(p for p in folder.glob("*.csv") if p.name != "label.csv")
            if not sensors:
                raise ValueError(f"No sensor files found in {folder}")
            records, masks = [], []
            for sensor in sensors:
                record, mask = prepare_sensor(sensor, dest, timestamps, labels, locations)
                records.append(record)
                masks.append(mask)
            with (dest / "availability.csv").open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow((CLOCK, *(sensor.stem for sensor in sensors)))
                writer.writerows((clock, *(mask[i] for mask in masks))
                                 for i, clock in enumerate(timestamps))
            manifest["sessions"].append({
                "session": folder.name, "label_rows": len(timestamps), "sensors": records,
            })
            print(f"[{number}/{len(folders)}] {folder.name}: "
                  f"{sum(r['rows'] for r in records):,} rows matched", flush=True)
        records = [r for s in manifest["sessions"] for r in s["sensors"]]
        manifest["totals"] = {
            "sessions": len(folders), "sensor_files": len(records),
            "sensor_rows": sum(r["rows"] for r in records),
            "label_rows": sum(s["label_rows"] for s in manifest["sessions"]),
            "unmatched_rows": sum(r["unmatched_rows"] for r in records),
            "embedded_label_columns_removed": sum(r["embedded_label_columns_removed"] for r in records),
            "embedded_label_mismatches": sum(r["embedded_label_mismatches"] for r in records),
        }
        (prepared / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        prepared.rename(output)
    return manifest


@dataclass(frozen=True)
class Sample:
    sync_timestamp: int
    timestamp: float
    features: tuple[float, ...]
    label: str


def iter_samples(session: Path, sensor: str):
    """Yield nine-channel features and a separate target; clocks are metadata."""
    if Path(sensor).name != sensor or Path(sensor).suffix != ".csv":
        raise ValueError("sensor must be a CSV filename within the session")
    with ((session / "inputs" / sensor).open(newline="") as inputs,
          (session / "targets" / sensor).open(newline="") as targets):
        x_reader, y_reader = csv.reader(inputs), csv.reader(targets)
        if tuple(next(x_reader, [])) != INPUT_COLUMNS or tuple(next(y_reader, [])) != TARGET_COLUMNS:
            raise ValueError("Unexpected prepared input/target schema")
        previous = None
        for x, y in zip_longest(x_reader, y_reader):
            if x is None or y is None:
                raise ValueError("Input and target row counts differ")
            if len(x) != len(INPUT_COLUMNS) or len(y) != len(TARGET_COLUMNS) or x[0] != y[0]:
                raise ValueError("Input and target clocks/fields differ")
            clock = int(x[0])
            if previous is not None and clock <= previous:
                raise ValueError("Prepared sensor clock is not strictly increasing")
            previous = clock
            yield Sample(clock, float(x[1]), tuple(map(float, x[2:])), y[1])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=BASE / "strength_training_data_clean")
    parser.add_argument("--output", type=Path, default=BASE / "strength_training_data_aligned")
    args = parser.parse_args()
    result = prepare_dataset(args.source, args.output)
    print(json.dumps(result["totals"], indent=2))


if __name__ == "__main__":
    main()
