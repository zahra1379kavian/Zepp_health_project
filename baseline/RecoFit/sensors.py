"""Exact-clock wrist/waist inputs without reading waist labels or using torch."""

from pathlib import Path
import shutil
import tempfile

import numpy as np

from strength_training.temporal_segmentation.constants import DT_MS, FEATURES
from strength_training.temporal_segmentation.data import load_arrays, load_sensor, write_json


SENSOR_MODES = ("wrist", "wrist-waist")
DEFAULT_WAIST_STATIONS = ("leg_raise",)


def aligned_waist(path, t):
    """Align one physical device exactly; absent ticks stay unavailable, never filled."""
    t = np.asarray(t)
    if not len(t) or np.any(np.diff(t) != DT_MS):
        raise ValueError("Waist alignment needs a complete 40 ms wrist grid")
    clocks, values = load_sensor(path, FEATURES[:6], allow_waist=True)
    if np.any((clocks-t[0]) % DT_MS):
        raise ValueError(f"{path}: waist clock does not exactly match the wrist grid")
    inside = (clocks >= t[0]) & (clocks <= t[-1])
    indices = ((clocks[inside]-t[0])//DT_MS).astype(int)
    x, available = np.zeros((len(t), 6), np.float32), np.zeros(len(t), bool)
    x[indices], available[indices] = values[inside], True
    return x, available


def select_waist(folder, t):
    """Choose greatest clock overlap; filename breaks ties. Never stitch devices."""
    candidates, selected = [], None
    for path in sorted(Path(folder).glob("*waist*.csv")):
        x, available = aligned_waist(path, t)
        count = int(available.sum())
        candidates.append({"file": path.name, "aligned_samples": count})
        if selected is None or count > selected[2].sum():
            selected = path, x, available
    if selected is None or not selected[2].any():
        raise ValueError(f"{folder}: no waist sensor overlaps the wrist grid")
    path, x, available = selected
    return x, available, {"selected": path.name, "candidates": candidates}


def prepare_waist(root, output, source=None):
    """Copy an immutable wrist dataset and add synchronized six-channel waist data.

    Keep original targets, references, masks and subject splits for comparable
    scoring. Training/inference intersect validity with waist availability; scoring
    retains original wrist coverage so waist dropouts cannot hide missed reps.
    """
    from .pipeline import manifest_at
    root, output = Path(root).resolve(), Path(output).resolve()
    manifest = manifest_at(root)
    source = Path(source or manifest["source"]).resolve()
    if (root == output or root in output.parents or output in root.parents or
            source == output or source in output.parents or output in source.parents):
        raise ValueError("Prepared inputs, source recordings and output must be separate directories")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".recofit-waist-", dir=output.parent) as temp:
        dest = Path(temp)/"data"
        shutil.copytree(root, dest)
        for row in manifest["sessions"]:
            arrays = load_arrays(root/row["session"])
            x, available, info = select_waist(source/row["source_session"], arrays["t"])
            path = dest/row["session"]
            np.save(path/"waist_x.npy", x, allow_pickle=False)
            np.save(path/"waist_available.npy", available, allow_pickle=False)
            row["waist_sensor"] = info
            row["waist_available_samples"] = int(available.sum())
            row["paired_valid_samples"] = int((arrays["valid"] & available).sum())
            print(f"Paired {row['session']}: {row['paired_valid_samples']:,} valid samples ({info['selected']})", flush=True)
        manifest["recofit_sensors"] = {
            "mode": "wrist-waist", "waist_features": FEATURES[:6],
            "source_dataset": str(root), "waist_source": str(source),
            "selection": "Most overlapping clock samples; filename breaks ties; no device stitching",
            "scoring": "Original wrist validity/references retained, including waist dropouts"}
        write_json(dest/"manifest.json", manifest)
        dest.rename(output)
    return manifest


def load_inputs(path, sensors="wrist"):
    if sensors not in SENSOR_MODES:
        raise ValueError(f"Unknown sensor mode: {sensors}")
    arrays = load_arrays(path)
    if sensors == "wrist-waist":
        try:
            x = np.load(Path(path)/"waist_x.npy", mmap_mode="r", allow_pickle=False)
            available = np.load(Path(path)/"waist_available.npy", mmap_mode="r", allow_pickle=False)
        except FileNotFoundError as error:
            raise ValueError("Paired inputs missing; use prepare-waist or run --sensors wrist-waist") from error
        if x.shape != (len(arrays["t"]), 6) or available.shape != arrays["t"].shape or available.dtype != bool:
            raise ValueError("Invalid prepared waist arrays")
        arrays["waist_x"], arrays["waist_available"] = x, available
        arrays["valid"] = arrays["valid"] & available
    return arrays
