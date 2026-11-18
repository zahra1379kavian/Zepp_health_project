"""Fixed task vocabulary; model inputs never contain these labels."""

DT_MS = 40
START_HISTORY = 6000 // DT_MS
STATION_STATES = ["rest", "at_station"]
FEATURES = [f"{sensor}_{axis}" for sensor in ("acc", "gyro", "mag") for axis in "xyz"]
CLASSES = ["NULL", "bench_press", "biceps_curls", "crunch", "deadlift", "dips",
           "hip_thrust", "lat_pulldown", "leg_raise", "lunges",
           "overhead_press_(ohp)", "plank", "pull-up", "push-up", "rowing", "squat"]
EXCLUDED_SESSIONS = {
    "20260604_JCLQJCLQ", "20260604_JCLQVGFFNHJI",
    "20260604_SSCDDQTQHOPK", "20260604_WTTANHJIOBWA",
}
EXCLUDED_SCENES = {"action", "spull", "other"}


def station(label):
    if label == "rest":
        return "NULL"
    for suffix in ("_phase1", "_phase2", "_transition"):
        if label.endswith(suffix):
            label = label[:-len(suffix)]
            break
    if label not in CLASSES[1:]:
        raise ValueError(f"Unknown annotation: {label!r}")
    return label
