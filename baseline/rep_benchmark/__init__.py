"""Common activity-agnostic wrist repetition benchmark."""

_REPORT_DIRECTORIES = {
    "pooled": "all_five_folds",
    "fold_5": "local_fold_5_only",
    "activities": "per_exercise_results",
    "examples": "prediction_examples",
}


def report_directory(parent, key):
    """Use descriptive names for new results and recognize earlier saved reports."""
    current = parent / _REPORT_DIRECTORIES[key]
    earlier = parent / key
    if earlier.is_dir() and not current.exists():
        return earlier
    return current
