"""
Assertion utilities for CI integration.

Most assertion logic is in DiffReport.assert_within(), but this module
provides utility functions for advanced use cases.
"""
from typing import Dict, List, Optional


def validate_thresholds(
    thresholds: Dict[str, any],
) -> tuple[bool, List[str]]:
    """
    Validate that threshold config is well-formed.

    Returns:
        (is_valid, list_of_errors)
    """
    errors = []

    for key, value in thresholds.items():
        if key in [
            "max_rows_removed_pct",
            "max_rows_added_pct",
            "max_null_rate_increase",
            "max_psi",
        ]:
            if not isinstance(value, (int, float)):
                errors.append(f"'{key}' must be numeric, got {type(value)}")
            elif value < 0:
                errors.append(f"'{key}' must be >= 0, got {value}")
        elif key in ["no_type_changes", "no_removed_columns", "no_critical"]:
            if not isinstance(value, bool):
                errors.append(f"'{key}' must be bool, got {type(value)}")
        elif key == "columns":
            if not isinstance(value, list):
                errors.append(f"'columns' must be list, got {type(value)}")
        else:
            errors.append(f"Unknown threshold key: '{key}'")

    return len(errors) == 0, errors
