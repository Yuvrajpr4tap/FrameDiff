"""
Schema diffing logic — detect column additions, removals, type changes, etc.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import pandas as pd


@dataclass
class SchemaDiff:
    """Represents schema-level changes between two DataFrames."""

    added_columns: List[str] = field(default_factory=list)
    removed_columns: List[str] = field(default_factory=list)
    type_changes: Dict[str, Dict[str, str]] = field(
        default_factory=dict
    )  # col → {"before": before_dtype, "after": after_dtype}
    nullable_changes: Dict[str, Dict[str, bool]] = field(
        default_factory=dict
    )  # col → {"before": before_nullable, "after": after_nullable}
    index_changes: Dict = field(default_factory=dict)
    issues: List["DiffIssue"] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        return {
            "added_columns": self.added_columns,
            "removed_columns": self.removed_columns,
            "type_changes": {k: v for k, v in self.type_changes.items()},
            "nullable_changes": {k: v for k, v in self.nullable_changes.items()},
            "index_changes": self.index_changes,
        }


@dataclass
class DiffIssue:
    """A single issue found in a diff."""

    severity: str  # "info" | "warning" | "critical"
    category: str  # "schema", "stats", "rows"
    message: str


def compare_schemas(before: pd.DataFrame, after: pd.DataFrame) -> SchemaDiff:
    """
    Compare DataFrame schemas and return a SchemaDiff.

    Args:
        before: DataFrame before changes
        after: DataFrame after changes

    Returns:
        SchemaDiff object with detected changes
    """
    diff = SchemaDiff()
    issues = []

    # Detect added and removed columns
    before_cols = set(before.columns)
    after_cols = set(after.columns)

    diff.added_columns = sorted(list(after_cols - before_cols))
    diff.removed_columns = sorted(list(before_cols - after_cols))

    # Severity: removed columns are critical
    for col in diff.removed_columns:
        issues.append(
            DiffIssue(
                severity="critical",
                category="schema",
                message=f"Column '{col}' was removed",
            )
        )

    # Severity: added columns are info (non-breaking change)
    for col in diff.added_columns:
        issues.append(
            DiffIssue(
                severity="info",
                category="schema",
                message=f"Column '{col}' was added",
            )
        )

    # Detect type changes in common columns
    common_cols = before_cols & after_cols
    for col in common_cols:
        before_dtype = str(before[col].dtype)
        after_dtype = str(after[col].dtype)
        if before_dtype != after_dtype:
            diff.type_changes[col] = {"before": before_dtype, "after": after_dtype}
            severity = _assess_type_change_severity(before_dtype, after_dtype)
            issues.append(
                DiffIssue(
                    severity=severity,
                    category="schema",
                    message=f"Column '{col}' dtype changed: {before_dtype} → {after_dtype}",
                )
            )

    # Detect nullable changes (columns that went from all non-null to some null or vice versa)
    for col in common_cols:
        before_has_null = before[col].isna().any()
        after_has_null = after[col].isna().any()
        if before_has_null != after_has_null:
            diff.nullable_changes[col] = {"before": before_has_null, "after": after_has_null}

    # Detect index changes
    if before.index.name != after.index.name or str(before.index.dtype) != str(
        after.index.dtype
    ):
        diff.index_changes = {
            "before_name": before.index.name,
            "after_name": after.index.name,
            "before_dtype": str(before.index.dtype),
            "after_dtype": str(after.index.dtype),
        }

    diff.issues = issues
    return diff


def _assess_type_change_severity(before_dtype: str, after_dtype: str) -> str:
    """
    Assess severity of a type change.

    Lossy conversions (float→int, object→category) are warnings.
    Otherwise, info.
    """
    lossy_patterns = [
        ("float", "int"),
        ("int", "uint"),
        ("uint", "int"),
        ("object", "category"),
        ("float64", "int64"),
        ("float32", "int32"),
    ]

    before_lower = before_dtype.lower()
    after_lower = after_dtype.lower()

    for from_t, to_t in lossy_patterns:
        if from_t in before_lower and to_t in after_lower:
            return "warning"

    return "info"
