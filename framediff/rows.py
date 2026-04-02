"""
Row-level diff logic — detect added, removed, and modified rows.
"""
from dataclasses import dataclass, field
from typing import Any, List, Optional, Dict
import pandas as pd
from .exceptions import DiffKeyError


@dataclass
class RowDiff:
    """Row-level summary of changes."""

    added_count: int = 0
    removed_count: int = 0
    modified_count: int = 0
    total_before: int = 0
    total_after: int = 0
    added_pct: float = 0.0
    removed_pct: float = 0.0
    sample_added: pd.DataFrame = field(default_factory=pd.DataFrame)
    sample_removed: pd.DataFrame = field(default_factory=pd.DataFrame)
    sample_modified: pd.DataFrame = field(default_factory=pd.DataFrame)
    modifications: Dict[str, int] = field(default_factory=dict)  # col → count of changes

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        return {
            "added_count": int(self.added_count),
            "removed_count": int(self.removed_count),
            "modified_count": int(self.modified_count),
            "total_before": int(self.total_before),
            "total_after": int(self.total_after),
            "added_pct": float(self.added_pct),
            "removed_pct": float(self.removed_pct),
            "sample_added": self.sample_added.to_dict(orient="records"),
            "sample_removed": self.sample_removed.to_dict(orient="records"),
            "sample_modified": self.sample_modified.to_dict(orient="records"),
            "modifications": {
                k: int(v) for k, v in self.modifications.items()
            },  # ensure int
        }


def _check_duplicate_keys(df: pd.DataFrame, keys: List[str], frame_label: str) -> None:
    """
    Check if the DataFrame has duplicate keys. Raises DiffKeyError if duplicates found.
    
    Args:
        df: DataFrame to check
        keys: List of key column names
        frame_label: Label for error message (e.g., "before", "after")
        
    Raises:
        DiffKeyError: If duplicate keys are found
    """
    duplicated = df.duplicated(subset=keys, keep=False)
    if duplicated.any():
        count = duplicated.sum()
        raise DiffKeyError(
            f"{frame_label} has {count} duplicate key(s) in column(s) {keys!r}. "
            f"framediff requires unique keys for row-level diff. "
            f"Use key=None for positional matching, or deduplicate first."
        )


def compare_rows(
    before: pd.DataFrame,
    after: pd.DataFrame,
    key: Optional[str | List[str]] = None,
) -> RowDiff:
    """
    Compare row-level changes between two DataFrames.

    Args:
        before: DataFrame before changes
        after: DataFrame after changes
        key: Column name(s) to use as join key. If None, use positional matching.

    Returns:
        RowDiff object summarizing added, removed, and modified rows
    """
    diff = RowDiff()
    diff.total_before = len(before)
    diff.total_after = len(after)

    if diff.total_before == 0 and diff.total_after == 0:
        return diff

    # BUG FIX: Validate key columns exist (mirrors validation in compare())
    if key is not None:
        key_cols = [key] if isinstance(key, str) else key
        for col in key_cols:
            if col not in before.columns:
                raise ValueError(
                    f"Key column {col!r} not found in before DataFrame. "
                    f"Available columns: {list(before.columns)}"
                )
            if col not in after.columns:
                raise ValueError(
                    f"Key column {col!r} not found in after DataFrame. "
                    f"Available columns: {list(after.columns)}"
                )

    if key is None:
        # Positional matching
        return _compare_rows_positional(before, after, diff)
    elif isinstance(key, str):
        # Single key
        return _compare_rows_with_key(before, after, [key], diff)
    else:
        # Composite key
        return _compare_rows_with_key(before, after, key, diff)


def _compare_rows_positional(
    before: pd.DataFrame, after: pd.DataFrame, diff: RowDiff
) -> RowDiff:
    """
    Compare rows positionally (by index).
    """
    min_len = min(len(before), len(after))
    max_len = max(len(before), len(after))

    if len(after) > len(before):
        diff.added_count = len(after) - len(before)
    else:
        diff.removed_count = len(before) - len(after)

    diff.added_pct = (
        100.0 * diff.added_count / len(before) if len(before) > 0 else 0.0
    )
    diff.removed_pct = (
        100.0 * diff.removed_count / len(before) if len(before) > 0 else 0.0
    )

    # Find modified rows in the overlapping range
    if min_len > 0:
        # Compare overlapping rows
        before_overlap = before.iloc[:min_len]
        after_overlap = after.iloc[:min_len]

        # Check which rows changed
        row_changes = []
        for i in range(min_len):
            if not before_overlap.iloc[i].equals(after_overlap.iloc[i]):
                row_changes.append(i)

        diff.modified_count = len(row_changes)

        # Sample modified rows (up to 10)
        if row_changes:
            sample_indices = row_changes[:10]
            modified_samples = []
            for i in sample_indices:
                before_row = before_overlap.iloc[i]
                after_row = after_overlap.iloc[i]

                # Create side-by-side before/after
                sample_dict = {}
                for col in before.columns:
                    before_val = before_row.get(col)
                    after_val = after_row.get(col) if col in after.columns else None
                    
                    # Safe comparison handling NA values
                    before_is_na = pd.isna(before_val)
                    after_is_na = pd.isna(after_val)
                    
                    # Check for difference
                    if before_is_na and after_is_na:
                        changed = False
                    elif before_is_na or after_is_na:
                        changed = True
                    else:
                        try:
                            changed = before_val != after_val
                        except (TypeError, ValueError):
                            # Fallback for unhashable or non-comparable types
                            changed = True
                    
                    if changed:
                        sample_dict[f"{col}__before"] = before_val
                        sample_dict[f"{col}__after"] = after_val

                if sample_dict:
                    modified_samples.append(sample_dict)

            if modified_samples:
                diff.sample_modified = pd.DataFrame(modified_samples)

    # Sample added rows
    if diff.added_count > 0:
        start_idx = len(before)
        added_indices = list(range(start_idx, min(start_idx + 10, len(after))))
        diff.sample_added = after.iloc[added_indices].reset_index(drop=True)

    # Sample removed rows
    if diff.removed_count > 0:
        removed_indices = list(range(len(after), min(len(after) + 10, len(before))))
        diff.sample_removed = before.iloc[removed_indices].reset_index(drop=True)

    return diff


def _compare_rows_with_key(
    before: pd.DataFrame, after: pd.DataFrame, keys: List[str], diff: RowDiff
) -> RowDiff:
    """
    Compare rows using one or more key columns (join semantics).
    Properly handles both unique and non-unique keys.
    """
    # Check for duplicate keys
    _check_duplicate_keys(before, keys, "before")
    _check_duplicate_keys(after, keys, "after")

    before_set = set(before[keys].apply(tuple, axis=1))
    after_set = set(after[keys].apply(tuple, axis=1))

    added_keys = after_set - before_set
    removed_keys = before_set - after_set

    diff.added_count = len(added_keys)
    diff.removed_count = len(removed_keys)
    diff.added_pct = (
        100.0 * diff.added_count / len(before) if len(before) > 0 else 0.0
    )
    diff.removed_pct = (
        100.0 * diff.removed_count / len(before) if len(before) > 0 else 0.0
    )

    # Count modified rows by comparing values for common keys
    modified_count = 0
    modified_samples: List[Dict[str, Any]] = []
    
    for key_vals in before_set & after_set:
        # Get all rows matching this key value(s)
        key_dict = dict(zip(keys, key_vals if isinstance(key_vals, tuple) else (key_vals,)))
        
        # Build mask for this key
        before_mask = pd.Series([True] * len(before))
        after_mask = pd.Series([True] * len(after))
        for k, v in key_dict.items():
            before_mask = before_mask & (before[k] == v)
            after_mask = after_mask & (after[k] == v)
        
        before_subset = before[before_mask].reset_index(drop=True)
        after_subset = after[after_mask].reset_index(drop=True)
        
        # Compare subsets row by row
        min_len = min(len(before_subset), len(after_subset))
        for i in range(min_len):
            before_row = before_subset.iloc[i]
            after_row = after_subset.iloc[i]
            
            # Check if any non-key column changed (only columns in both)
            changed = False
            changes_dict = {}
            common_cols = set(before.columns) & set(after.columns)
            for col in common_cols:
                if col not in keys:
                    before_val = before_row[col]
                    after_val = after_row[col]
                    # Use pandas null-safe comparison
                    if pd.isna(before_val) and pd.isna(after_val):
                        continue
                    # Safe comparison handling NA values
                    before_is_na = pd.isna(before_val)
                    after_is_na = pd.isna(after_val)
                    
                    if before_is_na or after_is_na or before_val != after_val:
                        changed = True
                        changes_dict[f"{col}__before"] = before_val
                        changes_dict[f"{col}__after"] = after_val
            
            if changed:
                modified_count += 1
                if len(modified_samples) < 10:
                    modified_samples.append(changes_dict)

    diff.modified_count = modified_count
    if modified_samples:
        diff.sample_modified = pd.DataFrame(modified_samples)

    # Sample added rows
    if diff.added_count > 0:
        added_df = after[after[keys].apply(tuple, axis=1).isin(list(added_keys)[:10])]
        diff.sample_added = added_df.reset_index(drop=True)

    # Sample removed rows
    if diff.removed_count > 0:
        removed_df = before[
            before[keys].apply(tuple, axis=1).isin(list(removed_keys)[:10])
        ]
        diff.sample_removed = removed_df.reset_index(drop=True)

    # Track which columns changed most
    if diff.modified_count > 0:
        modifications = {}
        common_cols = set(before.columns) & set(after.columns)
        for col in common_cols:
            if col not in keys:
                change_count = 0
                for key_vals in before_set & after_set:
                    key_dict = dict(zip(keys, key_vals if isinstance(key_vals, tuple) else (key_vals,)))
                    before_mask = pd.Series([True] * len(before))
                    after_mask = pd.Series([True] * len(after))
                    for k, v in key_dict.items():
                        before_mask = before_mask & (before[k] == v)
                        after_mask = after_mask & (after[k] == v)
                    
                    before_subset = before[before_mask][col]
                    after_subset = after[after_mask][col]
                    
                    # Count changes in this column for this key
                    for i in range(min(len(before_subset), len(after_subset))):
                        before_val = before_subset.iloc[i]
                        after_val = after_subset.iloc[i]
                        if pd.isna(before_val) and pd.isna(after_val):
                            continue
                        # Safe comparison handling NA values  
                        before_is_na = pd.isna(before_val)
                        after_is_na = pd.isna(after_val)
                        if before_is_na or after_is_na or before_val != after_val:
                            change_count += 1
                
                if change_count > 0:
                    modifications[col] = change_count

        diff.modifications = modifications

    return diff
