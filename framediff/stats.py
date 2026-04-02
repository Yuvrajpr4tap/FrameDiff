"""
Statistical diff logic — PSI, KL divergence, distribution analysis, etc.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency


@dataclass
class StatDiff:
    """Statistical summary of a single column's changes."""

    column: str
    dtype: str
    mean_delta: Optional[float] = None  # numeric only
    std_delta: Optional[float] = None  # numeric only
    null_rate_before: float = 0.0
    null_rate_after: float = 0.0
    null_rate_delta: float = 0.0
    cardinality_before: int = 0
    cardinality_after: int = 0
    distribution_method: str = "none"  # which method was used
    distribution_score: float = 0.0  # PSI, KL divergence, chi2, etc.
    distribution_label: str = "stable"  # "stable" | "moderate shift" | "large shift"
    new_categories: List[str] = field(default_factory=list)  # categorical only
    dropped_categories: List[str] = field(default_factory=list)  # categorical only
    severity: str = "info"
    value_change_rate: Optional[float] = None  # fraction of rows where value changed (low-cardinality only, key-based)

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        def serialize_category(cat):
            """Convert category value to JSON-serializable type."""
            if hasattr(cat, 'item'):  # numpy type
                return cat.item()
            return cat
        
        return {
            "column": self.column,
            "dtype": self.dtype,
            "mean_delta": (
                float(self.mean_delta) if self.mean_delta is not None else None
            ),
            "std_delta": float(self.std_delta) if self.std_delta is not None else None,
            "null_rate_before": float(self.null_rate_before),
            "null_rate_after": float(self.null_rate_after),
            "null_rate_delta": float(self.null_rate_delta),
            "cardinality_before": int(self.cardinality_before),
            "cardinality_after": int(self.cardinality_after),
            "distribution_method": self.distribution_method,
            "distribution_score": float(self.distribution_score),
            "distribution_label": self.distribution_label,
            "new_categories": [serialize_category(c) for c in self.new_categories],
            "dropped_categories": [serialize_category(c) for c in self.dropped_categories],
            "severity": self.severity,
            "value_change_rate": float(self.value_change_rate) if self.value_change_rate is not None else None,
        }


def compare_stats(
    before: pd.DataFrame,
    after: pd.DataFrame,
    stat_methods: List[str] | None = None,
    key: Optional[str | List[str]] = None,
) -> Dict[str, StatDiff]:
    """
    Compute statistical diffs for all overlapping columns.

    Args:
        before: DataFrame before changes
        after: DataFrame after changes
        stat_methods: List of methods ('auto', 'psi', 'kl', 'wasserstein')
        key: Column name(s) to use as join key for key-based value change rate calculation.
             If provided, enables value_change_rate calculation for low-cardinality columns.

    Returns:
        Dict mapping column name to StatDiff
    """
    if stat_methods is None:
        stat_methods = ["auto"]

    stats = {}
    common_cols = set(before.columns) & set(after.columns)

    # Prepare aligned data for key-based comparisons
    before_aligned = None
    after_aligned = None
    if key is not None:
        before_aligned = before.copy()
        after_aligned = after.copy()

    for col in sorted(common_cols):
        dtype = str(before[col].dtype)
        stat_diff = StatDiff(column=col, dtype=dtype)

        # Compute null rates
        before_null_count = before[col].isna().sum()
        after_null_count = after[col].isna().sum()
        
        if len(before) > 0:
            stat_diff.null_rate_before = float(before_null_count / len(before))
        if len(after) > 0:
            stat_diff.null_rate_after = float(after_null_count / len(after))
        stat_diff.null_rate_delta = abs(
            stat_diff.null_rate_after - stat_diff.null_rate_before
        )

        # Assess null rate change severity
        if stat_diff.null_rate_delta > 0.1:
            stat_diff.severity = "critical"
        elif stat_diff.null_rate_delta > 0.02:
            stat_diff.severity = "warning"

        # Get non-null data
        before_clean = before[col].dropna()
        after_clean = after[col].dropna()

        # Cardinality
        stat_diff.cardinality_before = before_clean.nunique()
        stat_diff.cardinality_after = after_clean.nunique()

        # Determine column type and compute distribution diff
        if pd.api.types.is_numeric_dtype(dtype):
            _compute_numeric_diff(
                before_clean, after_clean, stat_diff, stat_methods[0]
            )
        elif (isinstance(dtype, pd.CategoricalDtype) 
              or pd.api.types.is_object_dtype(dtype)
              or pd.api.types.is_string_dtype(dtype)):
            _compute_categorical_diff(before_clean, after_clean, stat_diff)
        elif pd.api.types.is_datetime64_any_dtype(dtype):
            _compute_datetime_diff(before_clean, after_clean, stat_diff)

        # Calculate value_change_rate for low-cardinality columns (≤10 unique values)
        if key is not None and stat_diff.cardinality_before <= 10 and stat_diff.cardinality_after <= 10:
            value_change_rate = _value_change_rate(before_aligned, after_aligned, col, key)
            if value_change_rate is not None:
                stat_diff.value_change_rate = value_change_rate
                # Apply severity upgrade rules based on value_change_rate
                if value_change_rate >= 0.50:
                    stat_diff.severity = "critical"
                elif value_change_rate >= 0.30:
                    if stat_diff.severity != "critical":
                        stat_diff.severity = "warning"

        # Overall severity from distribution
        if stat_diff.distribution_label == "large shift":
            stat_diff.severity = "critical"
        elif stat_diff.distribution_label == "moderate shift" and stat_diff.severity != "critical":
            stat_diff.severity = "warning"

        stats[col] = stat_diff

    return stats


def _value_change_rate(
    before: pd.DataFrame,
    after: pd.DataFrame,
    col: str,
    key: Optional[str | List[str]],
) -> Optional[float]:
    """
    Calculate the fraction of rows where a column value changed.

    This requires key-based matching to align rows. Returns None if key is not provided
    or if the key columns don't exist.

    Args:
        before: DataFrame before changes (with all columns including key)
        after: DataFrame after changes (with all columns including key)
        col: Column name to check for changes
        key: Column name(s) to use as join key

    Returns:
        Fraction of rows where column value changed (0.0-1.0), or None if alignment fails
    """
    if key is None:
        return None

    # Normalize key to list
    key_cols = [key] if isinstance(key, str) else key

    # Check if key columns exist in both frames
    if not all(k in before.columns for k in key_cols) or not all(k in after.columns for k in key_cols):
        return None

    try:
        # Merge on key to align rows
        merged = before[[*key_cols, col]].merge(
            after[[*key_cols, col]],
            on=key_cols,
            how="inner",
            suffixes=("_before", "_after"),
        )

        if len(merged) == 0:
            return None

        # Calculate fraction of rows where column value changed
        changed = (merged[f"{col}_before"] != merged[f"{col}_after"]).sum()
        total = len(merged)
        return float(changed / total)
    except (KeyError, TypeError, ValueError):
        # If merge fails, return None
        return None


def _compute_numeric_diff(
    before: pd.Series,
    after: pd.Series,
    stat_diff: StatDiff,
    method: str = "auto",
) -> None:
    """Compute distribution diff for numeric columns."""
    if len(before) == 0 and len(after) == 0:
        stat_diff.distribution_method = "none"
        return

    # Try to convert after to numeric if it's not already
    # (handles type mutations like int -> str)
    try:
        after_numeric = pd.to_numeric(after, errors="coerce")
    except (TypeError, ValueError):
        # If conversion fails entirely, treat as categorical
        _compute_categorical_diff(before, after, stat_diff)
        return
    
    # Check if most values could be converted
    if len(after_numeric) > 0 and after_numeric.isna().sum() > len(after) * 0.5:
        # More than 50% of values are NaN after conversion - likely a type mismatch
        # Treat as categorical instead
        _compute_categorical_diff(before, after, stat_diff)
        return

    # Compute descriptive stats
    if len(before) > 0:
        before_mean = before.mean()
        before_std = before.std() or 0.0
    else:
        before_mean = 0.0
        before_std = 0.0

    if len(after_numeric) > 0:
        after_mean = after_numeric.mean()
        after_std = after_numeric.std() or 0.0
    else:
        after_mean = 0.0
        after_std = 0.0

    stat_diff.mean_delta = float(abs(after_mean - before_mean))
    stat_diff.std_delta = float(abs(after_std - before_std))

    # Use PSI if >50 unique values, else categorical
    if stat_diff.cardinality_before > 50 or stat_diff.cardinality_after > 50:
        psi_score = _compute_psi(before, after_numeric, n_bins=10)
        stat_diff.distribution_method = "psi"
        stat_diff.distribution_score = float(psi_score)
        stat_diff.distribution_label = _psi_to_label(psi_score)
    else:
        # Treat as categorical
        _compute_categorical_diff(before, after, stat_diff)


def _compute_psi(before: pd.Series, after: pd.Series, n_bins: int = 10) -> float:
    """
    Compute Population Stability Index (PSI).

    PSI = sum( (after_pct - before_pct) * ln(after_pct / before_pct) )

    Args:
        before: Series of before values (non-null)
        after: Series of after values (non-null)
        n_bins: Number of bins for histograms

    Returns:
        PSI score (float)
    """
    if len(before) == 0 or len(after) == 0:
        return 0.0

    before_clean = pd.to_numeric(before, errors="coerce").dropna()
    after_clean = pd.to_numeric(after, errors="coerce").dropna()

    if len(before_clean) == 0 or len(after_clean) == 0:
        return 0.0

    # Define bins based on combined data
    all_data = pd.concat([before_clean, after_clean])
    bin_edges = pd.cut(all_data, bins=n_bins, duplicates="drop", retbins=True)[1]

    # Bin both series
    before_binned = pd.cut(before_clean, bins=bin_edges, include_lowest=True)
    after_binned = pd.cut(after_clean, bins=bin_edges, include_lowest=True)

    # Compute distributions
    before_dist = before_binned.value_counts(normalize=True, sort=False)
    after_dist = after_binned.value_counts(normalize=True, sort=False)

    # Align indices
    all_bins = set(before_dist.index) | set(after_dist.index)
    before_dist = before_dist.reindex(all_bins, fill_value=0.0)
    after_dist = after_dist.reindex(all_bins, fill_value=0.0)

    # Avoid log(0) by adding small epsilon
    epsilon = 1e-10
    before_dist = before_dist + epsilon
    after_dist = after_dist + epsilon

    # Renormalize to sum to 1
    before_dist = before_dist / before_dist.sum()
    after_dist = after_dist / after_dist.sum()

    # Compute PSI
    psi = np.sum((after_dist - before_dist) * np.log(after_dist / before_dist))
    return float(psi)


def _psi_to_label(psi: float) -> str:
    """Convert PSI score to human-readable label.
    
    Uses standard PSI thresholds:
    - PSI < 0.1: stable (no significant shift)
    - PSI 0.1-0.25: moderate shift
    - PSI >= 0.25: large shift (critical)
    """
    if psi < 0.1:
        return "stable"
    elif psi < 0.25:
        return "moderate shift"
    else:
        return "large shift"


def _compute_categorical_diff(
    before: pd.Series, after: pd.Series, stat_diff: StatDiff
) -> None:
    """Compute distribution diff for categorical/string columns."""
    if len(before) == 0 and len(after) == 0:
        stat_diff.distribution_method = "none"
        return

    before_cats = set(before.unique())
    after_cats = set(after.unique())

    # Sort categories, but handle mixed types gracefully
    def _safe_sort(items):
        """Sort items, handling mixed types by converting to strings."""
        try:
            return sorted(items)
        except TypeError:
            # Mixed types - sort by string representation
            return sorted(items, key=str)
    
    stat_diff.new_categories = _safe_sort(list(after_cats - before_cats))
    stat_diff.dropped_categories = _safe_sort(list(before_cats - after_cats))

    # Chi-squared test on value counts
    if len(before_cats) > 0 and len(after_cats) > 0:
        before_counts = before.value_counts()
        after_counts = after.value_counts()

        # Align on union of categories
        all_cats = set(before_counts.index) | set(after_counts.index)
        
        # Handle mixed types in categories
        try:
            before_counts = before_counts.reindex(all_cats, fill_value=0)
            after_counts = after_counts.reindex(all_cats, fill_value=0)
        except (TypeError, KeyError):
            # If reindex fails due to mixed types, convert to strings
            str_cats = {str(c) for c in all_cats}
            before_counts = before.astype(str).value_counts()
            after_counts = after.astype(str).value_counts()
            before_counts = before_counts.reindex(str_cats, fill_value=0)
            after_counts = after_counts.reindex(str_cats, fill_value=0)

        # Chi-squared statistic
        chi2_stat = np.sum((after_counts - before_counts) ** 2 / (before_counts + 1))
        stat_diff.distribution_score = float(chi2_stat)
        stat_diff.distribution_method = "chi2"

        # Heuristic label: use normalized chi2
        normalized_chi2 = chi2_stat / max(len(before), 1)
        if normalized_chi2 < 0.05:
            stat_diff.distribution_label = "stable"
        elif normalized_chi2 < 0.15:
            stat_diff.distribution_label = "moderate shift"
        else:
            stat_diff.distribution_label = "large shift"
    else:
        stat_diff.distribution_method = "none"


def _compute_datetime_diff(
    before: pd.Series, after: pd.Series, stat_diff: StatDiff
) -> None:
    """Compute range shift for datetime columns."""
    if len(before) == 0 and len(after) == 0:
        stat_diff.distribution_method = "none"
        return

    # FIX: Check if after column is still datetime (it may have changed type)
    if not pd.api.types.is_datetime64_any_dtype(after.dtype):
        # Type changed from datetime to something else - cannot compute datetime diff
        stat_diff.distribution_method = "none"
        return

    stat_diff.distribution_method = "datetime_range"

    if len(before) > 0:
        before_min = before.min()
        before_max = before.max()
    else:
        before_min = None
        before_max = None

    if len(after) > 0:
        after_min = after.min()
        after_max = after.max()
    else:
        after_min = None
        after_max = None

    # Rough measure of range shift
    if before_min is not None and after_min is not None:
        min_shift = abs((after_min - before_min).days)
        max_shift = abs((after_max - before_max).days) if before_max else 0
        total_shift = (min_shift + max_shift) / 2
        stat_diff.distribution_score = float(total_shift)
        if total_shift > 365:
            stat_diff.distribution_label = "large shift"
        elif total_shift > 30:
            stat_diff.distribution_label = "moderate shift"
        else:
            stat_diff.distribution_label = "stable"
