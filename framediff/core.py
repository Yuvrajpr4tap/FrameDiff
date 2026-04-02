"""
Core compare() function — the main entry point for framediff.
"""
from typing import Union, Optional, List
import pandas as pd

from .schema import compare_schemas
from .stats import compare_stats
from .rows import compare_rows
from .report import DiffReport
from .exceptions import InvalidFrameError

# Try to import polars at module level
try:
    import polars as pl
    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False


def _normalise(df: Union[pd.DataFrame, "pl.DataFrame"]) -> pd.DataFrame:
    """
    Normalize a DataFrame to pandas format.

    Supports both pandas and Polars DataFrames and LazyFrames.

    Args:
        df: pandas or Polars DataFrame/LazyFrame

    Returns:
        pandas DataFrame

    Raises:
        TypeError: If not a supported DataFrame type
    """
    if isinstance(df, pd.DataFrame):
        return df

    # Try Polars DataFrame/LazyFrame by class check
    type_name = type(df).__name__
    module_name = type(df).__module__
    
    if "polars" in module_name:
        if "DataFrame" in type_name:
            # Standard Polars DataFrame
            if HAS_POLARS and isinstance(df, pl.DataFrame):
                return df.to_pandas()
            # Fallback using method call
            try:
                if hasattr(df, 'to_pandas'):
                    return df.to_pandas()
            except Exception:
                pass
        elif "LazyFrame" in type_name:
            # Polars LazyFrame - collect first then convert
            try:
                if hasattr(df, 'collect'):
                    collected = df.collect()
                    if hasattr(collected, 'to_pandas'):
                        return collected.to_pandas()
            except Exception:
                pass

    raise TypeError(
        f"Unsupported frame type: {module_name}.{type_name}. "
        f"Expected pandas.DataFrame or polars.DataFrame"
    )


def compare(
    before: Union[pd.DataFrame, "pl.DataFrame"],
    after: Union[pd.DataFrame, "pl.DataFrame"],
    key: Optional[Union[str, List[str]]] = None,
    sample_size: Optional[int] = None,
    stat_methods: Optional[List[str]] = None,
    severity_thresholds: Optional[dict] = None,
) -> DiffReport:
    """
    Compare two DataFrames and return a comprehensive diff report.

    This is the main entry point for framediff. It performs schema, statistical,
    and row-level analysis on the two input frames and returns a DiffReport with
    full details.

    Args:
        before: DataFrame before changes (pandas or Polars)
        after: DataFrame after changes (pandas or Polars)
        key: Column name(s) to use as row join key. If None, uses positional matching.
             Can be a single column name (str) or list of column names for composite keys.
        sample_size: Optional row sample size for large frames. If provided, will sample
                     this many rows from each DataFrame for analysis (performance optimization).
                     Default: None (use all rows).
        stat_methods: List of statistical methods to use for distribution analysis.
                      Options: "auto" (default), "psi", "kl", "wasserstein", "chi2".
                      Default: ["auto"] which auto-selects per column type.
        severity_thresholds: Optional dict to override default severity thresholds.
                            Not currently used but reserved for future expansion.

    Returns:
        DiffReport: Comprehensive diff report containing:
            - schema: SchemaDiff with column add/remove/type changes
            - stats: Dict of StatDiff per column with distribution analysis
            - rows: RowDiff with added/removed/modified row counts
            - issues: List of DiffIssue objects with severity levels
            - severity: Aggregate severity ("info", "warning", "critical")

    Example:
        >>> import framediff as fd
        >>> import pandas as pd
        >>> df1 = pd.DataFrame({"id": [1, 2, 3], "value": [10.0, 20.0, 30.0]})
        >>> df2 = pd.DataFrame({"id": [1, 2, 4], "value": [10.0, 21.0, 40.0]})
        >>> report = fd.compare(df1, df2, key="id")
        >>> print(report.summary)
        >>> report.assert_within(max_rows_added_pct=5, max_rows_removed_pct=5)

    Raises:
        TypeError: If before or after are not supported DataFrame types
        ValueError: If key validation fails
    """
    # Normalize frames
    before_normalized = _normalise(before)
    after_normalized = _normalise(after)

    # BUG FIX #1: Validate key is not an empty list
    if isinstance(key, list) and len(key) == 0:
        raise ValueError(
            "key cannot be empty. Provide a column name (key='id') "
            "or a list with at least one column (key=['col_a', 'col_b'])."
        )

    # BUG FIX #2: Validate key columns exist in both DataFrames
    if key is not None:
        key_cols = [key] if isinstance(key, str) else key
        for col in key_cols:
            if col not in before_normalized.columns:
                raise ValueError(
                    f"Key column {col!r} not found in before DataFrame. "
                    f"Available columns: {list(before_normalized.columns)}"
                )
            if col not in after_normalized.columns:
                raise ValueError(
                    f"Key column {col!r} not found in after DataFrame. "
                    f"Available columns: {list(after_normalized.columns)}"
                )

    # Apply sampling if requested
    if sample_size is not None:
        if len(before_normalized) > sample_size:
            before_normalized = before_normalized.sample(
                n=sample_size, random_state=42
            )
        if len(after_normalized) > sample_size:
            after_normalized = after_normalized.sample(n=sample_size, random_state=42)

    if stat_methods is None:
        stat_methods = ["auto"]

    # Perform schema diff
    schema_diff = compare_schemas(before_normalized, after_normalized)

    # Perform statistical diff
    stats_diff = compare_stats(before_normalized, after_normalized, stat_methods, key=key)

    # Perform row diff
    rows_diff = compare_rows(before_normalized, after_normalized, key=key)

    # Create and return report
    report = DiffReport(
        schema=schema_diff,
        stats=stats_diff,
        rows=rows_diff,
    )

    return report
