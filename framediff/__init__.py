"""
framediff — Schema-aware, zero-config dataframe diffing and change-tracking library.

Main public API:
    - compare(): Entry point for comparing two DataFrames
    - DiffReport: Result object with full diff details
    - DiffThresholdError: Raised when assertions fail

Example:
    >>> import framediff as fd
    >>> import pandas as pd
    >>> df1 = pd.DataFrame({"id": [1, 2, 3], "value": [10, 20, 30]})
    >>> df2 = pd.DataFrame({"id": [1, 2, 4], "value": [10, 21, 40]})
    >>> report = fd.compare(df1, df2, key="id")
    >>> print(report.summary)
    >>> report.assert_within(max_rows_added_pct=5)
"""

from .core import compare
from .report import DiffReport
from .exceptions import DiffThresholdError, InvalidFrameError, DiffKeyError

__version__ = "0.2.0"
__all__ = ["compare", "DiffReport", "DiffThresholdError", "InvalidFrameError", "DiffKeyError"]
