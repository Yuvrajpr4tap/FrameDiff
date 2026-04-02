"""
Custom exceptions for framediff.
"""
from typing import Any, Optional


class DiffThresholdError(Exception):
    """Raised when a diff report violates specified thresholds in assert_within()."""

    def __init__(self, message: str, violations: Optional[list[Any]] = None):
        """
        Args:
            message: Human-readable error message
            violations: List of assertion violations
        """
        super().__init__(message)
        self.message = message
        self.violations = violations or []


class InvalidFrameError(Exception):
    """Raised when an invalid frame is passed to compare()."""

    pass


class DiffKeyError(Exception):
    """Raised when key column(s) contain duplicates or are invalid."""

    pass
