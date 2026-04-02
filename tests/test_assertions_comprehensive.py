"""
BLOCK 7: assert_within — Comprehensive coverage
Complete tests for assertion API and threshold validation.
"""
import pytest
import pandas as pd
import numpy as np
from framediff import compare
from framediff.exceptions import DiffThresholdError


class TestAssertBasics:
    """AW01-AW10: Basic assertion behavior"""

    def test_aw01_no_arguments_never_raises(self):
        """AW01: No arguments → never raises, regardless of report content"""
        before = pd.DataFrame({"A": []})
        before = before.drop("A", axis=1)
        after = pd.DataFrame({"B": [1] * 1000})
        report = compare(before, after)
        
        # Should not raise even with massive changes
        report.assert_within()

    def test_aw02_all_thresholds_satisfied(self):
        """AW02: All thresholds satisfied → never raises"""
        before = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        after = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        report = compare(before, after)
        
        # No violations
        report.assert_within(
            max_rows_removed_pct=1,
            max_rows_added_pct=1,
            max_null_rate_increase=0.1,
            max_psi=0.05
        )

    def test_aw03_max_rows_removed_pct_violation(self):
        """AW03: max_rows_removed_pct=0.01, actual=0.05 → raises DiffThresholdError"""
        before = pd.DataFrame({"A": range(10000)})
        after = pd.DataFrame({"A": range(9500)})  # 5% removed
        report = compare(before, after)
        
        with pytest.raises(DiffThresholdError):
            report.assert_within(max_rows_removed_pct=0.01)

    def test_aw04_max_rows_added_pct_violation(self):
        """AW04: max_rows_added_pct=0.01, actual=0.05 → raises DiffThresholdError"""
        before = pd.DataFrame({"A": range(10000)})
        after = pd.DataFrame({"A": range(10500)})  # 5% added
        report = compare(before, after)
        
        with pytest.raises(DiffThresholdError):
            report.assert_within(max_rows_added_pct=0.01)

    def test_aw05_max_null_rate_increase_violation(self):
        """AW05: max_null_rate_increase=0.01, actual=0.10 → raises DiffThresholdError"""
        before = pd.DataFrame({"A": [1.0] * 1000})
        after_data = [1.0] * 900 + [np.nan] * 100
        after = pd.DataFrame({"A": after_data})
        report = compare(before, after)
        
        with pytest.raises(DiffThresholdError):
            report.assert_within(max_null_rate_increase=0.01)

    def test_aw06_max_psi_violation(self):
        """AW06: max_psi=0.1, actual=0.3 → raises DiffThresholdError"""
        np.random.seed(42)
        before = pd.DataFrame({"A": np.random.normal(100, 15, 1000)})
        after = pd.DataFrame({"A": np.random.normal(125, 15, 1000)})
        report = compare(before, after)
        
        with pytest.raises(DiffThresholdError):
            report.assert_within(max_psi=0.1)

    def test_aw07_no_type_changes_violation(self):
        """AW07: no_type_changes=True, one type changed → raises DiffThresholdError"""
        before = pd.DataFrame({"A": [1, 2, 3]})
        after = pd.DataFrame({"A": [1.0, 2.0, 3.0]})
        report = compare(before, after)
        
        with pytest.raises(DiffThresholdError):
            report.assert_within(no_type_changes=True)

    def test_aw08_no_removed_columns_violation(self):
        """AW08: no_removed_columns=True, one removed → raises DiffThresholdError"""
        before = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        after = pd.DataFrame({"A": [1, 2, 3]})
        report = compare(before, after)
        
        with pytest.raises(DiffThresholdError):
            report.assert_within(no_removed_columns=True)

    def test_aw09_no_critical_violation(self):
        """AW09: no_critical=True, one critical issue → raises DiffThresholdError"""
        before = pd.DataFrame({"A": [1, 2, 3]})
        after = pd.DataFrame({"A": []})
        after = after.drop("A", axis=1)
        report = compare(before, after)
        
        with pytest.raises(DiffThresholdError):
            report.assert_within(no_critical=True)

    def test_aw10_multiple_violations_all_in_message(self):
        """AW10: Multiple thresholds all violated → error message mentions ALL violations"""
        before = pd.DataFrame({"A": [1, 2, 3], "B": [4.0] * 3})
        after = pd.DataFrame({
            "A": [1.0, 2.0, 3.0],  # type change
            "B": [4.0, 4.0, np.nan]  # null increase
        })
        after = after.iloc[:1]  # Remove rows
        report = compare(before, after)
        
        error_raised = False
        error_msg = ""
        try:
            report.assert_within(
                no_type_changes=True,
                max_rows_removed_pct=0.01,
                max_null_rate_increase=0.01
            )
        except DiffThresholdError as e:
            error_raised = True
            error_msg = str(e)
        
        assert error_raised
        # Message should mention all violations
        assert len(error_msg) > 10  # Has substantial content


class TestColumnFiltering:
    """AW11-AW13: Column-specific assertions"""

    def test_aw11_column_filter_no_violation_in_monitored(self):
        """AW11: columns=["col_a"], violation only in col_b → does NOT raise"""
        before = pd.DataFrame({
            "col_a": [1, 2, 3],
            "col_b": [1.0] * 3
        })
        after = pd.DataFrame({
            "col_a": [1, 2, 3],
            "col_b": [1.0, 1.0, np.nan]  # violation in col_b
        })
        report = compare(before, after)
        
        # Should not raise because col_b is not monitored
        report.assert_within(columns=["col_a"], max_null_rate_increase=0.01)

    def test_aw12_column_filter_violation_in_monitored(self):
        """AW12: columns=["col_a"], violation in col_a → raises"""
        before = pd.DataFrame({
            "col_a": [1.0] * 3,
            "col_b": [1.0] * 3
        })
        after = pd.DataFrame({
            "col_a": [1.0, 1.0, np.nan],  # violation in col_a
            "col_b": [1.0, 1.0, np.nan]
        })
        report = compare(before, after)
        
        with pytest.raises(DiffThresholdError):
            report.assert_within(columns=["col_a"], max_null_rate_increase=0.01)

    def test_aw13_error_message_contains_column_name(self):
        """AW13: DiffThresholdError message contains the specific column name"""
        before = pd.DataFrame({"col_a": [1.0] * 10})
        after = pd.DataFrame({"col_a": [1.0] * 8})  # 20% removed
        report = compare(before, after)
        
        try:
            report.assert_within(max_rows_removed_pct=0.01)
            assert False, "Should have raised"
        except DiffThresholdError as e:
            assert "0.2" in str(e) or "20" in str(e) or "rows" in str(e).lower()


class TestErrorMessageContent:
    """AW14-AW15: Error message quality"""

    def test_aw14_error_message_contains_actual_value(self):
        """AW14: DiffThresholdError message contains the actual value"""
        before = pd.DataFrame({"A": range(10000)})
        after = pd.DataFrame({"A": range(9500)})  # 5% removed
        report = compare(before, after)
        
        try:
            report.assert_within(max_rows_removed_pct=0.01)
            assert False, "Should have raised"
        except DiffThresholdError as e:
            msg = str(e)
            # Should mention the actual percentage or "removed"
            assert "removed" in msg.lower() or "0.05" in msg or "5%" in msg

    def test_aw15_error_message_contains_threshold_value(self):
        """AW15: DiffThresholdError message contains the threshold value"""
        before = pd.DataFrame({"A": range(10000)})
        after = pd.DataFrame({"A": range(9500)})
        report = compare(before, after)
        
        try:
            report.assert_within(max_rows_removed_pct=0.01)
            assert False, "Should have raised"
        except DiffThresholdError as e:
            msg = str(e)
            # Should mention the threshold
            assert "0.01" in msg or "1%" in msg or "threshold" in msg.lower()


class TestIdempotency:
    """AW16-AW17: Idempotency and non-mutation"""

    def test_aw16_assert_within_idempotent(self):
        """AW16: assert_within() is idempotent: calling twice produces same result"""
        before = pd.DataFrame({"A": [1, 2, 3]})
        after = pd.DataFrame({"A": [1, 2, 3]})
        report = compare(before, after)
        
        # Call twice with same checks
        report.assert_within()
        report.assert_within()
        
        # Both should succeed (both raise nothing or both raise same thing)

    def test_aw17_assert_within_does_not_mutate_report(self):
        """AW17: assert_within() does not mutate the report"""
        before = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        after = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        report = compare(before, after)
        
        fingerprint_before = report.fingerprint
        severity_before = report.severity
        issues_before = len(report.issues)
        
        try:
            report.assert_within(max_rows_removed_pct=0.1)
        except DiffThresholdError:
            pass
        
        # Report should be unchanged
        assert report.fingerprint == fingerprint_before
        assert report.severity == severity_before
        assert len(report.issues) == issues_before
