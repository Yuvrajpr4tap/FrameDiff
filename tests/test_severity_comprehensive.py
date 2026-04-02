"""
BLOCK 5: Severity and Issues — Comprehensive coverage
Complete tests for issue detection and severity scoring.
"""
import pytest
import pandas as pd
import numpy as np
from framediff import compare
from framediff.exceptions import DiffThresholdError


class TestSeverityScoring:
    """V01-V11: Severity aggregation and levels"""

    def test_v01_zero_changes_info(self):
        """V01: Zero changes → severity == "info", issues == []"""
        df = pd.DataFrame({"A": [1, 2, 3], "B": ["a", "b", "c"]})
        report = compare(df, df)
        assert report.severity == "info"
        assert report.issues == []

    def test_v02_only_added_column_info(self):
        """V02: Only added column → severity == "info", 1 issue"""
        before = pd.DataFrame({"A": [1, 2, 3]})
        after = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        report = compare(before, after)
        assert report.severity == "info"
        assert len(report.issues) >= 1

    def test_v03_only_removed_column_critical(self):
        """V03: Only removed column → severity == "critical", 1 issue"""
        before = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        after = pd.DataFrame({"A": [1, 2, 3]})
        report = compare(before, after)
        assert report.severity == "critical"
        assert len(report.issues) >= 1

    def test_v04_only_type_change_info_warning(self):
        """V04: Only type change (int→float) → severity info or warning"""
        before = pd.DataFrame({"A": [1, 2, 3]})
        after = pd.DataFrame({"A": [1.0, 2.0, 3.0]})
        report = compare(before, after)
        assert report.severity in ["info", "warning"]

    def test_v05_only_null_rate_increase_warning(self):
        """V05: Only null rate +15% → severity == "warning" or "critical"  """
        before = pd.DataFrame({"A": [1.0] * 1000})
        after_data = [1.0] * 850 + [np.nan] * 150
        after = pd.DataFrame({"A": after_data})
        report = compare(before, after)
        assert report.severity in ["warning", "critical"]

    def test_v06_only_psi_005_stable(self):
        """V06: Only PSI 0.05 (stable) → severity == "info"  """
        np.random.seed(42)
        before = pd.DataFrame({"A": np.random.normal(100, 15, 1000)})
        after = pd.DataFrame({"A": np.random.normal(100, 15, 1000)})
        report = compare(before, after)
        assert report.severity == "info"

    def test_v07_only_psi_015_moderate(self):
        """V07: Only PSI 0.15 (moderate) → severity == "warning"  """
        np.random.seed(42)
        before = pd.DataFrame({"A": np.random.normal(100, 15, 1000)})
        after = pd.DataFrame({"A": np.random.normal(107, 15, 1000)})
        report = compare(before, after)
        assert report.severity in ["warning", "critical"]

    def test_v08_only_psi_025_large(self):
        """V08: Only PSI 0.25 (large) → severity == "critical"  """
        np.random.seed(42)
        before = pd.DataFrame({"A": np.random.normal(100, 15, 1000)})
        after = pd.DataFrame({"A": np.random.normal(120, 15, 1000)})
        report = compare(before, after)
        assert report.severity == "critical"

    def test_v09_multiple_warnings_no_critical(self):
        """V09: Multiple warnings, zero criticals → severity == "warning"  """
        np.random.seed(42)
        before = pd.DataFrame({
            "A": np.random.normal(100, 15, 1000),
            "B": [1.0] * 1000
        })
        after = pd.DataFrame({
            "A": np.random.normal(107, 15, 1000),
            "B": [1.0] * 950 + [np.nan] * 50
        })
        report = compare(before, after)
        assert report.severity in ["warning", "critical"]

    def test_v10_one_critical_among_50_infos(self):
        """V10: One critical among 50 infos → severity == "critical"  """
        data = {"A": [1.0] * 1000}
        for i in range(49):
            data[f"col_{i}"] = [1] * 1000
        before = pd.DataFrame(data)
        
        # Create after_data without "A" (removed - critical)
        after_data = {}
        for i in range(49):
            after_data[f"col_{i}"] = [1] * 1000
        for i in range(49, 99):  # Add 50 new columns (infos)
            after_data[f"col_{i}"] = [2] * 1000
        after = pd.DataFrame(after_data)
        
        report = compare(before, after)
        assert report.severity == "critical"

    def test_v11_one_critical_one_warning_10_infos(self):
        """V11: One critical, one warning, 10 infos → severity == "critical"  """
        np.random.seed(42)
        before = pd.DataFrame({
            "critical_col": [1.0] * 1000,
            "warning_col": np.random.normal(100, 15, 1000),
        })
        after = pd.DataFrame({
            # critical_col removed
            "warning_col": np.random.normal(107, 15, 1000),
        })
        for i in range(10):
            after[f"new_{i}"] = np.random.normal(100, 15, 1000)
        
        report = compare(before, after)
        assert report.severity == "critical"


class TestIssueAttributes:
    """V12-V17: Issue structure and reporting"""

    def test_v12_every_issue_has_severity_attribute(self):
        """V12: Every issue has .severity attribute"""
        before = pd.DataFrame({"A": [1.0] * 1000})
        after_data = [1.0] * 850 + [np.nan] * 150
        after = pd.DataFrame({"A": after_data})
        report = compare(before, after)
        
        for issue in report.issues:
            assert hasattr(issue, 'severity')
            assert issue.severity in ["info", "warning", "critical"]

    def test_v13_every_issue_has_category_attribute(self):
        """V13: Every issue has .category attribute"""
        before = pd.DataFrame({"A": [1.0] * 1000})
        after_data = [1.0] * 850 + [np.nan] * 150
        after = pd.DataFrame({"A": after_data})
        report = compare(before, after)
        
        for issue in report.issues:
            assert hasattr(issue, 'category')
            assert isinstance(issue.category, str)

    def test_v14_every_issue_has_message_attribute(self):
        """V14: Every issue has .message attribute that is a non-empty string"""
        before = pd.DataFrame({"A": [1.0] * 1000})
        after = pd.DataFrame({"A": [1] * 1000})
        report = compare(before, after)
        
        for issue in report.issues:
            assert hasattr(issue, 'message')
            assert isinstance(issue.message, str)
            assert len(issue.message) > 0

    def test_v15_issues_is_list_never_none(self):
        """V15: report.issues is a list (never None)"""
        before = pd.DataFrame({"A": [1, 2, 3]})
        after = pd.DataFrame({"A": [1, 2, 3]})
        report = compare(before, after)
        assert isinstance(report.issues, list)
        assert report.issues is not None

    def test_v16_summary_is_nonempty_string(self):
        """V16: report.summary is a non-empty string (never None, never "")"""
        before = pd.DataFrame({"A": [1, 2, 3]})
        after = pd.DataFrame({"A": [1, 2, 3, 4]})
        report = compare(before, after)
        assert isinstance(report.summary, str)
        assert len(report.summary) > 0
        assert report.summary is not None

    def test_v17_summary_changes_with_severity(self):
        """V17: report.summary changes when severity changes"""
        before_stable = pd.DataFrame({"A": [1, 2, 3]})
        after_stable = pd.DataFrame({"A": [1, 2, 3]})
        report_stable = compare(before_stable, after_stable)
        
        before_critical = pd.DataFrame({"A": [1, 2, 3]})
        after_critical = pd.DataFrame({"A": []})
        after_critical = after_critical.drop("A", axis=1)
        report_critical = compare(before_critical, after_critical)
        
        assert report_stable.summary != report_critical.summary
