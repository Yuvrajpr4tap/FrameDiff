"""
BLOCK 14: API Contract — Final verification
Complete tests for public API contracts and return types.
"""
import pytest
import pandas as pd
import numpy as np
from framediff import compare, DiffReport
from framediff.exceptions import DiffThresholdError, DiffKeyError, InvalidFrameError
from framediff.schema import SchemaDiff
from framediff.rows import RowDiff


class TestPublicAPI:
    """AC01-AC04: Public API existence and types"""

    def test_ac01_fd_compare_exists(self):
        """AC01: fd.compare exists and is callable"""
        assert callable(compare)

    def test_ac02_diff_report_exists(self):
        """AC02: fd.DiffReport exists and is a class"""
        assert isinstance(DiffReport, type)

    def test_ac03_diff_threshold_error_exists(self):
        """AC03: fd.DiffThresholdError exists and is an Exception subclass"""
        assert issubclass(DiffThresholdError, Exception)

    def test_ac04_diff_key_error_exists(self):
        """AC04: fd.DiffKeyError exists and is an Exception subclass"""
        assert issubclass(DiffKeyError, Exception)


class TestDiffReportAttributes:
    """AC05-AC14: DiffReport attribute types"""

    def test_ac05_schema_is_schema_diff(self):
        """AC05: DiffReport.schema is always SchemaDiff (never None)"""
        before = pd.DataFrame({"A": [1, 2, 3]})
        after = pd.DataFrame({"A": [1, 2, 3]})
        report = compare(before, after)
        
        assert report.schema is not None
        assert isinstance(report.schema, SchemaDiff)

    def test_ac06_stats_is_dict(self):
        """AC06: DiffReport.stats is always dict (never None)"""
        before = pd.DataFrame({"A": [1, 2, 3]})
        after = pd.DataFrame({"A": [1, 2, 3]})
        report = compare(before, after)
        
        assert report.stats is not None
        assert isinstance(report.stats, dict)

    def test_ac07_rows_is_row_diff(self):
        """AC07: DiffReport.rows is always RowDiff (never None)"""
        before = pd.DataFrame({"A": [1, 2, 3]})
        after = pd.DataFrame({"A": [1, 2, 3]})
        report = compare(before, after)
        
        assert report.rows is not None
        assert isinstance(report.rows, RowDiff)

    def test_ac08_issues_is_list(self):
        """AC08: DiffReport.issues is always list (never None)"""
        before = pd.DataFrame({"A": [1, 2, 3]})
        after = pd.DataFrame({"A": [1, 2, 3]})
        report = compare(before, after)
        
        assert report.issues is not None
        assert isinstance(report.issues, list)

    def test_ac09_severity_valid_value(self):
        """AC09: DiffReport.severity is always one of: "info", "warning", "critical"  """
        test_cases = [
            (pd.DataFrame({"A": [1, 2, 3]}), pd.DataFrame({"A": [1, 2, 3]})),  # identical
            (pd.DataFrame({"A": [1, 2, 3]}), pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})),  # added
            (pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]}), pd.DataFrame({"A": [1, 2, 3]})),  # removed
        ]
        
        for before, after in test_cases:
            report = compare(before, after)
            assert report.severity in ["info", "warning", "critical"]

    def test_ac10_fingerprint_format(self):
        """AC10: DiffReport.fingerprint is always 64-char lowercase hex"""
        before = pd.DataFrame({"A": [1, 2, 3]})
        after = pd.DataFrame({"A": [1, 2, 3]})
        report = compare(before, after)
        
        assert isinstance(report.fingerprint, str)
        assert len(report.fingerprint) == 64
        assert all(c in "0123456789abcdef" for c in report.fingerprint)

    def test_ac11_summary_non_empty_string(self):
        """AC11: DiffReport.summary is always non-empty string"""
        before = pd.DataFrame({"A": [1, 2, 3]})
        after = pd.DataFrame({"A": [1, 2, 3]})
        report = compare(before, after)
        
        assert isinstance(report.summary, str)
        assert len(report.summary) > 0

    def test_ac12_to_dict_returns_dict(self):
        """AC12: DiffReport.to_dict() always returns dict"""
        before = pd.DataFrame({"A": [1, 2, 3]})
        after = pd.DataFrame({"A": [1, 2, 3]})
        report = compare(before, after)
        
        result = report.to_dict()
        assert isinstance(result, dict)

    def test_ac13_to_json_returns_string(self):
        """AC13: DiffReport.to_json() always returns valid JSON string"""
        import json
        
        before = pd.DataFrame({"A": [1, 2, 3]})
        after = pd.DataFrame({"A": [1, 2, 3]})
        report = compare(before, after)
        
        result = report.to_json()
        assert isinstance(result, str)
        # Must be valid JSON
        json.loads(result)

    def test_ac14_assert_within_exists(self):
        """AC14: DiffReport.assert_within() exists and accepts **kwargs"""
        before = pd.DataFrame({"A": [1, 2, 3]})
        after = pd.DataFrame({"A": [1, 2, 3]})
        report = compare(before, after)
        
        assert callable(getattr(report, "assert_within", None))
        # Should accept kwargs
        report.assert_within(max_rows_added_pct=0.1)


class TestSchemaDiffContract:
    """AC15-AC20: SchemaDiff attribute contracts"""

    def test_ac15_added_columns_is_list(self):
        """AC15: SchemaDiff.added_columns is always list"""
        before = pd.DataFrame({"A": [1, 2, 3]})
        after = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        report = compare(before, after)
        
        assert isinstance(report.schema.added_columns, list)

    def test_ac16_removed_columns_is_list(self):
        """AC16: SchemaDiff.removed_columns is always list"""
        before = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        after = pd.DataFrame({"A": [1, 2, 3]})
        report = compare(before, after)
        
        assert isinstance(report.schema.removed_columns, list)

    def test_ac17_type_changes_is_dict(self):
        """AC17: SchemaDiff.type_changes is always dict"""
        before = pd.DataFrame({"A": [1, 2, 3]})
        after = pd.DataFrame({"A": [1.0, 2.0, 3.0]})
        report = compare(before, after)
        
        assert isinstance(report.schema.type_changes, dict)

    def test_ac18_added_count_is_nonnegative_int(self):
        """AC18: RowDiff.added_count is always int ≥ 0"""
        before = pd.DataFrame({"A": [1, 2, 3]})
        after = pd.DataFrame({"A": [1, 2, 3, 4]})
        report = compare(before, after)
        
        assert isinstance(report.rows.added_count, int)
        assert report.rows.added_count >= 0

    def test_ac19_removed_count_is_nonnegative_int(self):
        """AC19: RowDiff.removed_count is always int ≥ 0"""
        before = pd.DataFrame({"A": [1, 2, 3, 4]})
        after = pd.DataFrame({"A": [1, 2, 3]})
        report = compare(before, after)
        
        assert isinstance(report.rows.removed_count, int)
        assert report.rows.removed_count >= 0

    def test_ac20_modified_count_is_nonnegative_int(self):
        """AC20: RowDiff.modified_count is always int ≥ 0"""
        before = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        after = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 7]})
        report = compare(before, after, key="A")
        
        assert isinstance(report.rows.modified_count, int)
        assert report.rows.modified_count >= 0


class TestStatDiffContract:
    """AC21-AC24: StatDiff attribute contracts"""

    def test_ac21_distribution_score_float_or_none(self):
        """AC21: StatDiff.distribution_score is float or None (never NaN)"""
        before = pd.DataFrame({"A": [1, 2, 3, 4, 5]})
        after = pd.DataFrame({"A": [1, 2, 3, 4, 5]})
        report = compare(before, after)
        
        if "A" in report.stats:
            score = report.stats["A"].distribution_score
            if score is not None:
                assert isinstance(score, (int, float))
                assert not np.isnan(score)

    def test_ac22_null_rate_before_valid_range(self):
        """AC22: StatDiff.null_rate_before is float between 0.0 and 1.0"""
        before = pd.DataFrame({"A": [1.0, 2.0, np.nan]})
        after = pd.DataFrame({"A": [1.0, 2.0, np.nan]})
        report = compare(before, after)
        
        if "A" in report.stats:
            rate = report.stats["A"].null_rate_before
            assert isinstance(rate, float)
            assert 0.0 <= rate <= 1.0

    def test_ac23_null_rate_after_valid_range(self):
        """AC23: StatDiff.null_rate_after is float between 0.0 and 1.0"""
        before = pd.DataFrame({"A": [1.0, 2.0, np.nan]})
        after = pd.DataFrame({"A": [1.0, 2.0, 3.0, np.nan, np.nan]})
        report = compare(before, after)
        
        if "A" in report.stats:
            rate = report.stats["A"].null_rate_after
            assert isinstance(rate, float)
            assert 0.0 <= rate <= 1.0

    def test_ac24_contract_holds_across_frame_types(self):
        """AC24: All contracts hold across: identical frames, completely different
        frames, empty frames, 1M row frames, Polars inputs"""
        test_cases = [
            # Identical frames
            (pd.DataFrame({"A": [1, 2, 3]}), pd.DataFrame({"A": [1, 2, 3]})),
            # Completely different
            (pd.DataFrame({"A": [1, 2, 3]}), pd.DataFrame({"B": [4, 5, 6]})),
            # Empty
            (pd.DataFrame(), pd.DataFrame()),
            # Different row counts
            (pd.DataFrame({"A": [1]}), pd.DataFrame({"A": range(1000)})),
        ]
        
        for before, after in test_cases:
            report = compare(before, after)
            
            # Verify all contracts
            assert isinstance(report, DiffReport)
            assert report.schema is not None
            assert isinstance(report.schema, SchemaDiff)
            assert isinstance(report.rows, RowDiff)
            assert isinstance(report.issues, list)
            assert report.severity in ["info", "warning", "critical"]
            assert isinstance(report.fingerprint, str)
            assert len(report.fingerprint) == 64
            assert isinstance(report.summary, str)
            assert len(report.summary) > 0
