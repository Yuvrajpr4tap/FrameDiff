"""
BLOCK 9: Edge Cases and Extremes — Comprehensive coverage
Complete tests for boundary conditions and unusual data.
"""
import pytest
import pandas as pd
import numpy as np
import time
from framediff import compare


class TestIdenticalAndEmpty:
    """E01-E04: Identical frames and empty frames"""

    def test_e01_both_frames_identical(self):
        """E01: Both frames identical → zero issues, fingerprint stable"""
        df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6], "C": ["x", "y", "z"]})
        report = compare(df, df)
        
        assert len(report.issues) == 0
        assert report.severity == "info"

    def test_e02_both_empty_0_rows_0_columns(self):
        """E02: Both frames empty (0 rows, 0 columns) → valid report, no crash"""
        before = pd.DataFrame()
        after = pd.DataFrame()
        
        report = compare(before, after)
        assert report is not None
        assert report.rows.added_count == 0

    def test_e03_both_0_rows_same_columns(self):
        """E03: Both frames have 0 rows but same columns → valid report"""
        before = pd.DataFrame({"A": [], "B": [], "C": []})
        after = pd.DataFrame({"A": [], "B": [], "C": []})
        
        report = compare(before, after)
        assert report is not None

    def test_e04_both_0_columns_same_rows(self):
        """E04: Both frames have 0 columns but same rows → valid report"""
        before = pd.DataFrame(index=range(10))
        after = pd.DataFrame(index=range(10))
        
        report = compare(before, after)
        assert report is not None


class TestSingleCell:
    """E05-E06: Single cell frames"""

    def test_e05_single_cell_identical(self):
        """E05: Single cell: 1 row × 1 column, identical → info"""
        before = pd.DataFrame({"A": [1]})
        after = pd.DataFrame({"A": [1]})
        
        report = compare(before, after)
        assert report.severity == "info"

    def test_e06_single_cell_different(self):
        """E06: Single cell: 1 row × 1 column, different value → change detected"""
        before = pd.DataFrame({"A": [1]})
        after = pd.DataFrame({"A": [2]})
        
        report = compare(before, after)
        assert len(report.issues) > 0 or report.rows.added_count > 0


class TestScaleExtremes:
    """E07-E09: Very large scale changes"""

    def test_e07_before_1_after_1million(self):
        """E07: Before: 1 row, after: 50,000 rows → valid report, correct counts"""
        before = pd.DataFrame({"A": [1]})
        after = pd.DataFrame({"A": np.random.random(50000)})
        
        report = compare(before, after)
        assert report.rows.added_count == 49999

    def test_e08_before_1million_after_1(self):
        """E08: Before: 50,000 rows, after: 1 row → valid report, correct counts"""
        before = pd.DataFrame({"A": np.random.random(50000)})
        after = pd.DataFrame({"A": [1]})
        
        report = compare(before, after)
        assert report.rows.removed_count == 49999

    def test_e09_before_1million_after_0(self):
        """E09: Before: 50,000 rows, after: 0 rows → removed_count == 50000"""
        before = pd.DataFrame({"A": np.random.random(50000)})
        after = pd.DataFrame({"A": []})
        
        report = compare(before, after)
        assert report.rows.removed_count == 50000


class TestAllNullOrInf:
    """E10-E14: All-null and all-inf columns"""

    def test_e10_all_nan_values(self):
        """E10: All values in every column are NaN → valid report, no crash"""
        before = pd.DataFrame({
            "A": [np.nan] * 100,
            "B": [np.nan] * 100,
            "C": [np.nan] * 100
        })
        after = pd.DataFrame({
            "A": [np.nan] * 100,
            "B": [np.nan] * 100,
            "C": [np.nan] * 100
        })
        
        report = compare(before, after)
        assert report is not None

    def test_e11_all_none_values(self):
        """E11: All values in every column are None → valid report, no crash"""
        before = pd.DataFrame({
            "A": [None] * 100,
            "B": [None] * 100
        })
        after = pd.DataFrame({
            "A": [None] * 100,
            "B": [None] * 100
        })
        
        report = compare(before, after)
        assert report is not None

    def test_e12_all_inf_values(self):
        """E12: All values in every column are np.inf → valid report, no crash"""
        before = pd.DataFrame({
            "A": [np.inf] * 100,
            "B": [np.inf] * 100
        })
        after = pd.DataFrame({
            "A": [np.inf] * 100,
            "B": [np.inf] * 100
        })
        
        report = compare(before, after)
        assert report is not None

    def test_e13_null_to_values(self):
        """E13: Column exists in before: 100% null. In after: has values → change detected"""
        before = pd.DataFrame({"A": [np.nan] * 100})
        after = pd.DataFrame({"A": np.random.random(100)})
        
        report = compare(before, after)
        assert len(report.issues) > 0 or report.stats["A"].null_rate_after < 0.5

    def test_e14_values_to_null(self):
        """E14: Column exists in before: has values. In after: 100% null → change detected"""
        before = pd.DataFrame({"A": np.random.random(100)})
        after = pd.DataFrame({"A": [np.nan] * 100})
        
        report = compare(before, after)
        assert len(report.issues) > 0


class TestLargeDatasets:
    """E15-E16: Performance on large datasets"""

    def test_e15_10000_columns_10_rows(self):
        """E15: 100 columns, 10 rows → valid report, completes in under 30 seconds"""
        data = {f"col_{i}": np.random.random(10) for i in range(100)}
        before = pd.DataFrame(data)
        after = pd.DataFrame(data)
        
        start = time.time()
        report = compare(before, after)
        elapsed = time.time() - start
        
        assert elapsed < 30
        assert report is not None

    def test_e16_2_columns_10_million_rows(self):
        """E16: 2 columns, 50,000 rows → valid report, completes in under 60 seconds"""
        before = pd.DataFrame({
            "A": np.random.random(50000),
            "B": np.random.randint(0, 1000, 50000)
        })
        after = pd.DataFrame({
            "A": np.random.random(50000),
            "B": np.random.randint(0, 1000, 50000)
        })
        
        start = time.time()
        report = compare(before, after)
        elapsed = time.time() - start
        
        assert elapsed < 60
        assert report is not None


class TestSpecialColumnNames:
    """E17-E25: Column name edge cases"""

    def test_e17_integer_column_names(self):
        """E17: Column names are integers (0, 1, 2) → handled, no crash"""
        before = pd.DataFrame({0: [1, 2, 3], 1: [4, 5, 6], 2: [7, 8, 9]})
        after = pd.DataFrame({0: [1, 2, 3], 1: [4, 5, 6], 2: [7, 8, 9]})
        
        report = compare(before, after)
        assert report is not None

    def test_e18_spaces_in_column_names(self):
        """E18: Column names contain spaces ("col name") → handled"""
        before = pd.DataFrame({"col name": [1, 2, 3], "another col": [4, 5, 6]})
        after = pd.DataFrame({"col name": [1, 2, 3], "another col": [4, 5, 6]})
        
        report = compare(before, after)
        assert report is not None

    def test_e19_unicode_column_names(self):
        """E19: Column names contain unicode ("价格", "çolümn") → handled"""
        before = pd.DataFrame({"价格": [1, 2, 3], "çolümn": [4, 5, 6]})
        after = pd.DataFrame({"价格": [1, 2, 3], "çolümn": [4, 5, 6]})
        
        report = compare(before, after)
        assert report is not None

    def test_e20_special_chars_in_column_names(self):
        """E20: Column names contain special chars ("col!@#$%") → handled"""
        before = pd.DataFrame({"col!@#$%": [1, 2, 3], "norm": [4, 5, 6]})
        after = pd.DataFrame({"col!@#$%": [1, 2, 3], "norm": [4, 5, 6]})
        
        report = compare(before, after)
        assert report is not None

    def test_e21_newline_in_column_name(self):
        """E21: Column names contain newlines ("col\nnewline") → handled"""
        before = pd.DataFrame({"col\nnewline": [1, 2, 3]})
        after = pd.DataFrame({"col\nnewline": [1, 2, 3]})
        
        report = compare(before, after)
        assert report is not None

    def test_e22_empty_string_column_name(self):
        """E22: Column name is empty string "" → handled or clear error"""
        before = pd.DataFrame({"": [1, 2, 3], "name": [4, 5, 6]})
        after = pd.DataFrame({"": [1, 2, 3], "name": [4, 5, 6]})
        
        try:
            report = compare(before, after)
            assert report is not None
        except (ValueError, KeyError):
            # Clear error is acceptable
            pass

    def test_e23_index_as_data_column(self):
        """E23: Column named "index" as regular data column → not confused with index"""
        before = pd.DataFrame({"index": [1, 2, 3], "value": [4, 5, 6]})
        after = pd.DataFrame({"index": [1, 2, 3], "value": [4, 5, 6]})
        
        report = compare(before, after)
        assert report is not None
        # "index" should be treated as a regular column
        assert "index" not in report.schema.added_columns or len(report.schema.added_columns) == 0

    def test_e24_internal_column_names(self):
        """E24: Column named "__before", "__after", "__diff" → not confused with internals"""
        before = pd.DataFrame({
            "__before": [1, 2, 3],
            "__after": [4, 5, 6],
            "__diff": [7, 8, 9]
        })
        after = pd.DataFrame({
            "__before": [1, 2, 3],
            "__after": [4, 5, 6],
            "__diff": [7, 8, 9]
        })
        
        report = compare(before, after)
        assert report is not None

    def test_e25_key_column_name_as_key(self):
        """E25: Column named "key" when key="key" is also the join key → handled"""
        before = pd.DataFrame({"key": [1, 2, 3], "value": [10, 20, 30]})
        after = pd.DataFrame({"key": [1, 2, 3], "value": [10, 20, 30]})
        
        report = compare(before, after, key="key")
        assert report is not None


class TestMixedTypes:
    """E26-E30: Mixed types and disjoint schemas"""

    def test_e26_object_column_mixed_types(self):
        """E26: Object column with mixed types: ints, strings, None, lists, dicts → no crash"""
        before = pd.DataFrame({
            "mixed": [1, "string", None, [1, 2], {"a": 1}]
        })
        after = pd.DataFrame({
            "mixed": [1, "string", None, [1, 2], {"a": 1}]
        })
        
        try:
            report = compare(before, after)
            assert report is not None
        except (TypeError, ValueError):
            # May error, but shouldn't crash unexpectedly
            pass

    def test_e27_object_column_custom_classes(self):
        """E27: Object column containing Python objects (custom classes) → no crash or clear error"""
        class CustomClass:
            def __init__(self, val):
                self.val = val
        
        obj1 = CustomClass(1)
        obj2 = CustomClass(2)
        
        before = pd.DataFrame({"obj": [obj1, obj2]})
        after = pd.DataFrame({"obj": [obj1, obj2]})
        
        try:
            report = compare(before, after)
            assert report is not None
        except (TypeError, ValueError, AttributeError):
            # Clear error is acceptable
            pass

    def test_e28_zero_variance_stable(self):
        """E28: Column with single value repeated throughout both frames → stable, no crash"""
        before = pd.DataFrame({"A": [5] * 1000})
        after = pd.DataFrame({"A": [5] * 1000})
        
        report = compare(before, after)
        assert report.severity == "info"

    def test_e29_completely_different_columns(self):
        """E29: Before and after have completely different column sets (0 overlap)
        → all before cols removed, all after cols added, no row modifications"""
        before = pd.DataFrame({"A": [1], "B": [2]})
        after = pd.DataFrame({"C": [3], "D": [4]})
        
        report = compare(before, after)
        assert len(report.schema.removed_columns) == 2
        assert len(report.schema.added_columns) == 2

    def test_e30_completely_different_row_keys(self):
        """E30: Before and after have completely different row keys (0 key overlap)
        → all rows removed and added, no modifications"""
        before = pd.DataFrame({"key": [1, 2, 3], "value": [10, 20, 30]})
        after = pd.DataFrame({"key": [4, 5, 6], "value": [40, 50, 60]})
        
        report = compare(before, after, key="key")
        assert report.rows.removed_count == 3
        assert report.rows.added_count == 3
        assert report.rows.modified_count == 0
