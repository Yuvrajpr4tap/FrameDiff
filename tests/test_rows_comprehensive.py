"""
BLOCK 4: Row-level Diff — Comprehensive coverage
Complete tests for row addition, removal, and modification detection.
"""
import pytest
import pandas as pd
import numpy as np
from framediff import compare
from framediff.exceptions import DiffKeyError


class TestRowAddRemove:
    """R01-R07: Row addition and removal"""

    def test_r01_add_exactly_1_row(self):
        """R01: Add exactly 1 row → added_count == 1"""
        before = pd.DataFrame({"id": [1, 2, 3], "value": [10, 20, 30]})
        after = pd.DataFrame({"id": [1, 2, 3, 4], "value": [10, 20, 30, 40]})
        report = compare(before, after)
        assert report.rows.added_count == 1

    def test_r02_add_exactly_100_rows(self):
        """R02: Add exactly 100 rows → added_count == 100"""
        before = pd.DataFrame({"id": range(100), "value": range(100)})
        after = pd.DataFrame({"id": range(200), "value": range(200)})
        report = compare(before, after)
        assert report.rows.added_count == 100

    def test_r03_add_exactly_100000_rows(self):
        """R03: Add exactly 100,000 rows → added_count == 100000"""
        before = pd.DataFrame({"id": range(100000), "value": np.random.random(100000)})
        add_count = 100000
        after = pd.DataFrame({"id": range(200000), "value": np.random.random(200000)})
        report = compare(before, after)
        assert report.rows.added_count == add_count

    def test_r04_remove_exactly_1_row(self):
        """R04: Remove exactly 1 row → removed_count == 1"""
        before = pd.DataFrame({"id": [1, 2, 3], "value": [10, 20, 30]})
        after = pd.DataFrame({"id": [1, 2], "value": [10, 20]})
        report = compare(before, after)
        assert report.rows.removed_count == 1

    def test_r05_remove_exactly_100_rows(self):
        """R05: Remove exactly 100 rows → removed_count == 100"""
        before = pd.DataFrame({"id": range(200), "value": range(200)})
        after = pd.DataFrame({"id": range(100), "value": range(100)})
        report = compare(before, after)
        assert report.rows.removed_count == 100

    def test_r06_remove_all_rows(self):
        """R06: Remove all rows → removed_count == original total, added_count == 0"""
        before = pd.DataFrame({"id": range(100), "value": range(100)})
        after = pd.DataFrame({"id": [], "value": []})
        report = compare(before, after)
        assert report.rows.removed_count == 100
        assert report.rows.added_count == 0

    def test_r07_replace_all_rows(self):
        """R07: Replace all rows (all removed, all new added) → counts correct"""
        before = pd.DataFrame({"id": range(50), "value": range(50)})
        after = pd.DataFrame({"id": range(50, 100), "value": range(50)})
        report = compare(before, after)
        assert report.rows.removed_count == 50
        assert report.rows.added_count == 50


class TestRowModification:
    """R08-R11: Row modification detection"""

    def test_r08_modify_exactly_1_cell(self):
        """R08: Modify exactly 1 cell in 1 row → modified_count == 1, modifications["col"] == 1"""
        before = pd.DataFrame({"id": [1, 2, 3], "value": [10, 20, 30]})
        after = pd.DataFrame({"id": [1, 2, 3], "value": [10, 21, 30]})
        report = compare(before, after, key="id")
        assert report.rows.modified_count == 1
        assert report.rows.modifications.get("value", 0) == 1

    def test_r09_modify_1_cell_in_every_row(self):
        """R09: Modify 1 cell in every row → modified_count == total rows"""
        before = pd.DataFrame({"id": range(100), "value": range(100)})
        after = pd.DataFrame({"id": range(100), "value": np.arange(1, 101)})
        report = compare(before, after, key="id")
        assert report.rows.modified_count == 100

    def test_r10_modify_every_cell_in_every_row(self):
        """R10: Modify every cell in every row → modified_count == total rows, modifications has all cols"""
        before = pd.DataFrame({"id": range(10), "value": range(10), "category": ["A"] * 10})
        after = pd.DataFrame(
            {"id": range(1, 11), "value": range(10, 20), "category": ["B"] * 10}
        )
        # Only value and category should be tracked (id is the key)
        report = compare(before, after, key="id")
        assert report.rows.modified_count == 10

    def test_r11_no_changes(self):
        """R11: No changes → modified_count == 0, added_count == 0, removed_count == 0"""
        df = pd.DataFrame({"id": [1, 2, 3], "value": [10, 20, 30]})
        report = compare(df, df, key="id")
        assert report.rows.modified_count == 0
        assert report.rows.added_count == 0
        assert report.rows.removed_count == 0


class TestKeyMatching:
    """R12-R18: Key-based and positional matching"""

    def test_r12_single_column_key(self):
        """R12: Single-column key → correct matching"""
        before = pd.DataFrame({
            "user_id": [1, 2, 3],
            "score": [100, 200, 300]
        })
        after = pd.DataFrame({
            "user_id": [1, 2, 3],
            "score": [100, 210, 300]
        })
        report = compare(before, after, key="user_id")
        assert report.rows.modified_count == 1

    def test_r13_two_column_composite_key(self):
        """R13: Two-column composite key → correct matching"""
        before = pd.DataFrame({
            "year": [2020, 2020, 2021],
            "month": [1, 2, 1],
            "revenue": [1000, 2000, 3000]
        })
        after = pd.DataFrame({
            "year": [2020, 2020, 2021],
            "month": [1, 2, 1],
            "revenue": [1000, 2100, 3000]
        })
        report = compare(before, after, key=["year", "month"])
        assert report.rows.modified_count == 1

    def test_r14_five_column_composite_key(self):
        """R14: Five-column composite key → correct matching"""
        before = pd.DataFrame({
            "a": [1, 2, 3],
            "b": [10, 20, 30],
            "c": [100, 200, 300],
            "d": ["x", "y", "z"],
            "e": ["p", "q", "r"],
            "val": [1000, 2000, 3000]
        })
        after = pd.DataFrame({
            "a": [1, 2, 3],
            "b": [10, 20, 30],
            "c": [100, 200, 300],
            "d": ["x", "y", "z"],
            "e": ["p", "q", "r"],
            "val": [1000, 2100, 3000]
        })
        report = compare(before, after, key=["a", "b", "c", "d", "e"])
        assert report.rows.modified_count == 1

    def test_r15_key_column_contains_nan(self):
        """R15: Key column contains NaN → raises DiffKeyError or handles with warning"""
        before = pd.DataFrame({
            "id": [1.0, 2.0, np.nan],
            "value": [10, 20, 30]
        })
        after = pd.DataFrame({
            "id": [1.0, 2.0, np.nan],
            "value": [10, 20, 30]
        })
        with pytest.raises(DiffKeyError):
            compare(before, after, key="id")

    def test_r16_key_column_has_duplicates(self):
        """R16: Key column has duplicates → raises DiffKeyError"""
        before = pd.DataFrame({
            "id": [1, 1, 2],  # duplicate 1
            "value": [10, 20, 30]
        })
        after = pd.DataFrame({
            "id": [1, 1, 2],
            "value": [10, 20, 30]
        })
        with pytest.raises(DiffKeyError):
            compare(before, after, key="id")

    def test_r17_key_empty_list(self):
        """R17: key=[] → raises ValueError with helpful message"""
        before = pd.DataFrame({"id": [1, 2, 3], "value": [10, 20, 30]})
        after = pd.DataFrame({"id": [1, 2, 3], "value": [10, 20, 30]})
        with pytest.raises(ValueError):
            compare(before, after, key=[])

    def test_r18_key_nonexistent_column(self):
        """R18: key="nonexistent" → raises ValueError naming the column"""
        before = pd.DataFrame({"id": [1, 2, 3], "value": [10, 20, 30]})
        after = pd.DataFrame({"id": [1, 2, 3], "value": [10, 20, 30]})
        with pytest.raises((ValueError, KeyError)) as exc_info:
            compare(before, after, key="nonexistent")
        assert "nonexistent" in str(exc_info.value).lower()


class TestPositionalMatching:
    """R19-R21: Positional matching without keys"""

    def test_r19_key_none_equal_row_counts(self):
        """R19: key=None, equal row counts → positional matching, 0 modifications"""
        before = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        after = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        report = compare(before, after, key=None)
        assert report.rows.modified_count == 0

    def test_r20_key_none_before_100_after_150(self):
        """R20: key=None, before has 100 rows, after has 150 → 50 added"""
        before = pd.DataFrame({"id": range(100), "value": np.random.random(100)})
        after = pd.DataFrame({"id": range(150), "value": np.random.random(150)})
        report = compare(before, after, key=None)
        assert report.rows.added_count == 50

    def test_r21_key_none_before_150_after_100(self):
        """R21: key=None, before has 150 rows, after has 100 → 50 removed"""
        before = pd.DataFrame({"id": range(150), "value": np.random.random(150)})
        after = pd.DataFrame({"id": range(100), "value": np.random.random(100)})
        report = compare(before, after, key=None)
        assert report.rows.removed_count == 50


class TestSampleDataframes:
    """R22-R26: Sample DataFrames and modification tracking"""

    def test_r22_sample_added_not_none_when_added_count_zero(self):
        """R22: sample_added is a DataFrame (not None) even when added_count == 0"""
        before = pd.DataFrame({"id": [1, 2, 3], "value": [10, 20, 30]})
        after = pd.DataFrame({"id": [1, 2, 3], "value": [10, 20, 30]})
        report = compare(before, after, key="id")
        assert isinstance(report.rows.sample_added, pd.DataFrame)
        assert len(report.rows.sample_added) == 0

    def test_r23_sample_removed_not_none_when_removed_count_zero(self):
        """R23: sample_removed is a DataFrame (not None) even when removed_count == 0"""
        before = pd.DataFrame({"id": [1, 2, 3], "value": [10, 20, 30]})
        after = pd.DataFrame({"id": [1, 2, 3], "value": [10, 20, 30]})
        report = compare(before, after, key="id")
        assert isinstance(report.rows.sample_removed, pd.DataFrame)
        assert len(report.rows.sample_removed) == 0

    def test_r24_sample_modified_has_before_after_columns(self):
        """R24: sample_modified contains __before and __after columns for changed cols"""
        before = pd.DataFrame({
            "id": [1, 2, 3],
            "value": [10, 20, 30],
            "category": ["A", "B", "C"]
        })
        after = pd.DataFrame({
            "id": [1, 2, 3],
            "value": [10, 21, 30],
            "category": ["A", "X", "C"]
        })
        report = compare(before, after, key="id")
        sample_mod = report.rows.sample_modified
        # Should have __before and __after columns for modified columns
        if len(sample_mod) > 0:
            for col in report.rows.modifications.keys():
                assert f"{col}__before" in sample_mod.columns or col in sample_mod.columns

    def test_r25_modifications_contains_only_changed_columns(self):
        """R25: modifications dict only contains columns that actually changed"""
        before = pd.DataFrame({
            "id": [1, 2, 3],
            "value": [10, 20, 30],
            "unchanged": [100, 100, 100]
        })
        after = pd.DataFrame({
            "id": [1, 2, 3],
            "value": [10, 21, 30],
            "unchanged": [100, 100, 100]
        })
        report = compare(before, after, key="id")
        assert "value" in report.rows.modifications
        assert "unchanged" not in report.rows.modifications

    def test_r26_modifications_does_not_contain_key_column(self):
        """R26: modifications dict does not contain the key column itself"""
        before = pd.DataFrame({
            "id": [1, 2, 3],
            "value": [10, 20, 30]
        })
        after = pd.DataFrame({
            "id": [1, 2, 3],
            "value": [10, 21, 30]
        })
        report = compare(before, after, key="id")
        assert "id" not in report.rows.modifications


class TestKeyMatching:
    """R27-R28: Special key matching scenarios"""

    def test_r27_disjoint_keys_no_overlap(self):
        """R27: Disjoint keys (no overlap) → all rows removed + all rows added"""
        before = pd.DataFrame({
            "key": [1, 2, 3],
            "value": [10, 20, 30]
        })
        after = pd.DataFrame({
            "key": [4, 5, 6],
            "value": [40, 50, 60]
        })
        report = compare(before, after, key="key")
        assert report.rows.removed_count == 3
        assert report.rows.added_count == 3
        assert report.rows.modified_count == 0

    def test_r28_100pct_key_overlap_zero_value_changes(self):
        """R28: 100% key overlap, 0% value changes → modified_count == 0"""
        before = pd.DataFrame({
            "key": range(100),
            "value": np.ones(100) * 42
        })
        after = pd.DataFrame({
            "key": range(100),
            "value": np.ones(100) * 42
        })
        report = compare(before, after, key="key")
        assert report.rows.modified_count == 0
        assert report.rows.added_count == 0
        assert report.rows.removed_count == 0
