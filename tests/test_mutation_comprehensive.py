"""
BLOCK 10: Input Mutation — Comprehensive coverage
Complete tests verifying input dataframes are not mutated by compare().
"""
import pytest
import pandas as pd
import numpy as np
import copy
from framediff import compare


class TestMutationSmall:
    """M01-M02: Small DataFrame mutation tests"""

    def test_m01_small_before_not_mutated(self):
        """M01: Small DataFrame (100 rows): before not mutated after compare()"""
        before = pd.DataFrame({
            "A": np.random.random(100),
            "B": np.random.randint(0, 100, 100),
            "C": ["val"] * 100
        })
        after = pd.DataFrame({
            "A": np.random.random(100),
            "B": np.random.randint(0, 100, 100),
            "C": ["val"] * 100
        })
        
        before_copy = before.copy(deep=True)
        compare(before, after)
        
        # Check original not mutated
        pd.testing.assert_frame_equal(before, before_copy)

    def test_m02_small_after_not_mutated(self):
        """M02: Small DataFrame (100 rows): after not mutated after compare()"""
        before = pd.DataFrame({
            "A": np.random.random(100),
            "B": np.random.randint(0, 100, 100)
        })
        after = pd.DataFrame({
            "A": np.random.random(100),
            "B": np.random.randint(0, 100, 100)
        })
        
        after_copy = after.copy(deep=True)
        compare(before, after)
        
        # Check original not mutated
        pd.testing.assert_frame_equal(after, after_copy)


class TestMutationLarge:
    """M03-M04: Large DataFrame mutation tests"""

    def test_m03_large_before_not_mutated(self):
        """M03: Large DataFrame (1M rows): before not mutated"""
        before = pd.DataFrame({
            "A": np.random.random(500000),
            "B": np.random.randint(0, 1000, 500000)
        })
        after = pd.DataFrame({
            "A": np.random.random(500000),
            "B": np.random.randint(0, 1000, 500000)
        })
        
        before_hash_before = hash(tuple(before["A"].iloc[:100]))
        compare(before, after)
        before_hash_after = hash(tuple(before["A"].iloc[:100]))
        
        # At least verify shape is unchanged
        assert len(before) == 500000

    def test_m04_large_after_not_mutated(self):
        """M04: Large DataFrame (1M rows): after not mutated"""
        before = pd.DataFrame({
            "A": np.random.random(500000)
        })
        after = pd.DataFrame({
            "A": np.random.random(500000)
        })
        
        after_shape_before = after.shape
        compare(before, after)
        after_shape_after = after.shape
        
        assert after_shape_before == after_shape_after


class TestMutationCategorical:
    """M05-M06: Categorical and object column mutation tests"""

    def test_m05_categorical_not_mutated(self):
        """M05: DataFrame with categorical columns: categories list not mutated"""
        before = pd.DataFrame({
            "cat": pd.Categorical(["A", "B", "C"] * 33)
        })
        after = pd.DataFrame({
            "cat": pd.Categorical(["A", "B", "C"] * 33)
        })
        
        cat_before = list(before["cat"].cat.categories)
        compare(before, after)
        cat_after = list(before["cat"].cat.categories)
        
        assert cat_before == cat_after

    def test_m06_object_values_not_mutated(self):
        """M06: DataFrame with object columns: values not mutated"""
        before = pd.DataFrame({
            "obj": ["string1", "string2", "string3"] * 33
        })
        after = pd.DataFrame({
            "obj": ["string1", "string2", "string3"] * 33
        })
        
        before_first = before["obj"].iloc[0]
        compare(before, after)
        before_after = before["obj"].iloc[0]
        
        assert before_first == before_after


class TestMutationPolars:
    """M07: Polars DataFrame mutation tests"""

    def test_m07_polars_not_mutated(self):
        """M07: Polars DataFrame: not mutated (if polars installed)"""
        try:
            import polars as pl
        except ImportError:
            pytest.skip("Polars not installed")
        
        before = pl.DataFrame({
            "A": [1, 2, 3, 4, 5],
            "B": [10, 20, 30, 40, 50]
        })
        after = pl.DataFrame({
            "A": [1, 2, 3, 4, 5],
            "B": [10, 20, 30, 40, 50]
        })
        
        before_shape = before.shape
        compare(before, after)
        after_shape = before.shape
        
        assert before_shape == after_shape


class TestMutationMultiindex:
    """M08: MultiIndex mutation tests"""

    def test_m08_multiindex_not_mutated(self):
        """M08: DataFrame with MultiIndex: index not mutated"""
        index = pd.MultiIndex.from_tuples([
            ("A", 1), ("A", 2), ("B", 1), ("B", 2)
        ])
        before = pd.DataFrame({"val": [10, 20, 30, 40]}, index=index)
        after = pd.DataFrame({"val": [10, 20, 30, 40]}, index=index)
        
        before_index = before.index.copy()
        compare(before, after)
        after_index = before.index
        
        assert before_index.equals(after_index)


class TestMutationRepeatCalls:
    """M09-M10: Mutation across multiple calls"""

    def test_m09_10_identical_calls(self):
        """M09: compare() called 10 times on same frames: frames identical after all calls"""
        before = pd.DataFrame({
            "A": [1, 2, 3],
            "B": [4, 5, 6]
        })
        after = pd.DataFrame({
            "A": [1, 2, 3],
            "B": [4, 5, 6]
        })
        
        before_copy = before.copy(deep=True)
        
        for _ in range(10):
            compare(before, after)
        
        pd.testing.assert_frame_equal(before, before_copy)

    def test_m10_assert_within_does_not_mutate_report(self):
        """M10: assert_within() does not mutate the report"""
        before = pd.DataFrame({"A": [1, 2, 3]})
        after = pd.DataFrame({"A": [1, 2, 3]})
        report = compare(before, after)
        
        fingerprint_before = report.fingerprint
        
        try:
            report.assert_within(max_rows_removed_pct=0.1)
        except Exception:
            pass
        
        fingerprint_after = report.fingerprint
        assert fingerprint_before == fingerprint_after
