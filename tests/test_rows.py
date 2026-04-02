"""
Tests for row-level diffing (framediff.rows).
"""
import pytest
import pandas as pd
from framediff.rows import compare_rows, RowDiff


def test_no_changes_positional(df_simple_before):
    """Test positional matching with no changes."""
    diff = compare_rows(df_simple_before, df_simple_before, key=None)

    assert diff.added_count == 0
    assert diff.removed_count == 0
    assert diff.modified_count == 0


def test_rows_added_positional(df_simple_before):
    """Test detection of added rows (positional)."""
    df_after = pd.concat(
        [df_simple_before, pd.DataFrame({"id": [6], "name": ["Frank"], "value": [600.0], "category": ["A"]})], 
        ignore_index=True
    )

    diff = compare_rows(df_simple_before, df_after, key=None)

    assert diff.added_count == 1
    assert diff.removed_count == 0
    assert diff.added_pct == pytest.approx(100.0 / 5, rel=0.01)


def test_rows_removed_positional(df_simple_before):
    """Test detection of removed rows (positional)."""
    df_after = df_simple_before.iloc[:-1]

    diff = compare_rows(df_simple_before, df_after, key=None)

    assert diff.added_count == 0
    assert diff.removed_count == 1
    assert diff.removed_pct == pytest.approx(100.0 / 5, rel=0.01)


def test_rows_modified_positional(df_simple_before):
    """Test detection of modified rows (positional)."""
    df_after = df_simple_before.copy()
    df_after.loc[0, "value"] = 999.0

    diff = compare_rows(df_simple_before, df_after, key=None)

    assert diff.modified_count >= 1
    assert not diff.sample_modified.empty


def test_composite_key_matching():
    """Test matching with composite (multi-column) key."""
    df_before = pd.DataFrame({
        "id": [1, 1, 2, 2],
        "subid": ["a", "b", "a", "b"],
        "value": [10, 20, 30, 40],
    })
    df_after = pd.DataFrame({
        "id": [1, 1, 2, 2],
        "subid": ["a", "b", "a", "b"],
        "value": [10, 21, 30, 40],
    })

    diff = compare_rows(df_before, df_after, key=["id", "subid"])

    # Second row changed (20 -> 21)
    assert diff.modified_count == 1
    assert diff.added_count == 0
    assert diff.removed_count == 0


def test_single_key_matching():
    """Test matching with single key column."""
    df_before = pd.DataFrame({
        "id": [1, 2, 3],
        "name": ["A", "B", "C"],
        "value": [10, 20, 30],
    })
    df_after = pd.DataFrame({
        "id": [1, 2, 4],  # 3 removed, 4 added
        "name": ["A", "B", "D"],
        "value": [10, 20, 40],
    })

    diff = compare_rows(df_before, df_after, key="id")

    assert diff.added_count == 1  # id=4
    assert diff.removed_count == 1  # id=3


def test_empty_frames():
    """Test row diff with empty DataFrames."""
    df1 = pd.DataFrame({"a": [], "b": []})
    df2 = pd.DataFrame({"a": [], "b": []})

    diff = compare_rows(df1, df2)

    assert diff.added_count == 0
    assert diff.removed_count == 0


def test_single_row_frames(df_single_row):
    """Test row diff with single-row DataFrames."""
    diff = compare_rows(df_single_row, df_single_row)

    assert diff.total_before == 1
    assert diff.total_after == 1
    assert diff.added_count == 0
    assert diff.removed_count == 0


def test_sample_sizes():
    """Test that sample sizes are limited (up to 10)."""
    df_before = pd.DataFrame({"id": range(100), "value": range(100)})
    df_after = pd.DataFrame({"id": range(100, 200), "value": range(100, 200)})

    diff = compare_rows(df_before, df_after, key="id")

    # All rows different (different ids)
    assert diff.added_count == 100
    assert diff.removed_count == 100
    # But sample should be capped at 10
    assert len(diff.sample_added) <= 10
    assert len(diff.sample_removed) <= 10


def test_invalid_key():
    """Test error when key column doesn't exist."""
    df1 = pd.DataFrame({"a": [1, 2, 3]})
    df2 = pd.DataFrame({"a": [1, 2, 3]})

    with pytest.raises(ValueError):
        compare_rows(df1, df2, key="nonexistent")


def test_percentage_calculations(df_simple_before):
    """Test that percentages are calculated correctly."""
    # df_simple_before has 5 rows. We take the first 3 and add 1 new, so df_after has 4 rows.
    # Positional comparison: rows 0-2 match, row 3 changed, row 4 removed = 1 removed
    df_after = pd.concat(
        [df_simple_before.iloc[:3], 
         pd.DataFrame([{"id": 6, "name": "Frank", "value": 600.0, "category": "A"}])],
        ignore_index=True
    )

    diff = compare_rows(df_simple_before, df_after, key=None)

    # Removed 2 rows out of 5 = 40%
    # After has 4 rows total: rows 0-2 from before (match) + 1 new row
    # So row 3 from before (id=4) is modified (doesn't match new row id=6)
    # and row 4 from before (id=5) is removed (no match in after)
    # But since we're comparing positionally, row 3 is modified (different), row 4 is removed
    # Actually: removed = total_before - len(df_after) if len(df_after) < len(df_simple_before)
    # 5 - 4 = 1 removed
    assert diff.removed_count == 1
    assert diff.removed_pct == pytest.approx(20.0, rel=0.01)  # 1/5 = 20%
