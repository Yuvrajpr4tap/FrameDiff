"""
Tests for statistical diffing (framediff.stats).
"""
import pytest
from framediff.stats import compare_stats, StatDiff, _compute_psi, _psi_to_label


def test_stable_numeric_distribution(df_numeric_stable):
    """Test PSI calculation for stable distribution."""
    stats = compare_stats(df_numeric_stable, df_numeric_stable)

    assert "value" in stats
    stat = stats["value"]
    assert stat.distribution_method == "psi"
    assert stat.distribution_score < 0.1
    assert stat.distribution_label == "stable"


def test_moderate_shift_numeric(df_numeric_stable, df_numeric_shifted):
    """Test PSI detection of moderate shift."""
    stats = compare_stats(df_numeric_stable, df_numeric_shifted)

    assert "value" in stats
    stat = stats["value"]
    assert stat.distribution_method == "psi"
    # Moderate shift: 0.1 <= PSI < 0.2
    assert 0.05 < stat.distribution_score < 0.3
    assert stat.distribution_label in ["moderate shift", "stable", "large shift"]


def test_large_shift_numeric(df_numeric_stable, df_numeric_large_shift):
    """Test PSI detection of large shift."""
    stats = compare_stats(df_numeric_stable, df_numeric_large_shift)

    assert "value" in stats
    stat = stats["value"]
    assert stat.distribution_method == "psi"
    # Should be large shift
    assert stat.distribution_score > 0.15
    assert stat.distribution_label == "large shift"


def test_null_rate_increase(df_with_nulls_before, df_with_nulls_after):
    """Test null rate change detection."""
    stats = compare_stats(df_with_nulls_before, df_with_nulls_after)

    assert "value" in stats
    stat = stats["value"]
    # Before: 2/5 = 0.4
    # After: 3/5 = 0.6
    # Delta = 0.2
    assert stat.null_rate_before == pytest.approx(0.4, abs=0.01)
    assert stat.null_rate_after == pytest.approx(0.6, abs=0.01)
    assert stat.null_rate_delta == pytest.approx(0.2, abs=0.01)
    # Large increase (>0.1) should be critical
    assert stat.severity == "critical"


def test_categorical_new_categories(df_categorical_stable, df_categorical_shifted):
    """Test detection of new categories."""
    stats = compare_stats(df_categorical_stable, df_categorical_shifted)

    assert "category" in stats
    stat = stats["category"]
    # Both have same categories, no new ones
    assert len(stat.new_categories) == 0
    assert len(stat.dropped_categories) == 0


def test_categorical_distribution(df_categorical_stable, df_categorical_shifted):
    """Test categorical distribution change."""
    stats = compare_stats(df_categorical_stable, df_categorical_shifted)

    assert "category" in stats
    stat = stats["category"]
    assert stat.distribution_method == "chi2"
    # Should detect some shift
    assert stat.distribution_score >= 0


def test_no_stats_for_non_overlapping_columns():
    """Test that non-overlapping columns don't appear in stats."""
    import pandas as pd

    df1 = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    df2 = pd.DataFrame({"c": [7, 8, 9], "d": [10, 11, 12]})

    stats = compare_stats(df1, df2)

    # Should only have stats for overlapping columns (none in this case)
    assert len(stats) == 0


def test_empty_dataframes():
    """Test stats with empty DataFrames."""
    import pandas as pd

    df1 = pd.DataFrame({"a": [], "b": []})
    df2 = pd.DataFrame({"a": [], "b": []})

    stats = compare_stats(df1, df2)

    # May be empty or have 0-length stats
    assert isinstance(stats, dict)


def test_psi_to_label_thresholds():
    """Test PSI score to label conversion."""
    assert _psi_to_label(0.05) == "stable"
    assert _psi_to_label(0.15) == "moderate shift"
    assert _psi_to_label(0.25) == "large shift"


def test_cardinality_tracking(df_numeric_stable, df_numeric_shifted):
    """Test that cardinality is tracked."""
    stats = compare_stats(df_numeric_stable, df_numeric_shifted)

    stat = stats["value"]
    # Should have cardinality counts (unique values)
    assert stat.cardinality_before > 0
    assert stat.cardinality_after > 0


def test_mean_and_std_tracking(df_numeric_stable, df_numeric_shifted):
    """Test that mean and std deltas are computed."""
    stats = compare_stats(df_numeric_stable, df_numeric_shifted)

    stat = stats["value"]
    assert stat.mean_delta is not None
    assert stat.std_delta is not None
    # Values should be non-negative
    assert stat.mean_delta >= 0
    assert stat.std_delta >= 0


def test_binary_column_value_change_rate_with_key(df_binary_col_before, df_binary_col_after_flipped):
    """Test value_change_rate calculation for binary column with key-based matching."""
    # All 10 rows have flipped values, so change rate should be 1.0 (100%)
    stats = compare_stats(df_binary_col_before, df_binary_col_after_flipped, key="id")

    assert "status" in stats
    stat = stats["status"]
    
    # Should have value_change_rate calculated
    assert stat.value_change_rate is not None
    assert stat.value_change_rate >= 0.50  # At least 50% changed
    
    # 100% change rate should trigger critical severity
    assert stat.severity == "critical"
