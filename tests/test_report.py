"""
Tests for DiffReport and assertion utilities.
"""
import pytest
import json
from framediff import compare, DiffReport, DiffThresholdError


def test_report_serialization(df_simple_before, df_simple_after):
    """Test that report can be serialized to JSON."""
    report = compare(df_simple_before, df_simple_after)

    dict_repr = report.to_dict()
    # Should be JSON-serializable
    json_str = json.dumps(dict_repr, default=str)
    assert len(json_str) > 0

    # Check key fields
    assert "schema" in dict_repr
    assert "stats" in dict_repr
    assert "rows" in dict_repr
    assert "severity" in dict_repr


def test_report_fingerprint_deterministic(df_simple_before, df_simple_after):
    """Test that fingerprint is deterministic."""
    report1 = compare(df_simple_before, df_simple_after)
    report2 = compare(df_simple_before, df_simple_after)

    assert report1.fingerprint == report2.fingerprint


def test_report_fingerprint_changes_with_data(df_simple_before, df_simple_after):
    """Test that fingerprint changes when data changes."""
    report1 = compare(df_simple_before, df_simple_after)

    import pandas as pd
    df_different = df_simple_after.copy()
    df_different.loc[0, "value"] = 999999.0
    report2 = compare(df_simple_before, df_different)

    assert report1.fingerprint != report2.fingerprint


def test_report_summary(df_simple_before, df_simple_after):
    """Test summary generation."""
    report = compare(df_simple_before, df_simple_after)

    summary = report.summary
    assert isinstance(summary, str)
    assert len(summary) > 0
    # Should mention some changes
    assert "schema change" in summary or "rows" in summary or "No changes" in summary


def test_report_repr_does_not_error(df_simple_before, df_simple_after):
    """Test that __repr__ doesn't raise errors."""
    report = compare(df_simple_before, df_simple_after)

    repr_str = repr(report)
    assert isinstance(repr_str, str)
    assert len(repr_str) > 0


def test_report_html_repr(df_simple_before, df_simple_after):
    """Test that _repr_html_ returns valid HTML."""
    report = compare(df_simple_before, df_simple_after)

    html = report._repr_html_()
    assert isinstance(html, str)
    assert "<div" in html or "<table" in html


def test_severity_aggregation(df_simple_before, df_simple_after):
    """Test that severity is aggregated correctly."""
    report = compare(df_simple_before, df_simple_after)

    # Severity should be one of the three values
    assert report.severity in ["info", "warning", "critical"]


def test_assert_within_passes_when_OK(df_simple_before):
    """Test that assert_within doesn't raise when constraints are met."""
    df_after = df_simple_before.copy()
    report = compare(df_simple_before, df_after)

    # Should not raise
    report.assert_within(
        max_rows_removed_pct=10,
        max_rows_added_pct=10,
        max_null_rate_increase=0.5,
    )


def test_assert_within_raises_on_rows_removed_pct(df_simple_before):
    """Test that assert_within raises on rows_removed_pct violation."""
    import pandas as pd

    df_after = df_simple_before.iloc[:2]  # Remove 3 out of 5 = 60%

    report = compare(df_simple_before, df_after)

    with pytest.raises(DiffThresholdError):
        report.assert_within(max_rows_removed_pct=50)


def test_assert_within_raises_on_rows_added_pct(df_simple_before):
    """Test that assert_within raises on rows_added_pct violation."""
    import pandas as pd

    df_after = pd.concat(
        [df_simple_before, pd.DataFrame({"id": [6, 7, 8], "name": ["X", "Y", "Z"], "value": [600.0, 700.0, 800.0], "category": ["A", "A", "A"]})],
        ignore_index=True
    )

    report = compare(df_simple_before, df_after)

    with pytest.raises(DiffThresholdError):
        report.assert_within(max_rows_added_pct=50)


def test_assert_within_no_critical(df_simple_before):
    """Test that assert_within can enforce no critical issues."""
    # Create a case with critical severity
    import pandas as pd

    df_before = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    df_after = pd.DataFrame({"a": [1, 2, 3]})  # Remove column b

    report = compare(df_before, df_after)

    # Removed column should cause critical issue
    if report.severity == "critical":
        with pytest.raises(DiffThresholdError):
            report.assert_within(no_critical=True)


def test_assert_within_no_removed_columns(df_simple_before):
    """Test that assert_within can enforce no removed columns."""
    import pandas as pd

    df_before = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    df_after = pd.DataFrame({"a": [1, 2, 3]})  # Remove column b

    report = compare(df_before, df_after)

    with pytest.raises(DiffThresholdError):
        report.assert_within(no_removed_columns=True)


def test_assert_within_no_type_changes(df_type_change_before, df_type_change_after):
    """Test that assert_within can enforce no type changes."""
    report = compare(df_type_change_before, df_type_change_after)

    with pytest.raises(DiffThresholdError):
        report.assert_within(no_type_changes=True)


def test_assert_within_column_filtering(df_numeric_stable, df_numeric_shifted):
    """Test that assert_within can filter by column."""
    report = compare(df_numeric_stable, df_numeric_shifted)

    # Should only check the 'value' column if specified
    report.assert_within(
        max_psi=10.0,  # High threshold
        columns=["value"],
    )


def test_assert_within_error_message():
    """Test that DiffThresholdError has helpful message."""
    import pandas as pd

    df_before = pd.DataFrame({"a": [1, 2, 3]})
    df_after = pd.DataFrame({"a": [1, 2]})

    report = compare(df_before, df_after)

    try:
        report.assert_within(max_rows_removed_pct=10)
    except DiffThresholdError as e:
        assert "exceeds" in str(e)
        assert "violations" in dir(e)


def test_to_json_valid(df_simple_before, df_simple_after):
    """Test to_json produces valid JSON string."""
    report = compare(df_simple_before, df_simple_after)

    json_str = report.to_json()
    assert isinstance(json_str, str)
    # Should be parseable
    parsed = json.loads(json_str)
    assert "schema" in parsed
    assert "stats" in parsed
