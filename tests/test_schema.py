"""
Tests for schema diffing (framediff.schema).
"""
import pytest
from framediff.schema import compare_schemas, SchemaDiff


def test_no_schema_changes(df_simple_before):
    """Test when schema is identical."""
    diff = compare_schemas(df_simple_before, df_simple_before)

    assert len(diff.added_columns) == 0
    assert len(diff.removed_columns) == 0
    assert len(diff.type_changes) == 0


def test_added_column(df_simple_before, df_simple_after):
    """Test detection of added columns."""
    diff = compare_schemas(df_simple_before, df_simple_after)

    assert "new_col" in diff.added_columns
    # Severity of added column should be info
    added_issues = [i for i in diff.issues if i.severity == "info"]
    assert any("new_col" in i.message for i in added_issues)


def test_removed_column(df_simple_before, df_simple_after):
    """Test detection of removed columns."""
    # Create a case where column is removed
    df_before = df_simple_before.copy()
    df_after = df_simple_before[["id", "name", "value"]].copy()

    diff = compare_schemas(df_before, df_after)

    assert "category" in diff.removed_columns
    # Severity of removed column should be critical
    critical_issues = [i for i in diff.issues if i.severity == "critical"]
    assert any("category" in i.message for i in critical_issues)


def test_type_change(df_type_change_before, df_type_change_after):
    """Test detection of type changes."""
    diff = compare_schemas(df_type_change_before, df_type_change_after)

    assert "amount" in diff.type_changes
    before_dtype, after_dtype = diff.type_changes["amount"]
    assert "float" in before_dtype.lower()
    assert "int" in after_dtype.lower()

    # Lossy change should be warning
    warning_issues = [i for i in diff.issues if i.severity == "warning"]
    assert any("amount" in i.message for i in warning_issues)


def test_multiple_column_changes(df_simple_before, df_simple_after):
    """Test multiple schema changes at once."""
    diff = compare_schemas(df_simple_before, df_simple_after)

    # Should detect the new_col addition
    assert "new_col" in diff.added_columns
    # Category column exists in both, so not in added/removed
    assert "category" not in diff.added_columns
    assert "category" not in diff.removed_columns


def test_empty_dataframes(df_empty):
    """Test schema diff of empty DataFrames."""
    diff = compare_schemas(df_empty, df_empty)

    assert len(diff.added_columns) == 0
    assert len(diff.removed_columns) == 0
    assert len(diff.type_changes) == 0


def test_single_row_schema(df_single_row):
    """Test schema diff with single row."""
    diff = compare_schemas(df_single_row, df_single_row)

    assert len(diff.added_columns) == 0
    assert len(diff.removed_columns) == 0
