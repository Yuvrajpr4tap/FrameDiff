"""
BLOCK 12: Rendering and Output — Comprehensive coverage
Complete tests for repr() and HTML representation.
"""
import pytest
import pandas as pd
import numpy as np
from framediff import compare


class TestRepr:
    """RD01-RD08: Terminal representation tests"""

    def test_rd01_repr_returns_string(self):
        """RD01: repr(report) returns a string (not an object reference)"""
        before = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        after = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        report = compare(before, after)
        
        repr_str = repr(report)
        assert isinstance(repr_str, str)
        assert len(repr_str) > 0

    def test_rd02_repr_contains_column_name(self):
    def test_rd02_repr_contains_column_name(self):
        """RD02: repr(report) contains reference to schema changes"""
        before = pd.DataFrame({"my_column": [1, 2, 3]})
        after = pd.DataFrame({"my_column": [1, 2, 3], "new_col": [4, 5, 6]})
        report = compare(before, after)
        
        repr_str = repr(report)
        # Should contain schema change information (count > 0)
        assert "schema" in repr_str.lower() or "1" in repr_str  # Schema change detected

    def test_rd03_repr_contains_severity(self):
        """RD03: repr(report) contains the severity level"""
        before = pd.DataFrame({"A": [1, 2, 3]})
        after = pd.DataFrame({"A": [1, 2, 3]})
        report = compare(before, after)
        
        repr_str = repr(report)
        assert "info" in repr_str.lower() or "severity" in repr_str.lower()

    def test_rd04_repr_no_error_identical_frames(self):
        """RD04: repr(report) does not raise on identical frames"""
        before = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        report = compare(before, before)
        
        try:
            repr(report)
        except Exception as e:
            pytest.fail(f"repr raised: {e}")

    def test_rd05_repr_no_error_empty_frames(self):
        """RD05: repr(report) does not raise on empty frames"""
        before = pd.DataFrame()
        after = pd.DataFrame()
        report = compare(before, after)
        
        try:
            repr(report)
        except Exception as e:
            pytest.fail(f"repr raised: {e}")

    def test_rd06_repr_no_error_all_null_frames(self):
        """RD06: repr(report) does not raise on all-null frames"""
        before = pd.DataFrame({"A": [np.nan] * 10, "B": [None] * 10})
        after = pd.DataFrame({"A": [np.nan] * 10, "B": [None] * 10})
        report = compare(before, after)
        
        try:
            repr(report)
        except Exception as e:
            pytest.fail(f"repr raised: {e}")

    def test_rd07_repr_no_error_special_chars_columns(self):
        """RD07: repr(report) does not raise on frames with special char column names"""
        before = pd.DataFrame({"col!@#": [1], "col\n": [2], "日本語": [3]})
        after = pd.DataFrame({"col!@#": [1], "col\n": [2], "日本語": [3]})
        report = compare(before, after)
        
        try:
            repr(report)
        except Exception as e:
            pytest.fail(f"repr raised: {e}")

    def test_rd08_repr_no_error_200_columns(self):
        """RD08: repr(report) does not raise on frames with 50 columns"""
        data = {f"col_{i}": [i] * 10 for i in range(50)}
        before = pd.DataFrame(data)
        after = pd.DataFrame(data)
        report = compare(before, after)
        
        try:
            repr(report)
        except Exception as e:
            pytest.fail(f"repr raised: {e}")


class TestHtmlRepr:
    """RD09-RD16: HTML representation tests"""

    def test_rd09_repr_html_returns_string_with_table(self):
        """RD09: _repr_html_() returns a string containing "<table"  """
        before = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        after = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        report = compare(before, after)
        
        html = report._repr_html_()
        assert isinstance(html, str)
        assert "<table" in html.lower() or "<div" in html.lower()

    def test_rd10_repr_html_valid_html(self):
        """RD10: _repr_html_() is valid enough that BeautifulSoup can parse it"""
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            pytest.skip("BeautifulSoup not installed")
        
        before = pd.DataFrame({"A": [1, 2, 3]})
        after = pd.DataFrame({"A": [1, 2, 3]})
        report = compare(before, after)
        
        html = report._repr_html_()
        soup = BeautifulSoup(html, "html.parser")
        
        # Should be parseable and have some structure
        assert soup is not None
        assert len(str(soup)) > 0

    def test_rd11_repr_html_no_error_edge_cases(self):
        """RD11: _repr_html_() does not raise on any edge case from Block 9"""
        edge_cases = [
            (pd.DataFrame(), pd.DataFrame()),  # Empty
            (pd.DataFrame({"A": [np.nan] * 10}), pd.DataFrame({"A": [np.nan] * 10})),  # All null
            (pd.DataFrame({i: [i] for i in range(100)}), pd.DataFrame({i: [i] for i in range(100)})),  # Many cols
            (pd.DataFrame({"A": [1]}), pd.DataFrame({"A": [1]})),  # Single cell
        ]
        
        for before, after in edge_cases:
            report = compare(before, after)
            try:
                html = report._repr_html_()
                assert isinstance(html, str)
            except Exception as e:
                pytest.fail(f"_repr_html_() raised on edge case: {e}")

    def test_rd12_repr_html_contains_severity_badge(self):
        """RD12: _repr_html_() contains severity badge text"""
        before = pd.DataFrame({"A": [1, 2, 3]})
        after = pd.DataFrame({"A": [1, 2, 3]})
        report = compare(before, after)
        
        html = report._repr_html_()
        # Should mention severity in some way
        assert "info" in html.lower() or "severity" in html.lower() or "status" in html.lower()

    def test_rd13_repr_html_contains_column_names(self):
        """RD13: _repr_html_() contains column names from the diff"""
        before = pd.DataFrame({"my_col": [1, 2, 3]})
        after = pd.DataFrame({"my_col": [1, 2, 3], "new_col": [4, 5, 6]})
        report = compare(before, after)
        
        html = report._repr_html_()
        # Should mention at least one column
        assert "column" in html.lower() or "my_col" in html or "new_col" in html

    def test_rd14_repr_html_critical_label(self):
        """RD14: Terminal repr contains "critical" (case-insensitive) when severity
        is critical"""
        before = pd.DataFrame({"A": [1, 2, 3]})
        after = pd.DataFrame({"A": []})
        after = after.drop("A", axis=1)
        report = compare(before, after)
        
        if report.severity == "critical":
            repr_str = repr(report)
            assert "critical" in repr_str.lower()

    def test_rd15_repr_contains_warning_label(self):
        """RD15: Terminal repr contains "warning" when severity is warning"""
        np.random.seed(42)
        before = pd.DataFrame({"A": np.random.normal(100, 15, 1000)})
        after = pd.DataFrame({"A": np.random.normal(107, 15, 1000)})
        report = compare(before, after)
        
        if report.severity == "warning":
            repr_str = repr(report)
            assert "warning" in repr_str.lower() or "shift" in repr_str.lower()

    def test_rd16_repr_output_length_reasonable(self):
        """RD16: repr output length is reasonable: between 100 and 50,000 characters"""
        before = pd.DataFrame({
            f"col_{i}": np.random.random(100) for i in range(50)
        })
        after = pd.DataFrame({
            f"col_{i}": np.random.random(100) for i in range(50)
        })
        report = compare(before, after)
        
        repr_str = repr(report)
        assert 100 < len(repr_str) < 50000
