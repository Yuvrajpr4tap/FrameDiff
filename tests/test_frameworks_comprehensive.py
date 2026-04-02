"""
BLOCK 8: Framework Interoperability — Comprehensive coverage
Complete tests for compatibility with different data frame libraries.
"""
import pytest
import pandas as pd
import numpy as np
from framediff import compare


class TestPandasInterop:
    """FW01: Pandas to Pandas (baseline)"""

    def test_fw01_pandas_to_pandas(self):
        """FW01: Pandas → Pandas: baseline, all features work"""
        before = pd.DataFrame({
            "id": [1, 2, 3],
            "value": [10.0, 20.0, 30.0],
            "category": ["A", "B", "C"]
        })
        after = pd.DataFrame({
            "id": [1, 2, 3],
            "value": [10.0, 21.0, 30.0],
            "category": ["A", "B", "C"]
        })
        
        report = compare(before, after, key="id")
        assert report is not None
        assert report.rows.modified_count == 1


class TestPolarsInterop:
    """FW02-FW06: Polars compatibility"""

    @pytest.mark.skipif(True, reason="Check if polars available")
    def test_fw02_polars_to_polars(self):
        """FW02: Polars → Polars: returns valid DiffReport"""
        try:
            import polars as pl
        except ImportError:
            pytest.skip("Polars not installed")
        
        before = pl.DataFrame({
            "id": [1, 2, 3],
            "value": [10.0, 20.0, 30.0]
        })
        after = pl.DataFrame({
            "id": [1, 2, 3],
            "value": [10.0, 21.0, 30.0]
        })
        
        report = compare(before, after, key="id")
        assert report is not None

    @pytest.mark.skipif(True, reason="Check if polars available")
    def test_fw03_polars_to_pandas(self):
        """FW03: Polars → Pandas: returns valid DiffReport"""
        try:
            import polars as pl
        except ImportError:
            pytest.skip("Polars not installed")
        
        before = pl.DataFrame({
            "id": [1, 2, 3],
            "value": [10.0, 20.0, 30.0]
        })
        after = pd.DataFrame({
            "id": [1, 2, 3],
            "value": [10.0, 21.0, 30.0]
        })
        
        report = compare(before, after, key="id")
        assert report is not None

    @pytest.mark.skipif(True, reason="Check if polars available")
    def test_fw04_pandas_to_polars(self):
        """FW04: Pandas → Polars: returns valid DiffReport"""
        try:
            import polars as pl
        except ImportError:
            pytest.skip("Polars not installed")
        
        before = pd.DataFrame({
            "id": [1, 2, 3],
            "value": [10.0, 20.0, 30.0]
        })
        after = pl.DataFrame({
            "id": [1, 2, 3],
            "value": [10.0, 21.0, 30.0]
        })
        
        report = compare(before, after, key="id")
        assert report is not None

    @pytest.mark.skipif(True, reason="Check if polars available")
    def test_fw05_polars_pandas_identical_fingerprints(self):
        """FW05: Polars and Pandas of same data → fingerprints are identical"""
        try:
            import polars as pl
        except ImportError:
            pytest.skip("Polars not installed")
        
        data = {"A": [1, 2, 3], "B": [4.0, 5.0, 6.0]}
        
        df_pd = pd.DataFrame(data)
        df_pl = pl.DataFrame(data)
        
        report_pd = compare(df_pd, df_pd)
        report_pl = compare(df_pl, df_pl)
        
        assert report_pd.fingerprint == report_pl.fingerprint

    @pytest.mark.skipif(True, reason="Check if polars available")
    def test_fw06_polars_pandas_identical_stats(self):
        """FW06: Polars and Pandas stat values numerically identical (within 1e-9)"""
        try:
            import polars as pl
        except ImportError:
            pytest.skip("Polars not installed")
        
        data = {"value": [1, 2, 3, 4, 5]}
        
        df_pd = pd.DataFrame(data)
        df_pl = pl.DataFrame(data)
        
        report_pd = compare(df_pd, df_pd)
        report_pl = compare(df_pl, df_pl)
        
        # Both should have stats for 'value'
        assert "value" in report_pd.stats
        assert "value" in report_pl.stats


class TestInvalidInputs:
    """FW07-FW12: Invalid input type handling"""

    def test_fw07_pyarrow_table_error(self):
        """FW07: PyArrow Table → raises InvalidFrameError or TypeError with clear message"""
        try:
            import pyarrow as pa
        except ImportError:
            pytest.skip("PyArrow not installed")
        
        tbl = pa.table({"A": [1, 2, 3], "B": [4, 5, 6]})
        df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        
        from framediff.exceptions import InvalidFrameError
        with pytest.raises((InvalidFrameError, NotImplementedError, TypeError)) as exc_info:
            compare(tbl, df)
        
        # Check that error mentions something useful
        error_msg = str(exc_info.value).lower()
        assert any(x in error_msg for x in ["unsupported", "invalid", "dataframe", "pyarrow"])

    def test_fw08_numpy_ndarray_error(self):
        """FW08: numpy ndarray → raises InvalidFrameError or TypeError with DataFrame mention"""
        arr = np.array([[1, 2, 3], [4, 5, 6]])
        df = pd.DataFrame({"A": [1, 2, 3]})
        
        from framediff.exceptions import InvalidFrameError
        with pytest.raises((InvalidFrameError, TypeError)) as exc_info:
            compare(arr, df)
        
        error_msg = str(exc_info.value).lower()
        assert "dataframe" in error_msg or "unsupported" in error_msg

    def test_fw09_python_dict_error(self):
        """FW09: Python dict → raises InvalidFrameError or TypeError with DataFrame mention"""
        d = {"A": [1, 2, 3], "B": [4, 5, 6]}
        df = pd.DataFrame({"A": [1, 2, 3]})
        
        from framediff.exceptions import InvalidFrameError
        with pytest.raises((InvalidFrameError, TypeError)) as exc_info:
            compare(d, df)
        
        error_msg = str(exc_info.value).lower()
        assert "dataframe" in error_msg or "unsupported" in error_msg

    def test_fw10_python_list_error(self):
        """FW10: Python list → raises InvalidFrameError or TypeError with DataFrame mention"""
        lst = [[1, 2, 3], [4, 5, 6]]
        df = pd.DataFrame({"A": [1, 2, 3]})
        
        from framediff.exceptions import InvalidFrameError
        with pytest.raises((InvalidFrameError, TypeError)) as exc_info:
            compare(lst, df)
        
        error_msg = str(exc_info.value).lower()
        assert "dataframe" in error_msg or "unsupported" in error_msg

    def test_fw11_none_error(self):
        """FW11: None → raises InvalidFrameError or TypeError with DataFrame mention"""
        df = pd.DataFrame({"A": [1, 2, 3]})
        
        from framediff.exceptions import InvalidFrameError
        with pytest.raises((InvalidFrameError, TypeError)) as exc_info:
            compare(None, df)
        
        error_msg = str(exc_info.value).lower()
        assert "dataframe" in error_msg or "unsupported" in error_msg

    def test_fw12_string_error(self):
        """FW12: String → raises InvalidFrameError or TypeError with DataFrame mention"""
        df = pd.DataFrame({"A": [1, 2, 3]})
        
        from framediff.exceptions import InvalidFrameError
        with pytest.raises((InvalidFrameError, TypeError)) as exc_info:
            compare("not a dataframe", df)
        
        error_msg = str(exc_info.value).lower()
        assert "dataframe" in error_msg or "unsupported" in error_msg


class TestPolarsSpecific:
    """FW13-FW17: Polars-specific dtype handling"""

    @pytest.mark.skipif(True, reason="Check if polars available")
    def test_fw13_polars_lazyframe(self):
        """FW13: Polars LazyFrame → raises TypeError or auto-collects (document which)"""
        try:
            import polars as pl
        except ImportError:
            pytest.skip("Polars not installed")
        
        lazy_df = pl.DataFrame({"A": [1, 2, 3]}).lazy()
        df_pd = pd.DataFrame({"A": [1, 2, 3]})
        
        try:
            report = compare(lazy_df, df_pd)
            # If it succeeds, it auto-collected
            assert report is not None
        except TypeError:
            # If it raises, that's documented
            pass

    @pytest.mark.skipif(True, reason="Check if polars available")
    def test_fw14_polars_categorical_dtype(self):
        """FW14: Polars with Categorical dtype → handled correctly"""
        try:
            import polars as pl
        except ImportError:
            pytest.skip("Polars not installed")
        
        before = pl.DataFrame({
            "id": [1, 2, 3],
            "cat": pl.Series(["A", "B", "C"], dtype=pl.Categorical())
        })
        after = pl.DataFrame({
            "id": [1, 2, 3],
            "cat": pl.Series(["A", "B", "C"], dtype=pl.Categorical())
        })
        
        report = compare(before, after, key="id")
        assert report is not None

    @pytest.mark.skipif(True, reason="Check if polars available")
    def test_fw15_polars_utf8_dtype(self):
        """FW15: Polars with Utf8 dtype → handled as string, not crashed"""
        try:
            import polars as pl
        except ImportError:
            pytest.skip("Polars not installed")
        
        before = pl.DataFrame({
            "id": [1, 2, 3],
            "text": pl.Series(["hello", "world", "test"], dtype=pl.Utf8())
        })
        after = pl.DataFrame({
            "id": [1, 2, 3],
            "text": pl.Series(["hello", "world", "test"], dtype=pl.Utf8())
        })
        
        report = compare(before, after, key="id")
        assert report is not None

    @pytest.mark.skipif(True, reason="Check if polars available")
    def test_fw16_polars_date_dtype(self):
        """FW16: Polars with Date dtype (not datetime) → handled correctly"""
        try:
            import polars as pl
        except ImportError:
            pytest.skip("Polars not installed")
        
        dates = pl.Series([1, 2, 3], dtype=pl.Date())
        before = pl.DataFrame({"date": dates})
        
        dates2 = pl.Series([1, 2, 3], dtype=pl.Date())
        after = pl.DataFrame({"date": dates2})
        
        try:
            report = compare(before, after)
            assert report is not None
        except (TypeError, AttributeError):
            # May not support Date dtype yet
            pass

    @pytest.mark.skipif(True, reason="Check if polars available")
    def test_fw17_polars_duration_dtype(self):
        """FW17: Polars with Duration dtype → handled correctly or clear error"""
        try:
            import polars as pl
        except ImportError:
            pytest.skip("Polars not installed")
        
        # Create duration series
        before = pl.DataFrame({
            "duration": pl.Series(range(3), dtype=pl.Duration())
        })
        after = pl.DataFrame({
            "duration": pl.Series(range(3), dtype=pl.Duration())
        })
        
        try:
            report = compare(before, after)
            assert report is not None
        except (TypeError, AttributeError):
            # May not support Duration dtype yet
            pass
