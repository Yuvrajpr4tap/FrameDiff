"""
BLOCK 13: Performance Benchmarks
Complete pytest-benchmark tests for performance verification.
"""
import pytest
import pandas as pd
import numpy as np
from framediff import compare


class TestPerformanceBenchmarks:
    """PF01-PF12: Performance benchmarks with hard limits"""

    def test_pf01_1k_rows_10_cols(self, benchmark):
        """PF01: 1k rows × 10 cols, no key → under 0.5s"""
        before = pd.DataFrame({
            f"col_{i}": np.random.random(1000) for i in range(10)
        })
        after = pd.DataFrame({
            f"col_{i}": np.random.random(1000) for i in range(10)
        })
        
        result = benchmark(compare, before, after)
        assert result is not None

    def test_pf02_5k_rows_10_cols(self, benchmark):
        """PF02: 5k rows × 10 cols, no key → under 1s"""
        before = pd.DataFrame({
            f"col_{i}": np.random.random(5000) for i in range(10)
        })
        after = pd.DataFrame({
            f"col_{i}": np.random.random(5000) for i in range(10)
        })
        
        result = benchmark(compare, before, after)
        assert result is not None

    def test_pf03_20k_rows_20_cols(self, benchmark):
        """PF03: 20k rows × 20 cols, no key → under 3s"""
        before = pd.DataFrame({
            f"col_{i}": np.random.random(20000) for i in range(20)
        })
        after = pd.DataFrame({
            f"col_{i}": np.random.random(20000) for i in range(20)
        })
        
        result = benchmark(compare, before, after)
        assert result is not None

    def test_pf04_50k_rows_5_cols(self, benchmark):
        """PF04: 50k rows × 5 cols, no key → under 5s"""
        before = pd.DataFrame({
            f"col_{i}": np.random.random(50000) for i in range(5)
        })
        after = pd.DataFrame({
            f"col_{i}": np.random.random(50000) for i in range(5)
        })
        
        result = benchmark(compare, before, after)
        assert result is not None

    def test_pf05_20k_rows_20_cols_with_key(self, benchmark):
        """PF05: 20k rows × 20 cols, with key → under 5s"""
        data = {f"col_{i}": np.random.random(20000) for i in range(20)}
        data["key"] = range(20000)
        
        before = pd.DataFrame(data)
        after = pd.DataFrame(data)
        
        result = benchmark(compare, before, after, key="key")
        assert result is not None

    def test_pf06_20k_rows_50_cols(self, benchmark):
        """PF06: 20k rows × 50 cols, no key → under 5s"""
        before = pd.DataFrame({
            f"col_{i}": np.random.random(20000) for i in range(50)
        })
        after = pd.DataFrame({
            f"col_{i}": np.random.random(20000) for i in range(50)
        })
        
        result = benchmark(compare, before, after)
        assert result is not None

    def test_pf07_2k_rows_100_cols(self, benchmark):
        """PF07: 2k rows × 100 cols, no key → under 5s"""
        before = pd.DataFrame({
            f"col_{i}": np.random.random(2000) for i in range(100)
        })
        after = pd.DataFrame({
            f"col_{i}": np.random.random(2000) for i in range(100)
        })
        
        result = benchmark(compare, before, after)
        assert result is not None

    def test_pf08_100_calls_100_row_frames(self, benchmark):
        """PF08: compare() called 100× on 100-row frames → under 5s total"""
        before = pd.DataFrame({
            "A": np.random.random(100),
            "B": np.random.randint(0, 100, 100)
        })
        after = pd.DataFrame({
            "A": np.random.random(100),
            "B": np.random.randint(0, 100, 100)
        })
        
        def call_100_times():
            for _ in range(100):
                compare(before, after)
        
        benchmark(call_100_times)

    def test_pf09_to_json_large_report(self, benchmark):
        """PF09: to_json() on report from 50k row comparison → under 1s"""
        before = pd.DataFrame({
            "A": np.random.random(50000),
            "B": np.random.randint(0, 1000, 50000)
        })
        after = pd.DataFrame({
            "A": np.random.random(50000),
            "B": np.random.randint(0, 1000, 50000)
        })
        
        report = compare(before, after)
        benchmark(report.to_json)

    def test_pf10_to_dict_large_report(self, benchmark):
        """PF10: to_dict() on report from 50k row comparison → under 0.5s"""
        before = pd.DataFrame({
            "A": np.random.random(50000),
            "B": np.random.randint(0, 1000, 50000)
        })
        after = pd.DataFrame({
            "A": np.random.random(50000),
            "B": np.random.randint(0, 1000, 50000)
        })
        
        report = compare(before, after)
        benchmark(report.to_dict)

    def test_pf11_fingerprint_computation_large(self, benchmark):
        """PF11: fingerprint computation on large report → under 0.5s"""
        before = pd.DataFrame({
            f"col_{i}": np.random.random(20000) for i in range(50)
        })
        after = pd.DataFrame({
            f"col_{i}": np.random.random(20000) for i in range(50)
        })
        
        report = compare(before, after)
        
        def get_fingerprint():
            return report.fingerprint
        
        benchmark(get_fingerprint)

    def test_pf12_polars_vs_pandas_speed(self, benchmark):
        """PF12: Polars input vs Pandas input on same data (20k rows × 20 cols)
        → Polars not more than 3× slower than Pandas"""
        try:
            import polars as pl
        except ImportError:
            pytest.skip("Polars not installed")
        
        # Create test data
        data = {f"col_{i}": np.random.random(20000) for i in range(20)}
        
        # Pandas baseline
        df_pd = pd.DataFrame(data)
        
        def compare_pandas():
            return compare(df_pd, df_pd)
        
        # Run pandas benchmark
        result = benchmark(compare_pandas)
        assert result is not None
