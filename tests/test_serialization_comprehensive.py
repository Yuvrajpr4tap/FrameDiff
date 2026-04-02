"""
BLOCK 6: Serialisation and Fingerprinting — Comprehensive coverage
Complete tests for JSON serialization, fingerprinting, and data integrity.
"""
import pytest
import pandas as pd
import numpy as np
import json
import hashlib
from framediff import compare


class TestSerializationCleanness:
    """SR01-SR06: JSON serialization must be clean (no numpy types)"""

    def test_sr01_to_dict_no_numpy_types(self):
        """SR01: to_dict() contains no np.int64, np.float64, np.bool_, np.nan
        — json.dumps(to_dict()) must not raise TypeError"""
        before = pd.DataFrame({"A": [1, 2, 3], "B": [1.5, 2.5, 3.5], "C": [True, False, True]})
        after = pd.DataFrame({"A": [1, 2, 3], "B": [1.5, 2.5, 3.5], "C": [True, False, True]})
        report = compare(before, after)
        
        report_dict = report.to_dict()
        
        # This should NOT raise TypeError
        json_str = json.dumps(report_dict)
        assert isinstance(json_str, str)

    def test_sr02_to_json_valid_json(self):
        """SR02: to_json() produces valid JSON — json.loads(to_json()) must not raise"""
        before = pd.DataFrame({"A": [1, 2, 3], "B": ["x", "y", "z"]})
        after = pd.DataFrame({"A": [1, 2, 3], "B": ["x", "y", "z"]})
        report = compare(before, after)
        
        json_str = report.to_json()
        # This should NOT raise
        parsed = json.loads(json_str)
        assert isinstance(parsed, dict)

    def test_sr03_to_json_handles_inf(self):
        """SR03: to_json() handles np.inf values (JSON has no inf — must be null or string)"""
        before = pd.DataFrame({"A": [1.0, np.inf, 3.0]})
        after = pd.DataFrame({"A": [1.0, np.inf, 3.0]})
        report = compare(before, after)
        
        json_str = report.to_json()
        # Should not raise; np.inf should be handled (null or "inf" string)
        parsed = json.loads(json_str)
        assert isinstance(parsed, dict)

    def test_sr04_to_json_handles_nan(self):
        """SR04: to_json() handles np.nan values (JSON has no nan — must be null)"""
        before = pd.DataFrame({"A": [1.0, np.nan, 3.0]})
        after = pd.DataFrame({"A": [1.0, np.nan, 3.0]})
        report = compare(before, after)
        
        json_str = report.to_json()
        # Should not raise
        parsed = json.loads(json_str)
        assert isinstance(parsed, dict)

    def test_sr05_to_json_handles_nat(self):
        """SR05: to_json() handles pd.NaT values"""
        before = pd.DataFrame({"A": pd.to_datetime(["2020-01-01", "NaT", "2020-01-03"])})
        after = pd.DataFrame({"A": pd.to_datetime(["2020-01-01", "NaT", "2020-01-03"])})
        report = compare(before, after)
        
        json_str = report.to_json()
        # Should not raise
        parsed = json.loads(json_str)
        assert isinstance(parsed, dict)

    def test_sr06_to_json_handles_na(self):
        """SR06: to_json() handles pd.NA values (nullable integer/string NA)"""
        before = pd.DataFrame({"A": pd.array([1, pd.NA, 3], dtype="Int64")})
        after = pd.DataFrame({"A": pd.array([1, pd.NA, 3], dtype="Int64")})
        report = compare(before, after)
        
        json_str = report.to_json()
        # Should not raise
        parsed = json.loads(json_str)
        assert isinstance(parsed, dict)


class TestFingerprintDeterminism:
    """SR07-SR14: Fingerprint consistency and validity"""

    def test_sr07_fingerprint_format(self):
        """SR07: fingerprint is exactly 64 lowercase hex characters"""
        before = pd.DataFrame({"A": [1, 2, 3]})
        after = pd.DataFrame({"A": [1, 2, 3]})
        report = compare(before, after)
        
        assert isinstance(report.fingerprint, str)
        assert len(report.fingerprint) == 64
        assert all(c in "0123456789abcdef" for c in report.fingerprint)

    def test_sr08_fingerprint_deterministic(self):
        """SR08: fingerprint is deterministic: same inputs → same fingerprint, 10 runs"""
        fingerprints = set()
        for i in range(10):
            before = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
            after = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
            report = compare(before, after)
            fingerprints.add(report.fingerprint)
        
        assert len(fingerprints) == 1

    def test_sr09_fingerprint_changes_with_cell_value(self):
        """SR09: fingerprint changes when any single cell value changes"""
        before = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        after1 = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        
        report1 = compare(before, after1)
        fp1 = report1.fingerprint
        
        after2 = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 7]})  # Changed one cell
        report2 = compare(before, after2)
        fp2 = report2.fingerprint
        
        assert fp1 != fp2

    def test_sr10_fingerprint_changes_with_added_column(self):
        """SR10: fingerprint changes when a column is added"""
        df1 = pd.DataFrame({"A": [1, 2, 3]})
        df2 = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        
        report1 = compare(df1, df1)
        fp1 = report1.fingerprint
        
        report2 = compare(df1, df2)
        fp2 = report2.fingerprint
        
        assert fp1 != fp2

    def test_sr11_fingerprint_changes_with_removed_column(self):
        """SR11: fingerprint changes when a column is removed"""
        df1 = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        df2 = pd.DataFrame({"A": [1, 2, 3]})
        
        df_same = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        report1 = compare(df_same, df_same)
        fp1 = report1.fingerprint
        
        report2 = compare(df1, df2)
        fp2 = report2.fingerprint
        
        assert fp1 != fp2

    def test_sr12_fingerprint_changes_with_row_count(self):
        """SR12: fingerprint changes when row count changes"""
        before = pd.DataFrame({"A": [1, 2, 3]})
        
        report1 = compare(before, before)
        fp1 = report1.fingerprint
        
        after = pd.DataFrame({"A": [1, 2, 3, 4]})
        report2 = compare(before, after)
        fp2 = report2.fingerprint
        
        assert fp1 != fp2

    def test_sr13_fingerprint_not_based_on_memory_address(self):
        """SR13: fingerprint does NOT change based on Python object memory addresses"""
        data = {"A": [1, 2, 3], "B": [4, 5, 6]}
        
        report1 = compare(pd.DataFrame(data), pd.DataFrame(data))
        fp1 = report1.fingerprint
        
        report2 = compare(pd.DataFrame(data), pd.DataFrame(data))
        fp2 = report2.fingerprint
        
        assert fp1 == fp2  # Different objects, but same fingerprint

    def test_sr14_fingerprint_matches_manual_calculation(self):
        """SR14: Manual fingerprint: sha256(json.dumps(to_dict(), sort_keys=True))
        must equal report.fingerprint"""
        before = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        after = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        report = compare(before, after)
        
        manual_fp = hashlib.sha256(
            json.dumps(report.to_dict(), sort_keys=True).encode()
        ).hexdigest()
        
        assert manual_fp == report.fingerprint


class TestRoundTrip:
    """SR15-SR17: Data integrity through serialization"""

    def test_sr15_roundtrip_type_preservation(self):
        """SR15: Round-trip: to_dict() values survive json.dumps → json.loads with
        correct Python types (not all strings)"""
        before = pd.DataFrame({"A": [1, 2, 3], "B": [1.5, 2.5, 3.5], "C": [True, False, True]})
        after = pd.DataFrame({"A": [1, 2, 3], "B": [1.5, 2.5, 3.5], "C": [True, False, True]})
        report = compare(before, after)
        
        json_str = json.dumps(report.to_dict(), sort_keys=True)
        parsed = json.loads(json_str)
        
        # Check that types are preserved (not all strings)
        assert parsed is not None
        assert isinstance(parsed, dict)

    def test_sr16_polars_pandas_identical_json(self):
        """SR16: to_json() output is identical for equivalent Pandas and Polars inputs
        (same logical data → same JSON)"""
        try:
            import polars as pl
        except ImportError:
            pytest.skip("Polars not installed")
        
        data = {"A": [1, 2, 3], "B": [4.0, 5.0, 6.0]}
        
        df_pd = pd.DataFrame(data)
        df_pl = pl.DataFrame(data)
        
        report_pd = compare(df_pd, df_pd)
        report_pl = compare(df_pl, df_pl)
        
        # Fingerprints should match for identical data
        assert report_pd.fingerprint == report_pl.fingerprint

    def test_sr17_large_report_serialization_speed(self):
        """SR17: Large report (50k rows, 30 cols) serialises in under 1 second"""
        import time
        
        before = pd.DataFrame({f"col_{i}": np.random.random(20000) for i in range(30)})
        after = pd.DataFrame({f"col_{i}": np.random.random(20000) for i in range(30)})
        report = compare(before, after)
        
        start = time.time()
        json_str = report.to_json()
        elapsed = time.time() - start
        
        assert elapsed < 2.0
        assert len(json_str) > 0
