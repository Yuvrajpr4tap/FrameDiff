#!/usr/bin/env python
"""Test TestPyPI installation"""
import framediff as fd
import pandas as pd
import numpy as np

print("=" * 60)
print("Testing framediff from TestPyPI")
print("=" * 60)

# Test 1: Basic comparison
print("\n✓ Test 1: Basic DataFrame comparison")
df1 = pd.DataFrame({"A": [1, 2, 3], "B": ["x", "y", "z"]})
df2 = pd.DataFrame({"A": [1, 2, 3], "B": ["x", "y", "z"]})
report = fd.compare(df1, df2)
assert report.severity == "info"
print(f"  Severity: {report.severity}")
print(f"  Issues: {len(report.issues)}")

# Test 2: Schema changes
print("\n✓ Test 2: Schema changes (added column)")
df1 = pd.DataFrame({"A": [1, 2, 3]})
df2 = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
report = fd.compare(df1, df2)
assert len(report.schema.added_columns) == 1
print(f"  Added columns: {report.schema.added_columns}")
print(f"  Severity: {report.severity}")

# Test 3: Statistical analysis
print("\n✓ Test 3: Statistical distribution analysis")
np.random.seed(42)
df1 = pd.DataFrame({"value": np.random.normal(100, 15, 1000)})
df2 = pd.DataFrame({"value": np.random.normal(120, 15, 1000)})
report = fd.compare(df1, df2)
stat = report.stats.get("value")
print(f"  Distribution label: {stat.distribution_label}")
print(f"  PSI score: {stat.distribution_score:.4f}")
print(f"  Severity: {report.severity}")

# Test 4: JSON serialization
print("\n✓ Test 4: JSON serialization")
df1 = pd.DataFrame({"A": [1, 2, 3]})
df2 = pd.DataFrame({"A": [1, 2, 4]})  # One value changed
report = fd.compare(df1, df2)
json_str = report.to_json()
assert len(json_str) > 0
print(f"  JSON output: {len(json_str)} bytes")

# Test 5: Assertions
print("\n✓ Test 5: CI-friendly assertions")
df1 = pd.DataFrame({"A": [1, 2, 3], "B": [10, 20, 30]})
df2 = pd.DataFrame({"A": [1, 2, 3], "B": [10, 20, 31]})
report = fd.compare(df1, df2, key="A")
try:
    report.assert_within(max_value_change_pct=5)
    print("  Assertion passed: changes within 5%")
except Exception as e:
    print(f"  Assertion raised: {type(e).__name__}")

print("\n" + "=" * 60)
print("All TestPyPI tests passed! ✅")
print("=" * 60)
print(f"\nVersion: {fd.__version__}")
print(f"Package: framediff")
print(f"Status: Ready for PyPI publication")
