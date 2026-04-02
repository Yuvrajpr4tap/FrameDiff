# framediff

**Schema-aware, zero-config dataframe diffing and change-tracking library for pandas and Polars.**

Framediff provides a single-function API to comprehensively compare two DataFrames. It detects schema changes, statistical distributions shifts, and row-level modifications — all with zero configuration required.

## Features

- ✨ **Zero configuration** — intelligent defaults detect schema, numeric distributions, categorical changes automatically
- 📊 **Statistical analysis** — PSI (Population Stability Index) for numeric columns, chi-squared for categorical
- 🔄 **Row-level tracking** — detect added, removed, and modified rows with sample data
- 🎯 **Severity scoring** — critical/warning/info levels for each finding
- 📝 **Enriched output** — serializable to JSON, pretty-print for terminal, Jupyter HTML with histograms
- 🧪 **CI-friendly assertions** — fail builds on drift thresholds
- 🔗 **Multi-framework** — pandas and Polars (auto-normalized internally)
- 🚀 **Performance** — handles 1M+ row frames in <10s

## Installation

```bash
pip install framediff
```

With optional dependencies for Polars support and rich terminal formatting:

```bash
pip install framediff[polars,rich]
```

## Quickstart

```python
import framediff as fd
import pandas as pd

# Load before/after data
before = pd.read_csv('data_v1.csv')
after = pd.read_csv('data_v2.csv')

# Compare in one line
report = fd.compare(before, after, key='customer_id')

# Inspect the report
print(report.summary)           # Human-readable summary
print(report.severity)          # "info" | "warning" | "critical"
print(report.fingerprint)       # Deterministic hash for reproducibility

# Assert constraints in CI
report.assert_within(
    max_rows_removed_pct=5,
    max_psi=0.15,
    no_critical=True
)
```

## Core API

### `compare(before, after, key=None, sample_size=None, stat_methods=["auto"])`

Main entry point. Performs schema, statistical, and row-level comparison.

**Args:**
- `before` (pd.DataFrame | pl.DataFrame): DataFrame before changes
- `after` (pd.DataFrame | pl.DataFrame): DataFrame after changes
- `key` (str | list[str], optional): Column(s) to join on. If None, uses positional matching
- `sample_size` (int, optional): Row sample for large frames (performance optimization)
- `stat_methods` (list[str]): Distribution methods. Default: `["auto"]`

**Returns:** `DiffReport` object

### DiffReport

Complete diff summary with schema, statistics, and row changes.

**Attributes:**

| Attribute | Type | Description |
|-----------|------|-------------|
| `schema` | `SchemaDiff` | Added/removed/changed columns, type changes |
| `stats` | `dict[str, StatDiff]` | Per-column distribution analysis |
| `rows` | `RowDiff` | Added/removed/modified row counts + samples |
| `severity` | `str` | Highest severity across all issues ("info" \| "warning" \| "critical") |
| `summary` | `str` | One-line human-readable change summary |
| `issues` | `list[DiffIssue]` | All detected issues with severity |
| `fingerprint` | `str` | Deterministic SHA256 hash for reproducibility |

**Methods:**

```python
report.to_dict()           # JSON-serializable dict
report.to_json()           # JSON string
report.assert_within(...)  # Raise if constraints violated
```

**Display:**

```python
print(report)              # Rich-formatted terminal table
display(report)            # Jupyter HTML with histograms
```

## Severity Scoring

Framediff assigns severity to each finding:

### Schema Changes
- **Removed column** → ⚠️ **Critical** (data loss)
- **Type change (lossy)** → ⚠️ **Warning** (e.g., float→int)
- **Type change (safe)** → ℹ️ **Info** (e.g., int→float)
- **Added column** → ℹ️ **Info**

### Statistical Shifts
For numeric columns with >50 unique values, computes **PSI (Population Stability Index)**:

| PSI Score | Label | Severity |
|-----------|-------|----------|
| PSI < 0.1 | Stable | ℹ️ Info |
| 0.1 ≤ PSI < 0.2 | Moderate shift | ⚠️ Warning |
| PSI ≥ 0.2 | Large shift | 🔴 Critical |

For categorical columns: chi-squared test on value counts; new/dropped categories reported.

### Row Changes
- **Null rate increase > 10%** → 🔴 **Critical**
- **Null rate increase 2–10%** → ⚠️ **Warning**
- **Added/removed rows > 20%** → ⚠️ **Warning**

### Overall Severity
Report severity = highest severity of any single issue.

## Assertions for CI

Assert constraints in continuous integration pipelines:

```python
from framediff import DiffThresholdError

report = fd.compare(before, after, key='id')

try:
    report.assert_within(
        max_rows_removed_pct=5,      # Fail if >5% removed
        max_rows_added_pct=10,        # Fail if >10% added
        max_null_rate_increase=0.05,  # Per column, fail if null rate jumps >5%
        max_psi=0.15,                 # Per numeric column, fail if PSI >0.15
        no_type_changes=True,         # Fail on any dtype change
        no_removed_columns=True,      # Fail if cols removed
        no_critical=True,             # Fail if any critical severity
        columns=['revenue', 'user_id'],  # Restrict to specific columns
    )
except DiffThresholdError as e:
    print(f"Data validation failed: {e}")
    exit(1)
```

## CI Integration Example

**GitHub Actions:**

```yaml
name: Data Quality Checks

on: [push, pull_request]

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: "3.11"
      
      - run: pip install framediff pytest pandas
      
      - name: Run data diff assertions
        run: |
          python -c "
          import framediff as fd
          import pandas as pd
          before = pd.read_csv('data/baseline.csv')
          after = pd.read_csv('data/candidate.csv')
          report = fd.compare(before, after, key='id')
          report.assert_within(
              max_rows_removed_pct=1,
              max_psi=0.1,
              no_critical=True
          )
          print(report.summary)
          "
```

## Framework Support

Framediff transparently handles both pandas and Polars DataFrames:

| Feature | Pandas | Polars | PyArrow |
|---------|--------|--------|---------|
| Schema analysis | ✅ | ✅ | Planned |
| Statistical diffs | ✅ | ✅ | Planned |
| Row-level tracking | ✅ | ✅ | Planned |
| JSON serialization | ✅ | ✅ | Planned |

All Polars DataFrames are internally normalized to pandas for computation using `.to_pandas()`.

## Examples

### Basic comparison

```python
import framediff as fd
import pandas as pd

df1 = pd.DataFrame({
    'id': [1, 2, 3],
    'age': [25.0, 30.0, 35.0],
    'status': ['active', 'inactive', 'active']
})

df2 = pd.DataFrame({
    'id': [1, 2, 4],
    'age': [25.5, 30.0, 40.0],
    'status': ['active', 'active', 'active']
})

report = fd.compare(df1, df2, key='id')
print(report)
```

### Comparing with schema normalization

```python
# DataFrames with different column orders and new columns are fine
df1 = pd.read_parquet('v1.parquet')
df2 = pd.read_parquet('v2.parquet')

report = fd.compare(df1, df2)
print(f"Added columns: {report.schema.added_columns}")
print(f"Type changes: {report.schema.type_changes}")
```

### Production data quality checks

```python
# Production pipeline: validate new data against baseline
baseline = pd.read_csv('baseline.csv')
new_data = pd.read_csv('new_data.csv')

report = fd.compare(baseline, new_data, key=['user_id', 'date'])

# Log the fingerprint for traceability
logger.info(f"Data quality check: {report.fingerprint}")

# Raise alert if critical issues
if report.severity == 'critical':
    send_alert(f"Critical data issues: {report.issues}")

# Check specific constraints
report.assert_within(
    max_null_rate_increase=0.02,
    no_removed_columns=True,
)
```

### Jupyter exploration

```python
import framediff as fd

report = fd.compare(baseline_df, new_df, key='id')
display(report)  # Renders HTML with severity badges and histograms
```

## Performance

On a modern laptop:

| Rows | Columns | Time |
|------|---------|------|
| 10K | 50 | ~0.1s |
| 100K | 50 | ~0.5s |
| 1M | 50 | ~3s |
| 1M | 20 | <10s |

For large DataFrames, use `sample_size` parameter:

```python
report = fd.compare(df_large, df_updated, sample_size=100000)
```

## API Reference

### Classes

**DiffReport**
- `severity: str` — Highest severity of any issue
- `summary: str` — One-line human summary
- `fingerprint: str` — Deterministic SHA256 hash
- `to_dict() → dict` — JSON-serializable representation
- `to_json() → str` — JSON string
- `assert_within(**kwargs) → None` — Raise on constraint violation

**SchemaDiff**
- `added_columns: list[str]`
- `removed_columns: list[str]`
- `type_changes: dict[str, tuple[str, str]]`
- `nullable_changes: dict[str, tuple[bool, bool]]`
- `index_changes: dict`

**StatDiff**
- `column: str` — Column name
- `dtype: str` — Data type
- `mean_delta: float` — Absolute change in mean (numeric only)
- `std_delta: float` — Absolute change in std (numeric only)
- `null_rate_before: float` — Fraction of nulls before
- `null_rate_after: float` — Fraction of nulls after
- `distribution_method: str` — Method used ("psi", "chi2", "datetime_range", "none")
- `distribution_score: float` — Score from method (PSI, chi-squared, etc.)
- `distribution_label: str` — "stable" | "moderate shift" | "large shift"
- `new_categories: list[str]` — Categories added (categorical only)
- `dropped_categories: list[str]` — Categories removed (categorical only)
- `severity: str` — Issue severity

**RowDiff**
- `added_count: int` — Number of rows added
- `removed_count: int` — Number of rows removed
- `modified_count: int` — Number of rows modified
- `added_pct: float` — Percentage of rows added
- `removed_pct: float` — Percentage of rows removed
- `sample_added: pd.DataFrame` — Up to 10 example added rows
- `sample_removed: pd.DataFrame` — Up to 10 example removed rows
- `sample_modified: pd.DataFrame` — Up to 10 example modified rows (before/after columns)

### Exceptions

**DiffThresholdError**
Raised by `assert_within()` when constraints are violated.

```python
except DiffThresholdError as e:
    print(e.message)      # Error message
    print(e.violations)   # List of violated constraints
```

## Requirements

- Python ≥ 3.9
- pandas ≥ 1.5
- numpy ≥ 1.21
- scipy ≥ 1.9

Optional:
- polars ≥ 0.19 (for Polars DataFrame support)
- rich ≥ 10.0 (for rich terminal formatting)

## Testing

Run the test suite:

```bash
pip install -e ".[dev]"
pytest tests/ -v --cov=framediff
```

## License

MIT License — see [LICENSE](LICENSE) for details.

## Contributing

Contributions welcome! Please open an issue or PR.

---
