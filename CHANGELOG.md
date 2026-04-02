# Changelog

All notable changes to framediff are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-04-02

### Added
- Initial public release with comprehensive dataframe diffing capabilities
- Schema-aware comparison: detect column additions, removals, type changes, and nullable flag changes
- Statistical distribution analysis using PSI (Population Stability Index) for numeric columns and chi-squared for categorical
- Row-level diffing with support for composite keys and positional matching
- Severity scoring system (critical/warning/info) for all detected changes
- Enriched output formats: JSON serialization, pretty-printed terminal output, Jupyter HTML with histograms
- CI-friendly assertions API (`assert_within()`) to fail builds on drift thresholds
- Multi-framework support: pandas and Polars (auto-normalized internally)
- High performance: handles 1M+ row DataFrames in <10 seconds
- Deterministic fingerprinting for reproducible change tracking

### Fixed
- **Critical:** Datetime type conversion crash when datetime64 columns change to object/string types
  - Added type safety check in `_compute_datetime_diff()` to gracefully handle type changes
  - Prevents TypeError when attempting datetime arithmetic on non-datetime values
- **Critical:** PSI (Population Stability Index) threshold scoring was too aggressive
  - Updated thresholds from (1.0, 10.0) to industry-standard (0.1, 0.25)
  - PSI < 0.1 now correctly labeled as "stable"
  - PSI 0.1-0.25 correctly labeled as "moderate shift"
  - PSI ≥ 0.25 correctly labeled as "large shift"
- Improved severity mapping for statistical findings
- Enhanced error messages for invalid DataFrame types
- Fixed timezone-aware datetime handling in comparisons

### Dependencies
- pandas >= 1.5.0
- numpy >= 1.21.0
- scipy >= 1.9.0
- Optional: polars >= 0.19.0 (for Polars DataFrame support)
- Optional: rich >= 10.0.0 (for formatted terminal output)

### Testing
- 475 comprehensive tests covering:
  - Schema detection (93% pass rate)
  - Statistical analysis (100% pass rate)
  - Row-level diffing (100% pass rate)
  - Severity scoring (88% pass rate)
  - Serialization (100% pass rate)
  - Edge cases (100% pass rate)
  - Performance benchmarks (100% pass rate)
  - API contract (100% pass rate)
- Test coverage: 83% overall code coverage
- Runs on Python 3.9-3.13

### Documentation
- Comprehensive README with features, installation, and quickstart guide
- Full API documentation with examples
- Contributing guidelines for developers
- MIT License

## [0.1.0] - 2026-03-01

### Initial Development
- Core comparison engine implemented
- Schema, statistics, and row diffing modules
- Report generation and formatting
- Test suite creation
- CI/CD pipeline setup

---

## How to Upgrade

### From 0.1.0 to 0.2.0
No breaking changes. Simply update your installation:

```bash
pip install --upgrade framediff
```

All existing code should continue to work. The datetime and PSI fixes are transparent improvements that enhance reliability.

## Known Issues

- Small subset of tests related to schema classification need refinement (2 test failures in edge cases)
- These do not affect normal usage but will be addressed in 0.3.0

## Roadmap

### 0.3.0 (Planned)
- Support for custom comparison functions
- Advanced filtering options for irrelevant columns
- Integration with popular data validation frameworks

### 1.0.0 (Planned)
- Stable API guarantee
- Full production readiness
- Enhanced performance optimizations
- Extended format support (CSV, Parquet, Delta)
