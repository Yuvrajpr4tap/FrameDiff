# Contributing to framediff

Thank you for your interest in contributing to framediff! We welcome contributions of all kinds, from bug reports and documentation improvements to new features and performance enhancements.

## Code of Conduct

Please be respectful and constructive in all interactions. We're committed to providing a welcoming environment for everyone regardless of background or experience level.

## Getting Started

### Prerequisites
- Python 3.9+
- Git
- Basic understanding of pandas/Polars DataFrames

### Development Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/framediff/framediff.git
   cd framediff
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install in development mode with all dependencies:**
   ```bash
   pip install -e ".[all]"
   ```

4. **Run tests to verify setup:**
   ```bash
   pytest tests/ -v
   ```

## Development Workflow

### Before Making Changes

1. Check existing [GitHub Issues](https://github.com/framediff/framediff/issues) to avoid duplicate work
2. For major features, open an issue first to discuss approach
3. Create a new branch for your work:
   ```bash
   git checkout -b feature/your-feature-name
   ```

### Code Standards

We follow these standards to keep code consistent and maintainable:

**Style Guide:**
- Use [PEP 8](https://www.python.org/dev/peps/pep-0008/) conventions
- Black for code formatting: `black framediff/`
- isort for import organization: `isort framediff/`
- Flake8 for linting: `flake8 framediff/`
- mypy for type checking: `mypy framediff/`

**Run all checks:**
```bash
black framediff/
isort framediff/
flake8 framediff/
mypy framediff/
```

### Testing

All changes must include tests. We use pytest.

**Run the test suite:**
```bash
pytest tests/ -v --cov=framediff
```

**Write tests for your changes:**
- Add tests in `tests/` directory matching the module being tested
- Test both normal cases and edge cases
- Use descriptive test names that explain what's being tested
- Aim for >80% code coverage

**Example test:**
```python
def test_my_feature():
    """Test that my feature does X when given Y."""
    input_data = pd.DataFrame(...)
    result = my_new_function(input_data)
    assert result.expected_attr == expected_value
```

### Documentation

- Update docstrings using standard Python docstring format
- Include type hints in function signatures
- Update README.md if adding user-facing features
- Update CHANGELOG.md with a summary of your changes

**Docstring example:**
```python
def compare_values(a: float, b: float) -> float:
    """
    Calculate the absolute difference between two values.
    
    Args:
        a: First value
        b: Second value
    
    Returns:
        Absolute difference
    """
    return abs(a - b)
```

## Submitting Changes

### Before Submitting a Pull Request

1. **Ensure all tests pass:**
   ```bash
   pytest tests/ -v
   ```

2. **Check code quality:**
   ```bash
   black framediff/
   isort framediff/
   flake8 framediff/
   mypy framediff/
   ```

3. **Update documentation and CHANGELOG.md**

4. **Create a clear commit message:**
   - Summarize the change in the first line (50 chars max)
   - Add more detail in subsequent paragraphs if needed
   - Reference any related issues: "Fixes #123"

### Pull Request Guidelines

1. Create a pull request with a descriptive title
2. Include a summary of changes in the description
3. Reference related issues if applicable
4. Ensure CI checks pass (tests, linting, coverage)
5. Respond to any review feedback promptly

**Good PR example:**
```
Title: Fix datetime type conversion in stats analysis

Description:
- Added type checking in _compute_datetime_diff() 
- Prevents crash when datetime64 converts to string
- Adds test case for datetime -> object conversion
- Fixes #42
```

## Types of Contributions

### Bug Reports
Please include:
- Python version and OS
- Minimal reproducible example
- Expected behavior vs actual behavior
- Full error traceback

### Feature Requests
Please include:
- Clear description of the feature
- Use case and motivation
- Proposed API (if applicable)
- Any potential impact on performance

### Documentation Improvements
- Fix typos and unclear explanations
- Add examples for features
- Improve API documentation
- Update README with new features

### Performance Improvements
- Profile code to demonstrate the improvement
- Include benchmark results before/after
- Ensure no regression in other areas

## Project Structure

```
framediff/
├── core.py          # Main compare() function
├── schema.py        # Schema comparison logic
├── stats.py         # Statistical analysis (PSI, distributions)
├── rows.py          # Row-level diffing
├── report.py        # Report generation and formatting
├── assertions.py    # CI-friendly assertions API
├── render.py        # Output formatting
├── exceptions.py    # Custom exception classes
└── __init__.py      # Package initialization

tests/
├── test_*.py        # Test modules (parallel to framediff/)
├── conftest.py      # Pytest fixtures and config
└── __pycache__/
```

## Key Modules

- **core.py:** Entry point; normalizes inputs and orchestrates comparison
- **schema.py:** Detects column changes, type conversions, nullable flags
- **stats.py:** Computes PSI, distribution analysis, statistical scoring
- **rows.py:** Performs row-level diffing with optional key-based matching
- **report.py:** Aggregates results and provides multiple output formats
- **assertions.py:** Provides CI-friendly assertion checks

## Common Tasks

### Adding a New Statistical Metric

1. Add computation function in `stats.py`
2. Update `StatDiff` dataclass if needed
3. Integrate into `compare_stats()` 
4. Add tests in `tests/test_stats_comprehensive.py`
5. Document in README.md

### Adding a New Output Format

1. Add method to `DiffReport` class in `report.py`
2. Implement formatting logic (JSON, HTML, etc.)
3. Add tests for the new format
4. Document the method in README

### Fixing a Bug

1. Create a minimal test that reproduces the bug
2. Fix the bug in the relevant module
3. Verify the test passes
4. Update CHANGELOG.md
5. Submit PR with test included

## Getting Help

- **Questions:** Open a GitHub Discussion
- **Bugs:** Open a GitHub Issue with reproducible example
- **Design discussion:** Comment on relevant issues/PRs

## Attribution

All contributors will be credited in release notes and the README.

Thank you for contributing to framediff! 🎉
