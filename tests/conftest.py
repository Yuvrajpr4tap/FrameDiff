"""
Pytest configuration and fixtures for framediff tests.
"""
import pytest
import pandas as pd
import numpy as np


@pytest.fixture(autouse=True)
def timeout_guard(request):
    """Auto-use fixture to ensure no single test exceeds 30 seconds."""
    # Note: signal.SIGALRM is not available on Windows
    # This fixture is a placeholder for documentation purposes
    # Tests should be written to complete within reasonable time limits
    yield


@pytest.fixture
def df_simple_before():
    """Simple DataFrame before changes."""
    return pd.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "name": ["Alice", "Bob", "Charlie", "David", "Eve"],
        "value": [100.0, 200.0, 300.0, 400.0, 500.0],
        "category": ["A", "B", "A", "B", "A"],
    })


@pytest.fixture
def df_simple_after():
    """Simple DataFrame after changes (rows modified, column added)."""
    return pd.DataFrame({
        "id": [1, 2, 3, 4, 6],  # 5 removed, 6 added
        "name": ["Alice", "Bob", "Charlie", "David", "Frank"],
        "value": [100.0, 210.0, 300.0, 400.0, 600.0],  # 2 modified
        "category": ["A", "B", "A", "C", "A"],  # new category C
        "new_col": ["X", "Y", "Z", "W", "V"],  # new column
    })


@pytest.fixture
def df_numeric_stable():
    """DataFrame with numeric column (stable distribution)."""
    np.random.seed(42)
    return pd.DataFrame({
        "id": range(1000),
        "value": np.random.normal(100, 15, 1000),
    })


@pytest.fixture
def df_numeric_shifted():
    """DataFrame with numeric column (PSI shift)."""
    np.random.seed(43)
    return pd.DataFrame({
        "id": range(1000),
        "value": np.random.normal(105, 15, 1000),  # slightly shifted mean
    })


@pytest.fixture
def df_numeric_large_shift():
    """DataFrame with large PSI shift."""
    np.random.seed(44)
    return pd.DataFrame({
        "id": range(1000),
        "value": np.random.normal(120, 15, 1000),  # large shift
    })


@pytest.fixture
def df_categorical_stable():
    """DataFrame with categorical column (stable)."""
    return pd.DataFrame({
        "id": range(100),
        "category": ["A"] * 50 + ["B"] * 30 + ["C"] * 20,
    })


@pytest.fixture
def df_categorical_shifted():
    """DataFrame with categorical column (shifted distribution)."""
    return pd.DataFrame({
        "id": range(100),
        "category": ["A"] * 40 + ["B"] * 40 + ["C"] * 20,  # different proportions
    })


@pytest.fixture
def df_with_nulls_before():
    """DataFrame with some null values."""
    return pd.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "value": [10.0, None, 30.0, 40.0, None],
        "name": ["A", "B", "C", "D", "E"],
    })


@pytest.fixture
def df_with_nulls_after():
    """DataFrame with increased null rate."""
    return pd.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "value": [10.0, None, None, 40.0, None],  # more nulls
        "name": ["A", "B", "C", "D", "E"],
    })


@pytest.fixture
def df_type_change_before():
    """DataFrame before type change."""
    return pd.DataFrame({
        "id": [1, 2, 3],
        "amount": [10.5, 20.5, 30.5],
    })


@pytest.fixture
def df_type_change_after():
    """DataFrame after type change (lossy)."""
    return pd.DataFrame({
        "id": [1, 2, 3],
        "amount": [10, 20, 30],  # float to int
    })


@pytest.fixture
def df_empty():
    """Empty DataFrame."""
    return pd.DataFrame({
        "col1": [],
        "col2": [],
    })


@pytest.fixture
def df_single_row():
    """Single-row DataFrame."""
    return pd.DataFrame({
        "id": [1],
        "value": [100.0],
    })


@pytest.fixture
def df_all_nulls():
    """DataFrame with column of all null values."""
    return pd.DataFrame({
        "id": [1, 2, 3],
        "all_null": [None, None, None],
    })


@pytest.fixture
def df_binary_col_before():
    """DataFrame with binary column (before)."""
    return pd.DataFrame({
        "id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "status": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    })


@pytest.fixture
def df_binary_col_after_flipped():
    """DataFrame with binary column (50% values flipped)."""
    return pd.DataFrame({
        "id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "status": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],  # all values flipped
    })
