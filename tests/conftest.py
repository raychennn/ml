"""Shared pytest fixtures for the performance-infrastructure test suite."""

import os
import sys
import tempfile
import shutil

import pytest
import numpy as np
import pandas as pd

# Ensure the project root is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def tmpdir_path():
    """Create and yield a temporary directory, cleaned up after the test."""
    path = tempfile.mkdtemp()
    yield path
    shutil.rmtree(path, ignore_errors=True)


@pytest.fixture
def synthetic_ohlcv():
    """Return a factory that generates synthetic OHLCV DataFrames."""

    def _make(n_bars: int = 500, seed: int = 42) -> pd.DataFrame:
        rng = np.random.RandomState(seed)
        close = 100.0 + np.cumsum(rng.randn(n_bars) * 0.5)
        return pd.DataFrame(
            {
                "timestamp": np.arange(1700000000, 1700000000 + n_bars * 3600, 3600)[
                    :n_bars
                ].astype(np.int64),
                "open": (close + rng.randn(n_bars) * 0.3).astype(np.float64),
                "high": (close + rng.uniform(0.1, 1.0, n_bars)).astype(np.float64),
                "low": (close - rng.uniform(0.1, 1.0, n_bars)).astype(np.float64),
                "close": close.astype(np.float64),
                "volume": rng.uniform(100, 10000, n_bars).astype(np.float64),
            }
        )

    return _make


@pytest.fixture
def metrics_path(tmpdir_path):
    """Return a path for a temporary JSONL metrics file."""
    return os.path.join(tmpdir_path, "metrics.jsonl")
