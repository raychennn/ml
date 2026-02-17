"""Tests for benchmarks.bench_prescreening module.

Tests all public and private functions in bench_prescreening.py with
small parameter values for speed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from benchmarks.bench_prescreening import (
    BenchmarkResult,
    _make_synthetic_ohlcv,
    _z_normalize,
    _prescreening_distance,
    bench_prescreening,
    bench_prescreening_scaling,
    run_all,
)
from benchmarks.exceptions import BenchmarkSetupError

# ---------------------------------------------------------------------------
# Tests for _make_synthetic_ohlcv
# ---------------------------------------------------------------------------


def test_make_synthetic_ohlcv_returns_correct_columns_and_rows():
    """Test that _make_synthetic_ohlcv returns DataFrame with correct columns and row count."""
    n_bars = 50
    df = _make_synthetic_ohlcv(n_bars=n_bars, seed=42)

    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["timestamp", "open", "high", "low", "close", "volume"]
    assert len(df) == n_bars


def test_make_synthetic_ohlcv_reproducibility():
    """Test that _make_synthetic_ohlcv produces identical results with same seed."""
    n_bars = 50
    seed = 123

    df1 = _make_synthetic_ohlcv(n_bars=n_bars, seed=seed)
    df2 = _make_synthetic_ohlcv(n_bars=n_bars, seed=seed)

    pd.testing.assert_frame_equal(df1, df2)


def test_make_synthetic_ohlcv_different_seeds_produce_different_data():
    """Test that _make_synthetic_ohlcv produces different results with different seeds."""
    n_bars = 50

    df1 = _make_synthetic_ohlcv(n_bars=n_bars, seed=42)
    df2 = _make_synthetic_ohlcv(n_bars=n_bars, seed=999)

    assert not df1["close"].equals(df2["close"])


def test_make_synthetic_ohlcv_with_small_n_bars():
    """Test that _make_synthetic_ohlcv works with small n_bars."""
    df = _make_synthetic_ohlcv(n_bars=5, seed=42)

    assert len(df) == 5
    assert list(df.columns) == ["timestamp", "open", "high", "low", "close", "volume"]


# ---------------------------------------------------------------------------
# Tests for _z_normalize
# ---------------------------------------------------------------------------


def test_z_normalize_produces_mean_zero_std_one():
    """Test that _z_normalize produces array with mean~0 and std~1."""
    rng = np.random.RandomState(42)
    arr = rng.randn(100) * 5.0 + 10.0

    result = _z_normalize(arr)

    assert isinstance(result, np.ndarray)
    assert np.abs(np.mean(result)) < 1e-10
    assert np.abs(np.std(result) - 1.0) < 1e-10


def test_z_normalize_with_constant_array_returns_zeros():
    """Test that _z_normalize with constant array returns zeros."""
    arr = np.ones(50) * 42.0

    result = _z_normalize(arr)

    assert isinstance(result, np.ndarray)
    assert np.allclose(result, 0.0)


def test_z_normalize_with_near_zero_std_returns_zeros():
    """Test that _z_normalize with very small std returns zeros."""
    arr = np.ones(50) * 100.0
    arr[0] = 100.0 + 1e-12

    result = _z_normalize(arr)

    assert np.allclose(result, 0.0)


def test_z_normalize_does_not_modify_input():
    """Test that _z_normalize returns a copy and doesn't modify input."""
    arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    arr_copy = arr.copy()

    result = _z_normalize(arr)

    np.testing.assert_array_equal(arr, arr_copy)
    assert result is not arr


# ---------------------------------------------------------------------------
# Tests for _prescreening_distance
# ---------------------------------------------------------------------------


def test_prescreening_distance_returns_non_negative_float():
    """Test that _prescreening_distance returns non-negative float."""
    rng = np.random.RandomState(42)
    ref_len = 20

    ref_close = rng.randn(ref_len)
    ref_znorm = _z_normalize(ref_close)
    target_close = rng.randn(50)

    distance = _prescreening_distance(ref_znorm, target_close, ref_len)

    assert isinstance(distance, float)
    assert distance >= 0.0


def test_prescreening_distance_with_identical_series_returns_near_zero():
    """Test that _prescreening_distance with identical series returns near-zero distance."""
    ref_len = 20
    rng = np.random.RandomState(42)
    series = rng.randn(ref_len)
    ref_znorm = _z_normalize(series)

    distance = _prescreening_distance(ref_znorm, series, ref_len)

    assert distance < 1e-10


def test_prescreening_distance_with_different_series_returns_positive():
    """Test that _prescreening_distance with different series returns positive distance."""
    ref_len = 20
    rng1 = np.random.RandomState(42)
    rng2 = np.random.RandomState(999)

    ref_close = rng1.randn(ref_len)
    ref_znorm = _z_normalize(ref_close)
    target_close = rng2.randn(50)

    distance = _prescreening_distance(ref_znorm, target_close, ref_len)

    assert distance > 0.0


def test_prescreening_distance_uses_tail_of_target():
    """Test that _prescreening_distance uses only the tail of target series."""
    ref_len = 10

    # Create reference
    ref_close = np.arange(ref_len, dtype=np.float64)
    ref_znorm = _z_normalize(ref_close)

    # Create target with matching tail
    target_close = np.concatenate(
        [np.array([999.0, 888.0, 777.0]), np.arange(ref_len, dtype=np.float64)]
    )

    distance = _prescreening_distance(ref_znorm, target_close, ref_len)

    assert distance < 1e-10


# ---------------------------------------------------------------------------
# Tests for bench_prescreening
# ---------------------------------------------------------------------------


def test_bench_prescreening_returns_two_results():
    """Test that bench_prescreening returns exactly 2 BenchmarkResult dicts."""
    results = bench_prescreening(n_symbols=5, ref_len=20, n_bars=50)

    assert isinstance(results, list)
    assert len(results) == 2


def test_bench_prescreening_results_have_correct_keys():
    """Test that bench_prescreening results have all required BenchmarkResult keys."""
    results = bench_prescreening(n_symbols=5, ref_len=20, n_bars=50)

    required_keys = {"name", "value", "unit", "detail"}
    for result in results:
        assert set(result.keys()) == required_keys


def test_bench_prescreening_first_result_is_time():
    """Test that first result from bench_prescreening is elapsed time."""
    results = bench_prescreening(n_symbols=5, ref_len=20, n_bars=50)

    time_result = results[0]
    assert "symbols" in time_result["name"]
    assert time_result["unit"] == "seconds"
    assert isinstance(time_result["value"], float)
    assert time_result["value"] >= 0.0


def test_bench_prescreening_second_result_is_throughput():
    """Test that second result from bench_prescreening is symbols/sec."""
    results = bench_prescreening(n_symbols=5, ref_len=20, n_bars=50)

    throughput_result = results[1]
    assert throughput_result["name"] == "prescreening_symbols_per_sec"
    assert throughput_result["unit"] == "symbols/sec"
    assert isinstance(throughput_result["value"], float)
    assert throughput_result["value"] > 0.0


def test_bench_prescreening_with_minimal_params():
    """Test that bench_prescreening works with minimal parameters."""
    results = bench_prescreening(n_symbols=2, ref_len=10, n_bars=20)

    assert len(results) == 2
    assert results[0]["value"] >= 0.0
    assert results[1]["value"] > 0.0


def test_bench_prescreening_throughput_increases_with_more_symbols():
    """Test that more symbols results in longer time but positive throughput."""
    results_small = bench_prescreening(n_symbols=2, ref_len=10, n_bars=20)
    results_large = bench_prescreening(n_symbols=10, ref_len=10, n_bars=20)

    # More symbols should take more time
    assert results_large[0]["value"] > results_small[0]["value"]

    # Both should have positive throughput
    assert results_small[1]["value"] > 0.0
    assert results_large[1]["value"] > 0.0


# ---------------------------------------------------------------------------
# Tests for bench_prescreening_scaling
# ---------------------------------------------------------------------------


def test_bench_prescreening_scaling_returns_non_empty_list():
    """Test that bench_prescreening_scaling returns a non-empty list."""
    # Note: We can't easily test this with small values since it uses hardcoded
    # values [50, 100, 200]. We'll test that it completes successfully.
    results = bench_prescreening_scaling()

    assert isinstance(results, list)
    assert len(results) > 0


def test_bench_prescreening_scaling_returns_three_results():
    """Test that bench_prescreening_scaling returns exactly 3 results."""
    results = bench_prescreening_scaling()

    assert len(results) == 3


def test_bench_prescreening_scaling_results_have_correct_structure():
    """Test that bench_prescreening_scaling results have correct structure."""
    results = bench_prescreening_scaling()

    required_keys = {"name", "value", "unit", "detail"}
    for result in results:
        assert set(result.keys()) == required_keys
        assert result["unit"] == "seconds"
        assert isinstance(result["value"], float)
        assert result["value"] >= 0.0


def test_bench_prescreening_scaling_results_have_unique_names():
    """Test that bench_prescreening_scaling results have unique names."""
    results = bench_prescreening_scaling()

    names = [r["name"] for r in results]
    assert len(names) == len(set(names))


# ---------------------------------------------------------------------------
# Tests for run_all
# ---------------------------------------------------------------------------


def test_run_all_returns_non_empty_list():
    """Test that run_all returns a non-empty list of BenchmarkResults."""
    results = run_all()

    assert isinstance(results, list)
    assert len(results) > 0


def test_run_all_results_have_correct_structure():
    """Test that all results from run_all have correct BenchmarkResult structure."""
    results = run_all()

    required_keys = {"name", "value", "unit", "detail"}
    for result in results:
        assert set(result.keys()) == required_keys
        assert isinstance(result["name"], str)
        assert isinstance(result["value"], (float, int, type(None)))
        assert isinstance(result["unit"], str)
        assert isinstance(result["detail"], str)


def test_run_all_includes_bench_prescreening_results():
    """Test that run_all includes results from bench_prescreening."""
    results = run_all()

    names = [r["name"] for r in results]
    assert any("prescreening" in name for name in names)


def test_run_all_includes_scaling_results():
    """Test that run_all includes results from bench_prescreening_scaling."""
    results = run_all()

    names = [r["name"] for r in results]
    assert any("scale" in name for name in names)


# ---------------------------------------------------------------------------
# Edge cases and error handling
# ---------------------------------------------------------------------------


def test_z_normalize_with_single_element():
    """Test that _z_normalize handles single-element array."""
    arr = np.array([42.0])

    result = _z_normalize(arr)

    assert len(result) == 1
    assert np.allclose(result, 0.0)


def test_prescreening_distance_with_exact_length_target():
    """Test _prescreening_distance when target length equals ref_len."""
    ref_len = 20
    rng = np.random.RandomState(42)

    ref_close = rng.randn(ref_len)
    ref_znorm = _z_normalize(ref_close)
    target_close = rng.randn(ref_len)

    distance = _prescreening_distance(ref_znorm, target_close, ref_len)

    assert isinstance(distance, float)
    assert distance >= 0.0


def test_z_normalize_with_negative_values():
    """Test that _z_normalize works correctly with negative values."""
    arr = np.array([-10.0, -5.0, 0.0, 5.0, 10.0])

    result = _z_normalize(arr)

    assert np.abs(np.mean(result)) < 1e-10
    assert np.abs(np.std(result) - 1.0) < 1e-10


def test_benchmark_result_type_checking():
    """Test that BenchmarkResult is properly structured."""
    result = BenchmarkResult(
        name="test", value=1.0, unit="seconds", detail="test detail"
    )

    assert result["name"] == "test"
    assert result["value"] == 1.0
    assert result["unit"] == "seconds"
    assert result["detail"] == "test detail"
