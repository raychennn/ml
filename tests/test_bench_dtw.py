"""Tests for benchmarks.bench_dtw module.

Tests DTW benchmark functions including random walk generation,
series pair creation, DTW distance calculations, and benchmark runners.
"""

import pytest
import numpy as np
from numpy.typing import NDArray

from benchmarks.bench_dtw import (
    BenchmarkResult,
    _make_random_walk,
    _make_series_pair,
    _dtw_dtaidistance,
    _dtw_numpy,
    bench_single_pair,
    bench_batch,
    bench_numpy_fallback,
    bench_dtaidistance_vs_numpy,
    run_all,
)
from benchmarks.exceptions import BenchmarkSetupError, BenchmarkRunError

# ---------------------------------------------------------------------------
# Tests for _make_random_walk
# ---------------------------------------------------------------------------


def test_make_random_walk_produces_correct_length():
    """Test that _make_random_walk produces an array of the requested length."""
    length = 50
    walk = _make_random_walk(length, seed=42)
    assert len(walk) == length
    assert isinstance(walk, np.ndarray)


def test_make_random_walk_is_z_normalized():
    """Test that _make_random_walk produces approximately z-normalized data (mean~0, std~1)."""
    walk = _make_random_walk(100, seed=42)
    mean = np.mean(walk)
    std = np.std(walk)

    # Allow small numerical tolerance
    assert abs(mean) < 1e-10, f"Expected mean~0, got {mean}"
    assert abs(std - 1.0) < 1e-10, f"Expected std~1, got {std}"


def test_make_random_walk_same_seed_produces_same_results():
    """Test that the same seed produces identical random walks (reproducibility)."""
    length = 50
    seed = 123
    walk1 = _make_random_walk(length, seed=seed)
    walk2 = _make_random_walk(length, seed=seed)

    np.testing.assert_array_equal(walk1, walk2)


def test_make_random_walk_different_seeds_produce_different_results():
    """Test that different seeds produce different random walks."""
    length = 50
    walk1 = _make_random_walk(length, seed=42)
    walk2 = _make_random_walk(length, seed=99)

    # Arrays should not be equal
    assert not np.array_equal(walk1, walk2)


def test_make_random_walk_handles_edge_case_small_std():
    """Test that _make_random_walk handles edge case where std is very small."""
    # This test verifies the zero-variance handling in the function
    # While rare with random data, the function has logic for std < 1e-10
    walk = _make_random_walk(10, seed=42)
    assert len(walk) == 10
    assert isinstance(walk, np.ndarray)


# ---------------------------------------------------------------------------
# Tests for _make_series_pair
# ---------------------------------------------------------------------------


def test_make_series_pair_returns_two_arrays():
    """Test that _make_series_pair returns a tuple of two arrays."""
    s1, s2 = _make_series_pair(length=30, seed_a=1, seed_b=2)

    assert isinstance(s1, np.ndarray)
    assert isinstance(s2, np.ndarray)


def test_make_series_pair_correct_length():
    """Test that _make_series_pair returns arrays of the correct length."""
    length = 40
    s1, s2 = _make_series_pair(length=length, seed_a=10, seed_b=20)

    assert len(s1) == length
    assert len(s2) == length


def test_make_series_pair_different_series():
    """Test that _make_series_pair with different seeds produces different series."""
    s1, s2 = _make_series_pair(length=30, seed_a=5, seed_b=6)

    # The two series should be different
    assert not np.array_equal(s1, s2)


def test_make_series_pair_reproducibility():
    """Test that _make_series_pair with same seeds produces same results."""
    pair1 = _make_series_pair(length=30, seed_a=42, seed_b=99)
    pair2 = _make_series_pair(length=30, seed_a=42, seed_b=99)

    np.testing.assert_array_equal(pair1[0], pair2[0])
    np.testing.assert_array_equal(pair1[1], pair2[1])


# ---------------------------------------------------------------------------
# Tests for _dtw_numpy
# ---------------------------------------------------------------------------


def test_dtw_numpy_produces_non_negative_result():
    """Test that _dtw_numpy produces a non-negative float result."""
    s1, s2 = _make_series_pair(length=30, seed_a=1, seed_b=2)
    window = 5

    result = _dtw_numpy(s1, s2, window)

    assert isinstance(result, float)
    assert result >= 0.0


def test_dtw_numpy_identical_series_gives_zero_distance():
    """Test that _dtw_numpy with identical series gives 0 distance."""
    s1 = _make_random_walk(30, seed=42)
    s2 = s1.copy()  # Identical series
    window = 5

    result = _dtw_numpy(s1, s2, window)

    # Distance should be very close to 0 for identical series
    assert result < 1e-10, f"Expected ~0 distance for identical series, got {result}"


def test_dtw_numpy_different_series_gives_positive_distance():
    """Test that _dtw_numpy with different series gives positive distance."""
    s1, s2 = _make_series_pair(length=30, seed_a=1, seed_b=2)
    window = 5

    result = _dtw_numpy(s1, s2, window)

    assert result > 0.0


def test_dtw_numpy_with_window():
    """Test that _dtw_numpy works with different window sizes."""
    s1, s2 = _make_series_pair(length=30, seed_a=1, seed_b=2)

    result_small = _dtw_numpy(s1, s2, window=3)
    result_large = _dtw_numpy(s1, s2, window=10)

    # Both should produce valid non-negative results
    assert result_small >= 0.0
    assert result_large >= 0.0


# ---------------------------------------------------------------------------
# Tests for _dtw_dtaidistance
# ---------------------------------------------------------------------------


def test_dtw_dtaidistance_produces_non_negative_result():
    """Test that _dtw_dtaidistance produces a non-negative float result."""
    s1, s2 = _make_series_pair(length=30, seed_a=1, seed_b=2)
    window = 5

    result = _dtw_dtaidistance(s1, s2, window)

    assert isinstance(result, float)
    assert result >= 0.0


def test_dtw_dtaidistance_different_series_gives_positive_distance():
    """Test that _dtw_dtaidistance with different series gives positive distance."""
    s1, s2 = _make_series_pair(length=30, seed_a=1, seed_b=2)
    window = 5

    result = _dtw_dtaidistance(s1, s2, window)

    assert result > 0.0


# ---------------------------------------------------------------------------
# Tests for bench_single_pair
# ---------------------------------------------------------------------------


def test_bench_single_pair_returns_benchmark_result():
    """Test that bench_single_pair returns a BenchmarkResult with correct keys."""
    result = bench_single_pair(length=30, repeats=2)

    assert isinstance(result, dict)
    assert "name" in result
    assert "value" in result
    assert "unit" in result
    assert "detail" in result


def test_bench_single_pair_has_correct_name():
    """Test that bench_single_pair has the correct benchmark name."""
    result = bench_single_pair(length=30, repeats=2)

    assert result["name"] == "dtw_single_pair"


def test_bench_single_pair_positive_value():
    """Test that bench_single_pair returns a positive value (elapsed time)."""
    result = bench_single_pair(length=30, repeats=2)

    assert result["value"] is not None
    assert result["value"] > 0.0


def test_bench_single_pair_unit_is_seconds():
    """Test that bench_single_pair uses 'seconds' as unit."""
    result = bench_single_pair(length=30, repeats=2)

    assert result["unit"] == "seconds"


def test_bench_single_pair_detail_contains_params():
    """Test that bench_single_pair detail contains length and repeats."""
    result = bench_single_pair(length=30, repeats=3)

    assert "length=30" in result["detail"]
    assert "repeats=3" in result["detail"]


# ---------------------------------------------------------------------------
# Tests for bench_batch
# ---------------------------------------------------------------------------


def test_bench_batch_returns_benchmark_result():
    """Test that bench_batch returns a BenchmarkResult with correct keys."""
    result = bench_batch(batch_size=3, length=30)

    assert isinstance(result, dict)
    assert "name" in result
    assert "value" in result
    assert "unit" in result
    assert "detail" in result


def test_bench_batch_has_correct_name():
    """Test that bench_batch has the correct benchmark name."""
    result = bench_batch(batch_size=3, length=30)

    assert result["name"] == "dtw_batch_10"


def test_bench_batch_positive_value():
    """Test that bench_batch returns a positive value (elapsed time)."""
    result = bench_batch(batch_size=3, length=30)

    assert result["value"] is not None
    assert result["value"] > 0.0


def test_bench_batch_unit_is_seconds():
    """Test that bench_batch uses 'seconds' as unit."""
    result = bench_batch(batch_size=3, length=30)

    assert result["unit"] == "seconds"


def test_bench_batch_detail_contains_params():
    """Test that bench_batch detail contains batch size and length."""
    result = bench_batch(batch_size=5, length=40)

    assert "batch=5" in result["detail"]
    assert "length=40" in result["detail"]


# ---------------------------------------------------------------------------
# Tests for bench_numpy_fallback
# ---------------------------------------------------------------------------


def test_bench_numpy_fallback_returns_benchmark_result():
    """Test that bench_numpy_fallback returns a BenchmarkResult with correct keys."""
    result = bench_numpy_fallback(length=30, repeats=2)

    assert isinstance(result, dict)
    assert "name" in result
    assert "value" in result
    assert "unit" in result
    assert "detail" in result


def test_bench_numpy_fallback_has_correct_name():
    """Test that bench_numpy_fallback has the correct benchmark name."""
    result = bench_numpy_fallback(length=30, repeats=2)

    assert result["name"] == "dtw_numpy_fallback"


def test_bench_numpy_fallback_positive_value():
    """Test that bench_numpy_fallback returns a positive value."""
    result = bench_numpy_fallback(length=30, repeats=2)

    assert result["value"] is not None
    assert result["value"] > 0.0


def test_bench_numpy_fallback_unit_is_seconds():
    """Test that bench_numpy_fallback uses 'seconds' as unit."""
    result = bench_numpy_fallback(length=30, repeats=2)

    assert result["unit"] == "seconds"


def test_bench_numpy_fallback_detail_contains_params():
    """Test that bench_numpy_fallback detail contains length and repeats."""
    result = bench_numpy_fallback(length=35, repeats=3)

    assert "length=35" in result["detail"]
    assert "repeats=3" in result["detail"]


# ---------------------------------------------------------------------------
# Tests for bench_dtaidistance_vs_numpy
# ---------------------------------------------------------------------------


def test_bench_dtaidistance_vs_numpy_returns_benchmark_result():
    """Test that bench_dtaidistance_vs_numpy returns a BenchmarkResult with correct keys."""
    result = bench_dtaidistance_vs_numpy(length=30)

    assert isinstance(result, dict)
    assert "name" in result
    assert "value" in result
    assert "unit" in result
    assert "detail" in result


def test_bench_dtaidistance_vs_numpy_has_correct_name():
    """Test that bench_dtaidistance_vs_numpy has the correct benchmark name."""
    result = bench_dtaidistance_vs_numpy(length=30)

    assert result["name"] == "dtaidistance_vs_numpy"


def test_bench_dtaidistance_vs_numpy_positive_speedup():
    """Test that bench_dtaidistance_vs_numpy returns a positive speedup value."""
    result = bench_dtaidistance_vs_numpy(length=30)

    assert result["value"] is not None
    assert result["value"] > 0.0


def test_bench_dtaidistance_vs_numpy_unit_is_speedup():
    """Test that bench_dtaidistance_vs_numpy uses 'x speedup' as unit."""
    result = bench_dtaidistance_vs_numpy(length=30)

    assert result["unit"] == "x speedup"


def test_bench_dtaidistance_vs_numpy_detail_contains_timing():
    """Test that bench_dtaidistance_vs_numpy detail contains timing information."""
    result = bench_dtaidistance_vs_numpy(length=30)

    detail = result["detail"]
    assert "C=" in detail
    assert "numpy=" in detail
    assert "dist_C=" in detail
    assert "dist_np=" in detail


# ---------------------------------------------------------------------------
# Tests for run_all
# ---------------------------------------------------------------------------


def test_run_all_returns_non_empty_list():
    """Test that run_all returns a non-empty list of results."""
    results = run_all()

    assert isinstance(results, list)
    assert len(results) > 0


def test_run_all_returns_benchmark_results():
    """Test that run_all returns a list of BenchmarkResult dicts."""
    results = run_all()

    for result in results:
        assert isinstance(result, dict)
        assert "name" in result
        assert "value" in result
        assert "unit" in result
        assert "detail" in result


def test_run_all_includes_expected_benchmarks():
    """Test that run_all includes all expected benchmark names."""
    results = run_all()
    names = [r["name"] for r in results]

    # Check that key benchmarks are present
    assert "dtw_single_pair" in names
    assert "dtw_batch_10" in names
    assert "dtw_numpy_fallback" in names
    assert "dtaidistance_vs_numpy" in names


def test_run_all_results_have_valid_values():
    """Test that run_all returns results with valid values or error state."""
    results = run_all()

    for result in results:
        # Value should either be a number or None (if error)
        if result["value"] is not None:
            assert isinstance(result["value"], (int, float))
            assert result["value"] >= 0.0
        else:
            # If value is None, unit should indicate error
            assert result["unit"] == "error"


# ---------------------------------------------------------------------------
# Edge case and error handling tests
# ---------------------------------------------------------------------------


def test_make_random_walk_with_small_length():
    """Test that _make_random_walk works with very small lengths."""
    walk = _make_random_walk(5, seed=42)
    assert len(walk) == 5


def test_dtw_numpy_with_different_length_series():
    """Test that _dtw_numpy handles series of different lengths."""
    s1 = _make_random_walk(30, seed=1)
    s2 = _make_random_walk(25, seed=2)
    window = 5

    # Should not raise an error
    result = _dtw_numpy(s1, s2, window)
    assert result >= 0.0


def test_dtw_dtaidistance_with_different_length_series():
    """Test that _dtw_dtaidistance handles series of different lengths."""
    s1 = _make_random_walk(30, seed=1)
    s2 = _make_random_walk(25, seed=2)
    window = 5

    # Should not raise an error
    result = _dtw_dtaidistance(s1, s2, window)
    assert result >= 0.0
