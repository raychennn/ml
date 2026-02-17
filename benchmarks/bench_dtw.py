"""DTW computation speed benchmarks.

Measures single-pair, batch, dtaidistance vs numpy fallback performance.
Uses synthetic z-normalized series for reproducibility.

Usage::

    python -m benchmarks.bench_dtw
"""

from __future__ import annotations

import time
import sys
from typing import TypedDict

import numpy as np
from numpy.typing import NDArray

from benchmarks.exceptions import BenchmarkRunError, BenchmarkSetupError


class BenchmarkResult(TypedDict):
    """Standard result dict returned by every benchmark function."""

    name: str
    value: float | None
    unit: str
    detail: str


# ---------------------------------------------------------------------------
# Synthetic data generators
# ---------------------------------------------------------------------------


def _make_random_walk(length: int, seed: int = 42) -> NDArray[np.float64]:
    """Generate a z-normalized random walk of *length* steps.

    Args:
        length: Number of data points in the walk.
        seed: Random seed for reproducibility.

    Returns:
        A 1-D numpy array of z-normalized cumulative random walk values.

    Raises:
        BenchmarkSetupError: If the random walk generation fails.
    """
    try:
        rng: np.random.RandomState = np.random.RandomState(seed)
        walk: NDArray[np.float64] = np.cumsum(rng.randn(length))
        std: np.floating = np.std(walk)
        if std < 1e-10:
            return np.zeros(length)
        return (walk - np.mean(walk)) / std
    except Exception as exc:
        raise BenchmarkSetupError(
            f"Failed to generate random walk (length={length}, seed={seed}): {exc}"
        ) from exc


def _make_series_pair(
    length: int = 120,
    seed_a: int = 42,
    seed_b: int = 99,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Create a pair of z-normalized random walk series.

    Args:
        length: Number of data points per series.
        seed_a: Random seed for the first series.
        seed_b: Random seed for the second series.

    Returns:
        A tuple of two 1-D numpy arrays.

    Raises:
        BenchmarkSetupError: If series generation fails.
    """
    try:
        return _make_random_walk(length, seed_a), _make_random_walk(length, seed_b)
    except BenchmarkSetupError:
        raise
    except Exception as exc:
        raise BenchmarkSetupError(
            f"Failed to create series pair (length={length}): {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# DTW via dtaidistance (C library)
# ---------------------------------------------------------------------------


def _dtw_dtaidistance(
    s1: NDArray[np.float64],
    s2: NDArray[np.float64],
    window: int,
) -> float:
    """Compute DTW distance using the dtaidistance C library.

    Args:
        s1: First z-normalized time series.
        s2: Second z-normalized time series.
        window: Sakoe-Chiba band width.

    Returns:
        The DTW distance as a float.
    """
    from dtaidistance import dtw as dtw_lib

    return float(dtw_lib.distance(s1, s2, window=window, use_c=True))


def _dtw_numpy(
    s1: NDArray[np.float64],
    s2: NDArray[np.float64],
    window: int,
) -> float:
    """Pure numpy DTW fallback -- mirrors ``DTWCalculator._dtw_numpy`` logic.

    Args:
        s1: First z-normalized time series.
        s2: Second z-normalized time series.
        window: Sakoe-Chiba band width.

    Returns:
        The normalized DTW cost.
    """
    n: int = len(s1)
    m: int = len(s2)
    cost: NDArray[np.float64] = np.full((n + 1, m + 1), np.inf)
    cost[0, 0] = 0.0
    for i in range(1, n + 1):
        j_start: int = max(1, i - window)
        j_end: int = min(m, i + window)
        for j in range(j_start, j_end + 1):
            d: float = abs(s1[i - 1] - s2[j - 1])
            cost[i, j] = d + min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])
    return float(cost[n, m] / max(n, m))


# ---------------------------------------------------------------------------
# Benchmark runners
# ---------------------------------------------------------------------------


def bench_single_pair(length: int = 120, repeats: int = 50) -> BenchmarkResult:
    """Benchmark a single DTW pair computation with dtaidistance.

    Args:
        length: Length of each synthetic series.
        repeats: Number of timed iterations.

    Returns:
        A ``BenchmarkResult`` dict with average elapsed time per call.

    Raises:
        BenchmarkRunError: If the benchmark execution fails.
    """
    try:
        s1: NDArray[np.float64]
        s2: NDArray[np.float64]
        s1, s2 = _make_series_pair(length)
        window: int = max(1, int(max(len(s1), len(s2)) * 0.12))

        # Warm up
        _dtw_dtaidistance(s1, s2, window)

        t0: float = time.perf_counter()
        for _ in range(repeats):
            _dtw_dtaidistance(s1, s2, window)
        elapsed: float = (time.perf_counter() - t0) / repeats
        return BenchmarkResult(
            name="dtw_single_pair",
            value=elapsed,
            unit="seconds",
            detail=f"length={length}, repeats={repeats}",
        )
    except (BenchmarkSetupError, BenchmarkRunError):
        raise
    except Exception as exc:
        raise BenchmarkRunError(f"bench_single_pair failed: {exc}") from exc


def bench_batch(batch_size: int = 10, length: int = 120) -> BenchmarkResult:
    """Benchmark a batch of DTW pair computations.

    Args:
        batch_size: Number of pairs to compute.
        length: Length of each synthetic series.

    Returns:
        A ``BenchmarkResult`` dict with total elapsed time.

    Raises:
        BenchmarkRunError: If the benchmark execution fails.
    """
    try:
        pairs: list[tuple[NDArray[np.float64], NDArray[np.float64]]] = [
            _make_series_pair(length, seed_a=i, seed_b=i + 100)
            for i in range(batch_size)
        ]
        window: int = max(1, int(length * 0.12))

        # Warm up
        _dtw_dtaidistance(pairs[0][0], pairs[0][1], window)

        t0: float = time.perf_counter()
        for s1, s2 in pairs:
            _dtw_dtaidistance(s1, s2, window)
        elapsed: float = time.perf_counter() - t0
        return BenchmarkResult(
            name="dtw_batch_10",
            value=elapsed,
            unit="seconds",
            detail=f"batch={batch_size}, length={length}",
        )
    except (BenchmarkSetupError, BenchmarkRunError):
        raise
    except Exception as exc:
        raise BenchmarkRunError(f"bench_batch failed: {exc}") from exc


def bench_numpy_fallback(length: int = 120, repeats: int = 10) -> BenchmarkResult:
    """Benchmark the pure-numpy DTW fallback implementation.

    Args:
        length: Length of each synthetic series.
        repeats: Number of timed iterations.

    Returns:
        A ``BenchmarkResult`` dict with average elapsed time per call.

    Raises:
        BenchmarkRunError: If the benchmark execution fails.
    """
    try:
        s1: NDArray[np.float64]
        s2: NDArray[np.float64]
        s1, s2 = _make_series_pair(length)
        window: int = max(1, int(length * 0.12))

        t0: float = time.perf_counter()
        for _ in range(repeats):
            _dtw_numpy(s1, s2, window)
        elapsed: float = (time.perf_counter() - t0) / repeats
        return BenchmarkResult(
            name="dtw_numpy_fallback",
            value=elapsed,
            unit="seconds",
            detail=f"length={length}, repeats={repeats}",
        )
    except (BenchmarkSetupError, BenchmarkRunError):
        raise
    except Exception as exc:
        raise BenchmarkRunError(f"bench_numpy_fallback failed: {exc}") from exc


def bench_dtaidistance_vs_numpy(length: int = 120) -> BenchmarkResult:
    """Compare dtaidistance C library speed against the numpy fallback.

    Args:
        length: Length of each synthetic series.

    Returns:
        A ``BenchmarkResult`` dict with the speedup factor (x).

    Raises:
        BenchmarkRunError: If the benchmark execution fails.
    """
    try:
        s1: NDArray[np.float64]
        s2: NDArray[np.float64]
        s1, s2 = _make_series_pair(length)
        window: int = max(1, int(length * 0.12))

        t_c: float = time.perf_counter()
        d_c: float = _dtw_dtaidistance(s1, s2, window)
        t_c = time.perf_counter() - t_c

        t_np: float = time.perf_counter()
        d_np: float = _dtw_numpy(s1, s2, window)
        t_np = time.perf_counter() - t_np

        speedup: float = t_np / t_c if t_c > 0 else float("inf")
        return BenchmarkResult(
            name="dtaidistance_vs_numpy",
            value=speedup,
            unit="x speedup",
            detail=f"C={t_c:.6f}s, numpy={t_np:.6f}s, dist_C={d_c:.4f}, dist_np={d_np:.4f}",
        )
    except (BenchmarkSetupError, BenchmarkRunError):
        raise
    except Exception as exc:
        raise BenchmarkRunError(f"bench_dtaidistance_vs_numpy failed: {exc}") from exc


def run_all() -> list[BenchmarkResult]:
    """Execute every DTW benchmark and return collected results.

    Returns:
        A list of ``BenchmarkResult`` dicts, one per benchmark (including
        error entries for benchmarks that failed).
    """
    results: list[BenchmarkResult] = []
    print("=" * 60)
    print("DTW Benchmark Suite")
    print("=" * 60)

    bench_fns: list = [
        bench_single_pair,
        bench_batch,
        bench_numpy_fallback,
        bench_dtaidistance_vs_numpy,
    ]
    for bench_fn in bench_fns:
        try:
            r: BenchmarkResult = bench_fn()
            results.append(r)
            print(
                f"  {r['name']:30s} {r['value']:.6f} {r['unit']:15s}  ({r['detail']})"
            )
        except Exception as e:
            print(f"  {bench_fn.__name__:30s} ERROR: {e}")
            results.append(
                BenchmarkResult(
                    name=bench_fn.__name__,
                    value=None,
                    unit="error",
                    detail=str(e),
                )
            )

    print()
    return results


if __name__ == "__main__":
    run_all()
