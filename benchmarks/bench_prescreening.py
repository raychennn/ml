"""Stage 1 prescreening throughput benchmarks.

Measures symbols/sec for z-normalized Euclidean distance prescreening
using synthetic OHLCV data.

Usage::

    python -m benchmarks.bench_prescreening
"""

from __future__ import annotations

import time
import sys
from typing import TypedDict

import numpy as np
import pandas as pd
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


def _make_synthetic_ohlcv(n_bars: int = 500, seed: int = 42) -> pd.DataFrame:
    """Generate a synthetic OHLCV DataFrame.

    Args:
        n_bars: Number of bars (rows) to generate.
        seed: Random seed for reproducibility.

    Returns:
        A DataFrame with columns: timestamp, open, high, low, close, volume.

    Raises:
        BenchmarkSetupError: If data generation fails.
    """
    try:
        rng: np.random.RandomState = np.random.RandomState(seed)
        close: NDArray[np.float64] = 100.0 + np.cumsum(rng.randn(n_bars) * 0.5)
        high: NDArray[np.float64] = close + rng.uniform(0.1, 1.0, n_bars)
        low: NDArray[np.float64] = close - rng.uniform(0.1, 1.0, n_bars)
        open_: NDArray[np.float64] = close + rng.randn(n_bars) * 0.3
        volume: NDArray[np.float64] = rng.uniform(100, 10000, n_bars)
        ts: NDArray[np.int64] = np.arange(1700000000, 1700000000 + n_bars * 3600, 3600)[
            :n_bars
        ]
        return pd.DataFrame(
            {
                "timestamp": ts,
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
            }
        )
    except Exception as exc:
        raise BenchmarkSetupError(
            f"Failed to generate synthetic OHLCV (n_bars={n_bars}, seed={seed}): {exc}"
        ) from exc


def _z_normalize(arr: NDArray[np.float64]) -> NDArray[np.float64]:
    """Z-normalize a 1-D array in place (zero mean, unit variance).

    Args:
        arr: Input array of float values.

    Returns:
        A z-normalized copy of *arr*, or zeros if std is negligible.
    """
    std: np.floating = np.std(arr)
    if std < 1e-10:
        return np.zeros_like(arr)
    return (arr - np.mean(arr)) / std


def _prescreening_distance(
    ref_znorm: NDArray[np.float64],
    target_close: NDArray[np.float64],
    ref_len: int,
) -> float:
    """Compute prescreening Euclidean distance between reference and target tail.

    Args:
        ref_znorm: Z-normalized reference close series.
        target_close: Raw close prices for the target symbol.
        ref_len: Length of the reference window (number of bars).

    Returns:
        Normalized Euclidean distance as a float.
    """
    tail: NDArray[np.float64] = target_close[-ref_len:]
    tail_znorm: NDArray[np.float64] = _z_normalize(tail)
    return float(np.sqrt(np.sum((ref_znorm - tail_znorm) ** 2)) / ref_len)


# ---------------------------------------------------------------------------
# Benchmark runners
# ---------------------------------------------------------------------------


def bench_prescreening(
    n_symbols: int = 100,
    ref_len: int = 120,
    n_bars: int = 500,
) -> list[BenchmarkResult]:
    """Benchmark prescreening throughput for a set of symbols.

    Args:
        n_symbols: Number of synthetic symbols to screen.
        ref_len: Length of the reference pattern in bars.
        n_bars: Number of bars per synthetic symbol.

    Returns:
        A list of two ``BenchmarkResult`` dicts: total time and
        symbols-per-second throughput.

    Raises:
        BenchmarkRunError: If the benchmark execution fails.
    """
    try:
        # Build synthetic symbol data
        symbol_data: dict[str, NDArray[np.float64]] = {}
        for i in range(n_symbols):
            df: pd.DataFrame = _make_synthetic_ohlcv(n_bars, seed=i)
            symbol_data[f"SYM{i:04d}USDT"] = df["close"].values

        ref_close: NDArray[np.float64] = _make_synthetic_ohlcv(ref_len, seed=9999)[
            "close"
        ].values
        ref_znorm: NDArray[np.float64] = _z_normalize(ref_close)

        # Warm up
        _prescreening_distance(ref_znorm, list(symbol_data.values())[0], ref_len)

        t0: float = time.perf_counter()
        distances: dict[str, float] = {}
        for sym, close_arr in symbol_data.items():
            if len(close_arr) >= ref_len:
                distances[sym] = _prescreening_distance(ref_znorm, close_arr, ref_len)
        elapsed: float = time.perf_counter() - t0

        symbols_per_sec: float = n_symbols / elapsed if elapsed > 0 else float("inf")

        return [
            BenchmarkResult(
                name=f"prescreening_{n_symbols}_symbols",
                value=elapsed,
                unit="seconds",
                detail=f"n_symbols={n_symbols}, ref_len={ref_len}, n_bars={n_bars}",
            ),
            BenchmarkResult(
                name="prescreening_symbols_per_sec",
                value=symbols_per_sec,
                unit="symbols/sec",
                detail=f"n_symbols={n_symbols}",
            ),
        ]
    except (BenchmarkSetupError, BenchmarkRunError):
        raise
    except Exception as exc:
        raise BenchmarkRunError(f"bench_prescreening failed: {exc}") from exc


def bench_prescreening_scaling() -> list[BenchmarkResult]:
    """Test prescreening throughput across increasing symbol counts.

    Returns:
        A list of ``BenchmarkResult`` dicts, one per symbol-count tier.

    Raises:
        BenchmarkRunError: If the benchmark execution fails.
    """
    try:
        results: list[BenchmarkResult] = []
        for n in [50, 100, 200]:
            t0: float = time.perf_counter()
            bench_prescreening(n_symbols=n)
            elapsed: float = time.perf_counter() - t0
            results.append(
                BenchmarkResult(
                    name=f"prescreening_scale_{n}",
                    value=elapsed,
                    unit="seconds",
                    detail=f"n_symbols={n}",
                )
            )
        return results
    except (BenchmarkSetupError, BenchmarkRunError):
        raise
    except Exception as exc:
        raise BenchmarkRunError(f"bench_prescreening_scaling failed: {exc}") from exc


def run_all() -> list[BenchmarkResult]:
    """Execute every prescreening benchmark and return collected results.

    Returns:
        A list of ``BenchmarkResult`` dicts for all benchmarks.
    """
    results: list[BenchmarkResult] = []
    print("=" * 60)
    print("Prescreening Benchmark Suite")
    print("=" * 60)

    for r in bench_prescreening(100):
        results.append(r)
        print(f"  {r['name']:40s} {r['value']:.6f} {r['unit']:15s}  ({r['detail']})")

    for r in bench_prescreening_scaling():
        results.append(r)
        print(f"  {r['name']:40s} {r['value']:.6f} {r['unit']:15s}  ({r['detail']})")

    print()
    return results


if __name__ == "__main__":
    run_all()
