"""Benchmark: Full scan cycle end-to-end.

Runs a complete scan cycle with synthetic data: prescreening, DTW, and ML
scoring.  No Binance API access is needed.

Usage::

    python -m benchmarks.bench_scan_cycle
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import time
from typing import TypedDict

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from benchmarks.exceptions import BenchmarkRunError, BenchmarkSetupError

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_SCHEMA: pa.Schema = pa.schema(
    [
        ("timestamp", pa.int64()),
        ("open", pa.float64()),
        ("high", pa.float64()),
        ("low", pa.float64()),
        ("close", pa.float64()),
        ("volume", pa.float64()),
    ]
)


class BenchmarkResult(TypedDict):
    """Standard result payload returned by every benchmark function."""

    name: str
    value: float | None
    unit: str
    detail: str


def _make_synthetic_ohlcv(n_bars: int = 500, seed: int = 42) -> pd.DataFrame:
    """Generate a synthetic OHLCV DataFrame for benchmarking.

    Args:
        n_bars: Number of candlestick bars to generate.
        seed: Random seed for reproducibility.

    Returns:
        A DataFrame with columns ``timestamp``, ``open``, ``high``,
        ``low``, ``close``, and ``volume``.
    """
    rng: np.random.RandomState = np.random.RandomState(seed)
    close: np.ndarray = 100.0 + np.cumsum(rng.randn(n_bars) * 0.5)
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


def _setup_synthetic_data(
    tmpdir: str,
    n_symbols: int = 20,
    n_bars: int = 500,
    timeframe: str = "4h",
) -> tuple[list[str], int, int]:
    """Create synthetic Parquet files in a temporary data directory.

    Args:
        tmpdir: Path to the temporary root directory.
        n_symbols: Number of synthetic symbols to generate.
        n_bars: Number of bars per symbol.
        timeframe: Timeframe label used in the directory structure.

    Returns:
        A tuple of ``(symbols, ref_start_ts, ref_end_ts)`` where *symbols*
        is the list of generated symbol names and the timestamps delimit a
        reference window inside the first symbol's data.

    Raises:
        BenchmarkSetupError: If directory creation or Parquet writing fails.
    """
    try:
        tf_dir: str = os.path.join(tmpdir, "parquet", f"timeframe={timeframe}")
        os.makedirs(tf_dir, exist_ok=True)

        symbols: list[str] = []
        for i in range(n_symbols):
            sym: str = f"SYM{i:04d}USDT"
            symbols.append(sym)
            df: pd.DataFrame = _make_synthetic_ohlcv(n_bars, seed=i)
            table: pa.Table = pa.Table.from_pandas(
                df, schema=_SCHEMA, preserve_index=False
            )
            pq.write_table(
                table, os.path.join(tf_dir, f"{sym}.parquet"), compression="snappy"
            )

        # Create a "reference" from the first symbol's data
        ref_df: pd.DataFrame = _make_synthetic_ohlcv(n_bars, seed=0)
        ref_start: int = int(ref_df["timestamp"].iloc[300])
        ref_end: int = int(ref_df["timestamp"].iloc[420])

        return symbols, ref_start, ref_end
    except Exception as exc:
        raise BenchmarkSetupError(
            f"Failed to set up synthetic data in {tmpdir}: {exc}"
        ) from exc


def bench_scan_cycle(
    n_symbols: int = 20,
    n_bars: int = 500,
    timeframe: str = "4h",
) -> BenchmarkResult:
    """Run a full scan cycle benchmark with synthetic data.

    The benchmark covers the entire pipeline: data loading, prescreening,
    DTW matching, and ML scoring.

    Args:
        n_symbols: Number of synthetic symbols to scan.
        n_bars: Number of bars per symbol.
        timeframe: Timeframe label for the scan.

    Returns:
        A :class:`BenchmarkResult` containing elapsed time in seconds.

    Raises:
        BenchmarkRunError: If the scan cycle fails during execution.
    """
    from config import SystemConfig
    from core.scanner_engine import ScannerEngine
    from data.data_manager import DataManager
    from references.reference_manager import ReferencePattern

    tmpdir: str = tempfile.mkdtemp()
    try:
        symbols, ref_start, ref_end = _setup_synthetic_data(
            tmpdir, n_symbols, n_bars, timeframe
        )

        config: SystemConfig = SystemConfig()
        config.data_root = tmpdir
        os.makedirs(os.path.join(tmpdir, "cache"), exist_ok=True)
        os.makedirs(os.path.join(tmpdir, "models"), exist_ok=True)

        data_mgr: DataManager = DataManager(config)
        scanner: ScannerEngine = ScannerEngine(config, data_mgr)

        ref: ReferencePattern = ReferencePattern(
            id="bench_ref",
            symbol=symbols[0],
            timeframe=timeframe,
            start_ts=ref_start,
            end_ts=ref_end,
            label="bench",
            description="Benchmark reference",
        )

        # Warm up: load data once
        data_mgr.bulk_read_timeframe(timeframe, tail_bars=200)

        t0: float = time.perf_counter()
        alerts = scanner.scan_timeframe(timeframe, [ref])
        elapsed: float = time.perf_counter() - t0

        return BenchmarkResult(
            name="scan_cycle_20_symbols",
            value=elapsed,
            unit="seconds",
            detail=f"symbols={n_symbols}, bars={n_bars}, alerts={len(alerts)}",
        )
    except BenchmarkSetupError:
        raise
    except Exception as exc:
        raise BenchmarkRunError(f"bench_scan_cycle failed: {exc}") from exc
    finally:
        shutil.rmtree(tmpdir)


def bench_prepare_for_dtw(repeats: int = 100) -> BenchmarkResult:
    """Benchmark ``DataNormalizer.prepare_for_dtw``.

    Args:
        repeats: Number of iterations to average over.

    Returns:
        A :class:`BenchmarkResult` with the mean elapsed time per call.

    Raises:
        BenchmarkRunError: If the normalization benchmark fails.
    """
    try:
        from config import SystemConfig
        from core.data_normalizer import DataNormalizer

        config: SystemConfig = SystemConfig()
        normalizer: DataNormalizer = DataNormalizer(config)
        df: pd.DataFrame = _make_synthetic_ohlcv(500, seed=42)

        # Warm up
        normalizer.prepare_for_dtw(df)

        t0: float = time.perf_counter()
        for _ in range(repeats):
            normalizer.prepare_for_dtw(df)
        elapsed: float = (time.perf_counter() - t0) / repeats

        return BenchmarkResult(
            name="prepare_for_dtw",
            value=elapsed,
            unit="seconds",
            detail=f"rows=500, repeats={repeats}",
        )
    except Exception as exc:
        raise BenchmarkRunError(f"bench_prepare_for_dtw failed: {exc}") from exc


def run_all() -> list[BenchmarkResult]:
    """Execute every benchmark in this module and return results.

    Returns:
        A list of :class:`BenchmarkResult` dictionaries, one per benchmark.
    """
    results: list[BenchmarkResult] = []
    print("=" * 60)
    print("Scan Cycle Benchmark Suite")
    print("=" * 60)

    for bench_fn in [bench_scan_cycle, bench_prepare_for_dtw]:
        try:
            r: BenchmarkResult = bench_fn()
            results.append(r)
            print(
                f"  {r['name']:30s} {r['value']:.6f} {r['unit']:15s}  ({r['detail']})"
            )
        except Exception as e:
            import traceback

            print(f"  {bench_fn.__name__:30s} ERROR: {e}")
            traceback.print_exc()
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
