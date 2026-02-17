"""Scaling Tests -- measure how performance changes with scale.

Tests scaling curves for:

* Symbol count: 10 to 200
* Reference count: 1 to 20
* Data volume: 100 to 10 000 rows per symbol
* Worker count: 1 to 8 (multiprocessing)

Usage::

    python -m benchmarks.scaling_tests
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import time
from typing import Any, TypedDict

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


def _setup_data(
    tmpdir: str,
    n_symbols: int,
    n_bars: int = 500,
    timeframe: str = "4h",
) -> list[str]:
    """Create synthetic Parquet files and supporting directories.

    Args:
        tmpdir: Path to the temporary root directory.
        n_symbols: Number of synthetic symbols to generate.
        n_bars: Number of bars per symbol.
        timeframe: Timeframe label used in the directory structure.

    Returns:
        A list of generated symbol names.

    Raises:
        BenchmarkSetupError: If directory creation or Parquet writing fails.
    """
    try:
        tf_dir: str = os.path.join(tmpdir, "parquet", f"timeframe={timeframe}")
        os.makedirs(tf_dir, exist_ok=True)
        os.makedirs(os.path.join(tmpdir, "cache"), exist_ok=True)
        os.makedirs(os.path.join(tmpdir, "models"), exist_ok=True)

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
        return symbols
    except Exception as exc:
        raise BenchmarkSetupError(f"Failed to set up data in {tmpdir}: {exc}") from exc


def _make_ref(
    symbols: list[str],
    seed: int = 0,
    n_bars: int = 500,
) -> Any:
    """Build a ``ReferencePattern`` from synthetic data.

    Args:
        symbols: Available symbol names (used to pick the reference symbol).
        seed: Random seed and reference identifier.
        n_bars: Number of bars in the synthetic source data.

    Returns:
        A ``ReferencePattern`` instance.
    """
    from references.reference_manager import ReferencePattern

    ref_df: pd.DataFrame = _make_synthetic_ohlcv(n_bars, seed=seed)
    return ReferencePattern(
        id=f"scale_ref_{seed}",
        symbol=symbols[min(seed, len(symbols) - 1)],
        timeframe="4h",
        start_ts=int(ref_df["timestamp"].iloc[300]),
        end_ts=int(ref_df["timestamp"].iloc[min(420, n_bars - 1)]),
        label="scale",
        description=f"Scale ref {seed}",
    )


# ---------------------------------------------------------------------------
# Scaling: symbol count
# ---------------------------------------------------------------------------


def scale_symbol_count() -> list[dict[str, Any]]:
    """Measure scan time as the number of symbols increases.

    Returns:
        A list of dicts with keys ``symbols``, ``time``, and ``alerts``.

    Raises:
        BenchmarkRunError: If a scaling step fails during execution.
    """
    print("\n--- Symbol Count Scaling ---")
    from config import SystemConfig
    from core.scanner_engine import ScannerEngine
    from data.data_manager import DataManager

    counts: list[int] = [10, 25, 50, 100, 200]
    results: list[dict[str, Any]] = []

    for n in counts:
        tmpdir: str = tempfile.mkdtemp()
        try:
            symbols: list[str] = _setup_data(tmpdir, n, n_bars=500)
            config: SystemConfig = SystemConfig()
            config.data_root = tmpdir
            data_mgr: DataManager = DataManager(config)
            scanner: ScannerEngine = ScannerEngine(config, data_mgr)
            ref = _make_ref(symbols, seed=0)

            t0: float = time.perf_counter()
            alerts = scanner.scan_timeframe("4h", [ref])
            elapsed: float = time.perf_counter() - t0

            results.append({"symbols": n, "time": elapsed, "alerts": len(alerts)})
            print(f"  {n:4d} symbols: {elapsed:.3f}s  ({len(alerts)} alerts)")
        except BenchmarkSetupError:
            raise
        except Exception as e:
            print(f"  {n:4d} symbols: ERROR - {e}")
        finally:
            shutil.rmtree(tmpdir)

    return results


# ---------------------------------------------------------------------------
# Scaling: reference count
# ---------------------------------------------------------------------------


def scale_reference_count() -> list[dict[str, Any]]:
    """Measure scan time as the number of reference patterns increases.

    Returns:
        A list of dicts with keys ``refs``, ``time``, and ``alerts``.

    Raises:
        BenchmarkRunError: If a scaling step fails during execution.
    """
    print("\n--- Reference Count Scaling ---")
    from config import SystemConfig
    from core.scanner_engine import ScannerEngine
    from data.data_manager import DataManager

    ref_counts: list[int] = [1, 5, 10, 20]
    n_symbols: int = 30
    results: list[dict[str, Any]] = []

    tmpdir: str = tempfile.mkdtemp()
    try:
        symbols: list[str] = _setup_data(tmpdir, n_symbols, n_bars=500)
        config: SystemConfig = SystemConfig()
        config.data_root = tmpdir
        data_mgr: DataManager = DataManager(config)
        scanner: ScannerEngine = ScannerEngine(config, data_mgr)

        for n_refs in ref_counts:
            refs = [_make_ref(symbols, seed=i) for i in range(n_refs)]

            t0: float = time.perf_counter()
            alerts = scanner.scan_timeframe("4h", refs)
            elapsed: float = time.perf_counter() - t0

            results.append({"refs": n_refs, "time": elapsed, "alerts": len(alerts)})
            print(f"  {n_refs:3d} references: {elapsed:.3f}s  ({len(alerts)} alerts)")
    except BenchmarkSetupError:
        raise
    except Exception as e:
        print(f"  ERROR: {e}")
    finally:
        shutil.rmtree(tmpdir)

    return results


# ---------------------------------------------------------------------------
# Scaling: data volume (rows per symbol)
# ---------------------------------------------------------------------------


def scale_data_volume() -> list[dict[str, Any]]:
    """Measure normalization time as per-symbol row count increases.

    Returns:
        A list of dicts with keys ``rows`` and ``time``.

    Raises:
        BenchmarkRunError: If a scaling step fails during execution.
    """
    print("\n--- Data Volume Scaling ---")
    from config import SystemConfig
    from core.data_normalizer import DataNormalizer

    row_counts: list[int] = [100, 500, 1000, 5000, 10000]
    results: list[dict[str, Any]] = []

    config: SystemConfig = SystemConfig()
    normalizer: DataNormalizer = DataNormalizer(config)

    for n_rows in row_counts:
        df: pd.DataFrame = _make_synthetic_ohlcv(n_rows, seed=42)

        t0: float = time.perf_counter()
        for _ in range(20):
            normalizer.prepare_for_dtw(df)
        elapsed: float = (time.perf_counter() - t0) / 20

        results.append({"rows": n_rows, "time": elapsed})
        print(f"  {n_rows:6d} rows: {elapsed*1000:.3f} ms")

    return results


# ---------------------------------------------------------------------------
# Scaling: worker count
# ---------------------------------------------------------------------------


def scale_worker_count() -> list[dict[str, Any]]:
    """Measure scan time as the number of worker processes increases.

    Returns:
        A list of dicts with keys ``workers``, ``time``, and ``alerts``.

    Raises:
        BenchmarkRunError: If a scaling step fails during execution.
    """
    print("\n--- Worker Count Scaling ---")
    from config import SystemConfig
    from core.scanner_engine import ScannerEngine
    from data.data_manager import DataManager

    worker_counts: list[int] = [1, 2, 4, 8]
    n_symbols: int = 50
    results: list[dict[str, Any]] = []

    tmpdir: str = tempfile.mkdtemp()
    try:
        symbols: list[str] = _setup_data(tmpdir, n_symbols, n_bars=500)
        config: SystemConfig = SystemConfig()
        config.data_root = tmpdir
        data_mgr: DataManager = DataManager(config)

        ref = _make_ref(symbols, seed=0)

        for n_workers in worker_counts:
            scanner: ScannerEngine = ScannerEngine(config, data_mgr)

            t0: float = time.perf_counter()
            alerts = scanner.scan_timeframe("4h", [ref], num_workers=n_workers)
            elapsed: float = time.perf_counter() - t0

            results.append(
                {"workers": n_workers, "time": elapsed, "alerts": len(alerts)}
            )
            print(f"  {n_workers} workers: {elapsed:.3f}s  ({len(alerts)} alerts)")
    except BenchmarkSetupError:
        raise
    except Exception as e:
        print(f"  ERROR: {e}")
    finally:
        shutil.rmtree(tmpdir)

    return results


def main() -> None:
    """Run all scaling tests and print a summary.

    Returns:
        None.
    """
    print("=" * 60)
    print("  SCALING TEST SUITE")
    print("=" * 60)

    all_results: dict[str, list[dict[str, Any]]] = {}

    for name, fn in [
        ("symbol_count", scale_symbol_count),
        ("reference_count", scale_reference_count),
        ("data_volume", scale_data_volume),
        ("worker_count", scale_worker_count),
    ]:
        try:
            all_results[name] = fn()
        except Exception as e:
            print(f"\n  [{name}] ERROR: {e}")
            import traceback

            traceback.print_exc()

    print("\n" + "=" * 60)
    print("  Scaling Summary")
    print("=" * 60)
    for name, results in all_results.items():
        print(f"\n  {name}:")
        for r in results:
            print(f"    {r}")


if __name__ == "__main__":
    main()
