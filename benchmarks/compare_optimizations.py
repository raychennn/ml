"""Optimization Comparison -- A/B benchmarks.

Compares alternative implementations to quantify optimization value:

* dtaidistance (C) vs numpy DTW
* With vs without prescreening
* Multiprocessing vs sequential DTW
* Snappy vs uncompressed Parquet
* BTC cache hit vs miss

Usage::

    python -m benchmarks.compare_optimizations
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


def _z_normalize(arr: np.ndarray) -> np.ndarray:
    """Z-normalize an array to zero mean and unit variance.

    Args:
        arr: Input numerical array.

    Returns:
        The z-normalized array, or an array of zeros if the standard
        deviation is negligible.
    """
    std: float = np.std(arr)
    if std < 1e-10:
        return np.zeros_like(arr)
    return (arr - np.mean(arr)) / std


# ---------------------------------------------------------------------------
# Comparison 1: dtaidistance vs numpy DTW
# ---------------------------------------------------------------------------


def compare_dtw_implementations() -> dict[str, Any]:
    """Compare dtaidistance (C extension) against a pure-numpy DTW fallback.

    Returns:
        A dict with keys ``comparison`` and ``speedup``.

    Raises:
        BenchmarkRunError: If the comparison fails during execution.
    """
    try:
        print("\n--- dtaidistance (C) vs numpy DTW ---")
        length: int = 120
        rng: np.random.RandomState = np.random.RandomState(42)
        s1: np.ndarray = _z_normalize(np.cumsum(rng.randn(length)))
        s2: np.ndarray = _z_normalize(np.cumsum(rng.randn(length)))
        window: int = max(1, int(length * 0.12))

        # dtaidistance
        from dtaidistance import dtw as dtw_lib

        dtw_lib.distance(s1, s2, window=window, use_c=True)  # warm up
        t0: float = time.perf_counter()
        for _ in range(100):
            dtw_lib.distance(s1, s2, window=window, use_c=True)
        t_c: float = (time.perf_counter() - t0) / 100

        # numpy fallback
        from benchmarks.bench_dtw import _dtw_numpy

        _dtw_numpy(s1, s2, window)  # warm up
        t0 = time.perf_counter()
        for _ in range(10):
            _dtw_numpy(s1, s2, window)
        t_np: float = (time.perf_counter() - t0) / 10

        speedup: float = t_np / t_c if t_c > 0 else float("inf")
        print(f"  dtaidistance (C): {t_c*1000:.3f} ms")
        print(f"  numpy fallback:   {t_np*1000:.3f} ms")
        print(f"  Speedup:          {speedup:.1f}x")
        return {"comparison": "dtw_c_vs_numpy", "speedup": speedup}
    except Exception as exc:
        raise BenchmarkRunError(f"compare_dtw_implementations failed: {exc}") from exc


# ---------------------------------------------------------------------------
# Comparison 2: with vs without prescreening
# ---------------------------------------------------------------------------


def compare_prescreening() -> dict[str, Any]:
    """Compare scan performance with and without prescreening.

    Returns:
        A dict with keys ``comparison`` and ``speedup``.

    Raises:
        BenchmarkRunError: If the comparison fails during execution.
    """
    from config import SystemConfig
    from core.scanner_engine import ScannerEngine
    from data.data_manager import DataManager
    from references.reference_manager import ReferencePattern

    print("\n--- With vs without prescreening ---")
    n_symbols: int = 40
    tmpdir: str = tempfile.mkdtemp()
    try:
        tf_dir: str = os.path.join(tmpdir, "parquet", "timeframe=4h")
        os.makedirs(tf_dir, exist_ok=True)
        os.makedirs(os.path.join(tmpdir, "cache"), exist_ok=True)
        os.makedirs(os.path.join(tmpdir, "models"), exist_ok=True)

        symbols: list[str] = []
        for i in range(n_symbols):
            sym: str = f"SYM{i:04d}USDT"
            symbols.append(sym)
            df: pd.DataFrame = _make_synthetic_ohlcv(500, seed=i)
            table: pa.Table = pa.Table.from_pandas(
                df, schema=_SCHEMA, preserve_index=False
            )
            pq.write_table(
                table, os.path.join(tf_dir, f"{sym}.parquet"), compression="snappy"
            )

        ref_df: pd.DataFrame = _make_synthetic_ohlcv(500, seed=0)
        ref: ReferencePattern = ReferencePattern(
            id="cmp_ref",
            symbol=symbols[0],
            timeframe="4h",
            start_ts=int(ref_df["timestamp"].iloc[300]),
            end_ts=int(ref_df["timestamp"].iloc[420]),
            label="compare",
            description="Compare reference",
        )

        # With prescreening (default: top 20%)
        config: SystemConfig = SystemConfig()
        config.data_root = tmpdir
        data_mgr: DataManager = DataManager(config)
        scanner: ScannerEngine = ScannerEngine(config, data_mgr)

        t0: float = time.perf_counter()
        alerts_with = scanner.scan_timeframe("4h", [ref])
        t_with: float = time.perf_counter() - t0

        # Without prescreening (keep 100%)
        config2: SystemConfig = SystemConfig()
        config2.data_root = tmpdir
        config2.prescreening_top_ratio = 1.0
        scanner2: ScannerEngine = ScannerEngine(config2, data_mgr)

        t0 = time.perf_counter()
        alerts_without = scanner2.scan_timeframe("4h", [ref])
        t_without: float = time.perf_counter() - t0

        speedup: float = t_without / t_with if t_with > 0 else float("inf")
        print(
            f"  With prescreening (top 20%):   {t_with:.3f}s  ({len(alerts_with)} alerts)"
        )
        print(
            f"  Without prescreening (100%):    {t_without:.3f}s  ({len(alerts_without)} alerts)"
        )
        print(f"  Speedup:                        {speedup:.1f}x")
        return {"comparison": "prescreening", "speedup": speedup}
    except Exception as exc:
        raise BenchmarkRunError(f"compare_prescreening failed: {exc}") from exc
    finally:
        shutil.rmtree(tmpdir)


# ---------------------------------------------------------------------------
# Comparison 3: Snappy vs uncompressed Parquet
# ---------------------------------------------------------------------------


def compare_parquet_compression() -> dict[str, Any]:
    """Compare Snappy-compressed vs uncompressed Parquet read/write speed.

    Returns:
        A dict with keys ``comparison`` and ``size_ratio``.

    Raises:
        BenchmarkRunError: If the comparison fails during execution.
    """
    try:
        print("\n--- Snappy vs uncompressed Parquet ---")
        df: pd.DataFrame = _make_synthetic_ohlcv(5000, seed=42)
        table: pa.Table = pa.Table.from_pandas(df, schema=_SCHEMA, preserve_index=False)
        tmpdir: str = tempfile.mkdtemp()

        try:
            results: dict[str, dict[str, float]] = {}
            for comp in ["snappy", "none"]:
                path: str = os.path.join(tmpdir, f"test_{comp}.parquet")

                # Write
                t0: float = time.perf_counter()
                for _ in range(20):
                    pq.write_table(table, path, compression=comp)
                write_time: float = (time.perf_counter() - t0) / 20

                file_size: int = os.path.getsize(path)

                # Read
                t0 = time.perf_counter()
                for _ in range(20):
                    pq.read_table(path).to_pandas()
                read_time: float = (time.perf_counter() - t0) / 20

                results[comp] = {
                    "write": write_time,
                    "read": read_time,
                    "size": file_size,
                }

            for comp, r in results.items():
                print(
                    f"  {comp:10s} write={r['write']*1000:.2f}ms  read={r['read']*1000:.2f}ms  "
                    f"size={r['size']//1024}KB"
                )

            size_ratio: float = results["none"]["size"] / results["snappy"]["size"]
            print(f"  Snappy compression ratio: {size_ratio:.2f}x smaller")
            return {"comparison": "parquet_compression", "size_ratio": size_ratio}
        finally:
            shutil.rmtree(tmpdir)
    except Exception as exc:
        raise BenchmarkRunError(f"compare_parquet_compression failed: {exc}") from exc


# ---------------------------------------------------------------------------
# Comparison 4: BTC cache hit vs miss
# ---------------------------------------------------------------------------


def compare_btc_cache() -> dict[str, Any]:
    """Compare BTC data read latency for cold vs warm (OS page cache) reads.

    Returns:
        A dict with keys ``comparison``, ``cold_ms``, and ``warm_ms``.

    Raises:
        BenchmarkRunError: If the comparison fails during execution.
    """
    from config import SystemConfig
    from data.data_manager import DataManager

    print("\n--- BTC data cache hit vs miss ---")
    tmpdir: str = tempfile.mkdtemp()
    try:
        tf_dir: str = os.path.join(tmpdir, "parquet", "timeframe=4h")
        os.makedirs(tf_dir, exist_ok=True)
        os.makedirs(os.path.join(tmpdir, "cache"), exist_ok=True)

        btc_df: pd.DataFrame = _make_synthetic_ohlcv(5000, seed=42)
        table: pa.Table = pa.Table.from_pandas(
            btc_df, schema=_SCHEMA, preserve_index=False
        )
        pq.write_table(
            table, os.path.join(tf_dir, "BTCUSDT.parquet"), compression="snappy"
        )

        config: SystemConfig = SystemConfig()
        config.data_root = tmpdir
        data_mgr: DataManager = DataManager(config)

        # Cold read (no cache)
        t0: float = time.perf_counter()
        for _ in range(20):
            data_mgr.read_symbol("BTCUSDT", "4h")
        t_cold: float = (time.perf_counter() - t0) / 20

        # Simulate repeated reads (Parquet is read from OS page cache after first read)
        t0 = time.perf_counter()
        for _ in range(20):
            data_mgr.read_symbol("BTCUSDT", "4h")
        t_warm: float = (time.perf_counter() - t0) / 20

        print(f"  First reads (cold):   {t_cold*1000:.3f} ms avg")
        print(f"  Repeated reads (warm): {t_warm*1000:.3f} ms avg")
        return {
            "comparison": "btc_cache",
            "cold_ms": t_cold * 1000,
            "warm_ms": t_warm * 1000,
        }
    except Exception as exc:
        raise BenchmarkRunError(f"compare_btc_cache failed: {exc}") from exc
    finally:
        shutil.rmtree(tmpdir)


def main() -> None:
    """Run all A/B optimization comparisons and print a summary.

    Returns:
        None.
    """
    print("=" * 60)
    print("  OPTIMIZATION COMPARISON SUITE")
    print("=" * 60)

    results: list[dict[str, Any]] = []
    for fn in [
        compare_dtw_implementations,
        compare_prescreening,
        compare_parquet_compression,
        compare_btc_cache,
    ]:
        try:
            r: dict[str, Any] = fn()
            results.append(r)
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback

            traceback.print_exc()

    print("\n" + "=" * 60)
    print("  Summary")
    print("=" * 60)
    for r in results:
        print(f"  {r['comparison']}: {r}")


if __name__ == "__main__":
    main()
