"""Load Test: 250+ symbol scan -- memory and timing verification.

Simulates a real scan workload with a large number of symbols and asserts
that memory consumption and wall-clock time remain within acceptable bounds.

Usage::

    python -m load_tests.test_large_symbol_count
"""

import os
import sys
import time
import tempfile
import shutil
import resource
from typing import Dict, List

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks.exceptions import (
    LoadTestError,
    LoadTestSetupError,
    LoadTestAssertionError,
)

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


def _make_synthetic_ohlcv(n_bars: int = 500, seed: int = 42) -> pd.DataFrame:
    """Generate a synthetic OHLCV DataFrame for testing.

    Args:
        n_bars: Number of candlestick bars to generate.
        seed: Random seed for reproducibility.

    Returns:
        A pandas DataFrame with columns timestamp, open, high, low, close,
        and volume.
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


def _get_peak_mb() -> float:
    """Return the peak resident-set size of the current process in megabytes.

    On macOS ``ru_maxrss`` is reported in bytes; on Linux it is in kilobytes.

    Returns:
        Peak RSS in megabytes.
    """
    usage = resource.getrusage(resource.RUSAGE_SELF)
    if sys.platform == "darwin":
        return usage.ru_maxrss / (1024 * 1024)
    return usage.ru_maxrss / 1024


def test_large_symbol_scan(
    n_symbols: int = 250, n_bars: int = 500
) -> Dict[str, object]:
    """Scan with 250+ symbols and verify memory and timing stay within limits.

    Args:
        n_symbols: Number of synthetic symbols to generate and scan.
        n_bars: Number of OHLCV bars per symbol.

    Returns:
        A dict with keys ``symbols``, ``time``, ``alerts``, and ``peak_mb``
        summarising the test run.

    Raises:
        LoadTestSetupError: If the temporary data environment cannot be built.
        LoadTestAssertionError: If the scan exceeds the time or memory
            threshold.
    """
    from config import SystemConfig
    from core.scanner_engine import ScannerEngine
    from data.data_manager import DataManager
    from references.reference_manager import ReferencePattern

    tmpdir: str = tempfile.mkdtemp()
    try:
        tf_dir: str = os.path.join(tmpdir, "parquet", "timeframe=4h")
        os.makedirs(tf_dir, exist_ok=True)
        os.makedirs(os.path.join(tmpdir, "cache"), exist_ok=True)
        os.makedirs(os.path.join(tmpdir, "models"), exist_ok=True)

        print(f"  Creating {n_symbols} synthetic symbols ({n_bars} bars each)...")
        symbols: List[str] = []
        t_setup: float = time.perf_counter()
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
        setup_time: float = time.perf_counter() - t_setup
        print(f"  Setup: {setup_time:.1f}s")

        config: SystemConfig = SystemConfig()
        config.data_root = tmpdir
        data_mgr: DataManager = DataManager(config)
        scanner: ScannerEngine = ScannerEngine(config, data_mgr)

        ref_df: pd.DataFrame = _make_synthetic_ohlcv(n_bars, seed=0)
        ref: ReferencePattern = ReferencePattern(
            id="large_test_ref",
            symbol=symbols[0],
            timeframe="4h",
            start_ts=int(ref_df["timestamp"].iloc[300]),
            end_ts=int(ref_df["timestamp"].iloc[420]),
            label="large_test",
        )

        mem_before: float = _get_peak_mb()
        t0: float = time.perf_counter()
        alerts = scanner.scan_timeframe("4h", [ref])
        scan_time: float = time.perf_counter() - t0
        mem_after: float = _get_peak_mb()

        symbols_per_sec: float = (
            n_symbols / scan_time if scan_time > 0 else float("inf")
        )

        print(f"\n  Results:")
        print(f"    Symbols:         {n_symbols}")
        print(f"    Scan time:       {scan_time:.2f}s")
        print(f"    Symbols/sec:     {symbols_per_sec:.0f}")
        print(f"    Alerts:          {len(alerts)}")
        print(f"    Peak RSS before: {mem_before:.0f} MB")
        print(f"    Peak RSS after:  {mem_after:.0f} MB")
        print(f"    Memory delta:    {mem_after - mem_before:.0f} MB")

        # Assertions
        max_time: int = 120  # 2 minutes max for 250 symbols
        if scan_time >= max_time:
            raise LoadTestAssertionError(
                f"Scan took {scan_time:.1f}s, expected < {max_time}s",
                metric="scan_time_seconds",
                actual=round(scan_time, 1),
                threshold=max_time,
            )

        max_memory_mb: int = 2000  # 2GB max
        if mem_after >= max_memory_mb:
            raise LoadTestAssertionError(
                f"Peak memory {mem_after:.0f}MB exceeds {max_memory_mb}MB limit",
                metric="peak_memory_mb",
                actual=round(mem_after),
                threshold=max_memory_mb,
            )

        print("  PASS")
        return {
            "symbols": n_symbols,
            "time": scan_time,
            "alerts": len(alerts),
            "peak_mb": mem_after,
        }

    finally:
        shutil.rmtree(tmpdir)


def main() -> int:
    """Run the large symbol count test and report results.

    Returns:
        Exit code: 0 if the test passed, 1 otherwise.
    """
    print("=" * 60)
    print("  LARGE SYMBOL COUNT TEST")
    print("=" * 60)

    try:
        test_large_symbol_scan(250)
        return 0
    except (AssertionError, LoadTestAssertionError) as e:
        print(f"  FAIL: {e}")
        return 1
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
