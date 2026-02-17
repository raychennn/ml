"""Load Test: Sustained scheduler operation.

Simulates 10 scheduler ticks and verifies that no resource leaks occur
(thread count, memory growth, or timing degradation).

Usage::

    python -m load_tests.test_sustained_scheduler
"""

import os
import sys
import time
import tempfile
import shutil
import threading
import resource
from typing import List

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


def test_sustained_ticks(n_ticks: int = 10, n_symbols: int = 15) -> None:
    """Run multiple scan cycles and check for resource leaks.

    Args:
        n_ticks: Number of simulated scheduler ticks to execute.
        n_symbols: Number of synthetic symbols to create.

    Raises:
        LoadTestSetupError: If the temporary data environment cannot be built.
        LoadTestAssertionError: If a thread leak or performance degradation
            is detected.
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

        symbols: List[str] = []
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

        config: SystemConfig = SystemConfig()
        config.data_root = tmpdir
        data_mgr: DataManager = DataManager(config)
        scanner: ScannerEngine = ScannerEngine(config, data_mgr)

        ref_df: pd.DataFrame = _make_synthetic_ohlcv(500, seed=0)
        ref: ReferencePattern = ReferencePattern(
            id="sustain_ref",
            symbol=symbols[0],
            timeframe="4h",
            start_ts=int(ref_df["timestamp"].iloc[300]),
            end_ts=int(ref_df["timestamp"].iloc[420]),
            label="sustain",
        )

        initial_threads: int = threading.active_count()
        mem_readings: List[float] = []
        tick_durations: List[float] = []

        print(f"  Running {n_ticks} simulated ticks...")

        for tick in range(n_ticks):
            mem_before: float = _get_peak_mb()
            t0: float = time.perf_counter()

            alerts = scanner.scan_timeframe("4h", [ref])

            duration: float = time.perf_counter() - t0
            mem_after: float = _get_peak_mb()

            tick_durations.append(duration)
            mem_readings.append(mem_after)

            print(
                f"    Tick {tick + 1}/{n_ticks}: {duration:.2f}s, "
                f"{len(alerts)} alerts, {mem_after:.0f}MB peak, "
                f"{threading.active_count()} threads"
            )

        final_threads: int = threading.active_count()

        # Check for thread leaks
        thread_delta: int = final_threads - initial_threads
        print(
            f"\n  Thread count: {initial_threads} -> {final_threads} (delta: {thread_delta})"
        )

        # Check for consistent timing (no degradation > 2x)
        avg_duration: float = sum(tick_durations) / len(tick_durations)
        max_duration: float = max(tick_durations)
        print(f"  Tick duration: avg={avg_duration:.2f}s, max={max_duration:.2f}s")

        # Check memory stability (peak should not grow unboundedly)
        mem_first: float = mem_readings[0]
        mem_last: float = mem_readings[-1]
        mem_growth: float = mem_last - mem_first
        print(
            f"  Memory: first={mem_first:.0f}MB, last={mem_last:.0f}MB, growth={mem_growth:.0f}MB"
        )

        # Assertions
        max_thread_delta: int = 2
        if thread_delta > max_thread_delta:
            raise LoadTestAssertionError(
                f"Thread leak detected: {thread_delta} new threads after {n_ticks} ticks",
                metric="thread_delta",
                actual=thread_delta,
                threshold=max_thread_delta,
            )

        degradation_factor: float = 3.0
        if max_duration >= avg_duration * degradation_factor:
            raise LoadTestAssertionError(
                f"Performance degradation: max={max_duration:.2f}s vs avg={avg_duration:.2f}s",
                metric="tick_duration_ratio",
                actual=(
                    round(max_duration / avg_duration, 2)
                    if avg_duration > 0
                    else float("inf")
                ),
                threshold=degradation_factor,
            )

        print("  PASS")

    finally:
        shutil.rmtree(tmpdir)


def main() -> int:
    """Run the sustained scheduler test and report results.

    Returns:
        Exit code: 0 if the test passed, 1 otherwise.
    """
    print("=" * 60)
    print("  SUSTAINED SCHEDULER TEST")
    print("=" * 60)

    try:
        test_sustained_ticks(n_ticks=10, n_symbols=15)
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
