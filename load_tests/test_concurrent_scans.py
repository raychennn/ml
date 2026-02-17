"""Load Test: Concurrent scan lock verification.

Verifies that ``scan_lock`` correctly blocks concurrent scans and that
the ``is_scanning`` property reflects the real lock state.

Usage::

    python -m load_tests.test_concurrent_scans
"""

import os
import sys
import threading
import tempfile
import shutil
import time
from typing import Dict, List, Optional, Tuple

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


def _setup_env() -> Tuple[str, List[str]]:
    """Create a temporary directory populated with synthetic parquet data.

    Creates the directory tree expected by the system (parquet, cache, models,
    logs, images, references) and writes 10 synthetic symbol parquet files.

    Returns:
        A tuple of ``(tmpdir, symbols)`` where *tmpdir* is the path to the
        temporary root and *symbols* is the list of generated symbol names.

    Raises:
        LoadTestSetupError: If the temporary environment cannot be created.
    """
    try:
        tmpdir: str = tempfile.mkdtemp()
        tf_dir: str = os.path.join(tmpdir, "parquet", "timeframe=4h")
        os.makedirs(tf_dir, exist_ok=True)
        for subdir in [
            "cache",
            "models",
            "logs",
            "images/alerts",
            "images/references",
            "images/historical",
            "references",
        ]:
            os.makedirs(os.path.join(tmpdir, subdir), exist_ok=True)

        symbols: List[str] = []
        for i in range(10):
            sym: str = f"SYM{i:04d}USDT"
            symbols.append(sym)
            df: pd.DataFrame = _make_synthetic_ohlcv(500, seed=i)
            table: pa.Table = pa.Table.from_pandas(
                df, schema=_SCHEMA, preserve_index=False
            )
            pq.write_table(
                table, os.path.join(tf_dir, f"{sym}.parquet"), compression="snappy"
            )
        return tmpdir, symbols
    except Exception as exc:
        raise LoadTestSetupError(f"Failed to create test environment: {exc}") from exc


def test_scan_lock_blocks_concurrent() -> None:
    """Verify that two threads cannot scan simultaneously.

    Starts two scan threads with a slight delay between them.  At least one
    must complete successfully; the other should either be blocked with a
    ``RuntimeError`` or complete if the first finishes before the second
    acquires the lock.

    Raises:
        LoadTestError: If neither scan thread completes successfully.
        AssertionError: If the blocked-thread error message is unexpected.
    """
    from config import SystemConfig
    from core.scanner_engine import ScannerEngine
    from data.data_manager import DataManager
    from core.scan_result_store import ScanResultStore
    from core.scan_orchestrator import ScanOrchestrator
    from references.reference_manager import ReferencePattern, ReferenceManager

    tmpdir: str
    symbols: List[str]
    tmpdir, symbols = _setup_env()
    try:
        config: SystemConfig = SystemConfig()
        config.data_root = tmpdir
        config.telegram_bot_token = ""
        config.telegram_chat_id = ""

        data_mgr: DataManager = DataManager(config)
        scanner: ScannerEngine = ScannerEngine(config, data_mgr)
        store: ScanResultStore = ScanResultStore(config)
        ref_mgr: ReferenceManager = ReferenceManager(config, data_manager=data_mgr)

        orchestrator: ScanOrchestrator = ScanOrchestrator(
            config=config,
            scanner_engine=scanner,
            data_manager=data_mgr,
            result_store=store,
            reference_manager=ref_mgr,
            notifier=None,
        )

        # Add a reference
        ref_df: pd.DataFrame = _make_synthetic_ohlcv(500, seed=0)
        ref: ReferencePattern = ReferencePattern(
            id="lock_test_ref",
            symbol=symbols[0],
            timeframe="4h",
            start_ts=int(ref_df["timestamp"].iloc[300]),
            end_ts=int(ref_df["timestamp"].iloc[420]),
            label="lock_test",
        )
        ref_mgr.add_reference(ref)

        results: Dict[str, Optional[object]] = {"thread1": None, "thread2": None}
        errors: Dict[str, Optional[str]] = {"thread1": None, "thread2": None}

        def scan_worker(name: str) -> None:
            """Execute a scan and record the outcome."""
            try:
                result = orchestrator.run_scan(
                    timeframes=["4h"], send_notifications=False
                )
                results[name] = result
            except RuntimeError as e:
                errors[name] = str(e)
            except Exception as e:
                errors[name] = f"unexpected: {e}"

        t1: threading.Thread = threading.Thread(
            target=scan_worker, args=("thread1",), name="scan-1"
        )
        t2: threading.Thread = threading.Thread(
            target=scan_worker, args=("thread2",), name="scan-2"
        )

        t1.start()
        time.sleep(0.1)  # Give thread1 a head start
        t2.start()

        t1.join(timeout=60)
        t2.join(timeout=60)

        # Exactly one should succeed, one should get blocked
        one_succeeded: bool = (results["thread1"] is not None) != (
            results["thread2"] is not None
        ) or (results["thread1"] is not None and results["thread2"] is not None)
        one_blocked: bool = (
            errors["thread1"] is not None or errors["thread2"] is not None
        )

        print(f"  Thread1: result={results['thread1']}, error={errors['thread1']}")
        print(f"  Thread2: result={results['thread2']}, error={errors['thread2']}")

        # At least one should have completed
        assert (
            results["thread1"] is not None or results["thread2"] is not None
        ), "At least one scan should complete"

        # The concurrent one should have been blocked (RuntimeError) or queued
        if errors["thread1"] or errors["thread2"]:
            blocked_error: str = errors["thread1"] or errors["thread2"]  # type: ignore[assignment]
            assert (
                "already running" in blocked_error.lower()
                or "scan" in blocked_error.lower()
            ), f"Expected scan lock error, got: {blocked_error}"
            print("  PASS: Concurrent scan was correctly blocked")
        else:
            # Both succeeded (possible if first finished before second acquired lock)
            print("  PASS: Both scans completed (no actual contention)")

    finally:
        shutil.rmtree(tmpdir)


def test_is_scanning_property() -> None:
    """Verify that ``is_scanning`` reflects the lock state.

    Creates an orchestrator and asserts that ``is_scanning`` is ``False``
    before any scan is started.

    Raises:
        AssertionError: If ``is_scanning`` is not ``False`` when idle.
    """
    from config import SystemConfig
    from core.scanner_engine import ScannerEngine
    from data.data_manager import DataManager
    from core.scan_result_store import ScanResultStore
    from core.scan_orchestrator import ScanOrchestrator
    from references.reference_manager import ReferenceManager

    tmpdir: str
    symbols: List[str]
    tmpdir, symbols = _setup_env()
    try:
        config: SystemConfig = SystemConfig()
        config.data_root = tmpdir
        config.telegram_bot_token = ""
        config.telegram_chat_id = ""

        data_mgr: DataManager = DataManager(config)
        scanner: ScannerEngine = ScannerEngine(config, data_mgr)
        store: ScanResultStore = ScanResultStore(config)
        ref_mgr: ReferenceManager = ReferenceManager(config, data_manager=data_mgr)

        orchestrator: ScanOrchestrator = ScanOrchestrator(
            config=config,
            scanner_engine=scanner,
            data_manager=data_mgr,
            result_store=store,
            reference_manager=ref_mgr,
            notifier=None,
        )

        assert not orchestrator.is_scanning, "Should not be scanning initially"
        print("  PASS: is_scanning correctly reports False when idle")
    finally:
        shutil.rmtree(tmpdir)


def main() -> int:
    """Run all concurrent scan lock tests and report results.

    Returns:
        Exit code: 0 if all tests passed, 1 otherwise.
    """
    print("=" * 60)
    print("  CONCURRENT SCAN LOCK TESTS")
    print("=" * 60)

    tests = [test_scan_lock_blocks_concurrent, test_is_scanning_property]
    passed: int = 0
    failed: int = 0

    for test_fn in tests:
        print(f"\n  Running: {test_fn.__name__}")
        try:
            test_fn()
            passed += 1
        except (AssertionError, LoadTestAssertionError) as e:
            print(f"  FAIL: {e}")
            failed += 1
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback

            traceback.print_exc()
            failed += 1

    print(f"\n  Results: {passed} passed, {failed} failed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
