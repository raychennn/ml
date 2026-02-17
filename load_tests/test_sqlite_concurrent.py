"""Load Test: Concurrent SQLite read/write.

Verifies that SQLite works correctly under concurrent access with WAL mode
enabled, including multi-writer and mixed reader/writer scenarios.

Usage::

    python -m load_tests.test_sqlite_concurrent
"""

import os
import sys
import threading
import time
import tempfile
import shutil
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks.exceptions import (
    LoadTestError,
    LoadTestSetupError,
    LoadTestAssertionError,
)


def _make_store(tmpdir: str) -> Any:
    """Create a ``ScanResultStore`` with a config pointing to *tmpdir*.

    Args:
        tmpdir: Path to the temporary data-root directory.

    Returns:
        A configured ``ScanResultStore`` instance.

    Raises:
        LoadTestSetupError: If the store cannot be created.
    """
    from config import SystemConfig
    from core.scan_result_store import ScanResultStore

    try:
        config: SystemConfig = SystemConfig()
        config.data_root = tmpdir
        os.makedirs(tmpdir, exist_ok=True)
        return ScanResultStore(config)
    except Exception as exc:
        raise LoadTestSetupError(f"Failed to create ScanResultStore: {exc}") from exc


def test_concurrent_writes(n_writers: int = 5, writes_per_thread: int = 20) -> None:
    """Run multiple threads writing scan results concurrently.

    Args:
        n_writers: Number of concurrent writer threads.
        writes_per_thread: Number of write operations each thread performs.

    Raises:
        LoadTestAssertionError: If any write operation fails or the
            completion count does not match the expected total.
    """
    tmpdir: str = tempfile.mkdtemp()

    try:
        store = _make_store(tmpdir)
        errors: List[str] = []
        lock: threading.Lock = threading.Lock()
        completed: Dict[str, int] = {"count": 0}

        def writer(thread_id: int) -> None:
            """Perform repeated start_run/finish_run cycles."""
            try:
                for i in range(writes_per_thread):
                    run_id = store.start_run(["4h"])
                    store.finish_run(run_id, total_alerts=0)
                    with lock:
                        completed["count"] += 1
            except Exception as e:
                with lock:
                    errors.append(f"Thread {thread_id}: {e}")

        threads: List[threading.Thread] = [
            threading.Thread(target=writer, args=(i,)) for i in range(n_writers)
        ]
        t0: float = time.perf_counter()
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)
        elapsed: float = time.perf_counter() - t0

        total_expected: int = n_writers * writes_per_thread
        print(f"  Writers: {n_writers}, writes each: {writes_per_thread}")
        print(f"  Completed: {completed['count']}/{total_expected}")
        print(f"  Errors: {len(errors)}")
        print(f"  Time: {elapsed:.2f}s")

        if errors:
            for e in errors[:5]:
                print(f"    {e}")

        if len(errors) != 0:
            raise LoadTestAssertionError(
                f"Got {len(errors)} errors: {errors[:3]}",
                metric="write_error_count",
                actual=len(errors),
                threshold=0,
            )
        assert (
            completed["count"] == total_expected
        ), f"Expected {total_expected} completions, got {completed['count']}"
        print("  PASS")
    finally:
        shutil.rmtree(tmpdir)


def test_concurrent_read_write(
    n_readers: int = 5,
    n_writers: int = 2,
    duration_s: int = 3,
) -> None:
    """Run readers and writers operating concurrently for a fixed duration.

    Args:
        n_readers: Number of concurrent reader threads.
        n_writers: Number of concurrent writer threads.
        duration_s: Duration in seconds to run the mixed workload.

    Raises:
        LoadTestAssertionError: If any read or write operation fails.
        AssertionError: If no reads or no writes complete.
    """
    tmpdir: str = tempfile.mkdtemp()

    try:
        store = _make_store(tmpdir)
        stop_event: threading.Event = threading.Event()
        errors: List[str] = []
        lock: threading.Lock = threading.Lock()
        counts: Dict[str, int] = {"reads": 0, "writes": 0}

        def writer(thread_id: int) -> None:
            """Continuously write scan runs until stopped."""
            while not stop_event.is_set():
                try:
                    run_id = store.start_run(["4h"])
                    store.finish_run(run_id, total_alerts=0)
                    with lock:
                        counts["writes"] += 1
                    time.sleep(0.01)
                except Exception as e:
                    with lock:
                        errors.append(f"Writer {thread_id}: {e}")

        def reader(thread_id: int) -> None:
            """Continuously read scan results until stopped."""
            while not stop_event.is_set():
                try:
                    store.get_total_results()
                    store.get_results(page=1, per_page=10)
                    with lock:
                        counts["reads"] += 1
                    time.sleep(0.005)
                except Exception as e:
                    with lock:
                        errors.append(f"Reader {thread_id}: {e}")

        threads: List[threading.Thread] = []
        for i in range(n_writers):
            threads.append(threading.Thread(target=writer, args=(i,), daemon=True))
        for i in range(n_readers):
            threads.append(threading.Thread(target=reader, args=(i,), daemon=True))

        for t in threads:
            t.start()
        time.sleep(duration_s)
        stop_event.set()
        for t in threads:
            t.join(timeout=5)

        print(f"  Writers: {n_writers}, Readers: {n_readers}, Duration: {duration_s}s")
        print(f"  Writes: {counts['writes']}, Reads: {counts['reads']}")
        print(f"  Errors: {len(errors)}")

        if errors:
            for e in errors[:5]:
                print(f"    {e}")

        if len(errors) != 0:
            raise LoadTestAssertionError(
                f"Got {len(errors)} errors during concurrent R/W",
                metric="rw_error_count",
                actual=len(errors),
                threshold=0,
            )
        assert counts["reads"] > 0, "No reads completed"
        assert counts["writes"] > 0, "No writes completed"
        print("  PASS")
    finally:
        shutil.rmtree(tmpdir)


def test_wal_mode_enabled() -> None:
    """Verify that WAL mode is enabled on the database.

    Creates a store, connects to the underlying SQLite file, and checks
    that ``PRAGMA journal_mode`` returns ``wal``.

    Raises:
        AssertionError: If WAL mode is not active.
    """
    import sqlite3

    tmpdir: str = tempfile.mkdtemp()

    try:
        store = _make_store(tmpdir)
        db_path: str = store.db_path

        conn: sqlite3.Connection = sqlite3.connect(db_path)
        try:
            mode: str = conn.execute("PRAGMA journal_mode").fetchone()[0]
            print(f"  Journal mode: {mode}")
            assert mode == "wal", f"Expected WAL mode, got {mode}"
            print("  PASS")
        finally:
            conn.close()
    finally:
        shutil.rmtree(tmpdir)


def main() -> int:
    """Run all SQLite concurrent access tests and report results.

    Returns:
        Exit code: 0 if all tests passed, 1 otherwise.
    """
    print("=" * 60)
    print("  SQLITE CONCURRENT ACCESS TESTS")
    print("=" * 60)

    tests = [
        test_concurrent_writes,
        test_concurrent_read_write,
        test_wal_mode_enabled,
    ]
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
