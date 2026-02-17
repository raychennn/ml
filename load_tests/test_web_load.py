"""Load Test: Flask dashboard under concurrent HTTP requests.

Tests the web dashboard with multiple concurrent clients hitting
health, results, and dashboard endpoints simultaneously.

Usage::

    python -m load_tests.test_web_load
"""

import os
import sys
import threading
import time
import tempfile
import shutil
from typing import Any, Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks.exceptions import (
    LoadTestError,
    LoadTestSetupError,
    LoadTestAssertionError,
)


def _setup_app() -> Tuple[Any, str]:
    """Create a test Flask app with minimal dependencies.

    Builds a temporary directory tree, configures a ``SystemConfig`` and
    ``ScanResultStore``, and returns a Flask application in testing mode.

    Returns:
        A tuple of ``(app, tmpdir)`` where *app* is the Flask test
        application and *tmpdir* is the temporary data root.

    Raises:
        LoadTestSetupError: If the app or temp environment cannot be created.
    """
    from config import SystemConfig
    from core.scan_result_store import ScanResultStore
    from web.app import create_app

    try:
        tmpdir: str = tempfile.mkdtemp()
        for subdir in [
            "parquet",
            "cache",
            "models",
            "logs",
            "images/alerts",
            "images/references",
            "images/historical",
            "references",
        ]:
            os.makedirs(os.path.join(tmpdir, subdir), exist_ok=True)

        config: SystemConfig = SystemConfig()
        config.data_root = tmpdir

        store: ScanResultStore = ScanResultStore(config)

        app = create_app(config=config, result_store=store)
        app.config["TESTING"] = True

        return app, tmpdir
    except Exception as exc:
        raise LoadTestSetupError(f"Failed to set up Flask test app: {exc}") from exc


def test_concurrent_health_checks(
    n_threads: int = 20, requests_per_thread: int = 10
) -> None:
    """Fire concurrent requests at ``/api/health``.

    Args:
        n_threads: Number of concurrent worker threads.
        requests_per_thread: Number of HTTP requests each thread sends.

    Raises:
        LoadTestAssertionError: If any request returns a non-200 status.
    """
    app, tmpdir = _setup_app()
    try:
        results: Dict[str, Any] = {"success": 0, "error": 0, "latencies": []}
        lock: threading.Lock = threading.Lock()

        def worker() -> None:
            """Send repeated health-check requests."""
            with app.test_client() as client:
                for _ in range(requests_per_thread):
                    t0: float = time.perf_counter()
                    try:
                        resp = client.get("/api/health")
                        latency: float = (time.perf_counter() - t0) * 1000
                        with lock:
                            results["latencies"].append(latency)
                            if resp.status_code == 200:
                                results["success"] += 1
                            else:
                                results["error"] += 1
                    except Exception:
                        with lock:
                            results["error"] += 1

        threads: List[threading.Thread] = [
            threading.Thread(target=worker) for _ in range(n_threads)
        ]

        t0: float = time.perf_counter()
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)
        total_time: float = time.perf_counter() - t0

        total_requests: int = n_threads * requests_per_thread
        latencies: List[float] = sorted(results["latencies"])
        avg_latency: float = sum(latencies) / len(latencies) if latencies else 0
        p95_latency: float = latencies[int(len(latencies) * 0.95)] if latencies else 0
        rps: float = total_requests / total_time if total_time > 0 else 0

        print(f"  Total requests:    {total_requests}")
        print(f"  Success:           {results['success']}")
        print(f"  Errors:            {results['error']}")
        print(f"  Total time:        {total_time:.2f}s")
        print(f"  Requests/sec:      {rps:.0f}")
        print(f"  Avg latency:       {avg_latency:.1f}ms")
        print(f"  P95 latency:       {p95_latency:.1f}ms")

        if results["error"] != 0:
            raise LoadTestAssertionError(
                f"Expected 0 errors, got {results['error']}",
                metric="error_count",
                actual=results["error"],
                threshold=0,
            )
        assert (
            results["success"] == total_requests
        ), f"Expected {total_requests} successes, got {results['success']}"
        print("  PASS")
    finally:
        shutil.rmtree(tmpdir)


def test_concurrent_results_page(
    n_threads: int = 10, requests_per_thread: int = 5
) -> None:
    """Fire concurrent requests at ``/api/results``.

    Args:
        n_threads: Number of concurrent worker threads.
        requests_per_thread: Number of HTTP requests each thread sends.

    Raises:
        LoadTestAssertionError: If any request returns a non-200 status.
    """
    app, tmpdir = _setup_app()
    try:
        results: Dict[str, int] = {"success": 0, "error": 0}
        lock: threading.Lock = threading.Lock()

        def worker() -> None:
            """Send repeated results-page requests."""
            with app.test_client() as client:
                for _ in range(requests_per_thread):
                    try:
                        resp = client.get("/api/results?page=1&per_page=10")
                        with lock:
                            if resp.status_code == 200:
                                results["success"] += 1
                            else:
                                results["error"] += 1
                    except Exception:
                        with lock:
                            results["error"] += 1

        threads: List[threading.Thread] = [
            threading.Thread(target=worker) for _ in range(n_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        total: int = n_threads * requests_per_thread
        print(
            f"  Results endpoint: {results['success']}/{total} success, {results['error']} errors"
        )
        if results["error"] != 0:
            raise LoadTestAssertionError(
                f"Expected 0 errors, got {results['error']}",
                metric="error_count",
                actual=results["error"],
                threshold=0,
            )
        print("  PASS")
    finally:
        shutil.rmtree(tmpdir)


def test_concurrent_dashboard_page(
    n_threads: int = 10, requests_per_thread: int = 5
) -> None:
    """Fire concurrent requests at ``/`` (dashboard HTML).

    Args:
        n_threads: Number of concurrent worker threads.
        requests_per_thread: Number of HTTP requests each thread sends.

    Raises:
        LoadTestAssertionError: If any request returns a non-200 status.
    """
    app, tmpdir = _setup_app()
    try:
        results: Dict[str, int] = {"success": 0, "error": 0}
        lock: threading.Lock = threading.Lock()

        def worker() -> None:
            """Send repeated dashboard-page requests."""
            with app.test_client() as client:
                for _ in range(requests_per_thread):
                    try:
                        resp = client.get("/")
                        with lock:
                            if resp.status_code == 200:
                                results["success"] += 1
                            else:
                                results["error"] += 1
                    except Exception:
                        with lock:
                            results["error"] += 1

        threads: List[threading.Thread] = [
            threading.Thread(target=worker) for _ in range(n_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        total: int = n_threads * requests_per_thread
        print(
            f"  Dashboard page: {results['success']}/{total} success, {results['error']} errors"
        )
        if results["error"] != 0:
            raise LoadTestAssertionError(
                f"Expected 0 errors, got {results['error']}",
                metric="error_count",
                actual=results["error"],
                threshold=0,
            )
        print("  PASS")
    finally:
        shutil.rmtree(tmpdir)


def main() -> int:
    """Run all web load tests and report results.

    Returns:
        Exit code: 0 if all tests passed, 1 otherwise.
    """
    print("=" * 60)
    print("  WEB LOAD TESTS")
    print("=" * 60)

    tests = [
        test_concurrent_health_checks,
        test_concurrent_results_page,
        test_concurrent_dashboard_page,
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
