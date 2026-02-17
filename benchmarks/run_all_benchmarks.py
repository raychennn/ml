"""Master Benchmark Runner.

Runs all benchmark suites, compares results to baselines, and reports
PASS / WARN / FAIL for every metric.

Usage::

    python -m benchmarks.run_all_benchmarks
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, TypedDict

from benchmarks.exceptions import BaselineLoadError

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class BenchmarkResult(TypedDict):
    """Standard result payload returned by every benchmark function."""

    name: str
    value: float | None
    unit: str
    detail: str


def _load_baselines() -> dict[str, Any]:
    """Load performance baselines from ``baselines.json``.

    Returns:
        A mapping of benchmark names to their baseline configuration
        (expected value, warn/fail multipliers, etc.).

    Raises:
        BaselineLoadError: If the file cannot be found, read, or parsed.
    """
    baselines_path: str = os.path.join(os.path.dirname(__file__), "baselines.json")
    try:
        with open(baselines_path, "r") as f:
            data: dict[str, Any] = json.load(f)
        return data["baselines"]
    except (FileNotFoundError, json.JSONDecodeError, KeyError) as exc:
        raise BaselineLoadError(
            f"Failed to load baselines from {baselines_path}: {exc}"
        ) from exc


def _evaluate(
    result: BenchmarkResult,
    baselines: dict[str, Any],
) -> tuple[str, str]:
    """Compare a benchmark result against its baseline.

    Args:
        result: The benchmark result to evaluate.
        baselines: Mapping of benchmark names to baseline definitions.

    Returns:
        A ``(status, message)`` tuple where *status* is one of
        ``"PASS"``, ``"WARN"``, ``"FAIL"``, ``"SKIP"``, or ``"ERROR"``.
    """
    name: str | None = result.get("name")
    value: float | None = result.get("value")

    if value is None:
        return "ERROR", "benchmark returned no value"

    if name not in baselines:
        return "SKIP", "no baseline defined"

    baseline: dict[str, Any] = baselines[name]
    expected: float = baseline["expected"]
    warn_mult: float = baseline["warn_multiplier"]
    fail_mult: float = baseline["fail_multiplier"]
    higher_is_better: bool = baseline.get("higher_is_better", False)

    if higher_is_better:
        # For throughput metrics: FAIL if value < expected * fail_mult
        if value < expected * fail_mult:
            return (
                "FAIL",
                f"{value:.4f} < {expected * fail_mult:.4f} (expected >= {expected} * {fail_mult})",
            )
        elif value < expected * warn_mult:
            return (
                "WARN",
                f"{value:.4f} < {expected * warn_mult:.4f} (expected >= {expected} * {warn_mult})",
            )
        else:
            return "PASS", f"{value:.4f} >= {expected * warn_mult:.4f}"
    else:
        # For latency metrics: FAIL if value > expected * fail_mult
        if value > expected * fail_mult:
            return (
                "FAIL",
                f"{value:.4f} > {expected * fail_mult:.4f} (expected <= {expected} * {fail_mult})",
            )
        elif value > expected * warn_mult:
            return (
                "WARN",
                f"{value:.4f} > {expected * warn_mult:.4f} (expected <= {expected} * {warn_mult})",
            )
        else:
            return "PASS", f"{value:.4f} <= {expected * warn_mult:.4f}"


def main() -> int:
    """Run all benchmark suites and evaluate against baselines.

    Returns:
        Exit code: ``1`` if any benchmark FAILed, ``0`` otherwise.
    """
    print("=" * 70)
    print("  CRYPTO PATTERN SYSTEM -- PERFORMANCE BENCHMARK SUITE")
    print("=" * 70)
    print()

    baselines: dict[str, Any] = _load_baselines()
    all_results: list[BenchmarkResult] = []
    suite_times: dict[str, float] = {}

    # Import and run each benchmark suite
    suites: list[tuple[str, Any]] = []
    try:
        from benchmarks.bench_dtw import run_all as run_dtw

        suites.append(("DTW", run_dtw))
    except ImportError as e:
        print(f"  [SKIP] bench_dtw: {e}")

    try:
        from benchmarks.bench_prescreening import run_all as run_prescreening

        suites.append(("Prescreening", run_prescreening))
    except ImportError as e:
        print(f"  [SKIP] bench_prescreening: {e}")

    try:
        from benchmarks.bench_parquet import run_all as run_parquet

        suites.append(("Parquet", run_parquet))
    except ImportError as e:
        print(f"  [SKIP] bench_parquet: {e}")

    try:
        from benchmarks.bench_ml_inference import run_all as run_ml

        suites.append(("ML Inference", run_ml))
    except ImportError as e:
        print(f"  [SKIP] bench_ml_inference: {e}")

    try:
        from benchmarks.bench_scan_cycle import run_all as run_scan

        suites.append(("Scan Cycle", run_scan))
    except ImportError as e:
        print(f"  [SKIP] bench_scan_cycle: {e}")

    for name, run_fn in suites:
        t0: float = time.perf_counter()
        try:
            results: list[BenchmarkResult] = run_fn()
            all_results.extend(results)
        except Exception as e:
            print(f"  [{name}] ERROR: {e}")
        suite_times[name] = time.perf_counter() - t0

    # Evaluation summary
    print("=" * 70)
    print("  RESULTS vs BASELINES")
    print("=" * 70)
    print()

    counts: dict[str, int] = {"PASS": 0, "WARN": 0, "FAIL": 0, "SKIP": 0, "ERROR": 0}

    for r in all_results:
        status: str
        msg: str
        status, msg = _evaluate(r, baselines)
        counts[status] += 1
        icon: str = {
            "PASS": "[PASS]",
            "WARN": "[WARN]",
            "FAIL": "[FAIL]",
            "SKIP": "[SKIP]",
            "ERROR": "[ERR ]",
        }[status]
        print(f"  {icon} {r['name']:40s} {msg}")

    # Summary
    print()
    print("-" * 70)
    total: int = sum(counts.values())
    print(
        f"  Total: {total}  |  PASS: {counts['PASS']}  |  WARN: {counts['WARN']}  |  "
        f"FAIL: {counts['FAIL']}  |  SKIP: {counts['SKIP']}  |  ERROR: {counts['ERROR']}"
    )
    print()

    print("  Suite durations:")
    for name, duration in suite_times.items():
        print(f"    {name:20s} {duration:.2f}s")
    print()

    # Exit code
    if counts["FAIL"] > 0:
        print("  OVERALL: FAIL")
        return 1
    elif counts["WARN"] > 0:
        print("  OVERALL: WARN (all within acceptable limits)")
        return 0
    else:
        print("  OVERALL: PASS")
        return 0


if __name__ == "__main__":
    sys.exit(main())
