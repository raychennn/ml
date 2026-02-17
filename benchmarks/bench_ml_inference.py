"""ML predictor inference latency benchmarks.

Measures feature extraction and model prediction time using synthetic
data and a mock model if no trained models exist.

Usage::

    python -m benchmarks.bench_ml_inference
"""

from __future__ import annotations

import time
import sys
import os
from dataclasses import dataclass
from typing import Any, TypedDict

import numpy as np
import pandas as pd
from numpy.typing import NDArray

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
        return pd.DataFrame(
            {
                "timestamp": np.arange(1700000000, 1700000000 + n_bars * 3600, 3600)[
                    :n_bars
                ],
                "open": close + rng.randn(n_bars) * 0.3,
                "high": close + rng.uniform(0.1, 1.0, n_bars),
                "low": close - rng.uniform(0.1, 1.0, n_bars),
                "close": close,
                "volume": rng.uniform(100, 10000, n_bars),
            }
        )
    except Exception as exc:
        raise BenchmarkSetupError(
            f"Failed to generate synthetic OHLCV (n_bars={n_bars}, seed={seed}): {exc}"
        ) from exc


@dataclass
class MockMatch:
    """Mock match result for benchmarking ML inference without real data."""

    symbol: str = "TESTUSDT"
    timeframe: str = "4h"
    score: float = 0.72
    price_distance: float = 0.18
    diff_distance: float = 0.22
    best_scale_factor: float = 1.0
    match_start_idx: int = 380
    match_end_idx: int = 500
    window_data: Any = None


def _make_mock_match_result() -> MockMatch:
    """Create a mock match result dataclass instance.

    Returns:
        A ``MockMatch`` with default field values suitable for benchmarking.

    Raises:
        BenchmarkSetupError: If mock creation fails unexpectedly.
    """
    try:
        return MockMatch()
    except Exception as exc:
        raise BenchmarkSetupError(f"Failed to create mock match result: {exc}") from exc


# ---------------------------------------------------------------------------
# Benchmark runners
# ---------------------------------------------------------------------------


def bench_feature_extraction(repeats: int = 50) -> BenchmarkResult:
    """Benchmark the feature extraction step in isolation.

    Args:
        repeats: Number of timed iterations.

    Returns:
        A ``BenchmarkResult`` dict with average extraction time per call.

    Raises:
        BenchmarkSetupError: If setup (config, data) fails.
        BenchmarkRunError: If the benchmark execution fails.
    """
    try:
        from config import SystemConfig
        from core.feature_extractor import FeatureExtractor

        config: SystemConfig = SystemConfig()
        extractor: FeatureExtractor = FeatureExtractor(config)

        window_df: pd.DataFrame = _make_synthetic_ohlcv(120, seed=42)
        btc_df: pd.DataFrame = _make_synthetic_ohlcv(500, seed=99)
        match: MockMatch = _make_mock_match_result()
    except (BenchmarkSetupError, BenchmarkRunError):
        raise
    except Exception as exc:
        raise BenchmarkSetupError(
            f"bench_feature_extraction setup failed: {exc}"
        ) from exc

    try:
        # Warm up
        extractor.extract(match, window_df, btc_df)

        t0: float = time.perf_counter()
        for _ in range(repeats):
            extractor.extract(match, window_df, btc_df)
        elapsed: float = (time.perf_counter() - t0) / repeats

        return BenchmarkResult(
            name="feature_extraction_single",
            value=elapsed,
            unit="seconds",
            detail=f"repeats={repeats}",
        )
    except (BenchmarkSetupError, BenchmarkRunError):
        raise
    except Exception as exc:
        raise BenchmarkRunError(f"bench_feature_extraction failed: {exc}") from exc


def bench_ml_predict_single(repeats: int = 50) -> BenchmarkResult:
    """Benchmark the full prediction pipeline (extraction + model).

    Args:
        repeats: Number of timed iterations.

    Returns:
        A ``BenchmarkResult`` dict with average prediction time per call.

    Raises:
        BenchmarkSetupError: If setup (config, predictor, data) fails.
        BenchmarkRunError: If the benchmark execution fails.
    """
    try:
        from config import SystemConfig
        from ml.predictor import MLPredictor

        config: SystemConfig = SystemConfig()
        predictor: MLPredictor = MLPredictor(config)
        predictor.load()

        window_df: pd.DataFrame = _make_synthetic_ohlcv(120, seed=42)
        btc_df: pd.DataFrame = _make_synthetic_ohlcv(500, seed=99)
        match: MockMatch = _make_mock_match_result()
    except (BenchmarkSetupError, BenchmarkRunError):
        raise
    except Exception as exc:
        raise BenchmarkSetupError(
            f"bench_ml_predict_single setup failed: {exc}"
        ) from exc

    try:
        # Warm up
        predictor.predict(match, window_df, btc_df)

        t0: float = time.perf_counter()
        for _ in range(repeats):
            predictor.predict(match, window_df, btc_df)
        elapsed: float = (time.perf_counter() - t0) / repeats

        mode: str = "trained" if predictor.is_loaded else "fallback"
        return BenchmarkResult(
            name="ml_inference_single",
            value=elapsed,
            unit="seconds",
            detail=f"repeats={repeats}, mode={mode}",
        )
    except (BenchmarkSetupError, BenchmarkRunError):
        raise
    except Exception as exc:
        raise BenchmarkRunError(f"bench_ml_predict_single failed: {exc}") from exc


def bench_ml_predict_batch(batch_size: int = 10) -> BenchmarkResult:
    """Benchmark batch prediction across multiple synthetic matches.

    Args:
        batch_size: Number of match items to predict.

    Returns:
        A ``BenchmarkResult`` dict with total batch prediction time.

    Raises:
        BenchmarkSetupError: If setup (config, predictor, data) fails.
        BenchmarkRunError: If the benchmark execution fails.
    """
    try:
        from config import SystemConfig
        from ml.predictor import MLPredictor

        config: SystemConfig = SystemConfig()
        predictor: MLPredictor = MLPredictor(config)
        predictor.load()

        items: list[tuple[MockMatch, pd.DataFrame, pd.DataFrame]] = []
        for i in range(batch_size):
            window_df: pd.DataFrame = _make_synthetic_ohlcv(120, seed=i)
            btc_df: pd.DataFrame = _make_synthetic_ohlcv(500, seed=99)
            match: MockMatch = _make_mock_match_result()
            items.append((match, window_df, btc_df))
    except (BenchmarkSetupError, BenchmarkRunError):
        raise
    except Exception as exc:
        raise BenchmarkSetupError(
            f"bench_ml_predict_batch setup failed: {exc}"
        ) from exc

    try:
        # Warm up
        predictor.predict(*items[0])

        t0: float = time.perf_counter()
        for match, wdf, bdf in items:
            predictor.predict(match, wdf, bdf)
        elapsed: float = time.perf_counter() - t0

        mode: str = "trained" if predictor.is_loaded else "fallback"
        return BenchmarkResult(
            name="ml_inference_batch_10",
            value=elapsed,
            unit="seconds",
            detail=f"batch={batch_size}, mode={mode}",
        )
    except (BenchmarkSetupError, BenchmarkRunError):
        raise
    except Exception as exc:
        raise BenchmarkRunError(f"bench_ml_predict_batch failed: {exc}") from exc


def run_all() -> list[BenchmarkResult]:
    """Execute every ML inference benchmark and return collected results.

    Returns:
        A list of ``BenchmarkResult`` dicts for all benchmarks (including
        error entries for benchmarks that failed).
    """
    results: list[BenchmarkResult] = []
    print("=" * 60)
    print("ML Inference Benchmark Suite")
    print("=" * 60)

    bench_fns: list = [
        bench_feature_extraction,
        bench_ml_predict_single,
        bench_ml_predict_batch,
    ]
    for bench_fn in bench_fns:
        try:
            r: BenchmarkResult = bench_fn()
            results.append(r)
            print(
                f"  {r['name']:30s} {r['value']:.6f} {r['unit']:15s}  ({r['detail']})"
            )
        except Exception as e:
            print(f"  {bench_fn.__name__:30s} ERROR: {e}")
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
