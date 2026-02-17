"""Parquet read/write speed benchmarks.

Measures single-file and bulk read/write performance using synthetic OHLCV
data with various compression codecs.

Usage::

    python -m benchmarks.bench_parquet
"""

from __future__ import annotations

import os
import time
import tempfile
import shutil
from typing import TypedDict

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from numpy.typing import NDArray

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


def _make_synthetic_ohlcv(n_rows: int = 5000, seed: int = 42) -> pd.DataFrame:
    """Generate a synthetic OHLCV DataFrame.

    Args:
        n_rows: Number of rows to generate.
        seed: Random seed for reproducibility.

    Returns:
        A DataFrame with columns: timestamp, open, high, low, close, volume.

    Raises:
        BenchmarkSetupError: If data generation fails.
    """
    try:
        rng: np.random.RandomState = np.random.RandomState(seed)
        close: NDArray[np.float64] = 100.0 + np.cumsum(rng.randn(n_rows) * 0.5)
        return pd.DataFrame(
            {
                "timestamp": np.arange(1700000000, 1700000000 + n_rows * 3600, 3600)[
                    :n_rows
                ].astype(np.int64),
                "open": close + rng.randn(n_rows) * 0.3,
                "high": close + rng.uniform(0.1, 1.0, n_rows),
                "low": close - rng.uniform(0.1, 1.0, n_rows),
                "close": close,
                "volume": rng.uniform(100, 10000, n_rows),
            }
        )
    except Exception as exc:
        raise BenchmarkSetupError(
            f"Failed to generate synthetic OHLCV (n_rows={n_rows}, seed={seed}): {exc}"
        ) from exc


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


# ---------------------------------------------------------------------------
# Benchmark runners
# ---------------------------------------------------------------------------


def bench_write_single(n_rows: int = 5000, repeats: int = 20) -> BenchmarkResult:
    """Benchmark writing a single Parquet file repeatedly.

    Args:
        n_rows: Number of rows in the synthetic DataFrame.
        repeats: Number of timed write iterations.

    Returns:
        A ``BenchmarkResult`` dict with average write time.

    Raises:
        BenchmarkSetupError: If temp directory or data creation fails.
        BenchmarkRunError: If the benchmark execution fails.
    """
    try:
        df: pd.DataFrame = _make_synthetic_ohlcv(n_rows)
        table: pa.Table = pa.Table.from_pandas(df, schema=_SCHEMA, preserve_index=False)
    except BenchmarkSetupError:
        raise
    except Exception as exc:
        raise BenchmarkSetupError(
            f"Failed to prepare data for bench_write_single: {exc}"
        ) from exc

    try:
        tmpdir: str = tempfile.mkdtemp()
    except Exception as exc:
        raise BenchmarkSetupError(f"Failed to create temp directory: {exc}") from exc

    try:
        path: str = os.path.join(tmpdir, "test.parquet")

        # Warm up
        pq.write_table(table, path, compression="snappy")

        t0: float = time.perf_counter()
        for _ in range(repeats):
            pq.write_table(table, path, compression="snappy")
        elapsed: float = (time.perf_counter() - t0) / repeats
    except Exception as exc:
        raise BenchmarkRunError(f"bench_write_single failed: {exc}") from exc
    finally:
        shutil.rmtree(tmpdir)

    return BenchmarkResult(
        name="parquet_write_single",
        value=elapsed,
        unit="seconds",
        detail=f"rows={n_rows}, repeats={repeats}",
    )


def bench_read_single(n_rows: int = 5000, repeats: int = 20) -> BenchmarkResult:
    """Benchmark reading a single Parquet file repeatedly.

    Args:
        n_rows: Number of rows in the synthetic DataFrame.
        repeats: Number of timed read iterations.

    Returns:
        A ``BenchmarkResult`` dict with average read time.

    Raises:
        BenchmarkSetupError: If temp directory or data creation fails.
        BenchmarkRunError: If the benchmark execution fails.
    """
    try:
        df: pd.DataFrame = _make_synthetic_ohlcv(n_rows)
        table: pa.Table = pa.Table.from_pandas(df, schema=_SCHEMA, preserve_index=False)
    except BenchmarkSetupError:
        raise
    except Exception as exc:
        raise BenchmarkSetupError(
            f"Failed to prepare data for bench_read_single: {exc}"
        ) from exc

    try:
        tmpdir: str = tempfile.mkdtemp()
    except Exception as exc:
        raise BenchmarkSetupError(f"Failed to create temp directory: {exc}") from exc

    try:
        path: str = os.path.join(tmpdir, "test.parquet")
        pq.write_table(table, path, compression="snappy")

        # Warm up
        pq.read_table(path).to_pandas()

        t0: float = time.perf_counter()
        for _ in range(repeats):
            pq.read_table(path).to_pandas()
        elapsed: float = (time.perf_counter() - t0) / repeats
    except Exception as exc:
        raise BenchmarkRunError(f"bench_read_single failed: {exc}") from exc
    finally:
        shutil.rmtree(tmpdir)

    return BenchmarkResult(
        name="parquet_read_single",
        value=elapsed,
        unit="seconds",
        detail=f"rows={n_rows}, repeats={repeats}",
    )


def bench_bulk_read(n_files: int = 50, n_rows: int = 5000) -> BenchmarkResult:
    """Benchmark reading many Parquet files sequentially.

    Args:
        n_files: Number of Parquet files to create and read.
        n_rows: Number of rows per file.

    Returns:
        A ``BenchmarkResult`` dict with total read time for all files.

    Raises:
        BenchmarkSetupError: If temp directory or file creation fails.
        BenchmarkRunError: If the benchmark execution fails.
    """
    try:
        tmpdir: str = tempfile.mkdtemp()
    except Exception as exc:
        raise BenchmarkSetupError(f"Failed to create temp directory: {exc}") from exc

    try:
        paths: list[str] = []
        for i in range(n_files):
            df: pd.DataFrame = _make_synthetic_ohlcv(n_rows, seed=i)
            table: pa.Table = pa.Table.from_pandas(
                df, schema=_SCHEMA, preserve_index=False
            )
            path: str = os.path.join(tmpdir, f"SYM{i:04d}.parquet")
            pq.write_table(table, path, compression="snappy")
            paths.append(path)

        # Warm up
        pq.read_table(paths[0]).to_pandas()

        t0: float = time.perf_counter()
        for path in paths:
            pq.read_table(path).to_pandas()
        elapsed: float = time.perf_counter() - t0
    except (BenchmarkSetupError, BenchmarkRunError):
        raise
    except Exception as exc:
        raise BenchmarkRunError(f"bench_bulk_read failed: {exc}") from exc
    finally:
        shutil.rmtree(tmpdir)

    return BenchmarkResult(
        name="parquet_bulk_read_50",
        value=elapsed,
        unit="seconds",
        detail=f"files={n_files}, rows_each={n_rows}",
    )


def bench_write_compression_comparison(
    n_rows: int = 5000,
) -> list[BenchmarkResult]:
    """Compare Parquet write speed and file size across compression codecs.

    Args:
        n_rows: Number of rows in the synthetic DataFrame.

    Returns:
        A list of ``BenchmarkResult`` dicts, one per compression codec.

    Raises:
        BenchmarkSetupError: If temp directory or data creation fails.
        BenchmarkRunError: If the benchmark execution fails.
    """
    try:
        df: pd.DataFrame = _make_synthetic_ohlcv(n_rows)
        table: pa.Table = pa.Table.from_pandas(df, schema=_SCHEMA, preserve_index=False)
    except BenchmarkSetupError:
        raise
    except Exception as exc:
        raise BenchmarkSetupError(
            f"Failed to prepare data for compression comparison: {exc}"
        ) from exc

    try:
        tmpdir: str = tempfile.mkdtemp()
    except Exception as exc:
        raise BenchmarkSetupError(f"Failed to create temp directory: {exc}") from exc

    results: list[BenchmarkResult] = []
    try:
        for comp in ["snappy", "gzip", "none"]:
            path: str = os.path.join(tmpdir, f"test_{comp}.parquet")
            t0: float = time.perf_counter()
            for _ in range(10):
                pq.write_table(table, path, compression=comp)
            write_time: float = (time.perf_counter() - t0) / 10

            file_size: int = os.path.getsize(path)

            t0 = time.perf_counter()
            for _ in range(10):
                pq.read_table(path).to_pandas()
            read_time: float = (time.perf_counter() - t0) / 10

            results.append(
                BenchmarkResult(
                    name=f"parquet_compression_{comp}",
                    value=write_time,
                    unit="seconds",
                    detail=f"write={write_time:.4f}s, read={read_time:.4f}s, size={file_size} bytes",
                )
            )
    except Exception as exc:
        raise BenchmarkRunError(
            f"bench_write_compression_comparison failed: {exc}"
        ) from exc
    finally:
        shutil.rmtree(tmpdir)

    return results


def run_all() -> list[BenchmarkResult]:
    """Execute every Parquet benchmark and return collected results.

    Returns:
        A list of ``BenchmarkResult`` dicts for all benchmarks.
    """
    results: list[BenchmarkResult] = []
    print("=" * 60)
    print("Parquet Benchmark Suite")
    print("=" * 60)

    for bench_fn in [bench_write_single, bench_read_single, bench_bulk_read]:
        try:
            r: BenchmarkResult = bench_fn()
            results.append(r)
            print(
                f"  {r['name']:30s} {r['value']:.6f} {r['unit']:15s}  ({r['detail']})"
            )
        except Exception as e:
            print(f"  {bench_fn.__name__:30s} ERROR: {e}")

    print("\n  Compression comparison:")
    try:
        for r in bench_write_compression_comparison():
            results.append(r)
            print(
                f"    {r['name']:28s} {r['value']:.6f} {r['unit']:15s}  ({r['detail']})"
            )
    except Exception as e:
        print(f"    compression_comparison ERROR: {e}")

    print()
    return results


if __name__ == "__main__":
    run_all()
