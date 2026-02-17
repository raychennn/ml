"""Profiler -- cProfile profiling for key system functions.

Profiles ``scan_timeframe``, ``prepare_for_dtw``, ``compute_similarity``,
and ``bulk_read_timeframe``.  Outputs a text summary and optional
flamegraph-compatible collapsed-stack format.

Usage::

    python -m benchmarks.profiler --target scan
    python -m benchmarks.profiler --target dtw
    python -m benchmarks.profiler --target normalize
    python -m benchmarks.profiler --target bulk_read
    python -m benchmarks.profiler --target all
"""

from __future__ import annotations

import argparse
import cProfile
import io
import os
import pstats
import shutil
import sys
import tempfile
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from benchmarks.exceptions import BenchmarkSetupError, ProfilerError

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


def _make_synthetic_ohlcv(n_bars: int = 500, seed: int = 42) -> pd.DataFrame:
    """Generate a synthetic OHLCV DataFrame for profiling.

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


def _setup_tmp_data(
    n_symbols: int = 20,
    timeframe: str = "4h",
) -> tuple[str, list[str]]:
    """Create a temporary data directory populated with synthetic Parquet files.

    Args:
        n_symbols: Number of synthetic symbols to generate.
        timeframe: Timeframe label used in the directory structure.

    Returns:
        A tuple of ``(tmpdir, symbols)`` where *tmpdir* is the root
        temporary directory and *symbols* is the list of symbol names.

    Raises:
        BenchmarkSetupError: If directory creation or Parquet writing fails.
    """
    try:
        tmpdir: str = tempfile.mkdtemp()
        tf_dir: str = os.path.join(tmpdir, "parquet", f"timeframe={timeframe}")
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
        return tmpdir, symbols
    except Exception as exc:
        raise BenchmarkSetupError(
            f"Failed to set up temporary profiling data: {exc}"
        ) from exc


def _print_stats(profiler: cProfile.Profile, top_n: int = 30) -> None:
    """Print the top cumulative-time entries from a cProfile run.

    Args:
        profiler: A completed cProfile profiler instance.
        top_n: Number of top entries to display.
    """
    s: io.StringIO = io.StringIO()
    ps: pstats.Stats = pstats.Stats(profiler, stream=s).sort_stats("cumulative")
    ps.print_stats(top_n)
    print(s.getvalue())


def _save_flamegraph_format(
    profiler: cProfile.Profile,
    output_path: str,
) -> None:
    """Save profile data in a collapsed-stack format for flamegraph tools.

    Args:
        profiler: A completed cProfile profiler instance.
        output_path: File path where the collapsed-stack output is written.
    """
    s: io.StringIO = io.StringIO()
    ps: pstats.Stats = pstats.Stats(profiler, stream=s).sort_stats("tottime")
    lines: list[str] = []
    for func, (cc, nc, tt, ct, callers) in ps.stats.items():
        filename: str
        lineno: int
        funcname: str
        filename, lineno, funcname = func
        short_file: str = os.path.basename(filename)
        lines.append(f"{short_file}:{funcname} {tt:.6f}")
    with open(output_path, "w") as f:
        f.write("\n".join(sorted(lines, key=lambda x: -float(x.split()[-1]))))
    print(f"Flamegraph data saved to: {output_path}")


def profile_scan(output_dir: str | None = None) -> None:
    """Profile ``scan_timeframe`` with synthetic data.

    Args:
        output_dir: Optional directory for flamegraph output files.

    Raises:
        ProfilerError: If profiling the scan target fails.
    """
    from config import SystemConfig
    from core.scanner_engine import ScannerEngine
    from data.data_manager import DataManager
    from references.reference_manager import ReferencePattern

    tmpdir: str
    symbols: list[str]
    tmpdir, symbols = _setup_tmp_data(20)
    try:
        config: SystemConfig = SystemConfig()
        config.data_root = tmpdir
        data_mgr: DataManager = DataManager(config)
        scanner: ScannerEngine = ScannerEngine(config, data_mgr)

        ref_df: pd.DataFrame = _make_synthetic_ohlcv(500, seed=0)
        ref: ReferencePattern = ReferencePattern(
            id="prof_ref",
            symbol=symbols[0],
            timeframe="4h",
            start_ts=int(ref_df["timestamp"].iloc[300]),
            end_ts=int(ref_df["timestamp"].iloc[420]),
            label="profile",
            description="Profile reference",
        )

        profiler: cProfile.Profile = cProfile.Profile()
        profiler.enable()
        scanner.scan_timeframe("4h", [ref])
        profiler.disable()

        print("\n=== Profile: scan_timeframe ===")
        _print_stats(profiler)
        if output_dir:
            _save_flamegraph_format(
                profiler, os.path.join(output_dir, "profile_scan.txt")
            )
    except BenchmarkSetupError:
        raise
    except Exception as exc:
        raise ProfilerError(f"profile_scan failed: {exc}") from exc
    finally:
        shutil.rmtree(tmpdir)


def profile_dtw(output_dir: str | None = None) -> None:
    """Profile ``compute_similarity`` over multiple iterations.

    Args:
        output_dir: Optional directory for flamegraph output files.

    Raises:
        ProfilerError: If profiling the DTW target fails.
    """
    try:
        from config import SystemConfig
        from core.dtw_calculator import DTWCalculator
        from core.data_normalizer import DataNormalizer

        config: SystemConfig = SystemConfig()
        calc: DTWCalculator = DTWCalculator(config)
        normalizer: DataNormalizer = DataNormalizer(config)

        ref_df: pd.DataFrame = _make_synthetic_ohlcv(120, seed=42)
        target_df: pd.DataFrame = _make_synthetic_ohlcv(500, seed=99)

        ref_prep: dict[str, Any] = normalizer.prepare_for_dtw(ref_df)
        target_prep: dict[str, Any] = normalizer.prepare_for_dtw(target_df)

        profiler: cProfile.Profile = cProfile.Profile()
        profiler.enable()
        for _ in range(20):
            calc.compute_similarity(
                ref_close_znorm=ref_prep["close_znorm"],
                ref_sma_diffs_znorm=ref_prep["sma_diffs_znorm"],
                ref_slope=ref_prep["slope"],
                target_close_znorm=target_prep["close_znorm"],
                target_sma_diffs_znorm=target_prep["sma_diffs_znorm"],
                target_slope=target_prep["slope"],
                symbol="TEST",
                timeframe="4h",
                target_df=target_prep["df"],
            )
        profiler.disable()

        print("\n=== Profile: compute_similarity (20 iterations) ===")
        _print_stats(profiler)
        if output_dir:
            _save_flamegraph_format(
                profiler, os.path.join(output_dir, "profile_dtw.txt")
            )
    except Exception as exc:
        raise ProfilerError(f"profile_dtw failed: {exc}") from exc


def profile_normalize(output_dir: str | None = None) -> None:
    """Profile ``prepare_for_dtw`` over many iterations.

    Args:
        output_dir: Optional directory for flamegraph output files.

    Raises:
        ProfilerError: If profiling the normalization target fails.
    """
    try:
        from config import SystemConfig
        from core.data_normalizer import DataNormalizer

        config: SystemConfig = SystemConfig()
        normalizer: DataNormalizer = DataNormalizer(config)
        df: pd.DataFrame = _make_synthetic_ohlcv(500, seed=42)

        profiler: cProfile.Profile = cProfile.Profile()
        profiler.enable()
        for _ in range(100):
            normalizer.prepare_for_dtw(df)
        profiler.disable()

        print("\n=== Profile: prepare_for_dtw (100 iterations) ===")
        _print_stats(profiler)
        if output_dir:
            _save_flamegraph_format(
                profiler, os.path.join(output_dir, "profile_normalize.txt")
            )
    except Exception as exc:
        raise ProfilerError(f"profile_normalize failed: {exc}") from exc


def profile_bulk_read(output_dir: str | None = None) -> None:
    """Profile ``bulk_read_timeframe`` with 50 synthetic symbols.

    Args:
        output_dir: Optional directory for flamegraph output files.

    Raises:
        ProfilerError: If profiling the bulk-read target fails.
    """
    from config import SystemConfig
    from data.data_manager import DataManager

    tmpdir: str
    symbols: list[str]
    tmpdir, symbols = _setup_tmp_data(50)
    try:
        config: SystemConfig = SystemConfig()
        config.data_root = tmpdir
        data_mgr: DataManager = DataManager(config)

        profiler: cProfile.Profile = cProfile.Profile()
        profiler.enable()
        for _ in range(5):
            data_mgr.bulk_read_timeframe("4h", tail_bars=200)
        profiler.disable()

        print("\n=== Profile: bulk_read_timeframe (5 iterations, 50 symbols) ===")
        _print_stats(profiler)
        if output_dir:
            _save_flamegraph_format(
                profiler, os.path.join(output_dir, "profile_bulk_read.txt")
            )
    except BenchmarkSetupError:
        raise
    except Exception as exc:
        raise ProfilerError(f"profile_bulk_read failed: {exc}") from exc
    finally:
        shutil.rmtree(tmpdir)


_TARGETS: dict[str, Any] = {
    "scan": profile_scan,
    "dtw": profile_dtw,
    "normalize": profile_normalize,
    "bulk_read": profile_bulk_read,
}


def main() -> None:
    """Parse CLI arguments and run the requested profiling target(s).

    Returns:
        None.
    """
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Profile crypto pattern system functions",
    )
    parser.add_argument(
        "--target",
        choices=list(_TARGETS.keys()) + ["all"],
        default="all",
        help="What to profile",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for flamegraph output files",
    )
    args: argparse.Namespace = parser.parse_args()

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    if args.target == "all":
        for name, fn in _TARGETS.items():
            print(f"\n{'=' * 60}")
            print(f"Profiling: {name}")
            print(f"{'=' * 60}")
            try:
                fn(args.output_dir)
            except Exception as e:
                print(f"  ERROR: {e}")
    else:
        _TARGETS[args.target](args.output_dir)


if __name__ == "__main__":
    main()
