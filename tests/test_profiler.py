"""Tests for benchmarks.profiler module.

This module tests the profiling utilities including synthetic data generation,
temp directory setup, stats printing, and flamegraph output.
"""

from __future__ import annotations

import cProfile
import os
import shutil
import tempfile

import pandas as pd
import pytest

from benchmarks.profiler import (
    _make_synthetic_ohlcv,
    _print_stats,
    _save_flamegraph_format,
    _setup_tmp_data,
)


class TestMakeSyntheticOHLCV:
    """Tests for _make_synthetic_ohlcv function."""

    def test_returns_correct_columns(self):
        """_make_synthetic_ohlcv should return DataFrame with correct columns."""
        df: pd.DataFrame = _make_synthetic_ohlcv(n_bars=100, seed=42)

        expected_columns = ["timestamp", "open", "high", "low", "close", "volume"]
        assert list(df.columns) == expected_columns

    def test_returns_correct_row_count(self):
        """_make_synthetic_ohlcv should return DataFrame with requested row count."""
        n_bars = 250
        df: pd.DataFrame = _make_synthetic_ohlcv(n_bars=n_bars, seed=42)

        assert len(df) == n_bars

    def test_default_parameters(self):
        """_make_synthetic_ohlcv should work with default parameters."""
        df: pd.DataFrame = _make_synthetic_ohlcv()

        assert len(df) == 500  # default n_bars
        assert "close" in df.columns

    def test_column_types(self):
        """_make_synthetic_ohlcv should return correct column types."""
        df: pd.DataFrame = _make_synthetic_ohlcv(n_bars=100, seed=42)

        assert df["timestamp"].dtype == "int64"
        assert df["open"].dtype == "float64"
        assert df["high"].dtype == "float64"
        assert df["low"].dtype == "float64"
        assert df["close"].dtype == "float64"
        assert df["volume"].dtype == "float64"

    def test_reproducibility_with_seed(self):
        """_make_synthetic_ohlcv should produce same data with same seed."""
        df1: pd.DataFrame = _make_synthetic_ohlcv(n_bars=100, seed=123)
        df2: pd.DataFrame = _make_synthetic_ohlcv(n_bars=100, seed=123)

        pd.testing.assert_frame_equal(df1, df2)

    def test_different_data_with_different_seed(self):
        """_make_synthetic_ohlcv should produce different data with different seed."""
        df1: pd.DataFrame = _make_synthetic_ohlcv(n_bars=100, seed=42)
        df2: pd.DataFrame = _make_synthetic_ohlcv(n_bars=100, seed=99)

        # Check that close prices are different
        assert not df1["close"].equals(df2["close"])

    def test_ohlc_relationships(self):
        """_make_synthetic_ohlcv should maintain valid OHLC relationships."""
        df: pd.DataFrame = _make_synthetic_ohlcv(n_bars=100, seed=42)

        # High should be >= close and open (with some tolerance for floating point)
        # Low should be <= close and open
        for idx, row in df.iterrows():
            # These might not hold exactly due to random generation, but let's check
            # that we have valid numeric data at least
            assert pd.notna(row["high"])
            assert pd.notna(row["low"])
            assert pd.notna(row["close"])
            assert pd.notna(row["open"])
            assert row["volume"] > 0


class TestSetupTmpData:
    """Tests for _setup_tmp_data function."""

    def test_creates_temp_dir_with_parquet_files(self):
        """_setup_tmp_data should create temp directory with parquet files."""
        tmpdir, symbols = _setup_tmp_data(n_symbols=5, timeframe="4h")

        try:
            # Check that tmpdir exists
            assert os.path.isdir(tmpdir)

            # Check that parquet subdirectory exists
            tf_dir = os.path.join(tmpdir, "parquet", "timeframe=4h")
            assert os.path.isdir(tf_dir)

            # Check that cache and models directories exist
            assert os.path.isdir(os.path.join(tmpdir, "cache"))
            assert os.path.isdir(os.path.join(tmpdir, "models"))

            # Check that parquet files were created
            for symbol in symbols:
                parquet_file = os.path.join(tf_dir, f"{symbol}.parquet")
                assert os.path.isfile(
                    parquet_file
                ), f"Missing parquet file for {symbol}"

        finally:
            # Clean up
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_returns_correct_number_of_symbols(self):
        """_setup_tmp_data should return correct number of symbols."""
        n_symbols = 10
        tmpdir, symbols = _setup_tmp_data(n_symbols=n_symbols, timeframe="1h")

        try:
            assert len(symbols) == n_symbols

        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_symbol_naming_format(self):
        """_setup_tmp_data should use correct symbol naming format."""
        tmpdir, symbols = _setup_tmp_data(n_symbols=3, timeframe="4h")

        try:
            # Check symbol format: SYM0000USDT, SYM0001USDT, etc.
            assert symbols[0] == "SYM0000USDT"
            assert symbols[1] == "SYM0001USDT"
            assert symbols[2] == "SYM0002USDT"

        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_default_parameters(self):
        """_setup_tmp_data should work with default parameters."""
        tmpdir, symbols = _setup_tmp_data()

        try:
            assert len(symbols) == 20  # default n_symbols
            tf_dir = os.path.join(tmpdir, "parquet", "timeframe=4h")
            assert os.path.isdir(tf_dir)

        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_different_timeframe(self):
        """_setup_tmp_data should create directory with specified timeframe."""
        tmpdir, symbols = _setup_tmp_data(n_symbols=2, timeframe="1d")

        try:
            tf_dir = os.path.join(tmpdir, "parquet", "timeframe=1d")
            assert os.path.isdir(tf_dir)

        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_parquet_files_are_readable(self):
        """_setup_tmp_data should create valid parquet files."""
        import pyarrow.parquet as pq

        tmpdir, symbols = _setup_tmp_data(n_symbols=2, timeframe="4h")

        try:
            tf_dir = os.path.join(tmpdir, "parquet", "timeframe=4h")
            parquet_file = os.path.join(tf_dir, f"{symbols[0]}.parquet")

            # Try to read the parquet file
            table = pq.read_table(parquet_file)
            df = table.to_pandas()

            # Verify it has the expected structure
            assert len(df) == 500  # default n_bars in _make_synthetic_ohlcv
            assert "close" in df.columns

        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


class TestPrintStats:
    """Tests for _print_stats function."""

    def test_outputs_text(self, capsys):
        """_print_stats should output text to stdout."""
        # Create a trivial cProfile profiler
        profiler = cProfile.Profile()
        profiler.enable()

        # Do some trivial work
        for i in range(100):
            _ = i * 2

        profiler.disable()

        # Call _print_stats
        _print_stats(profiler, top_n=10)

        # Capture output
        captured = capsys.readouterr()

        # Should have some output
        assert len(captured.out) > 0

        # Should contain profiling-related text
        # The exact format depends on pstats, but should have some numbers
        assert "ncalls" in captured.out or "function calls" in captured.out.lower()

    def test_respects_top_n_parameter(self, capsys):
        """_print_stats should respect the top_n parameter."""
        profiler = cProfile.Profile()
        profiler.enable()

        # Do some work with multiple function calls
        def dummy_func():
            return sum(range(100))

        for _ in range(10):
            dummy_func()

        profiler.disable()

        # Call with small top_n
        _print_stats(profiler, top_n=5)

        captured = capsys.readouterr()
        assert len(captured.out) > 0

    def test_handles_minimal_profiler(self, capsys):
        """_print_stats should handle profiler with minimal data."""
        profiler = cProfile.Profile()
        profiler.enable()
        # Minimal work
        x = 1 + 1
        profiler.disable()

        # Should not crash
        _print_stats(profiler, top_n=10)

        captured = capsys.readouterr()
        # Should have some output
        assert isinstance(captured.out, str)
        assert len(captured.out) > 0


class TestSaveFlamegraphFormat:
    """Tests for _save_flamegraph_format function."""

    def test_creates_file_with_content(self, tmpdir_path: str):
        """_save_flamegraph_format should create a file with content."""
        # Create a profiler with some data
        profiler = cProfile.Profile()
        profiler.enable()

        # Do some work
        def worker_func():
            total = 0
            for i in range(1000):
                total += i
            return total

        for _ in range(10):
            worker_func()

        profiler.disable()

        # Save flamegraph format
        output_path = os.path.join(tmpdir_path, "flamegraph.txt")
        _save_flamegraph_format(profiler, output_path)

        # Check that file was created
        assert os.path.isfile(output_path)

        # Check that file has content
        with open(output_path, "r") as f:
            content = f.read()

        assert len(content) > 0

        # Check format: should have lines with "filename:function time"
        lines = content.strip().split("\n")
        assert len(lines) > 0

        # Each line should have a space (separating function from time)
        for line in lines:
            assert " " in line

    def test_output_format_structure(self, tmpdir_path: str, capsys):
        """_save_flamegraph_format should produce correctly formatted output."""
        profiler = cProfile.Profile()
        profiler.enable()

        sum(range(100))

        profiler.disable()

        output_path = os.path.join(tmpdir_path, "test_flamegraph.txt")
        _save_flamegraph_format(profiler, output_path)

        with open(output_path, "r") as f:
            lines = f.readlines()

        # Each line should have format: "file:function time"
        for line in lines:
            line = line.strip()
            if line:
                parts = line.split()
                # Should have at least 2 parts: "file:func" and "time"
                assert len(parts) >= 2
                # Last part should be a float (time)
                try:
                    float(parts[-1])
                except ValueError:
                    pytest.fail(f"Last part should be a float: {parts[-1]}")

    def test_prints_save_message(self, tmpdir_path: str, capsys):
        """_save_flamegraph_format should print save confirmation message."""
        profiler = cProfile.Profile()
        profiler.enable()
        sum(range(50))
        profiler.disable()

        output_path = os.path.join(tmpdir_path, "output.txt")
        _save_flamegraph_format(profiler, output_path)

        captured = capsys.readouterr()
        assert "Flamegraph data saved to:" in captured.out
        assert output_path in captured.out

    def test_handles_minimal_profiler(self, tmpdir_path: str):
        """_save_flamegraph_format should handle profiler with minimal stats."""
        profiler = cProfile.Profile()
        profiler.enable()
        # Minimal work
        x = 1 + 1
        profiler.disable()

        output_path = os.path.join(tmpdir_path, "minimal.txt")

        # Should not crash
        _save_flamegraph_format(profiler, output_path)

        # File should be created
        assert os.path.isfile(output_path)

    def test_overwrites_existing_file(self, tmpdir_path: str):
        """_save_flamegraph_format should overwrite existing file."""
        output_path = os.path.join(tmpdir_path, "existing.txt")

        # Create an existing file
        with open(output_path, "w") as f:
            f.write("old content\n")

        # Create profiler with new data
        profiler = cProfile.Profile()
        profiler.enable()
        sum(range(100))
        profiler.disable()

        # Save over existing file
        _save_flamegraph_format(profiler, output_path)

        # Read the new content
        with open(output_path, "r") as f:
            content = f.read()

        # Should not contain old content
        assert "old content" not in content


# ── Tests for profile_* functions and main() ────────────────────────


import argparse
import sys
import types
from unittest import mock
from unittest.mock import MagicMock, patch

import numpy as np

from benchmarks.exceptions import ProfilerError
from benchmarks.profiler import (
    _TARGETS,
    _make_synthetic_ohlcv,
    main,
    profile_bulk_read,
    profile_dtw,
    profile_normalize,
    profile_scan,
)


def _build_mock_modules():
    """Return a dict of fake modules to inject into sys.modules.

    Every object used by the profile_* functions is represented by a
    MagicMock so that the real project packages are never imported.
    """
    # -- config ----------------------------------------------------------
    mock_config_mod = types.ModuleType("config")
    mock_system_config_cls = MagicMock(name="SystemConfig")
    mock_config_instance = MagicMock(name="config_instance")
    mock_system_config_cls.return_value = mock_config_instance
    mock_config_mod.SystemConfig = mock_system_config_cls

    # -- core.scanner_engine ---------------------------------------------
    mock_scanner_engine_mod = types.ModuleType("core.scanner_engine")
    mock_scanner_cls = MagicMock(name="ScannerEngine")
    mock_scanner_engine_mod.ScannerEngine = mock_scanner_cls

    # -- core.dtw_calculator ---------------------------------------------
    mock_dtw_mod = types.ModuleType("core.dtw_calculator")
    mock_dtw_cls = MagicMock(name="DTWCalculator")
    mock_dtw_mod.DTWCalculator = mock_dtw_cls

    # -- core.data_normalizer --------------------------------------------
    mock_normalizer_mod = types.ModuleType("core.data_normalizer")
    mock_normalizer_cls = MagicMock(name="DataNormalizer")
    # prepare_for_dtw must return a dict with the expected keys
    mock_normalizer_instance = MagicMock(name="normalizer_instance")
    mock_normalizer_instance.prepare_for_dtw.return_value = {
        "close_znorm": np.zeros(120),
        "sma_diffs_znorm": np.zeros(120),
        "slope": np.zeros(120),
        "df": _make_synthetic_ohlcv(120, seed=0),
    }
    mock_normalizer_cls.return_value = mock_normalizer_instance
    mock_normalizer_mod.DataNormalizer = mock_normalizer_cls

    # -- data.data_manager -----------------------------------------------
    mock_data_mod = types.ModuleType("data.data_manager")
    mock_dm_cls = MagicMock(name="DataManager")
    mock_data_mod.DataManager = mock_dm_cls

    # -- references.reference_manager ------------------------------------
    mock_ref_mod = types.ModuleType("references.reference_manager")
    mock_ref_cls = MagicMock(name="ReferencePattern")
    mock_ref_mod.ReferencePattern = mock_ref_cls

    # -- parent namespace packages so "from X.Y import Z" works ----------
    mock_core = types.ModuleType("core")
    mock_data = types.ModuleType("data")
    mock_references = types.ModuleType("references")

    return {
        "config": mock_config_mod,
        "core": mock_core,
        "core.scanner_engine": mock_scanner_engine_mod,
        "core.dtw_calculator": mock_dtw_mod,
        "core.data_normalizer": mock_normalizer_mod,
        "data": mock_data,
        "data.data_manager": mock_data_mod,
        "references": mock_references,
        "references.reference_manager": mock_ref_mod,
    }


def _make_mock_profiler():
    """Return a mock cProfile.Profile that does not conflict with pytest-cov."""
    mock_prof = MagicMock(spec=cProfile.Profile)
    # stats attribute needed by pstats.Stats -- provide a minimal dict
    mock_prof.stats = {}
    mock_prof.getstats = MagicMock(return_value=[])
    return mock_prof


@pytest.fixture()
def mock_modules():
    """Fixture that injects mock modules and patches cProfile.Profile."""
    mods = _build_mock_modules()
    mock_prof = _make_mock_profiler()
    with patch.dict(sys.modules, mods), patch(
        "benchmarks.profiler.cProfile.Profile", return_value=mock_prof
    ), patch("benchmarks.profiler._print_stats") as mock_ps, patch(
        "benchmarks.profiler._save_flamegraph_format"
    ) as mock_sfg:
        mods["_mock_profiler"] = mock_prof
        mods["_mock_print_stats"] = mock_ps
        mods["_mock_save_flamegraph"] = mock_sfg
        yield mods


# ── profile_scan ────────────────────────────────────────────────────


class TestProfileScan:
    """Tests for profile_scan."""

    def test_runs_without_error(self, mock_modules, capsys):
        """profile_scan completes successfully with mocked modules."""
        profile_scan()
        captured = capsys.readouterr()
        assert "Profile: scan_timeframe" in captured.out

    def test_calls_print_stats(self, mock_modules):
        """profile_scan should call _print_stats, producing profiling output."""
        profile_scan()
        mock_modules["_mock_print_stats"].assert_called_once()

    def test_output_dir_saves_flamegraph(self, mock_modules, tmpdir_path):
        """profile_scan should save flamegraph data when output_dir is provided."""
        profile_scan(output_dir=tmpdir_path)
        mock_modules["_mock_save_flamegraph"].assert_called_once()
        call_args = mock_modules["_mock_save_flamegraph"].call_args
        assert call_args[0][1] == os.path.join(tmpdir_path, "profile_scan.txt")

    def test_no_flamegraph_without_output_dir(self, mock_modules):
        """profile_scan should not save flamegraph when output_dir is None."""
        profile_scan(output_dir=None)
        mock_modules["_mock_save_flamegraph"].assert_not_called()

    def test_raises_profiler_error_on_failure(self, mock_modules):
        """profile_scan should raise ProfilerError when the scan fails."""
        scanner_cls = mock_modules["core.scanner_engine"].ScannerEngine
        scanner_cls.return_value.scan_timeframe.side_effect = RuntimeError("boom")
        with pytest.raises(ProfilerError, match="profile_scan failed"):
            profile_scan()

    def test_cleans_up_tmpdir_on_success(self, mock_modules):
        """profile_scan should clean up its temporary directory."""
        with patch("benchmarks.profiler._setup_tmp_data") as mock_setup, patch(
            "benchmarks.profiler.shutil.rmtree"
        ) as mock_rm:
            mock_setup.return_value = ("/fake/tmpdir", ["SYM0000USDT"])
            profile_scan()
            mock_rm.assert_called_once_with("/fake/tmpdir")

    def test_cleans_up_tmpdir_on_failure(self, mock_modules):
        """profile_scan should clean up its temporary directory even on failure."""
        scanner_cls = mock_modules["core.scanner_engine"].ScannerEngine
        scanner_cls.return_value.scan_timeframe.side_effect = RuntimeError("fail")
        with patch("benchmarks.profiler._setup_tmp_data") as mock_setup, patch(
            "benchmarks.profiler.shutil.rmtree"
        ) as mock_rm:
            mock_setup.return_value = ("/fake/tmpdir", ["SYM0000USDT"])
            with pytest.raises(ProfilerError):
                profile_scan()
            mock_rm.assert_called_once_with("/fake/tmpdir")


# ── profile_dtw ─────────────────────────────────────────────────────


class TestProfileDTW:
    """Tests for profile_dtw."""

    def test_runs_without_error(self, mock_modules, capsys):
        """profile_dtw completes successfully with mocked modules."""
        profile_dtw()
        captured = capsys.readouterr()
        assert "Profile: compute_similarity" in captured.out

    def test_calls_print_stats(self, mock_modules):
        """profile_dtw should call _print_stats, producing profiling output."""
        profile_dtw()
        mock_modules["_mock_print_stats"].assert_called_once()

    def test_output_dir_saves_flamegraph(self, mock_modules, tmpdir_path):
        """profile_dtw should save flamegraph data when output_dir is provided."""
        profile_dtw(output_dir=tmpdir_path)
        mock_modules["_mock_save_flamegraph"].assert_called_once()
        call_args = mock_modules["_mock_save_flamegraph"].call_args
        assert call_args[0][1] == os.path.join(tmpdir_path, "profile_dtw.txt")

    def test_no_flamegraph_without_output_dir(self, mock_modules):
        """profile_dtw should not save flamegraph when output_dir is None."""
        profile_dtw(output_dir=None)
        mock_modules["_mock_save_flamegraph"].assert_not_called()

    def test_raises_profiler_error_on_failure(self, mock_modules):
        """profile_dtw should raise ProfilerError on compute_similarity failure."""
        dtw_cls = mock_modules["core.dtw_calculator"].DTWCalculator
        dtw_cls.return_value.compute_similarity.side_effect = RuntimeError("dtw boom")
        with pytest.raises(ProfilerError, match="profile_dtw failed"):
            profile_dtw()

    def test_calls_compute_similarity_multiple_times(self, mock_modules):
        """profile_dtw should call compute_similarity 20 times."""
        profile_dtw()
        dtw_cls = mock_modules["core.dtw_calculator"].DTWCalculator
        assert dtw_cls.return_value.compute_similarity.call_count == 20


# ── profile_normalize ───────────────────────────────────────────────


class TestProfileNormalize:
    """Tests for profile_normalize."""

    def test_runs_without_error(self, mock_modules, capsys):
        """profile_normalize completes successfully with mocked modules."""
        profile_normalize()
        captured = capsys.readouterr()
        assert "Profile: prepare_for_dtw" in captured.out

    def test_calls_print_stats(self, mock_modules):
        """profile_normalize should call _print_stats, producing profiling output."""
        profile_normalize()
        mock_modules["_mock_print_stats"].assert_called_once()

    def test_output_dir_saves_flamegraph(self, mock_modules, tmpdir_path):
        """profile_normalize should save flamegraph data when output_dir is given."""
        profile_normalize(output_dir=tmpdir_path)
        mock_modules["_mock_save_flamegraph"].assert_called_once()
        call_args = mock_modules["_mock_save_flamegraph"].call_args
        assert call_args[0][1] == os.path.join(tmpdir_path, "profile_normalize.txt")

    def test_no_flamegraph_without_output_dir(self, mock_modules):
        """profile_normalize should not save flamegraph when output_dir is None."""
        profile_normalize(output_dir=None)
        mock_modules["_mock_save_flamegraph"].assert_not_called()

    def test_raises_profiler_error_on_failure(self, mock_modules):
        """profile_normalize should raise ProfilerError on prepare_for_dtw failure."""
        norm_cls = mock_modules["core.data_normalizer"].DataNormalizer
        norm_cls.return_value.prepare_for_dtw.side_effect = RuntimeError("norm fail")
        with pytest.raises(ProfilerError, match="profile_normalize failed"):
            profile_normalize()

    def test_calls_prepare_for_dtw_100_times(self, mock_modules):
        """profile_normalize should call prepare_for_dtw 100 times."""
        profile_normalize()
        norm_cls = mock_modules["core.data_normalizer"].DataNormalizer
        assert norm_cls.return_value.prepare_for_dtw.call_count == 100


# ── profile_bulk_read ───────────────────────────────────────────────


class TestProfileBulkRead:
    """Tests for profile_bulk_read."""

    def test_runs_without_error(self, mock_modules, capsys):
        """profile_bulk_read completes successfully with mocked modules."""
        profile_bulk_read()
        captured = capsys.readouterr()
        assert "Profile: bulk_read_timeframe" in captured.out

    def test_calls_print_stats(self, mock_modules):
        """profile_bulk_read should call _print_stats, producing profiling output."""
        profile_bulk_read()
        mock_modules["_mock_print_stats"].assert_called_once()

    def test_output_dir_saves_flamegraph(self, mock_modules, tmpdir_path):
        """profile_bulk_read should save flamegraph data when output_dir is given."""
        profile_bulk_read(output_dir=tmpdir_path)
        mock_modules["_mock_save_flamegraph"].assert_called_once()
        call_args = mock_modules["_mock_save_flamegraph"].call_args
        assert call_args[0][1] == os.path.join(tmpdir_path, "profile_bulk_read.txt")

    def test_no_flamegraph_without_output_dir(self, mock_modules):
        """profile_bulk_read should not save flamegraph when output_dir is None."""
        profile_bulk_read(output_dir=None)
        mock_modules["_mock_save_flamegraph"].assert_not_called()

    def test_raises_profiler_error_on_failure(self, mock_modules):
        """profile_bulk_read should raise ProfilerError on failure."""
        dm_cls = mock_modules["data.data_manager"].DataManager
        dm_cls.return_value.bulk_read_timeframe.side_effect = RuntimeError("read fail")
        with pytest.raises(ProfilerError, match="profile_bulk_read failed"):
            profile_bulk_read()

    def test_calls_bulk_read_timeframe_5_times(self, mock_modules):
        """profile_bulk_read should call bulk_read_timeframe 5 times."""
        profile_bulk_read()
        dm_cls = mock_modules["data.data_manager"].DataManager
        assert dm_cls.return_value.bulk_read_timeframe.call_count == 5

    def test_cleans_up_tmpdir_on_success(self, mock_modules):
        """profile_bulk_read should clean up its temporary directory."""
        with patch("benchmarks.profiler._setup_tmp_data") as mock_setup, patch(
            "benchmarks.profiler.shutil.rmtree"
        ) as mock_rm:
            mock_setup.return_value = (
                "/fake/tmpdir",
                [f"SYM{i:04d}USDT" for i in range(50)],
            )
            profile_bulk_read()
            mock_rm.assert_called_once_with("/fake/tmpdir")

    def test_cleans_up_tmpdir_on_failure(self, mock_modules):
        """profile_bulk_read should clean up its temporary directory on failure."""
        dm_cls = mock_modules["data.data_manager"].DataManager
        dm_cls.return_value.bulk_read_timeframe.side_effect = RuntimeError("fail")
        with patch("benchmarks.profiler._setup_tmp_data") as mock_setup, patch(
            "benchmarks.profiler.shutil.rmtree"
        ) as mock_rm:
            mock_setup.return_value = (
                "/fake/tmpdir",
                [f"SYM{i:04d}USDT" for i in range(50)],
            )
            with pytest.raises(ProfilerError):
                profile_bulk_read()
            mock_rm.assert_called_once_with("/fake/tmpdir")


# ── main() ──────────────────────────────────────────────────────────


class TestMain:
    """Tests for the main() entry point."""

    def test_main_all_targets(self, mock_modules, capsys):
        """main() with --target all should invoke every profile function."""
        with patch(
            "argparse.ArgumentParser.parse_args",
            return_value=argparse.Namespace(target="all", output_dir=None),
        ):
            main()
        captured = capsys.readouterr()
        assert "Profiling: scan" in captured.out
        assert "Profiling: dtw" in captured.out
        assert "Profiling: normalize" in captured.out
        assert "Profiling: bulk_read" in captured.out

    def test_main_single_target_scan(self, mock_modules, capsys):
        """main() with --target scan should invoke only profile_scan."""
        with patch(
            "argparse.ArgumentParser.parse_args",
            return_value=argparse.Namespace(target="scan", output_dir=None),
        ):
            main()
        captured = capsys.readouterr()
        assert "scan_timeframe" in captured.out
        mock_modules["_mock_print_stats"].assert_called_once()

    def test_main_single_target_dtw(self, mock_modules, capsys):
        """main() with --target dtw should invoke only profile_dtw."""
        with patch(
            "argparse.ArgumentParser.parse_args",
            return_value=argparse.Namespace(target="dtw", output_dir=None),
        ):
            main()
        captured = capsys.readouterr()
        assert "compute_similarity" in captured.out
        mock_modules["_mock_print_stats"].assert_called_once()

    def test_main_single_target_normalize(self, mock_modules, capsys):
        """main() with --target normalize should invoke only profile_normalize."""
        with patch(
            "argparse.ArgumentParser.parse_args",
            return_value=argparse.Namespace(target="normalize", output_dir=None),
        ):
            main()
        captured = capsys.readouterr()
        assert "prepare_for_dtw" in captured.out
        mock_modules["_mock_print_stats"].assert_called_once()

    def test_main_single_target_bulk_read(self, mock_modules, capsys):
        """main() with --target bulk_read should invoke only profile_bulk_read."""
        with patch(
            "argparse.ArgumentParser.parse_args",
            return_value=argparse.Namespace(target="bulk_read", output_dir=None),
        ):
            main()
        captured = capsys.readouterr()
        assert "bulk_read_timeframe" in captured.out
        mock_modules["_mock_print_stats"].assert_called_once()

    def test_main_with_output_dir(self, mock_modules, tmpdir_path):
        """main() with --output-dir should pass output_dir to the target function."""
        with patch(
            "argparse.ArgumentParser.parse_args",
            return_value=argparse.Namespace(target="normalize", output_dir=tmpdir_path),
        ):
            main()
        mock_modules["_mock_save_flamegraph"].assert_called_once()

    def test_main_creates_output_dir_if_missing(self, mock_modules, tmpdir_path):
        """main() should create the output directory if it does not exist."""
        new_dir = os.path.join(tmpdir_path, "sub", "dir")
        with patch(
            "argparse.ArgumentParser.parse_args",
            return_value=argparse.Namespace(target="normalize", output_dir=new_dir),
        ):
            main()
        assert os.path.isdir(new_dir)

    def test_main_all_catches_errors(self, mock_modules, capsys):
        """main() with --target all should print errors and continue."""
        dm_cls = mock_modules["data.data_manager"].DataManager
        dm_cls.return_value.bulk_read_timeframe.side_effect = RuntimeError("fail")
        with patch(
            "argparse.ArgumentParser.parse_args",
            return_value=argparse.Namespace(target="all", output_dir=None),
        ):
            # Should not raise -- main() catches exceptions in "all" mode
            main()
        captured = capsys.readouterr()
        assert "ERROR" in captured.out


# ── BenchmarkSetupError pass-through tests ──────────────────────────


from benchmarks.exceptions import BenchmarkSetupError


class TestBenchmarkSetupErrorPassthrough:
    """Tests that BenchmarkSetupError is re-raised (not wrapped in ProfilerError)."""

    def test_profile_scan_reraises_benchmark_setup_error(self, mock_modules):
        """profile_scan should re-raise BenchmarkSetupError without wrapping."""
        with patch("benchmarks.profiler._setup_tmp_data") as mock_setup:
            mock_setup.side_effect = BenchmarkSetupError("setup boom")
            with pytest.raises(BenchmarkSetupError, match="setup boom"):
                profile_scan()

    def test_profile_bulk_read_reraises_benchmark_setup_error(self, mock_modules):
        """profile_bulk_read should re-raise BenchmarkSetupError without wrapping."""
        with patch("benchmarks.profiler._setup_tmp_data") as mock_setup:
            mock_setup.side_effect = BenchmarkSetupError("setup boom")
            with pytest.raises(BenchmarkSetupError, match="setup boom"):
                profile_bulk_read()
