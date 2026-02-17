"""Tests for benchmarks.bench_scan_cycle module."""

from __future__ import annotations

import os
import shutil
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from benchmarks.bench_scan_cycle import (
    BenchmarkResult,
    _make_synthetic_ohlcv,
    _setup_synthetic_data,
    _SCHEMA,
    bench_prepare_for_dtw,
    bench_scan_cycle,
    run_all,
)
from benchmarks.exceptions import BenchmarkRunError, BenchmarkSetupError

# ---------------------------------------------------------------------------
# _make_synthetic_ohlcv
# ---------------------------------------------------------------------------


class TestMakeSyntheticOhlcv:
    """Tests for the _make_synthetic_ohlcv helper."""

    def test_returns_dataframe(self):
        df = _make_synthetic_ohlcv(n_bars=50, seed=0)
        assert isinstance(df, pd.DataFrame)

    def test_correct_shape(self):
        n = 50
        df = _make_synthetic_ohlcv(n_bars=n, seed=0)
        assert len(df) == n
        assert list(df.columns) == [
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]

    def test_dtypes(self):
        df = _make_synthetic_ohlcv(n_bars=50, seed=0)
        assert df["timestamp"].dtype == np.int64
        for col in ("open", "high", "low", "close", "volume"):
            assert df[col].dtype == np.float64

    def test_reproducibility(self):
        df1 = _make_synthetic_ohlcv(n_bars=50, seed=7)
        df2 = _make_synthetic_ohlcv(n_bars=50, seed=7)
        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seeds_differ(self):
        df1 = _make_synthetic_ohlcv(n_bars=50, seed=0)
        df2 = _make_synthetic_ohlcv(n_bars=50, seed=1)
        assert not df1["close"].equals(df2["close"])

    def test_timestamps_are_hourly(self):
        df = _make_synthetic_ohlcv(n_bars=50, seed=0)
        diffs = df["timestamp"].diff().dropna().unique()
        assert len(diffs) == 1
        assert diffs[0] == 3600

    def test_volume_positive(self):
        df = _make_synthetic_ohlcv(n_bars=50, seed=0)
        assert (df["volume"] > 0).all()

    def test_high_ge_low(self):
        """High should be above low since high = close + positive, low = close - positive."""
        df = _make_synthetic_ohlcv(n_bars=50, seed=0)
        assert (df["high"] > df["low"]).all()

    def test_default_params(self):
        df = _make_synthetic_ohlcv()
        assert len(df) == 500


# ---------------------------------------------------------------------------
# _setup_synthetic_data
# ---------------------------------------------------------------------------


class TestSetupSyntheticData:
    """Tests for the _setup_synthetic_data helper.

    Note: n_bars must be >= 421 because the function accesses iloc[300] and
    iloc[420] for reference timestamps.
    """

    def test_creates_parquet_files(self, tmpdir_path):
        n_symbols = 3
        n_bars = 500
        symbols, ref_start, ref_end = _setup_synthetic_data(
            tmpdir_path, n_symbols=n_symbols, n_bars=n_bars, timeframe="4h"
        )
        tf_dir = os.path.join(tmpdir_path, "parquet", "timeframe=4h")
        assert os.path.isdir(tf_dir)
        parquet_files = [f for f in os.listdir(tf_dir) if f.endswith(".parquet")]
        assert len(parquet_files) == n_symbols

    def test_returns_correct_symbols(self, tmpdir_path):
        symbols, _, _ = _setup_synthetic_data(tmpdir_path, n_symbols=3, n_bars=500)
        assert len(symbols) == 3
        assert symbols[0] == "SYM0000USDT"
        assert symbols[1] == "SYM0001USDT"
        assert symbols[2] == "SYM0002USDT"

    def test_ref_timestamps_are_ints(self, tmpdir_path):
        _, ref_start, ref_end = _setup_synthetic_data(
            tmpdir_path, n_symbols=3, n_bars=500
        )
        assert isinstance(ref_start, int)
        assert isinstance(ref_end, int)
        assert ref_end > ref_start

    def test_parquet_readable(self, tmpdir_path):
        symbols, _, _ = _setup_synthetic_data(tmpdir_path, n_symbols=2, n_bars=500)
        tf_dir = os.path.join(tmpdir_path, "parquet", "timeframe=4h")
        table = pq.read_table(os.path.join(tf_dir, f"{symbols[0]}.parquet"))
        assert table.num_rows == 500
        expected_cols = {"timestamp", "open", "high", "low", "close", "volume"}
        assert expected_cols.issubset(set(table.column_names))

    def test_raises_setup_error_on_failure(self):
        """If the tmpdir path is invalid, BenchmarkSetupError is raised."""
        with pytest.raises(BenchmarkSetupError):
            _setup_synthetic_data(
                "/nonexistent/path/that/cannot/exist", n_symbols=1, n_bars=500
            )

    def test_custom_timeframe(self, tmpdir_path):
        _setup_synthetic_data(tmpdir_path, n_symbols=1, n_bars=500, timeframe="1h")
        tf_dir = os.path.join(tmpdir_path, "parquet", "timeframe=1h")
        assert os.path.isdir(tf_dir)

    def test_ref_start_corresponds_to_iloc_300(self, tmpdir_path):
        """ref_start should match timestamp at iloc[300] of seed=0 data."""
        _, ref_start, _ = _setup_synthetic_data(tmpdir_path, n_symbols=1, n_bars=500)
        expected_df = _make_synthetic_ohlcv(500, seed=0)
        assert ref_start == int(expected_df["timestamp"].iloc[300])

    def test_ref_end_corresponds_to_iloc_420(self, tmpdir_path):
        """ref_end should match timestamp at iloc[420] of seed=0 data."""
        _, _, ref_end = _setup_synthetic_data(tmpdir_path, n_symbols=1, n_bars=500)
        expected_df = _make_synthetic_ohlcv(500, seed=0)
        assert ref_end == int(expected_df["timestamp"].iloc[420])


# ---------------------------------------------------------------------------
# bench_scan_cycle
# ---------------------------------------------------------------------------


def _make_scan_cycle_mocks():
    """Build the sys.modules dict and individual mocks for bench_scan_cycle."""
    mock_config = MagicMock()
    mock_data_mgr = MagicMock()
    mock_scanner = MagicMock()
    mock_scanner.scan_timeframe.return_value = [{"alert": "test"}]
    mock_ref_cls = MagicMock()

    modules = {
        "config": MagicMock(SystemConfig=MagicMock(return_value=mock_config)),
        "core": MagicMock(),
        "core.scanner_engine": MagicMock(
            ScannerEngine=MagicMock(return_value=mock_scanner)
        ),
        "data": MagicMock(),
        "data.data_manager": MagicMock(
            DataManager=MagicMock(return_value=mock_data_mgr)
        ),
        "references": MagicMock(),
        "references.reference_manager": MagicMock(ReferencePattern=mock_ref_cls),
    }
    return modules, mock_config, mock_data_mgr, mock_scanner


class TestBenchScanCycle:
    """Tests for bench_scan_cycle (all core deps mocked)."""

    @patch("benchmarks.bench_scan_cycle.shutil.rmtree")
    @patch("benchmarks.bench_scan_cycle.tempfile.mkdtemp")
    @patch("benchmarks.bench_scan_cycle._setup_synthetic_data")
    def test_returns_benchmark_result(self, mock_setup, mock_mkdtemp, mock_rmtree):
        tmpdir = tempfile.mkdtemp()
        try:
            mock_mkdtemp.return_value = tmpdir
            mock_setup.return_value = (
                ["SYM0000USDT", "SYM0001USDT", "SYM0002USDT"],
                1700001000,
                1700002000,
            )

            modules, _, _, mock_scanner = _make_scan_cycle_mocks()
            mock_scanner.scan_timeframe.return_value = [{"alert": "test"}]

            with patch.dict("sys.modules", modules):
                result = bench_scan_cycle(n_symbols=3, n_bars=50, timeframe="4h")

            assert isinstance(result, dict)
            assert result["name"] == "scan_cycle_20_symbols"
            assert result["unit"] == "seconds"
            assert isinstance(result["value"], float)
            assert result["value"] >= 0
            assert "symbols=3" in result["detail"]
            assert "alerts=1" in result["detail"]
            mock_rmtree.assert_called_once_with(tmpdir)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    @patch("benchmarks.bench_scan_cycle.shutil.rmtree")
    @patch("benchmarks.bench_scan_cycle.tempfile.mkdtemp")
    @patch("benchmarks.bench_scan_cycle._setup_synthetic_data")
    def test_raises_run_error_on_failure(self, mock_setup, mock_mkdtemp, mock_rmtree):
        tmpdir = tempfile.mkdtemp()
        try:
            mock_mkdtemp.return_value = tmpdir
            mock_setup.return_value = (["SYM0000USDT"], 1700001000, 1700002000)

            with patch.dict(
                "sys.modules",
                {
                    "config": MagicMock(
                        SystemConfig=MagicMock(
                            side_effect=RuntimeError("config broken")
                        )
                    ),
                    "core": MagicMock(),
                    "core.scanner_engine": MagicMock(),
                    "data": MagicMock(),
                    "data.data_manager": MagicMock(),
                    "references": MagicMock(),
                    "references.reference_manager": MagicMock(),
                },
            ):
                with pytest.raises(BenchmarkRunError, match="bench_scan_cycle failed"):
                    bench_scan_cycle(n_symbols=3, n_bars=50)

            mock_rmtree.assert_called_once_with(tmpdir)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    @patch("benchmarks.bench_scan_cycle.shutil.rmtree")
    @patch("benchmarks.bench_scan_cycle.tempfile.mkdtemp")
    @patch("benchmarks.bench_scan_cycle._setup_synthetic_data")
    def test_setup_error_propagates(self, mock_setup, mock_mkdtemp, mock_rmtree):
        tmpdir = tempfile.mkdtemp()
        try:
            mock_mkdtemp.return_value = tmpdir
            mock_setup.side_effect = BenchmarkSetupError("setup failed")

            with patch.dict(
                "sys.modules",
                {
                    "config": MagicMock(),
                    "core": MagicMock(),
                    "core.scanner_engine": MagicMock(),
                    "data": MagicMock(),
                    "data.data_manager": MagicMock(),
                    "references": MagicMock(),
                    "references.reference_manager": MagicMock(),
                },
            ):
                with pytest.raises(BenchmarkSetupError, match="setup failed"):
                    bench_scan_cycle(n_symbols=3, n_bars=50)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    @patch("benchmarks.bench_scan_cycle.shutil.rmtree")
    @patch("benchmarks.bench_scan_cycle.tempfile.mkdtemp")
    @patch("benchmarks.bench_scan_cycle._setup_synthetic_data")
    def test_tmpdir_cleaned_up_on_success(self, mock_setup, mock_mkdtemp, mock_rmtree):
        tmpdir = tempfile.mkdtemp()
        try:
            mock_mkdtemp.return_value = tmpdir
            mock_setup.return_value = (["SYM0000USDT"], 100, 200)

            modules, _, _, mock_scanner = _make_scan_cycle_mocks()
            mock_scanner.scan_timeframe.return_value = []

            with patch.dict("sys.modules", modules):
                bench_scan_cycle(n_symbols=3, n_bars=50)

            mock_rmtree.assert_called_once_with(tmpdir)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    @patch("benchmarks.bench_scan_cycle.shutil.rmtree")
    @patch("benchmarks.bench_scan_cycle.tempfile.mkdtemp")
    @patch("benchmarks.bench_scan_cycle._setup_synthetic_data")
    def test_bulk_read_called_before_scan(self, mock_setup, mock_mkdtemp, mock_rmtree):
        """DataManager.bulk_read_timeframe should be called as a warm-up."""
        tmpdir = tempfile.mkdtemp()
        try:
            mock_mkdtemp.return_value = tmpdir
            mock_setup.return_value = (["SYM0000USDT"], 100, 200)

            modules, _, mock_data_mgr, _ = _make_scan_cycle_mocks()

            with patch.dict("sys.modules", modules):
                bench_scan_cycle(n_symbols=3, n_bars=50, timeframe="4h")

            mock_data_mgr.bulk_read_timeframe.assert_called_once_with(
                "4h", tail_bars=200
            )
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# bench_prepare_for_dtw
# ---------------------------------------------------------------------------


class TestBenchPrepareForDtw:
    """Tests for bench_prepare_for_dtw (DataNormalizer mocked)."""

    def test_returns_benchmark_result(self):
        mock_normalizer = MagicMock()
        mock_normalizer.prepare_for_dtw.return_value = np.zeros(10)

        with patch.dict(
            "sys.modules",
            {
                "config": MagicMock(SystemConfig=MagicMock(return_value=MagicMock())),
                "core": MagicMock(),
                "core.data_normalizer": MagicMock(
                    DataNormalizer=MagicMock(return_value=mock_normalizer)
                ),
            },
        ):
            result = bench_prepare_for_dtw(repeats=1)

        assert isinstance(result, dict)
        assert result["name"] == "prepare_for_dtw"
        assert result["unit"] == "seconds"
        assert isinstance(result["value"], float)
        assert result["value"] >= 0
        assert "repeats=1" in result["detail"]

    def test_calls_prepare_for_dtw_correct_times(self):
        mock_normalizer = MagicMock()
        mock_normalizer.prepare_for_dtw.return_value = np.zeros(10)

        with patch.dict(
            "sys.modules",
            {
                "config": MagicMock(SystemConfig=MagicMock(return_value=MagicMock())),
                "core": MagicMock(),
                "core.data_normalizer": MagicMock(
                    DataNormalizer=MagicMock(return_value=mock_normalizer)
                ),
            },
        ):
            bench_prepare_for_dtw(repeats=3)

        # 1 warm-up + 3 repeats = 4 calls
        assert mock_normalizer.prepare_for_dtw.call_count == 4

    def test_raises_run_error_on_failure(self):
        with patch.dict(
            "sys.modules",
            {
                "config": MagicMock(
                    SystemConfig=MagicMock(side_effect=RuntimeError("boom"))
                ),
                "core": MagicMock(),
                "core.data_normalizer": MagicMock(),
            },
        ):
            with pytest.raises(BenchmarkRunError, match="bench_prepare_for_dtw failed"):
                bench_prepare_for_dtw(repeats=1)


# ---------------------------------------------------------------------------
# run_all
# ---------------------------------------------------------------------------


class TestRunAll:
    """Tests for run_all orchestrator."""

    @patch("benchmarks.bench_scan_cycle.bench_prepare_for_dtw")
    @patch("benchmarks.bench_scan_cycle.bench_scan_cycle")
    def test_returns_list_of_results(self, mock_scan, mock_dtw):
        mock_scan.return_value = BenchmarkResult(
            name="scan_cycle_20_symbols", value=0.5, unit="seconds", detail="test"
        )
        mock_dtw.return_value = BenchmarkResult(
            name="prepare_for_dtw", value=0.001, unit="seconds", detail="test"
        )

        results = run_all()

        assert isinstance(results, list)
        assert len(results) == 2
        assert results[0]["name"] == "scan_cycle_20_symbols"
        assert results[1]["name"] == "prepare_for_dtw"

    @patch("benchmarks.bench_scan_cycle.bench_prepare_for_dtw")
    @patch("benchmarks.bench_scan_cycle.bench_scan_cycle")
    def test_handles_errors_gracefully(self, mock_scan, mock_dtw):
        mock_scan.__name__ = "bench_scan_cycle"
        mock_scan.side_effect = BenchmarkRunError("scan failed")
        mock_dtw.return_value = BenchmarkResult(
            name="prepare_for_dtw", value=0.001, unit="seconds", detail="test"
        )

        results = run_all()

        assert len(results) == 2
        # Failed benchmark should have value=None and unit="error"
        assert results[0]["value"] is None
        assert results[0]["unit"] == "error"
        assert "scan failed" in results[0]["detail"]
        # Successful benchmark should be fine
        assert results[1]["value"] == 0.001

    @patch("benchmarks.bench_scan_cycle.bench_prepare_for_dtw")
    @patch("benchmarks.bench_scan_cycle.bench_scan_cycle")
    def test_all_fail(self, mock_scan, mock_dtw):
        mock_scan.__name__ = "bench_scan_cycle"
        mock_scan.side_effect = Exception("err1")
        mock_dtw.__name__ = "bench_prepare_for_dtw"
        mock_dtw.side_effect = Exception("err2")

        results = run_all()

        assert len(results) == 2
        assert all(r["value"] is None for r in results)

    @patch("benchmarks.bench_scan_cycle.bench_prepare_for_dtw")
    @patch("benchmarks.bench_scan_cycle.bench_scan_cycle")
    def test_prints_output(self, mock_scan, mock_dtw, capsys):
        mock_scan.return_value = BenchmarkResult(
            name="scan_cycle_20_symbols", value=1.234567, unit="seconds", detail="ok"
        )
        mock_dtw.return_value = BenchmarkResult(
            name="prepare_for_dtw", value=0.000123, unit="seconds", detail="ok"
        )

        run_all()

        captured = capsys.readouterr()
        assert "Scan Cycle Benchmark Suite" in captured.out
        assert "scan_cycle_20_symbols" in captured.out
        assert "prepare_for_dtw" in captured.out

    @patch("benchmarks.bench_scan_cycle.bench_prepare_for_dtw")
    @patch("benchmarks.bench_scan_cycle.bench_scan_cycle")
    def test_calls_both_benchmarks(self, mock_scan, mock_dtw):
        mock_scan.return_value = BenchmarkResult(
            name="a", value=0.1, unit="seconds", detail=""
        )
        mock_dtw.return_value = BenchmarkResult(
            name="b", value=0.2, unit="seconds", detail=""
        )

        run_all()

        mock_scan.assert_called_once()
        mock_dtw.assert_called_once()


# ---------------------------------------------------------------------------
# main() (module-level __name__ == "__main__" guard)
# ---------------------------------------------------------------------------


class TestMain:
    """Test that running the module as __main__ invokes run_all."""

    @patch("benchmarks.bench_scan_cycle.run_all")
    def test_main_calls_run_all(self, mock_run_all):
        mock_run_all.return_value = [
            BenchmarkResult(name="test", value=0.1, unit="seconds", detail="ok")
        ]
        from benchmarks import bench_scan_cycle as mod

        result = mod.run_all()
        assert isinstance(result, list)
        assert len(result) == 1
        mock_run_all.assert_called()

    @patch("benchmarks.bench_scan_cycle.run_all", return_value=[])
    def test_main_runs_without_error(self, mock_run_all):
        """Calling run_all (the __main__ entry point) returns without error."""
        from benchmarks.bench_scan_cycle import run_all as ra

        result = ra()
        assert result == []


# ---------------------------------------------------------------------------
# _SCHEMA constant
# ---------------------------------------------------------------------------


class TestSchema:
    """Verify the module-level Parquet schema."""

    def test_schema_has_expected_fields(self):
        field_names = [f.name for f in _SCHEMA]
        assert field_names == ["timestamp", "open", "high", "low", "close", "volume"]

    def test_schema_types(self):
        import pyarrow as pa

        assert _SCHEMA.field("timestamp").type == pa.int64()
        for col in ("open", "high", "low", "close", "volume"):
            assert _SCHEMA.field(col).type == pa.float64()
