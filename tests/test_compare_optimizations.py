"""Tests for benchmarks.compare_optimizations module.

Tests all functions: _make_synthetic_ohlcv, _z_normalize,
compare_dtw_implementations, compare_prescreening,
compare_parquet_compression, compare_btc_cache, and main.

External dependencies (dtaidistance, config, core, data, references) are
mocked to isolate the module under test and ensure fast execution.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from typing import Any
from unittest import mock
from unittest.mock import MagicMock, patch, call

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from benchmarks.compare_optimizations import (
    BenchmarkResult,
    _make_synthetic_ohlcv,
    _z_normalize,
    _SCHEMA,
    compare_dtw_implementations,
    compare_prescreening,
    compare_parquet_compression,
    compare_btc_cache,
    main,
)
from benchmarks.exceptions import BenchmarkRunError

# ---------------------------------------------------------------------------
# Tests for _make_synthetic_ohlcv
# ---------------------------------------------------------------------------


class TestMakeSyntheticOhlcv:
    """Tests for _make_synthetic_ohlcv helper."""

    def test_returns_dataframe(self):
        """Should return a pandas DataFrame."""
        df = _make_synthetic_ohlcv(n_bars=10, seed=42)
        assert isinstance(df, pd.DataFrame)

    def test_correct_number_of_rows(self):
        """Should have exactly n_bars rows."""
        df = _make_synthetic_ohlcv(n_bars=25, seed=0)
        assert len(df) == 25

    def test_expected_columns(self):
        """Should contain all OHLCV columns."""
        df = _make_synthetic_ohlcv(n_bars=10, seed=42)
        expected = {"timestamp", "open", "high", "low", "close", "volume"}
        assert set(df.columns) == expected

    def test_timestamp_dtype_is_int64(self):
        """Timestamp column should be int64."""
        df = _make_synthetic_ohlcv(n_bars=10, seed=42)
        assert df["timestamp"].dtype == np.int64

    def test_close_dtype_is_float64(self):
        """Close column should be float64."""
        df = _make_synthetic_ohlcv(n_bars=10, seed=42)
        assert df["close"].dtype == np.float64

    def test_volume_dtype_is_float64(self):
        """Volume column should be float64."""
        df = _make_synthetic_ohlcv(n_bars=10, seed=42)
        assert df["volume"].dtype == np.float64

    def test_timestamp_spacing_is_3600(self):
        """Timestamps should be spaced 3600 seconds apart."""
        df = _make_synthetic_ohlcv(n_bars=5, seed=42)
        diffs = df["timestamp"].diff().dropna().unique()
        assert len(diffs) == 1
        assert diffs[0] == 3600

    def test_reproducibility_same_seed(self):
        """Same seed should produce identical data."""
        df1 = _make_synthetic_ohlcv(n_bars=10, seed=99)
        df2 = _make_synthetic_ohlcv(n_bars=10, seed=99)
        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seeds_differ(self):
        """Different seeds should produce different data."""
        df1 = _make_synthetic_ohlcv(n_bars=10, seed=1)
        df2 = _make_synthetic_ohlcv(n_bars=10, seed=2)
        assert not df1["close"].equals(df2["close"])

    def test_high_above_low(self):
        """High should generally be above low (by construction)."""
        df = _make_synthetic_ohlcv(n_bars=50, seed=42)
        assert (df["high"] > df["low"]).all()

    def test_volume_positive(self):
        """Volume should be positive."""
        df = _make_synthetic_ohlcv(n_bars=20, seed=42)
        assert (df["volume"] > 0).all()

    def test_default_parameters(self):
        """Calling with defaults should produce 500 bars."""
        df = _make_synthetic_ohlcv()
        assert len(df) == 500


# ---------------------------------------------------------------------------
# Tests for _z_normalize
# ---------------------------------------------------------------------------


class TestZNormalize:
    """Tests for _z_normalize helper."""

    def test_mean_near_zero(self):
        """Normalized array should have mean near zero."""
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = _z_normalize(arr)
        assert abs(np.mean(result)) < 1e-10

    def test_std_near_one(self):
        """Normalized array should have std near one."""
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = _z_normalize(arr)
        assert abs(np.std(result) - 1.0) < 1e-10

    def test_constant_array_returns_zeros(self):
        """Constant array (zero std) should return all zeros."""
        arr = np.array([5.0, 5.0, 5.0, 5.0])
        result = _z_normalize(arr)
        np.testing.assert_array_equal(result, np.zeros(4))

    def test_preserves_length(self):
        """Output should have same length as input."""
        arr = np.arange(10, dtype=float)
        result = _z_normalize(arr)
        assert len(result) == 10

    def test_near_zero_std_returns_zeros(self):
        """Array with std < 1e-10 should return zeros."""
        arr = np.array([1.0, 1.0 + 1e-12, 1.0 - 1e-12])
        result = _z_normalize(arr)
        np.testing.assert_array_equal(result, np.zeros(3))


# ---------------------------------------------------------------------------
# Tests for _SCHEMA
# ---------------------------------------------------------------------------


class TestSchema:
    """Tests for the module-level Parquet schema."""

    def test_schema_is_pa_schema(self):
        """_SCHEMA should be a pyarrow Schema."""
        assert isinstance(_SCHEMA, pa.Schema)

    def test_schema_has_six_fields(self):
        """_SCHEMA should have 6 fields."""
        assert len(_SCHEMA) == 6

    def test_schema_field_names(self):
        """_SCHEMA should have the expected field names."""
        names = [f.name for f in _SCHEMA]
        assert names == ["timestamp", "open", "high", "low", "close", "volume"]


# ---------------------------------------------------------------------------
# Tests for compare_dtw_implementations
# ---------------------------------------------------------------------------


class TestCompareDtwImplementations:
    """Tests for compare_dtw_implementations."""

    @patch("benchmarks.compare_optimizations.time.perf_counter")
    def test_returns_dict_with_expected_keys(self, mock_time):
        """Should return dict with 'comparison' and 'speedup' keys."""
        mock_time.side_effect = [0.0, 1.0, 0.0, 2.0]

        mock_dtw_lib = MagicMock()
        mock_dtw_lib.distance.return_value = 1.5

        mock_dtw_numpy = MagicMock(return_value=1.5)

        with patch.dict(
            "sys.modules",
            {"dtaidistance": MagicMock(), "dtaidistance.dtw": mock_dtw_lib},
        ), patch(
            "benchmarks.compare_optimizations.time.perf_counter", mock_time
        ), patch(
            "benchmarks.bench_dtw._dtw_numpy", mock_dtw_numpy
        ):
            # Re-import to pick up mocked modules
            import importlib
            import benchmarks.compare_optimizations as mod

            importlib.reload(mod)

            result = mod.compare_dtw_implementations()

        assert result["comparison"] == "dtw_c_vs_numpy"
        assert "speedup" in result

    @patch("benchmarks.compare_optimizations.time.perf_counter")
    def test_speedup_is_positive(self, mock_time):
        """Speedup value should be positive."""
        # 100 iterations of C: elapsed = 1.0s => 0.01 per call
        # 10 iterations of numpy: elapsed = 2.0s => 0.2 per call
        # speedup = 0.2 / 0.01 = 20.0
        mock_time.side_effect = [0.0, 1.0, 0.0, 2.0]

        mock_dtw_lib = MagicMock()
        mock_dtw_lib.distance.return_value = 1.0

        mock_dtw_numpy_fn = MagicMock(return_value=1.0)

        mock_bench_dtw = MagicMock()
        mock_bench_dtw._dtw_numpy = mock_dtw_numpy_fn

        with patch.dict(
            "sys.modules",
            {
                "dtaidistance": MagicMock(),
                "dtaidistance.dtw": mock_dtw_lib,
                "benchmarks.bench_dtw": mock_bench_dtw,
            },
        ):
            result = compare_dtw_implementations()

        assert result["speedup"] > 0

    def test_raises_benchmark_run_error_on_failure(self):
        """Should raise BenchmarkRunError when an exception occurs."""
        with patch.dict(
            "sys.modules",
            {
                "dtaidistance": None,
                "dtaidistance.dtw": None,
            },
        ):
            with pytest.raises(
                BenchmarkRunError, match="compare_dtw_implementations failed"
            ):
                compare_dtw_implementations()

    @patch("benchmarks.compare_optimizations.time.perf_counter")
    def test_speedup_inf_when_t_c_zero(self, mock_time):
        """Speedup should be inf when C time is zero."""
        # warmup call, then perf_counter returns same value => t_c = 0
        mock_time.side_effect = [0.0, 0.0, 0.0, 1.0]

        mock_dtw_lib = MagicMock()
        mock_dtw_lib.distance.return_value = 1.0

        mock_dtw_numpy_fn = MagicMock(return_value=1.0)
        mock_bench_dtw = MagicMock()
        mock_bench_dtw._dtw_numpy = mock_dtw_numpy_fn

        with patch.dict(
            "sys.modules",
            {
                "dtaidistance": MagicMock(),
                "dtaidistance.dtw": mock_dtw_lib,
                "benchmarks.bench_dtw": mock_bench_dtw,
            },
        ):
            result = compare_dtw_implementations()

        assert result["speedup"] == float("inf")


# ---------------------------------------------------------------------------
# Tests for compare_prescreening
# ---------------------------------------------------------------------------


class TestComparePrescreening:
    """Tests for compare_prescreening."""

    def _build_mocks(self):
        """Build mock objects for SystemConfig, DataManager, ScannerEngine, ReferencePattern."""
        mock_config_cls = MagicMock()
        mock_config_instance = MagicMock()
        mock_config_cls.return_value = mock_config_instance

        mock_dm_cls = MagicMock()
        mock_dm_instance = MagicMock()
        mock_dm_cls.return_value = mock_dm_instance

        mock_scanner_cls = MagicMock()
        mock_scanner_instance = MagicMock()
        mock_scanner_instance.scan_timeframe.return_value = [{"alert": 1}]
        mock_scanner_cls.return_value = mock_scanner_instance

        mock_ref_cls = MagicMock()

        return mock_config_cls, mock_dm_cls, mock_scanner_cls, mock_ref_cls

    @patch("benchmarks.compare_optimizations.time.perf_counter")
    @patch("benchmarks.compare_optimizations.shutil.rmtree")
    @patch("benchmarks.compare_optimizations.tempfile.mkdtemp")
    @patch("benchmarks.compare_optimizations.os.makedirs")
    @patch("benchmarks.compare_optimizations.pq.write_table")
    def test_returns_dict_with_expected_keys(
        self, mock_write, mock_makedirs, mock_mkdtemp, mock_rmtree, mock_time
    ):
        """Should return dict with 'comparison' and 'speedup' keys."""
        mock_mkdtemp.return_value = "/tmp/fakedir"
        # scan_timeframe called twice: with prescreening then without
        # perf_counter calls: t0, t_with, t0, t_without
        mock_time.side_effect = [0.0, 1.0, 0.0, 3.0]

        mock_config_cls, mock_dm_cls, mock_scanner_cls, mock_ref_cls = (
            self._build_mocks()
        )

        with patch.dict(
            "sys.modules",
            {
                "config": MagicMock(SystemConfig=mock_config_cls),
                "core": MagicMock(),
                "core.scanner_engine": MagicMock(ScannerEngine=mock_scanner_cls),
                "data": MagicMock(),
                "data.data_manager": MagicMock(DataManager=mock_dm_cls),
                "references": MagicMock(),
                "references.reference_manager": MagicMock(
                    ReferencePattern=mock_ref_cls
                ),
            },
        ):
            result = compare_prescreening()

        assert result["comparison"] == "prescreening"
        assert "speedup" in result

    @patch("benchmarks.compare_optimizations.time.perf_counter")
    @patch("benchmarks.compare_optimizations.shutil.rmtree")
    @patch("benchmarks.compare_optimizations.tempfile.mkdtemp")
    @patch("benchmarks.compare_optimizations.os.makedirs")
    @patch("benchmarks.compare_optimizations.pq.write_table")
    def test_speedup_calculated_correctly(
        self, mock_write, mock_makedirs, mock_mkdtemp, mock_rmtree, mock_time
    ):
        """Speedup should be t_without / t_with."""
        mock_mkdtemp.return_value = "/tmp/fakedir"
        mock_time.side_effect = [0.0, 2.0, 0.0, 6.0]  # with=2s, without=6s => 3x

        mock_config_cls, mock_dm_cls, mock_scanner_cls, mock_ref_cls = (
            self._build_mocks()
        )

        with patch.dict(
            "sys.modules",
            {
                "config": MagicMock(SystemConfig=mock_config_cls),
                "core": MagicMock(),
                "core.scanner_engine": MagicMock(ScannerEngine=mock_scanner_cls),
                "data": MagicMock(),
                "data.data_manager": MagicMock(DataManager=mock_dm_cls),
                "references": MagicMock(),
                "references.reference_manager": MagicMock(
                    ReferencePattern=mock_ref_cls
                ),
            },
        ):
            result = compare_prescreening()

        assert result["speedup"] == pytest.approx(3.0)

    @patch("benchmarks.compare_optimizations.shutil.rmtree")
    @patch("benchmarks.compare_optimizations.tempfile.mkdtemp")
    @patch("benchmarks.compare_optimizations.os.makedirs")
    @patch("benchmarks.compare_optimizations.pq.write_table")
    def test_raises_benchmark_run_error_on_failure(
        self, mock_write, mock_makedirs, mock_mkdtemp, mock_rmtree
    ):
        """Should raise BenchmarkRunError when an exception occurs."""
        mock_mkdtemp.return_value = "/tmp/fakedir"

        # Make the scanner fail
        mock_scanner_cls = MagicMock()
        mock_scanner_cls.return_value.scan_timeframe.side_effect = RuntimeError("boom")

        mock_config_cls = MagicMock()
        mock_dm_cls = MagicMock()
        mock_ref_cls = MagicMock()

        with patch.dict(
            "sys.modules",
            {
                "config": MagicMock(SystemConfig=mock_config_cls),
                "core": MagicMock(),
                "core.scanner_engine": MagicMock(ScannerEngine=mock_scanner_cls),
                "data": MagicMock(),
                "data.data_manager": MagicMock(DataManager=mock_dm_cls),
                "references": MagicMock(),
                "references.reference_manager": MagicMock(
                    ReferencePattern=mock_ref_cls
                ),
            },
        ):
            with pytest.raises(BenchmarkRunError, match="compare_prescreening failed"):
                compare_prescreening()

    @patch("benchmarks.compare_optimizations.time.perf_counter")
    @patch("benchmarks.compare_optimizations.shutil.rmtree")
    @patch("benchmarks.compare_optimizations.tempfile.mkdtemp")
    @patch("benchmarks.compare_optimizations.os.makedirs")
    @patch("benchmarks.compare_optimizations.pq.write_table")
    def test_cleanup_called_even_on_success(
        self, mock_write, mock_makedirs, mock_mkdtemp, mock_rmtree, mock_time
    ):
        """Temp directory should be cleaned up via shutil.rmtree."""
        mock_mkdtemp.return_value = "/tmp/fakedir"
        mock_time.side_effect = [0.0, 1.0, 0.0, 2.0]

        mock_config_cls, mock_dm_cls, mock_scanner_cls, mock_ref_cls = (
            self._build_mocks()
        )

        with patch.dict(
            "sys.modules",
            {
                "config": MagicMock(SystemConfig=mock_config_cls),
                "core": MagicMock(),
                "core.scanner_engine": MagicMock(ScannerEngine=mock_scanner_cls),
                "data": MagicMock(),
                "data.data_manager": MagicMock(DataManager=mock_dm_cls),
                "references": MagicMock(),
                "references.reference_manager": MagicMock(
                    ReferencePattern=mock_ref_cls
                ),
            },
        ):
            compare_prescreening()

        mock_rmtree.assert_called_once_with("/tmp/fakedir")

    @patch("benchmarks.compare_optimizations.time.perf_counter")
    @patch("benchmarks.compare_optimizations.shutil.rmtree")
    @patch("benchmarks.compare_optimizations.tempfile.mkdtemp")
    @patch("benchmarks.compare_optimizations.os.makedirs")
    @patch("benchmarks.compare_optimizations.pq.write_table")
    def test_speedup_inf_when_t_with_zero(
        self, mock_write, mock_makedirs, mock_mkdtemp, mock_rmtree, mock_time
    ):
        """Speedup should be inf when prescreened time is zero."""
        mock_mkdtemp.return_value = "/tmp/fakedir"
        mock_time.side_effect = [0.0, 0.0, 0.0, 1.0]  # t_with=0

        mock_config_cls, mock_dm_cls, mock_scanner_cls, mock_ref_cls = (
            self._build_mocks()
        )

        with patch.dict(
            "sys.modules",
            {
                "config": MagicMock(SystemConfig=mock_config_cls),
                "core": MagicMock(),
                "core.scanner_engine": MagicMock(ScannerEngine=mock_scanner_cls),
                "data": MagicMock(),
                "data.data_manager": MagicMock(DataManager=mock_dm_cls),
                "references": MagicMock(),
                "references.reference_manager": MagicMock(
                    ReferencePattern=mock_ref_cls
                ),
            },
        ):
            result = compare_prescreening()

        assert result["speedup"] == float("inf")


# ---------------------------------------------------------------------------
# Tests for compare_parquet_compression
# ---------------------------------------------------------------------------


class TestCompareParquetCompression:
    """Tests for compare_parquet_compression."""

    def test_returns_dict_with_expected_keys(self):
        """Should return dict with 'comparison' and 'size_ratio' keys."""
        result = compare_parquet_compression()
        assert result["comparison"] == "parquet_compression"
        assert "size_ratio" in result

    def test_size_ratio_positive(self):
        """Size ratio should be a positive float."""
        result = compare_parquet_compression()
        assert result["size_ratio"] > 0

    def test_size_ratio_at_least_one(self):
        """Snappy-compressed should be same size or smaller => ratio >= 1."""
        result = compare_parquet_compression()
        assert result["size_ratio"] >= 1.0

    def test_raises_benchmark_run_error_on_failure(self):
        """Should raise BenchmarkRunError when an exception occurs."""
        with patch(
            "benchmarks.compare_optimizations._make_synthetic_ohlcv",
            side_effect=RuntimeError("data generation failed"),
        ):
            with pytest.raises(
                BenchmarkRunError, match="compare_parquet_compression failed"
            ):
                compare_parquet_compression()

    @patch("benchmarks.compare_optimizations.shutil.rmtree")
    @patch("benchmarks.compare_optimizations.tempfile.mkdtemp")
    def test_temp_directory_cleaned_up(self, mock_mkdtemp, mock_rmtree):
        """Temp directory should be cleaned up after execution."""
        real_tmpdir = tempfile.mkdtemp()
        mock_mkdtemp.return_value = real_tmpdir

        try:
            compare_parquet_compression()
        except Exception:
            pass

        mock_rmtree.assert_called_once_with(real_tmpdir)
        # Clean up the real dir if rmtree was mocked out
        if os.path.exists(real_tmpdir):
            shutil.rmtree(real_tmpdir)


# ---------------------------------------------------------------------------
# Tests for compare_btc_cache
# ---------------------------------------------------------------------------


class TestCompareBtcCache:
    """Tests for compare_btc_cache."""

    @patch("benchmarks.compare_optimizations.time.perf_counter")
    @patch("benchmarks.compare_optimizations.shutil.rmtree")
    @patch("benchmarks.compare_optimizations.tempfile.mkdtemp")
    @patch("benchmarks.compare_optimizations.os.makedirs")
    @patch("benchmarks.compare_optimizations.pq.write_table")
    def test_returns_dict_with_expected_keys(
        self, mock_write, mock_makedirs, mock_mkdtemp, mock_rmtree, mock_time
    ):
        """Should return dict with 'comparison', 'cold_ms', and 'warm_ms'."""
        mock_mkdtemp.return_value = "/tmp/fakedir"
        # cold: t0=0, end=1.0; warm: t0=0, end=0.5
        mock_time.side_effect = [0.0, 1.0, 0.0, 0.5]

        mock_config_cls = MagicMock()
        mock_dm_cls = MagicMock()
        mock_dm_cls.return_value.read_symbol.return_value = pd.DataFrame()

        with patch.dict(
            "sys.modules",
            {
                "config": MagicMock(SystemConfig=mock_config_cls),
                "data": MagicMock(),
                "data.data_manager": MagicMock(DataManager=mock_dm_cls),
            },
        ):
            result = compare_btc_cache()

        assert result["comparison"] == "btc_cache"
        assert "cold_ms" in result
        assert "warm_ms" in result

    @patch("benchmarks.compare_optimizations.time.perf_counter")
    @patch("benchmarks.compare_optimizations.shutil.rmtree")
    @patch("benchmarks.compare_optimizations.tempfile.mkdtemp")
    @patch("benchmarks.compare_optimizations.os.makedirs")
    @patch("benchmarks.compare_optimizations.pq.write_table")
    def test_cold_and_warm_ms_are_positive(
        self, mock_write, mock_makedirs, mock_mkdtemp, mock_rmtree, mock_time
    ):
        """Cold and warm ms values should be positive."""
        mock_mkdtemp.return_value = "/tmp/fakedir"
        mock_time.side_effect = [0.0, 1.0, 0.0, 0.5]

        mock_config_cls = MagicMock()
        mock_dm_cls = MagicMock()
        mock_dm_cls.return_value.read_symbol.return_value = pd.DataFrame()

        with patch.dict(
            "sys.modules",
            {
                "config": MagicMock(SystemConfig=mock_config_cls),
                "data": MagicMock(),
                "data.data_manager": MagicMock(DataManager=mock_dm_cls),
            },
        ):
            result = compare_btc_cache()

        assert result["cold_ms"] > 0
        assert result["warm_ms"] > 0

    @patch("benchmarks.compare_optimizations.time.perf_counter")
    @patch("benchmarks.compare_optimizations.shutil.rmtree")
    @patch("benchmarks.compare_optimizations.tempfile.mkdtemp")
    @patch("benchmarks.compare_optimizations.os.makedirs")
    @patch("benchmarks.compare_optimizations.pq.write_table")
    def test_ms_values_computed_correctly(
        self, mock_write, mock_makedirs, mock_mkdtemp, mock_rmtree, mock_time
    ):
        """cold_ms and warm_ms should be (elapsed / 20) * 1000."""
        mock_mkdtemp.return_value = "/tmp/fakedir"
        # cold: elapsed = 2.0, per-read = 0.1, ms = 100.0
        # warm: elapsed = 0.4, per-read = 0.02, ms = 20.0
        mock_time.side_effect = [0.0, 2.0, 0.0, 0.4]

        mock_config_cls = MagicMock()
        mock_dm_cls = MagicMock()
        mock_dm_cls.return_value.read_symbol.return_value = pd.DataFrame()

        with patch.dict(
            "sys.modules",
            {
                "config": MagicMock(SystemConfig=mock_config_cls),
                "data": MagicMock(),
                "data.data_manager": MagicMock(DataManager=mock_dm_cls),
            },
        ):
            result = compare_btc_cache()

        assert result["cold_ms"] == pytest.approx(100.0)
        assert result["warm_ms"] == pytest.approx(20.0)

    @patch("benchmarks.compare_optimizations.shutil.rmtree")
    @patch("benchmarks.compare_optimizations.tempfile.mkdtemp")
    @patch("benchmarks.compare_optimizations.os.makedirs")
    @patch("benchmarks.compare_optimizations.pq.write_table")
    def test_raises_benchmark_run_error_on_failure(
        self, mock_write, mock_makedirs, mock_mkdtemp, mock_rmtree
    ):
        """Should raise BenchmarkRunError on internal failure."""
        mock_mkdtemp.return_value = "/tmp/fakedir"

        mock_config_cls = MagicMock()
        mock_dm_cls = MagicMock()
        mock_dm_cls.return_value.read_symbol.side_effect = RuntimeError("read failed")

        with patch.dict(
            "sys.modules",
            {
                "config": MagicMock(SystemConfig=mock_config_cls),
                "data": MagicMock(),
                "data.data_manager": MagicMock(DataManager=mock_dm_cls),
            },
        ):
            with pytest.raises(BenchmarkRunError, match="compare_btc_cache failed"):
                compare_btc_cache()

    @patch("benchmarks.compare_optimizations.time.perf_counter")
    @patch("benchmarks.compare_optimizations.shutil.rmtree")
    @patch("benchmarks.compare_optimizations.tempfile.mkdtemp")
    @patch("benchmarks.compare_optimizations.os.makedirs")
    @patch("benchmarks.compare_optimizations.pq.write_table")
    def test_cleanup_called(
        self, mock_write, mock_makedirs, mock_mkdtemp, mock_rmtree, mock_time
    ):
        """Temp directory should be cleaned up via shutil.rmtree."""
        mock_mkdtemp.return_value = "/tmp/fakedir"
        mock_time.side_effect = [0.0, 1.0, 0.0, 0.5]

        mock_config_cls = MagicMock()
        mock_dm_cls = MagicMock()
        mock_dm_cls.return_value.read_symbol.return_value = pd.DataFrame()

        with patch.dict(
            "sys.modules",
            {
                "config": MagicMock(SystemConfig=mock_config_cls),
                "data": MagicMock(),
                "data.data_manager": MagicMock(DataManager=mock_dm_cls),
            },
        ):
            compare_btc_cache()

        mock_rmtree.assert_called_once_with("/tmp/fakedir")

    @patch("benchmarks.compare_optimizations.shutil.rmtree")
    @patch("benchmarks.compare_optimizations.tempfile.mkdtemp")
    @patch("benchmarks.compare_optimizations.os.makedirs")
    @patch("benchmarks.compare_optimizations.pq.write_table")
    def test_cleanup_called_on_error(
        self, mock_write, mock_makedirs, mock_mkdtemp, mock_rmtree
    ):
        """Temp directory should be cleaned up even when an error occurs."""
        mock_mkdtemp.return_value = "/tmp/fakedir"

        mock_config_cls = MagicMock()
        mock_dm_cls = MagicMock()
        mock_dm_cls.return_value.read_symbol.side_effect = RuntimeError("fail")

        with patch.dict(
            "sys.modules",
            {
                "config": MagicMock(SystemConfig=mock_config_cls),
                "data": MagicMock(),
                "data.data_manager": MagicMock(DataManager=mock_dm_cls),
            },
        ):
            with pytest.raises(BenchmarkRunError):
                compare_btc_cache()

        mock_rmtree.assert_called_once_with("/tmp/fakedir")


# ---------------------------------------------------------------------------
# Tests for BenchmarkResult TypedDict
# ---------------------------------------------------------------------------


class TestBenchmarkResultTypedDict:
    """Tests for the BenchmarkResult TypedDict definition."""

    def test_can_create_valid_instance(self):
        """Should be able to create a valid BenchmarkResult dict."""
        result: BenchmarkResult = {
            "name": "test",
            "value": 1.23,
            "unit": "seconds",
            "detail": "some detail",
        }
        assert result["name"] == "test"
        assert result["value"] == 1.23

    def test_value_can_be_none(self):
        """Value field should accept None."""
        result: BenchmarkResult = {
            "name": "test",
            "value": None,
            "unit": "error",
            "detail": "failed",
        }
        assert result["value"] is None


# ---------------------------------------------------------------------------
# Tests for main
# ---------------------------------------------------------------------------


class TestMain:
    """Tests for the main() entry point."""

    @patch("benchmarks.compare_optimizations.compare_btc_cache")
    @patch("benchmarks.compare_optimizations.compare_parquet_compression")
    @patch("benchmarks.compare_optimizations.compare_prescreening")
    @patch("benchmarks.compare_optimizations.compare_dtw_implementations")
    def test_main_calls_all_comparisons(self, mock_dtw, mock_pre, mock_pq, mock_btc):
        """main() should call all four comparison functions."""
        mock_dtw.return_value = {"comparison": "dtw", "speedup": 5.0}
        mock_pre.return_value = {"comparison": "prescreening", "speedup": 2.0}
        mock_pq.return_value = {"comparison": "parquet", "size_ratio": 1.5}
        mock_btc.return_value = {
            "comparison": "btc_cache",
            "cold_ms": 10.0,
            "warm_ms": 5.0,
        }

        main()

        mock_dtw.assert_called_once()
        mock_pre.assert_called_once()
        mock_pq.assert_called_once()
        mock_btc.assert_called_once()

    @patch("benchmarks.compare_optimizations.compare_btc_cache")
    @patch("benchmarks.compare_optimizations.compare_parquet_compression")
    @patch("benchmarks.compare_optimizations.compare_prescreening")
    @patch("benchmarks.compare_optimizations.compare_dtw_implementations")
    def test_main_returns_none(self, mock_dtw, mock_pre, mock_pq, mock_btc):
        """main() should return None."""
        mock_dtw.return_value = {"comparison": "dtw", "speedup": 5.0}
        mock_pre.return_value = {"comparison": "prescreening", "speedup": 2.0}
        mock_pq.return_value = {"comparison": "parquet", "size_ratio": 1.5}
        mock_btc.return_value = {
            "comparison": "btc_cache",
            "cold_ms": 10.0,
            "warm_ms": 5.0,
        }

        result = main()
        assert result is None

    @patch("benchmarks.compare_optimizations.compare_btc_cache")
    @patch("benchmarks.compare_optimizations.compare_parquet_compression")
    @patch("benchmarks.compare_optimizations.compare_prescreening")
    @patch("benchmarks.compare_optimizations.compare_dtw_implementations")
    def test_main_continues_on_individual_failure(
        self, mock_dtw, mock_pre, mock_pq, mock_btc
    ):
        """main() should continue running even if one comparison raises."""
        mock_dtw.side_effect = BenchmarkRunError("dtw failed")
        mock_pre.return_value = {"comparison": "prescreening", "speedup": 2.0}
        mock_pq.side_effect = BenchmarkRunError("parquet failed")
        mock_btc.return_value = {
            "comparison": "btc_cache",
            "cold_ms": 10.0,
            "warm_ms": 5.0,
        }

        # Should not raise
        main()

        # All functions should still be called
        mock_dtw.assert_called_once()
        mock_pre.assert_called_once()
        mock_pq.assert_called_once()
        mock_btc.assert_called_once()

    @patch("benchmarks.compare_optimizations.compare_btc_cache")
    @patch("benchmarks.compare_optimizations.compare_parquet_compression")
    @patch("benchmarks.compare_optimizations.compare_prescreening")
    @patch("benchmarks.compare_optimizations.compare_dtw_implementations")
    def test_main_all_fail_no_crash(self, mock_dtw, mock_pre, mock_pq, mock_btc):
        """main() should not crash even if all comparisons fail."""
        mock_dtw.side_effect = Exception("fail1")
        mock_pre.side_effect = Exception("fail2")
        mock_pq.side_effect = Exception("fail3")
        mock_btc.side_effect = Exception("fail4")

        # Should not raise
        main()
