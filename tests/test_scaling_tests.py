"""Comprehensive tests for benchmarks/scaling_tests.py

Every public and private function is exercised through mocks so that
no real project infrastructure (config, scanner, data manager, etc.)
is required.

The scaling functions use *local* imports (``from config import SystemConfig``,
``from core.scanner_engine import ScannerEngine``, etc.) so we mock them at
their source modules rather than on ``benchmarks.scaling_tests``.
"""

from __future__ import annotations

import os
import sys
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from benchmarks.exceptions import BenchmarkSetupError

# ---------------------------------------------------------------------------
# Module under test
# ---------------------------------------------------------------------------
from benchmarks.scaling_tests import (
    BenchmarkResult,
    _make_synthetic_ohlcv,
    _setup_data,
    _make_ref,
    scale_symbol_count,
    scale_reference_count,
    scale_data_volume,
    scale_worker_count,
    main,
    _SCHEMA,
)

# ---------------------------------------------------------------------------
# Helpers: mock the external classes that are locally imported
# ---------------------------------------------------------------------------


def _mock_scanner(alerts=None):
    """Return (MockScannerEngine_cls, mock_scanner_instance)."""
    scanner = MagicMock()
    scanner.scan_timeframe.return_value = alerts if alerts is not None else []
    cls = MagicMock(return_value=scanner)
    return cls, scanner


def _mock_normalizer():
    """Return (MockDataNormalizer_cls, mock_normalizer_instance)."""
    norm = MagicMock()
    norm.prepare_for_dtw.return_value = np.array([1.0])
    cls = MagicMock(return_value=norm)
    return cls, norm


# ===================================================================
# _make_synthetic_ohlcv
# ===================================================================


class TestMakeSyntheticOhlcv:
    """Tests for the _make_synthetic_ohlcv helper."""

    def test_returns_dataframe(self):
        df = _make_synthetic_ohlcv(10)
        assert isinstance(df, pd.DataFrame)

    def test_column_names(self):
        df = _make_synthetic_ohlcv(10)
        assert list(df.columns) == [
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]

    def test_row_count(self):
        for n in (1, 5, 50):
            df = _make_synthetic_ohlcv(n)
            assert len(df) == n

    def test_dtypes_are_numeric(self):
        df = _make_synthetic_ohlcv(10)
        assert df["timestamp"].dtype == np.int64
        for col in ("open", "high", "low", "close", "volume"):
            assert df[col].dtype == np.float64

    def test_reproducibility_with_same_seed(self):
        df1 = _make_synthetic_ohlcv(20, seed=7)
        df2 = _make_synthetic_ohlcv(20, seed=7)
        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seeds_differ(self):
        df1 = _make_synthetic_ohlcv(20, seed=0)
        df2 = _make_synthetic_ohlcv(20, seed=1)
        assert not df1["close"].equals(df2["close"])

    def test_timestamps_are_hourly(self):
        df = _make_synthetic_ohlcv(10)
        diffs = df["timestamp"].diff().dropna().unique()
        assert list(diffs) == [3600]

    def test_volume_positive(self):
        df = _make_synthetic_ohlcv(100)
        assert (df["volume"] > 0).all()

    def test_high_ge_low(self):
        """High should be above close while low should be below, so high > low."""
        df = _make_synthetic_ohlcv(200)
        assert (df["high"] > df["low"]).all()


# ===================================================================
# _setup_data
# ===================================================================


class TestSetupData:
    """Tests for the _setup_data helper."""

    def test_creates_directories(self, tmp_path):
        _setup_data(str(tmp_path), n_symbols=2, n_bars=10)
        assert (tmp_path / "parquet" / "timeframe=4h").is_dir()
        assert (tmp_path / "cache").is_dir()
        assert (tmp_path / "models").is_dir()

    def test_returns_symbol_list(self, tmp_path):
        syms = _setup_data(str(tmp_path), n_symbols=3, n_bars=10)
        assert syms == ["SYM0000USDT", "SYM0001USDT", "SYM0002USDT"]

    def test_parquet_files_written(self, tmp_path):
        syms = _setup_data(str(tmp_path), n_symbols=2, n_bars=10)
        tf_dir = tmp_path / "parquet" / "timeframe=4h"
        for sym in syms:
            assert (tf_dir / f"{sym}.parquet").exists()

    def test_parquet_row_count(self, tmp_path):
        _setup_data(str(tmp_path), n_symbols=1, n_bars=15)
        pf = tmp_path / "parquet" / "timeframe=4h" / "SYM0000USDT.parquet"
        table = pq.read_table(str(pf))
        assert table.num_rows == 15

    def test_parquet_has_expected_columns(self, tmp_path):
        _setup_data(str(tmp_path), n_symbols=1, n_bars=15)
        pf = tmp_path / "parquet" / "timeframe=4h" / "SYM0000USDT.parquet"
        table = pq.read_table(str(pf))
        expected = {"timestamp", "open", "high", "low", "close", "volume"}
        assert expected.issubset(set(table.column_names))

    def test_custom_timeframe(self, tmp_path):
        _setup_data(str(tmp_path), n_symbols=1, n_bars=5, timeframe="1h")
        assert (tmp_path / "parquet" / "timeframe=1h").is_dir()

    def test_raises_benchmark_setup_error_on_failure(self):
        """Passing an invalid path should trigger BenchmarkSetupError."""
        with pytest.raises(BenchmarkSetupError):
            _setup_data("/dev/null/bad\x00path", n_symbols=1, n_bars=5)


# ===================================================================
# _make_ref
# ===================================================================


class TestMakeRef:
    """Tests for the _make_ref helper."""

    @patch("benchmarks.scaling_tests._make_synthetic_ohlcv")
    @patch("references.reference_manager.ReferencePattern")
    def test_returns_reference_pattern(self, MockRefPat, mock_ohlcv):
        df = pd.DataFrame(
            {
                "timestamp": np.arange(
                    1700000000, 1700000000 + 500 * 3600, 3600, dtype=np.int64
                ),
            }
        )
        mock_ohlcv.return_value = df
        sentinel = MagicMock()
        MockRefPat.return_value = sentinel

        result = _make_ref(["SYMAUSDT", "SYMBUSDT"], seed=0, n_bars=500)

        assert result is sentinel
        MockRefPat.assert_called_once()

    @patch("benchmarks.scaling_tests._make_synthetic_ohlcv")
    @patch("references.reference_manager.ReferencePattern")
    def test_id_contains_seed(self, MockRefPat, mock_ohlcv):
        df = pd.DataFrame(
            {
                "timestamp": np.arange(
                    1700000000, 1700000000 + 500 * 3600, 3600, dtype=np.int64
                ),
            }
        )
        mock_ohlcv.return_value = df
        MockRefPat.return_value = MagicMock()

        _make_ref(["SYMAUSDT"], seed=3, n_bars=500)
        call_kwargs = MockRefPat.call_args.kwargs
        assert call_kwargs["id"] == "scale_ref_3"

    @patch("benchmarks.scaling_tests._make_synthetic_ohlcv")
    @patch("references.reference_manager.ReferencePattern")
    def test_seed_clamps_to_last_symbol(self, MockRefPat, mock_ohlcv):
        df = pd.DataFrame(
            {
                "timestamp": np.arange(
                    1700000000, 1700000000 + 500 * 3600, 3600, dtype=np.int64
                ),
            }
        )
        mock_ohlcv.return_value = df
        MockRefPat.return_value = MagicMock()

        _make_ref(["ONLY"], seed=10, n_bars=500)
        call_kwargs = MockRefPat.call_args.kwargs
        assert call_kwargs["symbol"] == "ONLY"


# ===================================================================
# scale_symbol_count
# ===================================================================


class TestScaleSymbolCount:
    """Tests for scale_symbol_count."""

    @patch("benchmarks.scaling_tests.shutil.rmtree")
    @patch("benchmarks.scaling_tests.tempfile.mkdtemp", return_value="/tmp/fake")
    @patch("benchmarks.scaling_tests._make_ref", return_value=MagicMock())
    @patch(
        "benchmarks.scaling_tests._setup_data",
        return_value=[f"SYM{i:04d}USDT" for i in range(10)],
    )
    def test_returns_list_of_dicts(
        self, mock_setup, mock_ref, mock_mkdtemp, mock_rmtree
    ):
        scanner_cls, scanner = _mock_scanner(["alert1"])

        with patch.dict(
            "sys.modules",
            {
                "config": MagicMock(SystemConfig=MagicMock()),
                "core.scanner_engine": MagicMock(ScannerEngine=scanner_cls),
                "data.data_manager": MagicMock(DataManager=MagicMock()),
            },
        ):
            results = scale_symbol_count()

        assert isinstance(results, list)
        assert len(results) == 5  # counts = [10, 25, 50, 100, 200]
        for r in results:
            assert "symbols" in r
            assert "time" in r
            assert "alerts" in r

    @patch("benchmarks.scaling_tests.shutil.rmtree")
    @patch("benchmarks.scaling_tests.tempfile.mkdtemp", return_value="/tmp/fake")
    @patch("benchmarks.scaling_tests._make_ref", return_value=MagicMock())
    @patch("benchmarks.scaling_tests._setup_data", return_value=["SYM0000USDT"])
    def test_cleanup_called_per_count(
        self, mock_setup, mock_ref, mock_mkdtemp, mock_rmtree
    ):
        scanner_cls, _ = _mock_scanner([])

        with patch.dict(
            "sys.modules",
            {
                "config": MagicMock(SystemConfig=MagicMock()),
                "core.scanner_engine": MagicMock(ScannerEngine=scanner_cls),
                "data.data_manager": MagicMock(DataManager=MagicMock()),
            },
        ):
            scale_symbol_count()

        assert mock_rmtree.call_count == 5

    @patch("benchmarks.scaling_tests.shutil.rmtree")
    @patch("benchmarks.scaling_tests.tempfile.mkdtemp", return_value="/tmp/fake")
    @patch("benchmarks.scaling_tests._make_ref", return_value=MagicMock())
    @patch("benchmarks.scaling_tests._setup_data", return_value=["SYM0000USDT"])
    def test_scanner_error_handled_gracefully(
        self, mock_setup, mock_ref, mock_mkdtemp, mock_rmtree
    ):
        scanner_cls, scanner = _mock_scanner()
        scanner.scan_timeframe.side_effect = RuntimeError("boom")

        with patch.dict(
            "sys.modules",
            {
                "config": MagicMock(SystemConfig=MagicMock()),
                "core.scanner_engine": MagicMock(ScannerEngine=scanner_cls),
                "data.data_manager": MagicMock(DataManager=MagicMock()),
            },
        ):
            results = scale_symbol_count()

        # Error is caught per iteration; results list has no successful entries
        assert isinstance(results, list)
        assert len(results) == 0

    @patch("benchmarks.scaling_tests.shutil.rmtree")
    @patch("benchmarks.scaling_tests.tempfile.mkdtemp", return_value="/tmp/fake")
    @patch(
        "benchmarks.scaling_tests._setup_data",
        side_effect=BenchmarkSetupError("bad setup"),
    )
    def test_setup_error_propagates(self, mock_setup, mock_mkdtemp, mock_rmtree):
        with patch.dict(
            "sys.modules",
            {
                "config": MagicMock(SystemConfig=MagicMock()),
                "core.scanner_engine": MagicMock(ScannerEngine=MagicMock()),
                "data.data_manager": MagicMock(DataManager=MagicMock()),
            },
        ):
            with pytest.raises(BenchmarkSetupError):
                scale_symbol_count()

    @patch("benchmarks.scaling_tests.shutil.rmtree")
    @patch("benchmarks.scaling_tests.tempfile.mkdtemp", return_value="/tmp/fake")
    @patch("benchmarks.scaling_tests._make_ref", return_value=MagicMock())
    @patch("benchmarks.scaling_tests._setup_data", return_value=["SYM0000USDT"])
    def test_symbol_counts_in_results(
        self, mock_setup, mock_ref, mock_mkdtemp, mock_rmtree
    ):
        scanner_cls, _ = _mock_scanner(["a"])

        with patch.dict(
            "sys.modules",
            {
                "config": MagicMock(SystemConfig=MagicMock()),
                "core.scanner_engine": MagicMock(ScannerEngine=scanner_cls),
                "data.data_manager": MagicMock(DataManager=MagicMock()),
            },
        ):
            results = scale_symbol_count()

        assert [r["symbols"] for r in results] == [10, 25, 50, 100, 200]


# ===================================================================
# scale_reference_count
# ===================================================================


class TestScaleReferenceCount:
    """Tests for scale_reference_count."""

    @patch("benchmarks.scaling_tests.shutil.rmtree")
    @patch("benchmarks.scaling_tests.tempfile.mkdtemp", return_value="/tmp/fake")
    @patch("benchmarks.scaling_tests._make_ref", return_value=MagicMock())
    @patch(
        "benchmarks.scaling_tests._setup_data",
        return_value=[f"SYM{i:04d}USDT" for i in range(30)],
    )
    def test_returns_list_of_dicts(
        self, mock_setup, mock_ref, mock_mkdtemp, mock_rmtree
    ):
        scanner_cls, scanner = _mock_scanner(["a1", "a2"])

        with patch.dict(
            "sys.modules",
            {
                "config": MagicMock(SystemConfig=MagicMock()),
                "core.scanner_engine": MagicMock(ScannerEngine=scanner_cls),
                "data.data_manager": MagicMock(DataManager=MagicMock()),
            },
        ):
            results = scale_reference_count()

        assert isinstance(results, list)
        assert len(results) == 4  # ref_counts = [1, 5, 10, 20]
        for r in results:
            assert "refs" in r
            assert "time" in r
            assert "alerts" in r
            assert r["alerts"] == 2

    @patch("benchmarks.scaling_tests.shutil.rmtree")
    @patch("benchmarks.scaling_tests.tempfile.mkdtemp", return_value="/tmp/fake")
    @patch("benchmarks.scaling_tests._make_ref", return_value=MagicMock())
    @patch(
        "benchmarks.scaling_tests._setup_data",
        return_value=[f"SYM{i:04d}USDT" for i in range(30)],
    )
    def test_correct_ref_counts(self, mock_setup, mock_ref, mock_mkdtemp, mock_rmtree):
        scanner_cls, _ = _mock_scanner([])

        with patch.dict(
            "sys.modules",
            {
                "config": MagicMock(SystemConfig=MagicMock()),
                "core.scanner_engine": MagicMock(ScannerEngine=scanner_cls),
                "data.data_manager": MagicMock(DataManager=MagicMock()),
            },
        ):
            results = scale_reference_count()

        assert [r["refs"] for r in results] == [1, 5, 10, 20]

    @patch("benchmarks.scaling_tests.shutil.rmtree")
    @patch("benchmarks.scaling_tests.tempfile.mkdtemp", return_value="/tmp/fake")
    @patch(
        "benchmarks.scaling_tests._setup_data", side_effect=BenchmarkSetupError("bad")
    )
    def test_setup_error_propagates(self, mock_setup, mock_mkdtemp, mock_rmtree):
        with patch.dict(
            "sys.modules",
            {
                "config": MagicMock(SystemConfig=MagicMock()),
                "core.scanner_engine": MagicMock(ScannerEngine=MagicMock()),
                "data.data_manager": MagicMock(DataManager=MagicMock()),
            },
        ):
            with pytest.raises(BenchmarkSetupError):
                scale_reference_count()

    @patch("benchmarks.scaling_tests.shutil.rmtree")
    @patch("benchmarks.scaling_tests.tempfile.mkdtemp", return_value="/tmp/fake")
    @patch("benchmarks.scaling_tests._make_ref", return_value=MagicMock())
    @patch(
        "benchmarks.scaling_tests._setup_data",
        return_value=[f"SYM{i:04d}USDT" for i in range(30)],
    )
    def test_scanner_error_handled(
        self, mock_setup, mock_ref, mock_mkdtemp, mock_rmtree
    ):
        scanner_cls, scanner = _mock_scanner()
        scanner.scan_timeframe.side_effect = RuntimeError("scan fail")

        with patch.dict(
            "sys.modules",
            {
                "config": MagicMock(SystemConfig=MagicMock()),
                "core.scanner_engine": MagicMock(ScannerEngine=scanner_cls),
                "data.data_manager": MagicMock(DataManager=MagicMock()),
            },
        ):
            results = scale_reference_count()

        assert isinstance(results, list)

    @patch("benchmarks.scaling_tests.shutil.rmtree")
    @patch("benchmarks.scaling_tests.tempfile.mkdtemp", return_value="/tmp/fake")
    @patch("benchmarks.scaling_tests._make_ref", return_value=MagicMock())
    @patch("benchmarks.scaling_tests._setup_data", return_value=["SYM0000USDT"])
    def test_cleanup_always_called(
        self, mock_setup, mock_ref, mock_mkdtemp, mock_rmtree
    ):
        scanner_cls, scanner = _mock_scanner()
        scanner.scan_timeframe.side_effect = RuntimeError("fail")

        with patch.dict(
            "sys.modules",
            {
                "config": MagicMock(SystemConfig=MagicMock()),
                "core.scanner_engine": MagicMock(ScannerEngine=scanner_cls),
                "data.data_manager": MagicMock(DataManager=MagicMock()),
            },
        ):
            scale_reference_count()

        mock_rmtree.assert_called_once_with("/tmp/fake")


# ===================================================================
# scale_data_volume
# ===================================================================


class TestScaleDataVolume:
    """Tests for scale_data_volume."""

    @patch("benchmarks.scaling_tests._make_synthetic_ohlcv")
    def test_returns_list_of_dicts(self, mock_ohlcv):
        mock_ohlcv.return_value = pd.DataFrame({"close": [100.0, 101.0]})
        norm_cls, norm = _mock_normalizer()

        with patch.dict(
            "sys.modules",
            {
                "config": MagicMock(SystemConfig=MagicMock()),
                "core.data_normalizer": MagicMock(DataNormalizer=norm_cls),
            },
        ):
            results = scale_data_volume()

        assert isinstance(results, list)
        assert len(results) == 5  # row_counts = [100, 500, 1000, 5000, 10000]
        for r in results:
            assert "rows" in r
            assert "time" in r
            assert r["time"] >= 0

    @patch("benchmarks.scaling_tests._make_synthetic_ohlcv")
    def test_correct_row_counts(self, mock_ohlcv):
        mock_ohlcv.return_value = pd.DataFrame({"close": [1.0]})
        norm_cls, _ = _mock_normalizer()

        with patch.dict(
            "sys.modules",
            {
                "config": MagicMock(SystemConfig=MagicMock()),
                "core.data_normalizer": MagicMock(DataNormalizer=norm_cls),
            },
        ):
            results = scale_data_volume()

        assert [r["rows"] for r in results] == [100, 500, 1000, 5000, 10000]

    @patch("benchmarks.scaling_tests._make_synthetic_ohlcv")
    def test_normalizer_called_20_times_per_row_count(self, mock_ohlcv):
        mock_ohlcv.return_value = pd.DataFrame({"close": [1.0]})
        norm_cls, norm = _mock_normalizer()

        with patch.dict(
            "sys.modules",
            {
                "config": MagicMock(SystemConfig=MagicMock()),
                "core.data_normalizer": MagicMock(DataNormalizer=norm_cls),
            },
        ):
            scale_data_volume()

        # 5 row_counts * 20 iterations = 100
        assert norm.prepare_for_dtw.call_count == 100

    @patch("benchmarks.scaling_tests._make_synthetic_ohlcv")
    def test_time_is_averaged_over_20_runs(self, mock_ohlcv):
        mock_ohlcv.return_value = pd.DataFrame({"close": [1.0]})
        norm_cls, _ = _mock_normalizer()

        with patch.dict(
            "sys.modules",
            {
                "config": MagicMock(SystemConfig=MagicMock()),
                "core.data_normalizer": MagicMock(DataNormalizer=norm_cls),
            },
        ):
            results = scale_data_volume()

        # Each time should be very small (mocked calls are near-instant)
        for r in results:
            assert r["time"] < 1.0


# ===================================================================
# scale_worker_count
# ===================================================================


class TestScaleWorkerCount:
    """Tests for scale_worker_count."""

    @patch("benchmarks.scaling_tests.shutil.rmtree")
    @patch("benchmarks.scaling_tests.tempfile.mkdtemp", return_value="/tmp/fake")
    @patch("benchmarks.scaling_tests._make_ref", return_value=MagicMock())
    @patch(
        "benchmarks.scaling_tests._setup_data",
        return_value=[f"SYM{i:04d}USDT" for i in range(50)],
    )
    def test_returns_list_of_dicts(
        self, mock_setup, mock_ref, mock_mkdtemp, mock_rmtree
    ):
        scanner_cls, _ = _mock_scanner(["alert"])

        with patch.dict(
            "sys.modules",
            {
                "config": MagicMock(SystemConfig=MagicMock()),
                "core.scanner_engine": MagicMock(ScannerEngine=scanner_cls),
                "data.data_manager": MagicMock(DataManager=MagicMock()),
            },
        ):
            results = scale_worker_count()

        assert isinstance(results, list)
        assert len(results) == 4  # worker_counts = [1, 2, 4, 8]
        for r in results:
            assert "workers" in r
            assert "time" in r
            assert "alerts" in r

    @patch("benchmarks.scaling_tests.shutil.rmtree")
    @patch("benchmarks.scaling_tests.tempfile.mkdtemp", return_value="/tmp/fake")
    @patch("benchmarks.scaling_tests._make_ref", return_value=MagicMock())
    @patch(
        "benchmarks.scaling_tests._setup_data",
        return_value=[f"SYM{i:04d}USDT" for i in range(50)],
    )
    def test_correct_worker_counts(
        self, mock_setup, mock_ref, mock_mkdtemp, mock_rmtree
    ):
        scanner_cls, _ = _mock_scanner([])

        with patch.dict(
            "sys.modules",
            {
                "config": MagicMock(SystemConfig=MagicMock()),
                "core.scanner_engine": MagicMock(ScannerEngine=scanner_cls),
                "data.data_manager": MagicMock(DataManager=MagicMock()),
            },
        ):
            results = scale_worker_count()

        assert [r["workers"] for r in results] == [1, 2, 4, 8]

    @patch("benchmarks.scaling_tests.shutil.rmtree")
    @patch("benchmarks.scaling_tests.tempfile.mkdtemp", return_value="/tmp/fake")
    @patch("benchmarks.scaling_tests._make_ref", return_value=MagicMock())
    @patch(
        "benchmarks.scaling_tests._setup_data",
        return_value=[f"SYM{i:04d}USDT" for i in range(50)],
    )
    def test_num_workers_passed_to_scanner(
        self, mock_setup, mock_ref, mock_mkdtemp, mock_rmtree
    ):
        scanner_cls, scanner = _mock_scanner([])

        with patch.dict(
            "sys.modules",
            {
                "config": MagicMock(SystemConfig=MagicMock()),
                "core.scanner_engine": MagicMock(ScannerEngine=scanner_cls),
                "data.data_manager": MagicMock(DataManager=MagicMock()),
            },
        ):
            scale_worker_count()

        for c in scanner.scan_timeframe.call_args_list:
            assert "num_workers" in c.kwargs

    @patch("benchmarks.scaling_tests.shutil.rmtree")
    @patch("benchmarks.scaling_tests.tempfile.mkdtemp", return_value="/tmp/fake")
    @patch(
        "benchmarks.scaling_tests._setup_data", side_effect=BenchmarkSetupError("bad")
    )
    def test_setup_error_propagates(self, mock_setup, mock_mkdtemp, mock_rmtree):
        with patch.dict(
            "sys.modules",
            {
                "config": MagicMock(SystemConfig=MagicMock()),
                "core.scanner_engine": MagicMock(ScannerEngine=MagicMock()),
                "data.data_manager": MagicMock(DataManager=MagicMock()),
            },
        ):
            with pytest.raises(BenchmarkSetupError):
                scale_worker_count()

    @patch("benchmarks.scaling_tests.shutil.rmtree")
    @patch("benchmarks.scaling_tests.tempfile.mkdtemp", return_value="/tmp/fake")
    @patch("benchmarks.scaling_tests._make_ref", return_value=MagicMock())
    @patch("benchmarks.scaling_tests._setup_data", return_value=["SYM0000USDT"])
    def test_cleanup_on_error(self, mock_setup, mock_ref, mock_mkdtemp, mock_rmtree):
        scanner_cls, scanner = _mock_scanner()
        scanner.scan_timeframe.side_effect = RuntimeError("fail")

        with patch.dict(
            "sys.modules",
            {
                "config": MagicMock(SystemConfig=MagicMock()),
                "core.scanner_engine": MagicMock(ScannerEngine=scanner_cls),
                "data.data_manager": MagicMock(DataManager=MagicMock()),
            },
        ):
            scale_worker_count()

        mock_rmtree.assert_called_once_with("/tmp/fake")

    @patch("benchmarks.scaling_tests.shutil.rmtree")
    @patch("benchmarks.scaling_tests.tempfile.mkdtemp", return_value="/tmp/fake")
    @patch("benchmarks.scaling_tests._make_ref", return_value=MagicMock())
    @patch(
        "benchmarks.scaling_tests._setup_data",
        return_value=[f"SYM{i:04d}USDT" for i in range(50)],
    )
    def test_scanner_created_per_worker(
        self, mock_setup, mock_ref, mock_mkdtemp, mock_rmtree
    ):
        """ScannerEngine is instantiated once per worker count."""
        scanner_cls, _ = _mock_scanner([])

        with patch.dict(
            "sys.modules",
            {
                "config": MagicMock(SystemConfig=MagicMock()),
                "core.scanner_engine": MagicMock(ScannerEngine=scanner_cls),
                "data.data_manager": MagicMock(DataManager=MagicMock()),
            },
        ):
            scale_worker_count()

        # 4 worker counts -> 4 ScannerEngine instantiations
        assert scanner_cls.call_count == 4


# ===================================================================
# main
# ===================================================================


class TestMain:
    """Tests for the main() orchestrator."""

    @patch("benchmarks.scaling_tests.scale_worker_count")
    @patch("benchmarks.scaling_tests.scale_data_volume")
    @patch("benchmarks.scaling_tests.scale_reference_count")
    @patch("benchmarks.scaling_tests.scale_symbol_count")
    def test_calls_all_scaling_functions(self, mock_sym, mock_ref, mock_vol, mock_wrk):
        mock_sym.return_value = [{"symbols": 10, "time": 0.1, "alerts": 0}]
        mock_ref.return_value = [{"refs": 1, "time": 0.1, "alerts": 0}]
        mock_vol.return_value = [{"rows": 100, "time": 0.01}]
        mock_wrk.return_value = [{"workers": 1, "time": 0.1, "alerts": 0}]

        main()

        mock_sym.assert_called_once()
        mock_ref.assert_called_once()
        mock_vol.assert_called_once()
        mock_wrk.assert_called_once()

    @patch("benchmarks.scaling_tests.scale_worker_count")
    @patch("benchmarks.scaling_tests.scale_data_volume")
    @patch("benchmarks.scaling_tests.scale_reference_count")
    @patch("benchmarks.scaling_tests.scale_symbol_count")
    def test_handles_exception_in_one_function(
        self, mock_sym, mock_ref, mock_vol, mock_wrk
    ):
        mock_sym.side_effect = RuntimeError("kaboom")
        mock_ref.return_value = [{"refs": 1, "time": 0.1, "alerts": 0}]
        mock_vol.return_value = [{"rows": 100, "time": 0.01}]
        mock_wrk.return_value = [{"workers": 1, "time": 0.1, "alerts": 0}]

        # Should NOT raise -- main catches per-function errors
        main()

    @patch("benchmarks.scaling_tests.scale_worker_count")
    @patch("benchmarks.scaling_tests.scale_data_volume")
    @patch("benchmarks.scaling_tests.scale_reference_count")
    @patch("benchmarks.scaling_tests.scale_symbol_count")
    def test_returns_none(self, mock_sym, mock_ref, mock_vol, mock_wrk):
        mock_sym.return_value = []
        mock_ref.return_value = []
        mock_vol.return_value = []
        mock_wrk.return_value = []

        result = main()
        assert result is None

    @patch("benchmarks.scaling_tests.scale_worker_count")
    @patch("benchmarks.scaling_tests.scale_data_volume")
    @patch("benchmarks.scaling_tests.scale_reference_count")
    @patch("benchmarks.scaling_tests.scale_symbol_count")
    def test_continues_after_failure(self, mock_sym, mock_ref, mock_vol, mock_wrk):
        """Even if the first function fails, the rest should still be called."""
        mock_sym.side_effect = RuntimeError("fail")
        mock_ref.side_effect = RuntimeError("fail")
        mock_vol.return_value = [{"rows": 100, "time": 0.01}]
        mock_wrk.return_value = [{"workers": 1, "time": 0.1, "alerts": 0}]

        main()

        # All four should have been attempted
        mock_sym.assert_called_once()
        mock_ref.assert_called_once()
        mock_vol.assert_called_once()
        mock_wrk.assert_called_once()


# ===================================================================
# _SCHEMA constant
# ===================================================================


class TestSchema:
    """Tests for the module-level _SCHEMA constant."""

    def test_schema_field_names(self):
        assert _SCHEMA.names == [
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]

    def test_schema_types(self):
        assert _SCHEMA.field("timestamp").type == pa.int64()
        for col in ("open", "high", "low", "close", "volume"):
            assert _SCHEMA.field(col).type == pa.float64()

    def test_schema_is_pyarrow_schema(self):
        assert isinstance(_SCHEMA, pa.Schema)

    def test_schema_has_six_fields(self):
        assert len(_SCHEMA) == 6


# ===================================================================
# BenchmarkResult TypedDict
# ===================================================================


class TestBenchmarkResult:
    """Tests for the BenchmarkResult TypedDict."""

    def test_is_typeddict(self):
        # TypedDict subclasses dict at runtime
        r: BenchmarkResult = {
            "name": "test",
            "value": 1.5,
            "unit": "seconds",
            "detail": "some detail",
        }
        assert isinstance(r, dict)

    def test_annotations(self):
        annotations = BenchmarkResult.__annotations__
        assert "name" in annotations
        assert "value" in annotations
        assert "unit" in annotations
        assert "detail" in annotations
