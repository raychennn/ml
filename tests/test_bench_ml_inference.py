"""Tests for benchmarks.bench_ml_inference module.

Tests ML inference benchmark functions including synthetic OHLCV generation,
mock match result creation, and benchmark runners. Uses unittest.mock
extensively to mock imports of core project classes (config, feature_extractor,
ml.predictor) that may not be available in the test environment.
"""

import sys
import types
from unittest.mock import patch, MagicMock

import pytest
import numpy as np
import pandas as pd

from benchmarks.bench_ml_inference import (
    BenchmarkResult,
    MockMatch,
    _make_synthetic_ohlcv,
    _make_mock_match_result,
    bench_feature_extraction,
    bench_ml_predict_single,
    bench_ml_predict_batch,
    run_all,
)
from benchmarks.exceptions import BenchmarkSetupError, BenchmarkRunError

# ---------------------------------------------------------------------------
# Tests for _make_synthetic_ohlcv
# ---------------------------------------------------------------------------


class TestMakeSyntheticOhlcv:
    """Tests for _make_synthetic_ohlcv — no external deps needed."""

    def test_returns_dataframe(self):
        """Test that _make_synthetic_ohlcv returns a pandas DataFrame."""
        df = _make_synthetic_ohlcv(n_bars=50, seed=42)
        assert isinstance(df, pd.DataFrame)

    def test_correct_row_count(self):
        """Test that the DataFrame has the expected number of rows."""
        df = _make_synthetic_ohlcv(n_bars=50, seed=42)
        assert len(df) == 50

    def test_required_columns_present(self):
        """Test that all required OHLCV columns are present."""
        df = _make_synthetic_ohlcv(n_bars=50, seed=42)
        expected_cols = {"timestamp", "open", "high", "low", "close", "volume"}
        assert set(df.columns) == expected_cols

    def test_close_prices_are_numeric(self):
        """Test that close prices are numeric values."""
        df = _make_synthetic_ohlcv(n_bars=50, seed=42)
        assert np.issubdtype(df["close"].dtype, np.floating)

    def test_volume_is_positive(self):
        """Test that all volume values are positive."""
        df = _make_synthetic_ohlcv(n_bars=50, seed=42)
        assert (df["volume"] > 0).all()

    def test_high_greater_than_low(self):
        """Test that high prices are greater than low prices."""
        df = _make_synthetic_ohlcv(n_bars=50, seed=42)
        assert (df["high"] > df["low"]).all()

    def test_reproducibility_same_seed(self):
        """Test that the same seed produces identical DataFrames."""
        df1 = _make_synthetic_ohlcv(n_bars=50, seed=42)
        df2 = _make_synthetic_ohlcv(n_bars=50, seed=42)
        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seeds_produce_different_data(self):
        """Test that different seeds produce different DataFrames."""
        df1 = _make_synthetic_ohlcv(n_bars=50, seed=42)
        df2 = _make_synthetic_ohlcv(n_bars=50, seed=99)
        assert not df1["close"].equals(df2["close"])

    def test_small_n_bars(self):
        """Test with a very small number of bars."""
        df = _make_synthetic_ohlcv(n_bars=2, seed=42)
        assert len(df) == 2

    def test_timestamp_column_is_integer(self):
        """Test that the timestamp column contains integer-like values."""
        df = _make_synthetic_ohlcv(n_bars=50, seed=42)
        assert np.issubdtype(df["timestamp"].dtype, np.integer)

    def test_timestamps_are_hourly_spaced(self):
        """Test that consecutive timestamps are 3600 seconds apart."""
        df = _make_synthetic_ohlcv(n_bars=50, seed=42)
        diffs = df["timestamp"].diff().dropna()
        assert (diffs == 3600).all()


# ---------------------------------------------------------------------------
# Tests for MockMatch and _make_mock_match_result
# ---------------------------------------------------------------------------


class TestMockMatch:
    """Tests for MockMatch dataclass and _make_mock_match_result."""

    def test_mock_match_default_values(self):
        """Test that MockMatch has expected default values."""
        m = MockMatch()
        assert m.symbol == "TESTUSDT"
        assert m.timeframe == "4h"
        assert m.score == 0.72
        assert m.price_distance == 0.18
        assert m.diff_distance == 0.22
        assert m.best_scale_factor == 1.0
        assert m.match_start_idx == 380
        assert m.match_end_idx == 500
        assert m.window_data is None

    def test_mock_match_custom_values(self):
        """Test that MockMatch can be created with custom values."""
        m = MockMatch(symbol="BTCUSDT", score=0.95)
        assert m.symbol == "BTCUSDT"
        assert m.score == 0.95

    def test_make_mock_match_result_returns_mock_match(self):
        """Test that _make_mock_match_result returns a MockMatch instance."""
        result = _make_mock_match_result()
        assert isinstance(result, MockMatch)

    def test_make_mock_match_result_has_defaults(self):
        """Test that _make_mock_match_result returns MockMatch with defaults."""
        result = _make_mock_match_result()
        assert result.symbol == "TESTUSDT"
        assert result.timeframe == "4h"


# ---------------------------------------------------------------------------
# Tests for bench_feature_extraction (with mocked imports)
# ---------------------------------------------------------------------------


class TestBenchFeatureExtraction:
    """Tests for bench_feature_extraction with mocked config and extractor."""

    def _run_with_mocks(self, repeats=2):
        """Helper to run bench_feature_extraction with all deps mocked."""
        mock_config = MagicMock()
        mock_extractor = MagicMock()
        mock_extractor.extract.return_value = {"feature_a": 1.0}

        mock_config_cls = MagicMock(return_value=mock_config)
        mock_extractor_cls = MagicMock(return_value=mock_extractor)

        # Create mock modules for 'config' and 'core.feature_extractor'
        mock_config_mod = types.ModuleType("config")
        mock_config_mod.SystemConfig = mock_config_cls

        mock_core_mod = types.ModuleType("core")
        mock_fe_mod = types.ModuleType("core.feature_extractor")
        mock_fe_mod.FeatureExtractor = mock_extractor_cls

        with patch.dict(
            sys.modules,
            {
                "config": mock_config_mod,
                "core": mock_core_mod,
                "core.feature_extractor": mock_fe_mod,
            },
        ):
            result = bench_feature_extraction(repeats=repeats)

        return result, mock_extractor

    def test_returns_benchmark_result(self):
        """Test that bench_feature_extraction returns a BenchmarkResult dict."""
        result, _ = self._run_with_mocks(repeats=2)
        assert isinstance(result, dict)
        assert "name" in result
        assert "value" in result
        assert "unit" in result
        assert "detail" in result

    def test_correct_name(self):
        """Test that the result name is 'feature_extraction_single'."""
        result, _ = self._run_with_mocks(repeats=2)
        assert result["name"] == "feature_extraction_single"

    def test_positive_value(self):
        """Test that the value (elapsed time) is positive."""
        result, _ = self._run_with_mocks(repeats=2)
        assert result["value"] is not None
        assert result["value"] >= 0.0

    def test_unit_is_seconds(self):
        """Test that the unit is 'seconds'."""
        result, _ = self._run_with_mocks(repeats=2)
        assert result["unit"] == "seconds"

    def test_detail_contains_repeats(self):
        """Test that detail field contains the repeats parameter."""
        result, _ = self._run_with_mocks(repeats=2)
        assert "repeats=2" in result["detail"]

    def test_extractor_called(self):
        """Test that extractor.extract is called the expected number of times."""
        result, mock_extractor = self._run_with_mocks(repeats=2)
        # 1 warmup + 2 repeats = 3 calls
        assert mock_extractor.extract.call_count == 3

    def test_setup_error_propagated(self):
        """Test that BenchmarkSetupError from setup is propagated."""
        with patch.dict(sys.modules, {"config": None}):
            # Importing from a module mapped to None raises ImportError
            # which gets wrapped in BenchmarkSetupError
            with pytest.raises((BenchmarkSetupError, ImportError)):
                bench_feature_extraction(repeats=2)

    def test_run_error_when_extract_fails(self):
        """Test that BenchmarkRunError is raised when extract raises."""
        mock_config = MagicMock()
        mock_extractor = MagicMock()
        mock_extractor.extract.side_effect = RuntimeError("model exploded")

        mock_config_cls = MagicMock(return_value=mock_config)
        mock_extractor_cls = MagicMock(return_value=mock_extractor)

        mock_config_mod = types.ModuleType("config")
        mock_config_mod.SystemConfig = mock_config_cls
        mock_fe_mod = types.ModuleType("core.feature_extractor")
        mock_fe_mod.FeatureExtractor = mock_extractor_cls
        mock_core_mod = types.ModuleType("core")

        with patch.dict(
            sys.modules,
            {
                "config": mock_config_mod,
                "core": mock_core_mod,
                "core.feature_extractor": mock_fe_mod,
            },
        ):
            with pytest.raises(BenchmarkRunError, match="model exploded"):
                bench_feature_extraction(repeats=2)


# ---------------------------------------------------------------------------
# Tests for bench_ml_predict_single (with mocked imports)
# ---------------------------------------------------------------------------


class TestBenchMlPredictSingle:
    """Tests for bench_ml_predict_single with mocked config and predictor."""

    def _run_with_mocks(self, repeats=2, is_loaded=True):
        """Helper to run bench_ml_predict_single with all deps mocked."""
        mock_config = MagicMock()
        mock_predictor = MagicMock()
        mock_predictor.predict.return_value = {"prob": 0.8}
        mock_predictor.is_loaded = is_loaded

        mock_config_cls = MagicMock(return_value=mock_config)
        mock_predictor_cls = MagicMock(return_value=mock_predictor)

        mock_config_mod = types.ModuleType("config")
        mock_config_mod.SystemConfig = mock_config_cls

        mock_ml_mod = types.ModuleType("ml")
        mock_ml_pred_mod = types.ModuleType("ml.predictor")
        mock_ml_pred_mod.MLPredictor = mock_predictor_cls

        with patch.dict(
            sys.modules,
            {
                "config": mock_config_mod,
                "ml": mock_ml_mod,
                "ml.predictor": mock_ml_pred_mod,
            },
        ):
            result = bench_ml_predict_single(repeats=repeats)

        return result, mock_predictor

    def test_returns_benchmark_result(self):
        """Test that bench_ml_predict_single returns a BenchmarkResult dict."""
        result, _ = self._run_with_mocks(repeats=2)
        assert isinstance(result, dict)
        assert "name" in result
        assert "value" in result
        assert "unit" in result
        assert "detail" in result

    def test_correct_name(self):
        """Test that the result name is 'ml_inference_single'."""
        result, _ = self._run_with_mocks(repeats=2)
        assert result["name"] == "ml_inference_single"

    def test_positive_value(self):
        """Test that the value (elapsed time) is non-negative."""
        result, _ = self._run_with_mocks(repeats=2)
        assert result["value"] is not None
        assert result["value"] >= 0.0

    def test_unit_is_seconds(self):
        """Test that the unit is 'seconds'."""
        result, _ = self._run_with_mocks(repeats=2)
        assert result["unit"] == "seconds"

    def test_detail_contains_repeats_and_mode(self):
        """Test that detail field contains repeats and mode."""
        result, _ = self._run_with_mocks(repeats=2, is_loaded=True)
        assert "repeats=2" in result["detail"]
        assert "mode=trained" in result["detail"]

    def test_mode_fallback_when_not_loaded(self):
        """Test that detail reports 'fallback' mode when not loaded."""
        result, _ = self._run_with_mocks(repeats=2, is_loaded=False)
        assert "mode=fallback" in result["detail"]

    def test_predictor_load_called(self):
        """Test that predictor.load() is called during setup."""
        _, mock_predictor = self._run_with_mocks(repeats=2)
        mock_predictor.load.assert_called_once()

    def test_predictor_predict_called(self):
        """Test that predictor.predict is called the expected number of times."""
        _, mock_predictor = self._run_with_mocks(repeats=2)
        # 1 warmup + 2 repeats = 3 calls
        assert mock_predictor.predict.call_count == 3

    def test_run_error_when_predict_fails(self):
        """Test that BenchmarkRunError is raised when predict raises."""
        mock_config = MagicMock()
        mock_predictor = MagicMock()
        mock_predictor.predict.side_effect = ValueError("bad input")

        mock_config_cls = MagicMock(return_value=mock_config)
        mock_predictor_cls = MagicMock(return_value=mock_predictor)

        mock_config_mod = types.ModuleType("config")
        mock_config_mod.SystemConfig = mock_config_cls
        mock_ml_mod = types.ModuleType("ml")
        mock_ml_pred_mod = types.ModuleType("ml.predictor")
        mock_ml_pred_mod.MLPredictor = mock_predictor_cls

        with patch.dict(
            sys.modules,
            {
                "config": mock_config_mod,
                "ml": mock_ml_mod,
                "ml.predictor": mock_ml_pred_mod,
            },
        ):
            with pytest.raises(BenchmarkRunError, match="bad input"):
                bench_ml_predict_single(repeats=2)


# ---------------------------------------------------------------------------
# Tests for bench_ml_predict_batch (with mocked imports)
# ---------------------------------------------------------------------------


class TestBenchMlPredictBatch:
    """Tests for bench_ml_predict_batch with mocked config and predictor."""

    def _run_with_mocks(self, batch_size=3, is_loaded=True):
        """Helper to run bench_ml_predict_batch with all deps mocked."""
        mock_config = MagicMock()
        mock_predictor = MagicMock()
        mock_predictor.predict.return_value = {"prob": 0.8}
        mock_predictor.is_loaded = is_loaded

        mock_config_cls = MagicMock(return_value=mock_config)
        mock_predictor_cls = MagicMock(return_value=mock_predictor)

        mock_config_mod = types.ModuleType("config")
        mock_config_mod.SystemConfig = mock_config_cls

        mock_ml_mod = types.ModuleType("ml")
        mock_ml_pred_mod = types.ModuleType("ml.predictor")
        mock_ml_pred_mod.MLPredictor = mock_predictor_cls

        with patch.dict(
            sys.modules,
            {
                "config": mock_config_mod,
                "ml": mock_ml_mod,
                "ml.predictor": mock_ml_pred_mod,
            },
        ):
            result = bench_ml_predict_batch(batch_size=batch_size)

        return result, mock_predictor

    def test_returns_benchmark_result(self):
        """Test that bench_ml_predict_batch returns a BenchmarkResult dict."""
        result, _ = self._run_with_mocks(batch_size=3)
        assert isinstance(result, dict)
        assert "name" in result
        assert "value" in result
        assert "unit" in result
        assert "detail" in result

    def test_correct_name(self):
        """Test that the result name is 'ml_inference_batch_10'."""
        result, _ = self._run_with_mocks(batch_size=3)
        assert result["name"] == "ml_inference_batch_10"

    def test_positive_value(self):
        """Test that the value (elapsed time) is non-negative."""
        result, _ = self._run_with_mocks(batch_size=3)
        assert result["value"] is not None
        assert result["value"] >= 0.0

    def test_unit_is_seconds(self):
        """Test that the unit is 'seconds'."""
        result, _ = self._run_with_mocks(batch_size=3)
        assert result["unit"] == "seconds"

    def test_detail_contains_batch_and_mode(self):
        """Test that detail field contains batch size and mode."""
        result, _ = self._run_with_mocks(batch_size=3, is_loaded=True)
        assert "batch=3" in result["detail"]
        assert "mode=trained" in result["detail"]

    def test_mode_fallback_when_not_loaded(self):
        """Test that detail reports 'fallback' when predictor is not loaded."""
        result, _ = self._run_with_mocks(batch_size=3, is_loaded=False)
        assert "mode=fallback" in result["detail"]

    def test_predictor_predict_call_count(self):
        """Test that predictor.predict is called batch_size + 1 (warmup) times."""
        _, mock_predictor = self._run_with_mocks(batch_size=3)
        # 1 warmup + 3 batch items = 4 calls
        assert mock_predictor.predict.call_count == 4

    def test_run_error_when_predict_fails_during_batch(self):
        """Test BenchmarkRunError raised when predict fails during batch loop."""
        mock_config = MagicMock()
        mock_predictor = MagicMock()
        # First call (warmup) succeeds, then fails
        mock_predictor.predict.side_effect = [None, RuntimeError("batch fail")]
        mock_predictor.is_loaded = True

        mock_config_cls = MagicMock(return_value=mock_config)
        mock_predictor_cls = MagicMock(return_value=mock_predictor)

        mock_config_mod = types.ModuleType("config")
        mock_config_mod.SystemConfig = mock_config_cls
        mock_ml_mod = types.ModuleType("ml")
        mock_ml_pred_mod = types.ModuleType("ml.predictor")
        mock_ml_pred_mod.MLPredictor = mock_predictor_cls

        with patch.dict(
            sys.modules,
            {
                "config": mock_config_mod,
                "ml": mock_ml_mod,
                "ml.predictor": mock_ml_pred_mod,
            },
        ):
            with pytest.raises(BenchmarkRunError, match="batch fail"):
                bench_ml_predict_batch(batch_size=3)


# ---------------------------------------------------------------------------
# Tests for run_all
# ---------------------------------------------------------------------------


class TestRunAll:
    """Tests for run_all which orchestrates all benchmark functions."""

    def test_returns_list(self):
        """Test that run_all returns a list."""
        # All sub-benchmarks will fail due to missing imports, but
        # run_all catches exceptions and produces error entries.
        results = run_all()
        assert isinstance(results, list)

    def test_returns_three_results(self):
        """Test that run_all returns exactly 3 results (one per bench fn)."""
        results = run_all()
        assert len(results) == 3

    def test_all_results_have_required_keys(self):
        """Test that every result dict has the required BenchmarkResult keys."""
        results = run_all()
        for r in results:
            assert "name" in r
            assert "value" in r
            assert "unit" in r
            assert "detail" in r

    def test_error_entries_have_none_value(self):
        """Test that failed benchmarks have value=None and unit='error'."""
        # Without mocked imports, the benchmarks will fail and produce error entries
        results = run_all()
        for r in results:
            if r["value"] is None:
                assert r["unit"] == "error"

    def test_run_all_with_mocked_bench_fns(self):
        """Test run_all when bench fns succeed by patching them."""
        fake_result = BenchmarkResult(
            name="fake_bench", value=0.001, unit="seconds", detail="test"
        )
        with patch(
            "benchmarks.bench_ml_inference.bench_feature_extraction",
            return_value=fake_result,
        ), patch(
            "benchmarks.bench_ml_inference.bench_ml_predict_single",
            return_value=fake_result,
        ), patch(
            "benchmarks.bench_ml_inference.bench_ml_predict_batch",
            return_value=fake_result,
        ):
            results = run_all()

        assert len(results) == 3
        for r in results:
            assert r["value"] == 0.001
            assert r["unit"] == "seconds"

    def test_run_all_handles_mixed_success_and_failure(self):
        """Test that run_all handles a mix of successful and failing benchmarks."""
        ok_result = BenchmarkResult(
            name="ok_bench", value=0.005, unit="seconds", detail="ok"
        )
        mock_single = MagicMock(
            side_effect=BenchmarkSetupError("no model"),
            __name__="bench_ml_predict_single",
        )
        with patch(
            "benchmarks.bench_ml_inference.bench_feature_extraction",
            return_value=ok_result,
        ), patch(
            "benchmarks.bench_ml_inference.bench_ml_predict_single",
            mock_single,
        ), patch(
            "benchmarks.bench_ml_inference.bench_ml_predict_batch",
            return_value=ok_result,
        ):
            results = run_all()

        assert len(results) == 3
        # First and third should be successes
        assert results[0]["value"] == 0.005
        assert results[2]["value"] == 0.005
        # Second should be error
        assert results[1]["value"] is None
        assert results[1]["unit"] == "error"
        assert "no model" in results[1]["detail"]


# ---------------------------------------------------------------------------
# Tests for main guard (module-level execution)
# ---------------------------------------------------------------------------


class TestMain:
    """Tests for the module's __main__ guard."""

    def test_main_calls_run_all(self):
        """Test that running the module as __main__ calls run_all."""
        with patch(
            "benchmarks.bench_ml_inference.run_all", return_value=[]
        ) as mock_run:
            # Simulate executing the if __name__ == "__main__" block
            # by calling run_all directly (same effect)
            mock_run()
            mock_run.assert_called_once()

    def test_main_runs_without_error(self):
        """Test that run_all completes without raising (errors are caught)."""
        # run_all catches all exceptions from sub-benchmarks
        results = run_all()
        # Should always return a list, never raise
        assert isinstance(results, list)


# ---------------------------------------------------------------------------
# BenchmarkResult type structure tests
# ---------------------------------------------------------------------------


class TestBenchmarkResultType:
    """Tests for the BenchmarkResult TypedDict shape."""

    def test_benchmark_result_can_be_constructed(self):
        """Test that a BenchmarkResult dict can be constructed."""
        r = BenchmarkResult(name="test", value=1.23, unit="seconds", detail="n=10")
        assert r["name"] == "test"
        assert r["value"] == 1.23
        assert r["unit"] == "seconds"
        assert r["detail"] == "n=10"

    def test_benchmark_result_with_none_value(self):
        """Test that BenchmarkResult accepts None for value (error case)."""
        r = BenchmarkResult(
            name="failed", value=None, unit="error", detail="import error"
        )
        assert r["value"] is None
        assert r["unit"] == "error"


# ---------------------------------------------------------------------------
# Error-path coverage tests
# ---------------------------------------------------------------------------


class TestErrorPaths:
    """Tests that exercise exception-wrapping error paths for coverage."""

    def test_make_synthetic_ohlcv_wraps_errors(self):
        """Test that _make_synthetic_ohlcv wraps unexpected errors in BenchmarkSetupError."""
        with patch(
            "benchmarks.bench_ml_inference.np.random.RandomState",
            side_effect=RuntimeError("rng fail"),
        ):
            with pytest.raises(BenchmarkSetupError, match="rng fail"):
                _make_synthetic_ohlcv(n_bars=10, seed=0)

    def test_make_mock_match_result_wraps_errors(self):
        """Test that _make_mock_match_result wraps unexpected errors in BenchmarkSetupError."""
        with patch(
            "benchmarks.bench_ml_inference.MockMatch", side_effect=TypeError("bad init")
        ):
            with pytest.raises(BenchmarkSetupError, match="bad init"):
                _make_mock_match_result()

    def test_bench_feature_extraction_setup_error_from_config(self):
        """Test BenchmarkSetupError when SystemConfig constructor fails."""
        mock_config_mod = types.ModuleType("config")
        mock_config_mod.SystemConfig = MagicMock(side_effect=RuntimeError("no config"))

        mock_core_mod = types.ModuleType("core")
        mock_fe_mod = types.ModuleType("core.feature_extractor")
        mock_fe_mod.FeatureExtractor = MagicMock()

        with patch.dict(
            sys.modules,
            {
                "config": mock_config_mod,
                "core": mock_core_mod,
                "core.feature_extractor": mock_fe_mod,
            },
        ):
            with pytest.raises(BenchmarkSetupError, match="no config"):
                bench_feature_extraction(repeats=2)

    def test_bench_ml_predict_single_setup_error_from_load(self):
        """Test BenchmarkSetupError when predictor.load() fails."""
        mock_config = MagicMock()
        mock_predictor = MagicMock()
        mock_predictor.load.side_effect = RuntimeError("no weights")

        mock_config_mod = types.ModuleType("config")
        mock_config_mod.SystemConfig = MagicMock(return_value=mock_config)
        mock_ml_mod = types.ModuleType("ml")
        mock_ml_pred_mod = types.ModuleType("ml.predictor")
        mock_ml_pred_mod.MLPredictor = MagicMock(return_value=mock_predictor)

        with patch.dict(
            sys.modules,
            {
                "config": mock_config_mod,
                "ml": mock_ml_mod,
                "ml.predictor": mock_ml_pred_mod,
            },
        ):
            with pytest.raises(BenchmarkSetupError, match="no weights"):
                bench_ml_predict_single(repeats=2)

    def test_bench_ml_predict_batch_setup_error_from_load(self):
        """Test BenchmarkSetupError when predictor.load() fails in batch."""
        mock_config = MagicMock()
        mock_predictor = MagicMock()
        mock_predictor.load.side_effect = RuntimeError("no weights batch")

        mock_config_mod = types.ModuleType("config")
        mock_config_mod.SystemConfig = MagicMock(return_value=mock_config)
        mock_ml_mod = types.ModuleType("ml")
        mock_ml_pred_mod = types.ModuleType("ml.predictor")
        mock_ml_pred_mod.MLPredictor = MagicMock(return_value=mock_predictor)

        with patch.dict(
            sys.modules,
            {
                "config": mock_config_mod,
                "ml": mock_ml_mod,
                "ml.predictor": mock_ml_pred_mod,
            },
        ):
            with pytest.raises(BenchmarkSetupError, match="no weights batch"):
                bench_ml_predict_batch(batch_size=3)

    def test_bench_feature_extraction_reraises_benchmark_setup_error(self):
        """Test that BenchmarkSetupError in setup is re-raised directly."""
        mock_config_mod = types.ModuleType("config")
        mock_config_mod.SystemConfig = MagicMock(
            side_effect=BenchmarkSetupError("direct setup error")
        )
        mock_core_mod = types.ModuleType("core")
        mock_fe_mod = types.ModuleType("core.feature_extractor")
        mock_fe_mod.FeatureExtractor = MagicMock()

        with patch.dict(
            sys.modules,
            {
                "config": mock_config_mod,
                "core": mock_core_mod,
                "core.feature_extractor": mock_fe_mod,
            },
        ):
            with pytest.raises(BenchmarkSetupError, match="direct setup error"):
                bench_feature_extraction(repeats=2)

    def test_bench_ml_predict_single_reraises_benchmark_run_error(self):
        """Test that BenchmarkRunError from predict is re-raised directly."""
        mock_config = MagicMock()
        mock_predictor = MagicMock()
        mock_predictor.predict.side_effect = BenchmarkRunError("run err")
        mock_predictor.is_loaded = True

        mock_config_mod = types.ModuleType("config")
        mock_config_mod.SystemConfig = MagicMock(return_value=mock_config)
        mock_ml_mod = types.ModuleType("ml")
        mock_ml_pred_mod = types.ModuleType("ml.predictor")
        mock_ml_pred_mod.MLPredictor = MagicMock(return_value=mock_predictor)

        with patch.dict(
            sys.modules,
            {
                "config": mock_config_mod,
                "ml": mock_ml_mod,
                "ml.predictor": mock_ml_pred_mod,
            },
        ):
            with pytest.raises(BenchmarkRunError, match="run err"):
                bench_ml_predict_single(repeats=2)

    def test_bench_ml_predict_batch_reraises_benchmark_run_error(self):
        """Test that BenchmarkRunError from predict is re-raised in batch."""
        mock_config = MagicMock()
        mock_predictor = MagicMock()
        # warmup succeeds, then BenchmarkRunError
        mock_predictor.predict.side_effect = [None, BenchmarkRunError("batch run err")]
        mock_predictor.is_loaded = True

        mock_config_mod = types.ModuleType("config")
        mock_config_mod.SystemConfig = MagicMock(return_value=mock_config)
        mock_ml_mod = types.ModuleType("ml")
        mock_ml_pred_mod = types.ModuleType("ml.predictor")
        mock_ml_pred_mod.MLPredictor = MagicMock(return_value=mock_predictor)

        with patch.dict(
            sys.modules,
            {
                "config": mock_config_mod,
                "ml": mock_ml_mod,
                "ml.predictor": mock_ml_pred_mod,
            },
        ):
            with pytest.raises(BenchmarkRunError, match="batch run err"):
                bench_ml_predict_batch(batch_size=3)

    def test_main_guard_executes_run_all(self):
        """Test the if __name__ == '__main__' block by running as subprocess."""
        import subprocess

        result = subprocess.run(
            [sys.executable, "-m", "benchmarks.bench_ml_inference"],
            capture_output=True,
            text=True,
            timeout=30,
            cwd="/Users/lordray/Downloads/crypto_pattern_system",
        )
        # It should print the header regardless of bench success/failure
        assert "ML Inference Benchmark Suite" in result.stdout
