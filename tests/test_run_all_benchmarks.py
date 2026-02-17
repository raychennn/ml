"""Tests for benchmarks.run_all_benchmarks module.

This module tests the benchmark runner infrastructure, including baseline
loading, result evaluation, and status determination.
"""

from __future__ import annotations

import json
import os
from typing import Any
from unittest import mock

import pytest

from benchmarks.exceptions import BaselineLoadError
from benchmarks.run_all_benchmarks import (
    BenchmarkResult,
    _evaluate,
    _load_baselines,
    main,
)


class TestLoadBaselines:
    """Tests for _load_baselines function."""

    def test_load_baselines_returns_dict_with_expected_keys(self):
        """_load_baselines should return dict with expected benchmark keys."""
        baselines: dict[str, Any] = _load_baselines()

        # Check that baselines is a dict
        assert isinstance(baselines, dict)

        # Check for some expected keys from baselines.json
        expected_keys = [
            "dtw_single_pair",
            "dtw_batch_10",
            "prescreening_100_symbols",
            "prescreening_symbols_per_sec",
            "parquet_write_single",
            "ml_inference_single",
        ]

        for key in expected_keys:
            assert key in baselines, f"Expected key '{key}' not found in baselines"

        # Verify structure of a baseline entry
        dtw_baseline = baselines["dtw_single_pair"]
        assert "expected" in dtw_baseline
        assert "warn_multiplier" in dtw_baseline
        assert "fail_multiplier" in dtw_baseline
        assert isinstance(dtw_baseline["expected"], (int, float))
        assert isinstance(dtw_baseline["warn_multiplier"], (int, float))
        assert isinstance(dtw_baseline["fail_multiplier"], (int, float))

    def test_load_baselines_raises_error_if_file_missing(self, tmpdir_path: str):
        """_load_baselines should raise BaselineLoadError if file is missing."""
        # Mock the baselines path to point to a non-existent file
        fake_path = os.path.join(tmpdir_path, "nonexistent.json")

        with mock.patch(
            "benchmarks.run_all_benchmarks.os.path.join", return_value=fake_path
        ):
            with pytest.raises(BaselineLoadError, match="Failed to load baselines"):
                _load_baselines()

    def test_load_baselines_raises_error_on_invalid_json(self, tmpdir_path: str):
        """_load_baselines should raise BaselineLoadError on malformed JSON."""
        # Create a file with invalid JSON
        invalid_json_path = os.path.join(tmpdir_path, "invalid.json")
        with open(invalid_json_path, "w") as f:
            f.write("{ this is not valid json }")

        with mock.patch(
            "benchmarks.run_all_benchmarks.os.path.join", return_value=invalid_json_path
        ):
            with pytest.raises(BaselineLoadError, match="Failed to load baselines"):
                _load_baselines()

    def test_load_baselines_raises_error_on_missing_baselines_key(
        self, tmpdir_path: str
    ):
        """_load_baselines should raise BaselineLoadError if 'baselines' key missing."""
        # Create a valid JSON file but without the 'baselines' key
        missing_key_path = os.path.join(tmpdir_path, "missing_key.json")
        with open(missing_key_path, "w") as f:
            json.dump({"version": "1.0.0", "data": {}}, f)

        with mock.patch(
            "benchmarks.run_all_benchmarks.os.path.join", return_value=missing_key_path
        ):
            with pytest.raises(BaselineLoadError, match="Failed to load baselines"):
                _load_baselines()


class TestEvaluate:
    """Tests for _evaluate function."""

    def test_evaluate_pass_within_normal_range_latency(self):
        """_evaluate should return PASS when latency value is within normal range."""
        result: BenchmarkResult = {
            "name": "test_benchmark",
            "value": 0.01,
            "unit": "seconds",
            "detail": "test detail",
        }
        baselines: dict[str, Any] = {
            "test_benchmark": {
                "expected": 0.01,
                "warn_multiplier": 1.5,
                "fail_multiplier": 3.0,
                "higher_is_better": False,
            }
        }

        status, message = _evaluate(result, baselines)

        assert status == "PASS"
        assert "0.0100" in message
        assert "<=" in message

    def test_evaluate_warn_exceeds_warn_threshold(self):
        """_evaluate should return WARN when value exceeds warn but not fail threshold."""
        result: BenchmarkResult = {
            "name": "test_benchmark",
            "value": 0.02,  # 2x expected, between warn (1.5x) and fail (3x)
            "unit": "seconds",
            "detail": "test detail",
        }
        baselines: dict[str, Any] = {
            "test_benchmark": {
                "expected": 0.01,
                "warn_multiplier": 1.5,
                "fail_multiplier": 3.0,
                "higher_is_better": False,
            }
        }

        status, message = _evaluate(result, baselines)

        assert status == "WARN"
        assert "0.0200" in message
        assert ">" in message

    def test_evaluate_fail_exceeds_fail_threshold(self):
        """_evaluate should return FAIL when value exceeds fail threshold."""
        result: BenchmarkResult = {
            "name": "test_benchmark",
            "value": 0.04,  # 4x expected, exceeds fail threshold (3x)
            "unit": "seconds",
            "detail": "test detail",
        }
        baselines: dict[str, Any] = {
            "test_benchmark": {
                "expected": 0.01,
                "warn_multiplier": 1.5,
                "fail_multiplier": 3.0,
                "higher_is_better": False,
            }
        }

        status, message = _evaluate(result, baselines)

        assert status == "FAIL"
        assert "0.0400" in message
        assert ">" in message

    def test_evaluate_skip_when_benchmark_not_in_baselines(self):
        """_evaluate should return SKIP when benchmark name not in baselines."""
        result: BenchmarkResult = {
            "name": "unknown_benchmark",
            "value": 0.01,
            "unit": "seconds",
            "detail": "test detail",
        }
        baselines: dict[str, Any] = {
            "test_benchmark": {
                "expected": 0.01,
                "warn_multiplier": 1.5,
                "fail_multiplier": 3.0,
            }
        }

        status, message = _evaluate(result, baselines)

        assert status == "SKIP"
        assert "no baseline defined" in message

    def test_evaluate_error_when_value_is_none(self):
        """_evaluate should return ERROR when value is None."""
        result: BenchmarkResult = {
            "name": "test_benchmark",
            "value": None,
            "unit": "seconds",
            "detail": "test detail",
        }
        baselines: dict[str, Any] = {
            "test_benchmark": {
                "expected": 0.01,
                "warn_multiplier": 1.5,
                "fail_multiplier": 3.0,
            }
        }

        status, message = _evaluate(result, baselines)

        assert status == "ERROR"
        assert "no value" in message

    def test_evaluate_higher_is_better_throughput_pass(self):
        """_evaluate should handle higher_is_better=True correctly (PASS)."""
        result: BenchmarkResult = {
            "name": "throughput_test",
            "value": 250.0,  # Well above expected * warn_mult (200 * 0.5 = 100)
            "unit": "symbols/sec",
            "detail": "test detail",
        }
        baselines: dict[str, Any] = {
            "throughput_test": {
                "expected": 200,
                "warn_multiplier": 0.5,  # WARN if < 100
                "fail_multiplier": 0.2,  # FAIL if < 40
                "higher_is_better": True,
            }
        }

        status, message = _evaluate(result, baselines)

        assert status == "PASS"
        assert "250.0000" in message
        assert ">=" in message

    def test_evaluate_higher_is_better_throughput_warn(self):
        """_evaluate should return WARN for throughput below warn threshold."""
        result: BenchmarkResult = {
            "name": "throughput_test",
            "value": 80.0,  # Below warn threshold (100) but above fail (40)
            "unit": "symbols/sec",
            "detail": "test detail",
        }
        baselines: dict[str, Any] = {
            "throughput_test": {
                "expected": 200,
                "warn_multiplier": 0.5,  # WARN if < 100
                "fail_multiplier": 0.2,  # FAIL if < 40
                "higher_is_better": True,
            }
        }

        status, message = _evaluate(result, baselines)

        assert status == "WARN"
        assert "80.0000" in message
        assert "<" in message

    def test_evaluate_higher_is_better_throughput_fail(self):
        """_evaluate should return FAIL for throughput below fail threshold."""
        result: BenchmarkResult = {
            "name": "throughput_test",
            "value": 30.0,  # Below fail threshold (40)
            "unit": "symbols/sec",
            "detail": "test detail",
        }
        baselines: dict[str, Any] = {
            "throughput_test": {
                "expected": 200,
                "warn_multiplier": 0.5,  # WARN if < 100
                "fail_multiplier": 0.2,  # FAIL if < 40
                "higher_is_better": True,
            }
        }

        status, message = _evaluate(result, baselines)

        assert status == "FAIL"
        assert "30.0000" in message
        assert "<" in message


class TestMain:
    """Tests for main function integration scenarios."""

    def test_evaluate_logic_for_passing_benchmark(self):
        """Verify evaluation logic returns PASS for good results."""
        passing_result: BenchmarkResult = {
            "name": "dtw_single_pair",
            "value": 0.005,  # Well within expected
            "unit": "seconds",
            "detail": "test",
        }
        baselines: dict[str, Any] = {
            "dtw_single_pair": {
                "expected": 0.01,
                "warn_multiplier": 1.5,
                "fail_multiplier": 3.0,
                "higher_is_better": False,
            }
        }

        status, _ = _evaluate(passing_result, baselines)
        assert status == "PASS"

    def test_evaluate_logic_for_failing_benchmark(self):
        """Verify evaluation logic returns FAIL for bad results."""
        failing_result: BenchmarkResult = {
            "name": "dtw_single_pair",
            "value": 0.05,  # 5x expected, exceeds fail threshold
            "unit": "seconds",
            "detail": "test",
        }
        baselines: dict[str, Any] = {
            "dtw_single_pair": {
                "expected": 0.01,
                "warn_multiplier": 1.5,
                "fail_multiplier": 3.0,
                "higher_is_better": False,
            }
        }

        status, _ = _evaluate(failing_result, baselines)
        assert status == "FAIL"


# ── Helper constants for main() tests ──────────────────────────────

_FAKE_BASELINES: dict[str, Any] = {
    "dtw_single_pair": {
        "expected": 0.01,
        "warn_multiplier": 1.5,
        "fail_multiplier": 3.0,
        "higher_is_better": False,
    },
    "prescreening_100_symbols": {
        "expected": 0.5,
        "warn_multiplier": 2.0,
        "fail_multiplier": 5.0,
        "higher_is_better": False,
    },
}

_PASSING_DTW_RESULTS: list[BenchmarkResult] = [
    {"name": "dtw_single_pair", "value": 0.005, "unit": "seconds", "detail": "ok"},
]

_PASSING_PRESCREENING_RESULTS: list[BenchmarkResult] = [
    {
        "name": "prescreening_100_symbols",
        "value": 0.3,
        "unit": "seconds",
        "detail": "ok",
    },
]

_FAILING_DTW_RESULTS: list[BenchmarkResult] = [
    {"name": "dtw_single_pair", "value": 0.05, "unit": "seconds", "detail": "slow"},
]

_WARN_DTW_RESULTS: list[BenchmarkResult] = [
    {"name": "dtw_single_pair", "value": 0.02, "unit": "seconds", "detail": "warn"},
]

# Patch targets for the five benchmark suite imports inside main()
_DTW_PATCH = "benchmarks.bench_dtw"
_PRESCREENING_PATCH = "benchmarks.bench_prescreening"
_PARQUET_PATCH = "benchmarks.bench_parquet"
_ML_PATCH = "benchmarks.bench_ml_inference"
_SCAN_PATCH = "benchmarks.bench_scan_cycle"


def _make_bench_module(results: list[BenchmarkResult]):
    """Create a fake module object with a run_all() that returns *results*."""
    mod = mock.MagicMock()
    mod.run_all.return_value = results
    return mod


def _patch_all_imports_passing():
    """Return a stack of patches where every suite import succeeds with passing data."""
    dtw_mod = _make_bench_module(_PASSING_DTW_RESULTS)
    pre_mod = _make_bench_module(_PASSING_PRESCREENING_RESULTS)
    empty_mod = _make_bench_module([])

    original_import = (
        __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__
    )

    def _fake_import(name, *args, **kwargs):
        mapping = {
            _DTW_PATCH: dtw_mod,
            _PRESCREENING_PATCH: pre_mod,
            _PARQUET_PATCH: empty_mod,
            _ML_PATCH: empty_mod,
            _SCAN_PATCH: empty_mod,
        }
        if name in mapping:
            return mapping[name]
        return original_import(name, *args, **kwargs)

    return _fake_import, dtw_mod, pre_mod


class TestMainFunction:
    """Tests that exercise the main() function end-to-end with mocked benchmarks."""

    @mock.patch(
        "benchmarks.run_all_benchmarks._load_baselines", return_value=_FAKE_BASELINES
    )
    def test_main_returns_0_when_all_pass(self, mock_baselines, capsys):
        """main() should return 0 when every benchmark passes."""
        dtw_mod = _make_bench_module(_PASSING_DTW_RESULTS)
        pre_mod = _make_bench_module(_PASSING_PRESCREENING_RESULTS)
        empty_mod = _make_bench_module([])

        with mock.patch.dict(
            "sys.modules",
            {
                _DTW_PATCH: dtw_mod,
                _PRESCREENING_PATCH: pre_mod,
                _PARQUET_PATCH: empty_mod,
                _ML_PATCH: empty_mod,
                _SCAN_PATCH: empty_mod,
            },
        ):
            exit_code = main()

        assert exit_code == 0
        captured = capsys.readouterr()
        assert "OVERALL: PASS" in captured.out

    @mock.patch(
        "benchmarks.run_all_benchmarks._load_baselines", return_value=_FAKE_BASELINES
    )
    def test_main_returns_1_when_any_benchmark_fails(self, mock_baselines, capsys):
        """main() should return 1 when at least one benchmark FAILs."""
        dtw_mod = _make_bench_module(_FAILING_DTW_RESULTS)
        pre_mod = _make_bench_module(_PASSING_PRESCREENING_RESULTS)
        empty_mod = _make_bench_module([])

        with mock.patch.dict(
            "sys.modules",
            {
                _DTW_PATCH: dtw_mod,
                _PRESCREENING_PATCH: pre_mod,
                _PARQUET_PATCH: empty_mod,
                _ML_PATCH: empty_mod,
                _SCAN_PATCH: empty_mod,
            },
        ):
            exit_code = main()

        assert exit_code == 1
        captured = capsys.readouterr()
        assert "OVERALL: FAIL" in captured.out

    @mock.patch(
        "benchmarks.run_all_benchmarks._load_baselines", return_value=_FAKE_BASELINES
    )
    def test_main_prints_pass_warn_fail_status_labels(self, mock_baselines, capsys):
        """main() should print [PASS], [WARN], [FAIL] labels in the output."""
        # One PASS, one WARN, one FAIL
        mixed_results: list[BenchmarkResult] = [
            {"name": "dtw_single_pair", "value": 0.005, "unit": "s", "detail": ""},
            {
                "name": "prescreening_100_symbols",
                "value": 1.5,
                "unit": "s",
                "detail": "",
            },
        ]
        # With the baselines: dtw_single_pair 0.005 <= 0.015 -> PASS
        # prescreening: 1.5 > 1.0 (0.5*2.0) but <= 2.5 (0.5*5.0) -> WARN
        dtw_mod = _make_bench_module([mixed_results[0]])
        pre_mod = _make_bench_module([mixed_results[1]])
        empty_mod = _make_bench_module([])

        with mock.patch.dict(
            "sys.modules",
            {
                _DTW_PATCH: dtw_mod,
                _PRESCREENING_PATCH: pre_mod,
                _PARQUET_PATCH: empty_mod,
                _ML_PATCH: empty_mod,
                _SCAN_PATCH: empty_mod,
            },
        ):
            exit_code = main()

        captured = capsys.readouterr()
        assert "[PASS]" in captured.out
        assert "[WARN]" in captured.out
        # WARN overall still returns 0
        assert exit_code == 0
        assert "OVERALL: WARN" in captured.out

    @mock.patch(
        "benchmarks.run_all_benchmarks._load_baselines", return_value=_FAKE_BASELINES
    )
    def test_main_prints_fail_status(self, mock_baselines, capsys):
        """main() should print [FAIL] for benchmarks exceeding the fail threshold."""
        dtw_mod = _make_bench_module(_FAILING_DTW_RESULTS)
        empty_mod = _make_bench_module([])

        with mock.patch.dict(
            "sys.modules",
            {
                _DTW_PATCH: dtw_mod,
                _PRESCREENING_PATCH: empty_mod,
                _PARQUET_PATCH: empty_mod,
                _ML_PATCH: empty_mod,
                _SCAN_PATCH: empty_mod,
            },
        ):
            exit_code = main()

        captured = capsys.readouterr()
        assert "[FAIL]" in captured.out
        assert exit_code == 1

    @mock.patch("benchmarks.run_all_benchmarks._load_baselines")
    def test_main_handles_missing_baselines_gracefully(self, mock_baselines, capsys):
        """main() should propagate BaselineLoadError when baselines file is missing."""
        mock_baselines.side_effect = BaselineLoadError("baselines.json not found")

        with pytest.raises(BaselineLoadError, match="baselines.json not found"):
            main()

    @mock.patch(
        "benchmarks.run_all_benchmarks._load_baselines", return_value=_FAKE_BASELINES
    )
    def test_main_handles_benchmark_exception_gracefully(self, mock_baselines, capsys):
        """main() should catch exceptions from individual benchmark suites and continue."""
        # DTW suite raises an exception; prescreening still runs fine
        dtw_mod = _make_bench_module([])
        dtw_mod.run_all.side_effect = RuntimeError("DTW computation exploded")
        pre_mod = _make_bench_module(_PASSING_PRESCREENING_RESULTS)
        empty_mod = _make_bench_module([])

        with mock.patch.dict(
            "sys.modules",
            {
                _DTW_PATCH: dtw_mod,
                _PRESCREENING_PATCH: pre_mod,
                _PARQUET_PATCH: empty_mod,
                _ML_PATCH: empty_mod,
                _SCAN_PATCH: empty_mod,
            },
        ):
            exit_code = main()

        captured = capsys.readouterr()
        assert "ERROR" in captured.out
        assert "DTW computation exploded" in captured.out
        # Prescreening result should still be evaluated
        assert "[PASS]" in captured.out
        assert exit_code == 0

    @mock.patch(
        "benchmarks.run_all_benchmarks._load_baselines", return_value=_FAKE_BASELINES
    )
    def test_main_handles_import_error_for_missing_suite(self, mock_baselines, capsys):
        """main() should skip suites that fail to import and print [SKIP]."""
        # Remove bench_dtw from sys.modules so the import fails
        pre_mod = _make_bench_module(_PASSING_PRESCREENING_RESULTS)
        empty_mod = _make_bench_module([])

        # Patch only some modules; leave DTW unpatched so import fails
        with mock.patch.dict(
            "sys.modules",
            {
                _PRESCREENING_PATCH: pre_mod,
                _PARQUET_PATCH: empty_mod,
                _ML_PATCH: empty_mod,
                _SCAN_PATCH: empty_mod,
            },
        ):
            # Force DTW import to fail by removing it if present and blocking it
            with mock.patch.dict("sys.modules", {_DTW_PATCH: None}):
                exit_code = main()

        captured = capsys.readouterr()
        assert "[SKIP] bench_dtw" in captured.out
        assert exit_code == 0

    @mock.patch(
        "benchmarks.run_all_benchmarks._load_baselines", return_value=_FAKE_BASELINES
    )
    def test_main_prints_suite_durations(self, mock_baselines, capsys):
        """main() should print duration for each benchmark suite."""
        dtw_mod = _make_bench_module(_PASSING_DTW_RESULTS)
        pre_mod = _make_bench_module(_PASSING_PRESCREENING_RESULTS)
        empty_mod = _make_bench_module([])

        with mock.patch.dict(
            "sys.modules",
            {
                _DTW_PATCH: dtw_mod,
                _PRESCREENING_PATCH: pre_mod,
                _PARQUET_PATCH: empty_mod,
                _ML_PATCH: empty_mod,
                _SCAN_PATCH: empty_mod,
            },
        ):
            exit_code = main()

        captured = capsys.readouterr()
        assert "Suite durations:" in captured.out
        assert "DTW" in captured.out

    @mock.patch(
        "benchmarks.run_all_benchmarks._load_baselines", return_value=_FAKE_BASELINES
    )
    def test_main_prints_summary_counts(self, mock_baselines, capsys):
        """main() should print Total / PASS / WARN / FAIL / SKIP / ERROR counts."""
        dtw_mod = _make_bench_module(_PASSING_DTW_RESULTS)
        empty_mod = _make_bench_module([])

        with mock.patch.dict(
            "sys.modules",
            {
                _DTW_PATCH: dtw_mod,
                _PRESCREENING_PATCH: empty_mod,
                _PARQUET_PATCH: empty_mod,
                _ML_PATCH: empty_mod,
                _SCAN_PATCH: empty_mod,
            },
        ):
            main()

        captured = capsys.readouterr()
        assert "Total: 1" in captured.out
        assert "PASS: 1" in captured.out

    @mock.patch("benchmarks.run_all_benchmarks._load_baselines", return_value={})
    def test_main_skip_status_for_results_with_no_baseline(
        self, mock_baselines, capsys
    ):
        """main() should show [SKIP] for benchmarks without a matching baseline entry."""
        dtw_mod = _make_bench_module(_PASSING_DTW_RESULTS)
        empty_mod = _make_bench_module([])

        with mock.patch.dict(
            "sys.modules",
            {
                _DTW_PATCH: dtw_mod,
                _PRESCREENING_PATCH: empty_mod,
                _PARQUET_PATCH: empty_mod,
                _ML_PATCH: empty_mod,
                _SCAN_PATCH: empty_mod,
            },
        ):
            exit_code = main()

        captured = capsys.readouterr()
        assert "[SKIP]" in captured.out
        assert "no baseline defined" in captured.out
        assert exit_code == 0

    @mock.patch(
        "benchmarks.run_all_benchmarks._load_baselines", return_value=_FAKE_BASELINES
    )
    def test_main_error_status_for_none_value(self, mock_baselines, capsys):
        """main() should show [ERR ] for benchmark results with value=None."""
        dtw_mod = _make_bench_module(
            [
                {
                    "name": "dtw_single_pair",
                    "value": None,
                    "unit": "s",
                    "detail": "broken",
                },
            ]
        )
        empty_mod = _make_bench_module([])

        with mock.patch.dict(
            "sys.modules",
            {
                _DTW_PATCH: dtw_mod,
                _PRESCREENING_PATCH: empty_mod,
                _PARQUET_PATCH: empty_mod,
                _ML_PATCH: empty_mod,
                _SCAN_PATCH: empty_mod,
            },
        ):
            exit_code = main()

        captured = capsys.readouterr()
        assert "[ERR ]" in captured.out
        assert "ERROR: 1" in captured.out
        assert exit_code == 0
