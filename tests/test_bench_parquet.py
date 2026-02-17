"""Tests for benchmarks.bench_parquet module."""

from __future__ import annotations

import pytest
import pandas as pd

from benchmarks.bench_parquet import (
    BenchmarkResult,
    _make_synthetic_ohlcv,
    bench_write_single,
    bench_read_single,
    bench_bulk_read,
    bench_write_compression_comparison,
    run_all,
)


class TestMakeSyntheticOHLCV:
    """Tests for _make_synthetic_ohlcv function."""

    def test_correct_columns(self):
        """Test that generated DataFrame has correct columns."""
        df = _make_synthetic_ohlcv(n_rows=100, seed=42)
        expected_columns = ["timestamp", "open", "high", "low", "close", "volume"]
        assert list(df.columns) == expected_columns

    def test_correct_row_count(self):
        """Test that generated DataFrame has correct number of rows."""
        n_rows = 100
        df = _make_synthetic_ohlcv(n_rows=n_rows, seed=42)
        assert len(df) == n_rows

    def test_column_types(self):
        """Test that columns have expected data types."""
        df = _make_synthetic_ohlcv(n_rows=100, seed=42)
        assert df["timestamp"].dtype == "int64"
        assert df["open"].dtype == "float64"
        assert df["high"].dtype == "float64"
        assert df["low"].dtype == "float64"
        assert df["close"].dtype == "float64"
        assert df["volume"].dtype == "float64"

    def test_reproducibility_with_seed(self):
        """Test that same seed produces same data."""
        df1 = _make_synthetic_ohlcv(n_rows=100, seed=42)
        df2 = _make_synthetic_ohlcv(n_rows=100, seed=42)
        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seed_produces_different_data(self):
        """Test that different seeds produce different data."""
        df1 = _make_synthetic_ohlcv(n_rows=100, seed=42)
        df2 = _make_synthetic_ohlcv(n_rows=100, seed=99)
        # Timestamps should be same but other values should differ
        pd.testing.assert_series_equal(df1["timestamp"], df2["timestamp"])
        with pytest.raises(AssertionError):
            pd.testing.assert_series_equal(df1["close"], df2["close"])


class TestBenchWriteSingle:
    """Tests for bench_write_single function."""

    def test_returns_valid_benchmark_result(self):
        """Test that function returns a valid BenchmarkResult dict."""
        result = bench_write_single(n_rows=100, repeats=2)

        assert isinstance(result, dict)
        assert "name" in result
        assert "value" in result
        assert "unit" in result
        assert "detail" in result

    def test_positive_value(self):
        """Test that benchmark value is positive."""
        result = bench_write_single(n_rows=100, repeats=2)
        assert result["value"] is not None
        assert result["value"] > 0

    def test_correct_name(self):
        """Test that benchmark has correct name."""
        result = bench_write_single(n_rows=100, repeats=2)
        assert result["name"] == "parquet_write_single"

    def test_correct_unit(self):
        """Test that benchmark uses correct unit."""
        result = bench_write_single(n_rows=100, repeats=2)
        assert result["unit"] == "seconds"

    def test_detail_contains_parameters(self):
        """Test that detail string contains parameter information."""
        result = bench_write_single(n_rows=100, repeats=2)
        assert "rows=100" in result["detail"]
        assert "repeats=2" in result["detail"]


class TestBenchReadSingle:
    """Tests for bench_read_single function."""

    def test_returns_valid_benchmark_result(self):
        """Test that function returns a valid BenchmarkResult dict."""
        result = bench_read_single(n_rows=100, repeats=2)

        assert isinstance(result, dict)
        assert "name" in result
        assert "value" in result
        assert "unit" in result
        assert "detail" in result

    def test_positive_value(self):
        """Test that benchmark value is positive."""
        result = bench_read_single(n_rows=100, repeats=2)
        assert result["value"] is not None
        assert result["value"] > 0

    def test_correct_name(self):
        """Test that benchmark has correct name."""
        result = bench_read_single(n_rows=100, repeats=2)
        assert result["name"] == "parquet_read_single"

    def test_correct_unit(self):
        """Test that benchmark uses correct unit."""
        result = bench_read_single(n_rows=100, repeats=2)
        assert result["unit"] == "seconds"

    def test_detail_contains_parameters(self):
        """Test that detail string contains parameter information."""
        result = bench_read_single(n_rows=100, repeats=2)
        assert "rows=100" in result["detail"]
        assert "repeats=2" in result["detail"]


class TestBenchBulkRead:
    """Tests for bench_bulk_read function."""

    def test_returns_valid_benchmark_result(self):
        """Test that function returns a valid BenchmarkResult dict."""
        result = bench_bulk_read(n_files=3, n_rows=100)

        assert isinstance(result, dict)
        assert "name" in result
        assert "value" in result
        assert "unit" in result
        assert "detail" in result

    def test_positive_value(self):
        """Test that benchmark value is positive."""
        result = bench_bulk_read(n_files=3, n_rows=100)
        assert result["value"] is not None
        assert result["value"] > 0

    def test_correct_name(self):
        """Test that benchmark has correct name."""
        result = bench_bulk_read(n_files=3, n_rows=100)
        assert result["name"] == "parquet_bulk_read_50"

    def test_correct_unit(self):
        """Test that benchmark uses correct unit."""
        result = bench_bulk_read(n_files=3, n_rows=100)
        assert result["unit"] == "seconds"

    def test_detail_contains_parameters(self):
        """Test that detail string contains parameter information."""
        result = bench_bulk_read(n_files=3, n_rows=100)
        assert "files=3" in result["detail"]
        assert "rows_each=100" in result["detail"]


class TestBenchWriteCompressionComparison:
    """Tests for bench_write_compression_comparison function."""

    def test_returns_list(self):
        """Test that function returns a list."""
        results = bench_write_compression_comparison(n_rows=100)
        assert isinstance(results, list)

    def test_returns_three_results(self):
        """Test that function returns 3 results for snappy, gzip, and none."""
        results = bench_write_compression_comparison(n_rows=100)
        assert len(results) == 3

    def test_all_results_are_valid(self):
        """Test that all results are valid BenchmarkResult dicts."""
        results = bench_write_compression_comparison(n_rows=100)

        for result in results:
            assert isinstance(result, dict)
            assert "name" in result
            assert "value" in result
            assert "unit" in result
            assert "detail" in result

    def test_all_values_are_positive(self):
        """Test that all benchmark values are positive."""
        results = bench_write_compression_comparison(n_rows=100)

        for result in results:
            assert result["value"] is not None
            assert result["value"] > 0

    def test_compression_names(self):
        """Test that results have correct names for each compression type."""
        results = bench_write_compression_comparison(n_rows=100)
        names = [r["name"] for r in results]

        assert "parquet_compression_snappy" in names
        assert "parquet_compression_gzip" in names
        assert "parquet_compression_none" in names

    def test_all_use_seconds_unit(self):
        """Test that all results use seconds as unit."""
        results = bench_write_compression_comparison(n_rows=100)

        for result in results:
            assert result["unit"] == "seconds"

    def test_details_contain_metrics(self):
        """Test that detail strings contain write, read, and size information."""
        results = bench_write_compression_comparison(n_rows=100)

        for result in results:
            detail = result["detail"]
            assert "write=" in detail
            assert "read=" in detail
            assert "size=" in detail
            assert "bytes" in detail


class TestRunAll:
    """Tests for run_all function."""

    def test_returns_list(self):
        """Test that run_all returns a list."""
        results = run_all()
        assert isinstance(results, list)

    def test_returns_non_empty_list(self):
        """Test that run_all returns a non-empty list."""
        results = run_all()
        assert len(results) > 0

    def test_all_results_are_valid(self):
        """Test that all results are valid BenchmarkResult dicts."""
        results = run_all()

        for result in results:
            assert isinstance(result, dict)
            assert "name" in result
            assert "value" in result
            assert "unit" in result
            assert "detail" in result

    def test_includes_write_single(self):
        """Test that results include write_single benchmark."""
        results = run_all()
        names = [r["name"] for r in results]
        assert "parquet_write_single" in names

    def test_includes_read_single(self):
        """Test that results include read_single benchmark."""
        results = run_all()
        names = [r["name"] for r in results]
        assert "parquet_read_single" in names

    def test_includes_bulk_read(self):
        """Test that results include bulk_read benchmark."""
        results = run_all()
        names = [r["name"] for r in results]
        assert "parquet_bulk_read_50" in names

    def test_includes_compression_benchmarks(self):
        """Test that results include all compression benchmarks."""
        results = run_all()
        names = [r["name"] for r in results]

        assert "parquet_compression_snappy" in names
        assert "parquet_compression_gzip" in names
        assert "parquet_compression_none" in names

    def test_expected_number_of_results(self):
        """Test that run_all returns expected number of results (6 total)."""
        results = run_all()
        # 3 individual benchmarks + 3 compression benchmarks = 6 total
        assert len(results) == 6
