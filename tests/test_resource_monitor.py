"""Tests for the resource monitoring module."""

import os
import threading
import time

import pytest

from monitoring.resource_monitor import (
    ResourceMonitor,
    get_peak_memory_mb,
    get_directory_size,
    get_file_count,
    get_file_size,
    _format_bytes,
)


class TestUtilityFunctions:
    """Test standalone utility functions."""

    def test_get_peak_memory_mb_returns_positive_float(self):
        """Peak memory should always be a positive float value."""
        peak = get_peak_memory_mb()
        assert isinstance(peak, float)
        assert peak > 0

    def test_get_directory_size_with_real_dir(self, tmpdir_path):
        """Directory size should be >= 0 for a real directory."""
        # Create some files with known sizes
        file1 = os.path.join(tmpdir_path, "file1.txt")
        file2 = os.path.join(tmpdir_path, "file2.txt")
        with open(file1, "w") as f:
            f.write("a" * 100)
        with open(file2, "w") as f:
            f.write("b" * 200)

        size = get_directory_size(tmpdir_path)
        assert isinstance(size, int)
        assert size >= 300  # At least 300 bytes from our files

    def test_get_directory_size_with_nonexistent_dir(self):
        """Nonexistent directory should return 0."""
        size = get_directory_size("/nonexistent/path/that/does/not/exist")
        assert size == 0

    def test_get_file_count_without_extension_filter(self, tmpdir_path):
        """File count should count all files when no extension filter is provided."""
        # Create files with different extensions
        open(os.path.join(tmpdir_path, "file1.txt"), "w").close()
        open(os.path.join(tmpdir_path, "file2.parquet"), "w").close()
        open(os.path.join(tmpdir_path, "file3.png"), "w").close()

        count = get_file_count(tmpdir_path)
        assert count == 3

    def test_get_file_count_with_extension_filter(self, tmpdir_path):
        """File count should filter by extension when provided."""
        open(os.path.join(tmpdir_path, "file1.txt"), "w").close()
        open(os.path.join(tmpdir_path, "file2.parquet"), "w").close()
        open(os.path.join(tmpdir_path, "file3.parquet"), "w").close()

        count = get_file_count(tmpdir_path, ".parquet")
        assert count == 2

    def test_get_file_count_with_subdirectories(self, tmpdir_path):
        """File count should recurse through subdirectories."""
        subdir = os.path.join(tmpdir_path, "subdir")
        os.makedirs(subdir)
        open(os.path.join(tmpdir_path, "file1.txt"), "w").close()
        open(os.path.join(subdir, "file2.txt"), "w").close()

        count = get_file_count(tmpdir_path, ".txt")
        assert count == 2

    def test_get_file_count_with_nonexistent_dir(self):
        """Nonexistent directory should return 0."""
        count = get_file_count("/nonexistent/path")
        assert count == 0

    def test_get_file_size_with_known_file(self, tmpdir_path):
        """File size should return correct size for an existing file."""
        file_path = os.path.join(tmpdir_path, "test.txt")
        content = "test content with known length"
        with open(file_path, "w") as f:
            f.write(content)

        size = get_file_size(file_path)
        assert size == len(content)

    def test_get_file_size_with_nonexistent_file(self):
        """Nonexistent file should return 0."""
        size = get_file_size("/nonexistent/file.txt")
        assert size == 0

    def test_format_bytes_in_bytes(self):
        """Bytes formatter should handle byte values correctly."""
        assert _format_bytes(512) == "512.0 B"

    def test_format_bytes_in_kilobytes(self):
        """Bytes formatter should handle kilobyte values correctly."""
        assert _format_bytes(1024) == "1.0 KB"
        assert _format_bytes(2048) == "2.0 KB"

    def test_format_bytes_in_megabytes(self):
        """Bytes formatter should handle megabyte values correctly."""
        assert _format_bytes(1024 * 1024) == "1.0 MB"
        assert _format_bytes(1536 * 1024) == "1.5 MB"

    def test_format_bytes_in_gigabytes(self):
        """Bytes formatter should handle gigabyte values correctly."""
        assert _format_bytes(1024 * 1024 * 1024) == "1.0 GB"
        assert _format_bytes(2048 * 1024 * 1024) == "2.0 GB"


class TestResourceMonitor:
    """Test the ResourceMonitor class."""

    def test_init_sets_data_root(self, tmpdir_path):
        """ResourceMonitor initialization should set data_root attribute."""
        monitor = ResourceMonitor(tmpdir_path)
        assert monitor.data_root == tmpdir_path

    def test_snapshot_returns_expected_structure(self, tmpdir_path):
        """Snapshot should return a dict with required keys."""
        monitor = ResourceMonitor(tmpdir_path)
        snap = monitor.snapshot()

        assert isinstance(snap, dict)
        assert "timestamp" in snap
        assert "memory" in snap
        assert "disk" in snap
        assert "threads" in snap

    def test_memory_info_returns_peak_rss_mb(self, tmpdir_path):
        """Memory info should include peak_rss_mb field."""
        monitor = ResourceMonitor(tmpdir_path)
        mem_info = monitor._memory_info()

        assert isinstance(mem_info, dict)
        assert "peak_rss_mb" in mem_info
        assert isinstance(mem_info["peak_rss_mb"], (int, float))
        assert mem_info["peak_rss_mb"] > 0

    def test_thread_info_returns_active_count_and_names(self, tmpdir_path):
        """Thread info should include active_count and thread_names."""
        monitor = ResourceMonitor(tmpdir_path)
        thread_info = monitor._thread_info()

        assert isinstance(thread_info, dict)
        assert "active_count" in thread_info
        assert "thread_names" in thread_info
        assert isinstance(thread_info["active_count"], int)
        assert isinstance(thread_info["thread_names"], list)
        assert thread_info["active_count"] > 0
        assert len(thread_info["thread_names"]) > 0

    def test_disk_info_structure(self, tmpdir_path):
        """Disk info should return proper structure with all categories."""
        # Create some test directories
        os.makedirs(os.path.join(tmpdir_path, "parquet"), exist_ok=True)
        os.makedirs(os.path.join(tmpdir_path, "images"), exist_ok=True)
        os.makedirs(os.path.join(tmpdir_path, "models"), exist_ok=True)
        os.makedirs(os.path.join(tmpdir_path, "logs"), exist_ok=True)

        monitor = ResourceMonitor(tmpdir_path)
        disk_info = monitor._disk_info()

        assert isinstance(disk_info, dict)
        assert "parquet" in disk_info
        assert "images" in disk_info
        assert "models" in disk_info
        assert "logs" in disk_info
        assert "sqlite" in disk_info
        assert "total" in disk_info

    def test_print_report_runs_without_error(self, tmpdir_path, capsys):
        """Print report should execute without raising errors."""
        monitor = ResourceMonitor(tmpdir_path)
        monitor.print_report()

        captured = capsys.readouterr()
        assert "RESOURCE MONITOR SNAPSHOT" in captured.out
        assert "Memory:" in captured.out
        assert "Peak RSS:" in captured.out
        assert "Disk Usage" in captured.out
        assert "Threads" in captured.out
