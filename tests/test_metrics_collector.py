"""Tests for the metrics collection module."""

import json
import os
import threading
import time
from unittest.mock import patch, mock_open

import pytest

from monitoring.metrics_collector import MetricsCollector
from benchmarks.exceptions import MetricsWriteError, MetricsReadError


class TestMetricsCollectorInit:
    """Test MetricsCollector initialization."""

    def test_init_creates_parent_directory(self, tmpdir_path):
        """Initialization should create parent directory if it doesn't exist."""
        metrics_path = os.path.join(tmpdir_path, "logs", "metrics.jsonl")
        assert not os.path.exists(os.path.dirname(metrics_path))

        collector = MetricsCollector(metrics_path)

        assert os.path.exists(os.path.dirname(metrics_path))
        assert collector.metrics_path == metrics_path

    def test_init_with_default_path(self):
        """Initialization with no path should use default."""
        collector = MetricsCollector()
        expected = os.path.join("data", "logs", "metrics.jsonl")
        assert collector.metrics_path == expected


class TestRecordMethods:
    """Test various record_* methods."""

    def test_record_scan_start(self, metrics_path):
        """record_scan_start should write event with correct fields."""
        collector = MetricsCollector(metrics_path)
        collector.record_scan_start(timeframes=["4h", "1d"], run_id=123)

        events = collector.read_events()
        assert len(events) == 1
        event = events[0]
        assert event["event"] == "scan_start"
        assert event["timeframes"] == ["4h", "1d"]
        assert event["run_id"] == 123
        assert "ts" in event

    def test_record_scan_end(self, metrics_path):
        """record_scan_end should write event with duration, alerts, and status."""
        collector = MetricsCollector(metrics_path)
        collector.record_scan_end(
            timeframes=["4h"],
            duration_s=12.345,
            total_alerts=5,
            run_id=456,
            status="completed",
        )

        events = collector.read_events()
        assert len(events) == 1
        event = events[0]
        assert event["event"] == "scan_end"
        assert event["timeframes"] == ["4h"]
        assert event["duration_s"] == 12.345
        assert event["total_alerts"] == 5
        assert event["run_id"] == 456
        assert event["status"] == "completed"
        assert "ts" in event

    def test_record_scan_timeframe(self, metrics_path):
        """record_scan_timeframe should write event with timeframe details."""
        collector = MetricsCollector(metrics_path)
        collector.record_scan_timeframe(
            timeframe="4h", duration_s=5.678, symbols_scanned=100, alerts=3
        )

        events = collector.read_events()
        assert len(events) == 1
        event = events[0]
        assert event["event"] == "scan_timeframe"
        assert event["timeframe"] == "4h"
        assert event["duration_s"] == 5.678
        assert event["symbols_scanned"] == 100
        assert event["alerts"] == 3
        assert "ts" in event

    def test_record_api_request(self, metrics_path):
        """record_api_request should write event with endpoint and status."""
        collector = MetricsCollector(metrics_path)
        collector.record_api_request(
            endpoint="/api/scan", status_code=200, duration_ms=123.4
        )

        events = collector.read_events()
        assert len(events) == 1
        event = events[0]
        assert event["event"] == "api_request"
        assert event["endpoint"] == "/api/scan"
        assert event["status_code"] == 200
        assert event["duration_ms"] == 123.4
        assert "ts" in event

    def test_record_api_error(self, metrics_path):
        """record_api_error should write event with error details."""
        collector = MetricsCollector(metrics_path)
        collector.record_api_error(
            endpoint="/api/scan", error="Connection timeout", status_code=500
        )

        events = collector.read_events()
        assert len(events) == 1
        event = events[0]
        assert event["event"] == "api_error"
        assert event["endpoint"] == "/api/scan"
        assert event["error"] == "Connection timeout"
        assert event["status_code"] == 500
        assert "ts" in event

    def test_record_resource_snapshot(self, metrics_path):
        """record_resource_snapshot should write event with snapshot data."""
        collector = MetricsCollector(metrics_path)
        snapshot = {
            "timestamp": "2026-02-16T12:00:00Z",
            "memory": {"peak_rss_mb": 128.5},
            "disk": {"total": 1024},
            "threads": {"active_count": 5},
        }
        collector.record_resource_snapshot(snapshot)

        events = collector.read_events()
        assert len(events) == 1
        event = events[0]
        assert event["event"] == "resource_snapshot"
        assert event["data"] == snapshot
        assert "ts" in event

    def test_record_event_arbitrary(self, metrics_path):
        """record_event should write arbitrary named events."""
        collector = MetricsCollector(metrics_path)
        collector.record_event("custom_event", foo="bar", count=42)

        events = collector.read_events()
        assert len(events) == 1
        event = events[0]
        assert event["event"] == "custom_event"
        assert event["foo"] == "bar"
        assert event["count"] == 42
        assert "ts" in event


class TestReadMethods:
    """Test read and query methods."""

    def test_read_events_returns_all_when_no_filter(self, metrics_path):
        """read_events should return all events when no filter is specified."""
        collector = MetricsCollector(metrics_path)
        collector.record_event("event1", data=1)
        collector.record_event("event2", data=2)
        collector.record_event("event3", data=3)

        events = collector.read_events(limit=100)
        assert len(events) == 3
        assert events[0]["event"] == "event1"
        assert events[1]["event"] == "event2"
        assert events[2]["event"] == "event3"

    def test_read_events_filters_by_event_type(self, metrics_path):
        """read_events should filter by event_type when specified."""
        collector = MetricsCollector(metrics_path)
        collector.record_scan_start(timeframes=["4h"], run_id=1)
        collector.record_api_request(endpoint="/test", status_code=200)
        collector.record_scan_start(timeframes=["1d"], run_id=2)

        events = collector.read_events(event_type="scan_start", limit=100)
        assert len(events) == 2
        assert all(e["event"] == "scan_start" for e in events)

    def test_read_events_respects_limit(self, metrics_path):
        """read_events should respect the limit parameter."""
        collector = MetricsCollector(metrics_path)
        for i in range(10):
            collector.record_event(f"event{i}", index=i)

        events = collector.read_events(limit=3)
        assert len(events) == 3
        # Should get the last 3 events
        assert events[0]["index"] == 7
        assert events[1]["index"] == 8
        assert events[2]["index"] == 9

    def test_read_events_returns_empty_for_nonexistent_file(self, tmpdir_path):
        """read_events should return empty list when file doesn't exist."""
        nonexistent_path = os.path.join(tmpdir_path, "nonexistent.jsonl")
        collector = MetricsCollector(nonexistent_path)

        events = collector.read_events()
        assert events == []

    def test_get_scan_stats_returns_correct_aggregates(self, metrics_path):
        """get_scan_stats should return correct aggregated statistics."""
        collector = MetricsCollector(metrics_path)
        collector.record_scan_end(["4h"], duration_s=10.0, total_alerts=5)
        collector.record_scan_end(["1d"], duration_s=20.0, total_alerts=15)
        collector.record_scan_end(["1h"], duration_s=5.0, total_alerts=10)

        stats = collector.get_scan_stats(last_n=50)

        assert stats["count"] == 3
        assert stats["avg_duration_s"] == 11.67  # (10 + 20 + 5) / 3
        assert stats["max_duration_s"] == 20.0
        assert stats["total_alerts"] == 30
        assert stats["avg_alerts_per_scan"] == 10.0

    def test_get_scan_stats_returns_count_zero_for_empty(self, metrics_path):
        """get_scan_stats should return count=0 when no scan events exist."""
        collector = MetricsCollector(metrics_path)

        stats = collector.get_scan_stats()
        assert stats == {"count": 0}


class TestThreadSafety:
    """Test thread-safety of concurrent operations."""

    def test_concurrent_writes_dont_corrupt_file(self, metrics_path):
        """Concurrent writes from multiple threads should not corrupt the file."""
        collector = MetricsCollector(metrics_path)
        num_threads = 10
        events_per_thread = 20

        def write_events(thread_id):
            for i in range(events_per_thread):
                collector.record_event(
                    "concurrent_test", thread_id=thread_id, event_num=i
                )

        threads = []
        for tid in range(num_threads):
            t = threading.Thread(target=write_events, args=(tid,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # Verify all events were written
        events = collector.read_events(limit=1000)
        assert len(events) == num_threads * events_per_thread

        # Verify no JSON corruption
        with open(metrics_path, "r") as f:
            lines = [line.strip() for line in f if line.strip()]
            assert len(lines) == num_threads * events_per_thread
            for line in lines:
                # Should be valid JSON
                parsed = json.loads(line)
                assert "event" in parsed
                assert "ts" in parsed


class TestErrorHandling:
    """Test error handling scenarios."""

    def test_write_error_raises_metrics_write_error(self, tmpdir_path):
        """Writing to a read-only location should raise MetricsWriteError."""
        metrics_path = os.path.join(tmpdir_path, "metrics.jsonl")
        collector = MetricsCollector(metrics_path)

        # Mock open to raise OSError
        with patch("builtins.open", side_effect=OSError("Permission denied")):
            with pytest.raises(MetricsWriteError) as exc_info:
                collector.record_event("test")

            assert "Failed to write metric event" in str(exc_info.value)
            assert metrics_path in str(exc_info.value)

    def test_read_error_raises_metrics_read_error(self, metrics_path):
        """Reading a corrupted file should raise MetricsReadError."""
        # Create the file first
        collector = MetricsCollector(metrics_path)
        collector.record_event("test")

        # Mock open to raise OSError during read
        with patch("builtins.open", side_effect=OSError("Read error")):
            with pytest.raises(MetricsReadError) as exc_info:
                collector.read_events()

            assert "Failed to read metrics" in str(exc_info.value)
            assert metrics_path in str(exc_info.value)

    def test_read_events_skips_invalid_json_lines(self, metrics_path):
        """read_events should skip lines that are not valid JSON."""
        # Write some valid and invalid JSON lines directly
        with open(metrics_path, "w") as f:
            f.write('{"event": "valid1", "ts": "2026-02-16T12:00:00Z"}\n')
            f.write("invalid json line\n")
            f.write('{"event": "valid2", "ts": "2026-02-16T12:01:00Z"}\n')
            f.write('{"incomplete": \n')
            f.write('{"event": "valid3", "ts": "2026-02-16T12:02:00Z"}\n')

        collector = MetricsCollector(metrics_path)
        events = collector.read_events(limit=100)

        # Should only get the 3 valid events
        assert len(events) == 3
        assert events[0]["event"] == "valid1"
        assert events[1]["event"] == "valid2"
        assert events[2]["event"] == "valid3"
