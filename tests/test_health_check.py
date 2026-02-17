"""Tests for monitoring.health_check module.

Tests the health check blueprint endpoints:
- /detailed - comprehensive system health information
- /resources - disk and memory resource snapshot
- /metrics - scan metrics and recent errors
"""

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, Mock, patch

import pytest
from flask import Flask

from monitoring.health_check import health_bp


@pytest.fixture
def app():
    """Create a minimal Flask app with health_bp registered."""
    test_app = Flask(__name__)
    test_app.config["TESTING"] = True
    test_app.register_blueprint(health_bp, url_prefix="/api/health")
    return test_app


@pytest.fixture
def client(app):
    """Create a test client for the Flask app."""
    return app.test_client()


class TestDetailedHealthEndpoint:
    """Test suite for GET /api/health/detailed endpoint."""

    def test_detailed_returns_200_status_code(self, client):
        """Test that /detailed returns HTTP 200."""
        response = client.get("/api/health/detailed")
        assert response.status_code == 200

    def test_detailed_returns_json_with_status_field(self, client):
        """Test that /detailed returns JSON with a 'status' field."""
        response = client.get("/api/health/detailed")
        data = response.get_json()
        assert "status" in data
        assert data["status"] == "ok"

    def test_detailed_includes_timestamp_field(self, client):
        """Test that /detailed includes a timestamp field."""
        response = client.get("/api/health/detailed")
        data = response.get_json()
        assert "timestamp" in data
        # Verify it's a valid ISO format timestamp
        timestamp_str = data["timestamp"]
        parsed_ts = datetime.fromisoformat(timestamp_str)
        assert parsed_ts is not None

    def test_detailed_includes_checks_dict(self, client):
        """Test that /detailed includes a checks dictionary."""
        response = client.get("/api/health/detailed")
        data = response.get_json()
        assert "checks" in data
        assert isinstance(data["checks"], dict)

    def test_detailed_has_unavailable_checks_when_no_config(self, client):
        """Test that /detailed shows unavailable status when no config is set."""
        response = client.get("/api/health/detailed")
        data = response.get_json()
        checks = data["checks"]

        # Database should be unavailable without RESULT_STORE
        assert "database" in checks
        assert checks["database"]["status"] == "unavailable"

        # ML models should be unavailable without SCANNER_ENGINE
        assert "ml_models" in checks
        assert checks["ml_models"]["status"] == "unavailable"

    def test_detailed_includes_thread_count(self, client):
        """Test that /detailed includes thread count information."""
        response = client.get("/api/health/detailed")
        data = response.get_json()
        checks = data["checks"]

        assert "thread_count" in checks
        assert "active" in checks["thread_count"]
        assert "names" in checks["thread_count"]
        assert isinstance(checks["thread_count"]["active"], int)
        assert isinstance(checks["thread_count"]["names"], list)
        assert checks["thread_count"]["active"] > 0

    def test_detailed_with_mock_store_shows_database_status(self, app, client):
        """Test that /detailed shows database status when RESULT_STORE is set."""
        # Create a mock result store
        mock_store = MagicMock()
        mock_store.get_total_results.return_value = 42
        mock_store.get_last_scan_time.return_value = "2026-02-16T12:00:00Z"

        # Set the mock store in app config
        app.config["RESULT_STORE"] = mock_store

        response = client.get("/api/health/detailed")
        data = response.get_json()
        checks = data["checks"]

        assert checks["database"]["status"] == "ok"
        assert checks["database"]["total_results"] == 42
        assert checks["database"]["last_scan_time"] == "2026-02-16T12:00:00Z"
        mock_store.get_total_results.assert_called_once()
        mock_store.get_last_scan_time.assert_called_once()

    def test_detailed_with_store_error_shows_degraded(self, app, client):
        """Test that /detailed shows degraded status when store raises error."""
        # Create a mock that raises an exception
        mock_store = MagicMock()
        mock_store.get_total_results.side_effect = Exception(
            "Database connection failed"
        )

        app.config["RESULT_STORE"] = mock_store

        response = client.get("/api/health/detailed")
        data = response.get_json()

        assert data["status"] == "degraded"
        assert data["checks"]["database"]["status"] == "error"
        assert "Database connection failed" in data["checks"]["database"]["error"]

    def test_detailed_with_mock_config_shows_data_dir_status(
        self, app, client, tmpdir_path
    ):
        """Test that /detailed shows data_dir status when SYSTEM_CONFIG is set."""
        # Create a mock config with an existing directory
        mock_config = MagicMock()
        mock_config.data_root = tmpdir_path

        app.config["SYSTEM_CONFIG"] = mock_config

        response = client.get("/api/health/detailed")
        data = response.get_json()
        checks = data["checks"]

        assert "data_dir" in checks
        assert checks["data_dir"]["status"] == "ok"
        assert checks["data_dir"]["path"] == tmpdir_path

    def test_detailed_with_missing_data_dir_shows_degraded(self, app, client):
        """Test that /detailed shows degraded status when data_dir doesn't exist."""
        # Create a mock config with a non-existent directory
        mock_config = MagicMock()
        mock_config.data_root = "/nonexistent/path/to/data"

        app.config["SYSTEM_CONFIG"] = mock_config

        response = client.get("/api/health/detailed")
        data = response.get_json()

        assert data["status"] == "degraded"
        assert data["checks"]["data_dir"]["status"] == "error"
        assert data["checks"]["data_dir"]["path"] == "/nonexistent/path/to/data"

    def test_detailed_with_ml_models_loaded(self, app, client):
        """Test that /detailed shows ML model status when scanner is available."""
        # Create a mock scanner with loaded models
        mock_predictor = MagicMock()
        mock_predictor.is_loaded = True
        mock_predictor.models = ["model1", "model2", "model3"]

        mock_scanner = MagicMock()
        mock_scanner.predictor = mock_predictor

        app.config["SCANNER_ENGINE"] = mock_scanner

        response = client.get("/api/health/detailed")
        data = response.get_json()
        checks = data["checks"]

        assert checks["ml_models"]["status"] == "ok"
        assert checks["ml_models"]["count"] == 3

    def test_detailed_with_ml_models_not_loaded(self, app, client):
        """Test that /detailed shows fallback status when ML models not loaded."""
        # Create a mock scanner with no loaded models
        mock_predictor = MagicMock()
        mock_predictor.is_loaded = False
        mock_predictor.models = []

        mock_scanner = MagicMock()
        mock_scanner.predictor = mock_predictor

        app.config["SCANNER_ENGINE"] = mock_scanner

        response = client.get("/api/health/detailed")
        data = response.get_json()
        checks = data["checks"]

        assert checks["ml_models"]["status"] == "fallback"
        assert checks["ml_models"]["count"] == 0

    def test_detailed_with_watchdog_all_alive(self, app, client):
        """Test that /detailed shows threads OK when all threads are alive."""
        # Create a mock watchdog with all threads alive
        mock_watchdog = MagicMock()
        mock_watchdog.status.return_value = {
            "worker-1": {"alive": True, "last_seen": "2026-02-16T12:00:00Z"},
            "worker-2": {"alive": True, "last_seen": "2026-02-16T12:00:01Z"},
        }

        app.config["WATCHDOG"] = mock_watchdog

        response = client.get("/api/health/detailed")
        data = response.get_json()
        checks = data["checks"]

        assert checks["threads"]["status"] == "ok"
        assert checks["threads"]["monitored"] == 2
        assert checks["threads"]["dead"] == []
        assert len(checks["threads"]["details"]) == 2

    def test_detailed_with_watchdog_dead_threads(self, app, client):
        """Test that /detailed shows degraded when threads are dead."""
        # Create a mock watchdog with dead threads
        mock_watchdog = MagicMock()
        mock_watchdog.status.return_value = {
            "worker-1": {"alive": True, "last_seen": "2026-02-16T12:00:00Z"},
            "worker-2": {"alive": False, "last_seen": "2026-02-16T11:00:00Z"},
            "worker-3": {"alive": False, "last_seen": "2026-02-16T10:00:00Z"},
        }

        app.config["WATCHDOG"] = mock_watchdog

        response = client.get("/api/health/detailed")
        data = response.get_json()

        assert data["status"] == "degraded"
        assert data["checks"]["threads"]["status"] == "error"
        assert data["checks"]["threads"]["monitored"] == 3
        assert set(data["checks"]["threads"]["dead"]) == {"worker-2", "worker-3"}

    def test_detailed_without_watchdog(self, client):
        """Test that /detailed shows not_configured when watchdog is absent."""
        response = client.get("/api/health/detailed")
        data = response.get_json()
        checks = data["checks"]

        assert checks["threads"]["status"] == "not_configured"


class TestResourcesEndpoint:
    """Test suite for GET /api/health/resources endpoint."""

    def test_resources_returns_500_when_no_system_config(self, client):
        """Test that /resources returns 500 error when SYSTEM_CONFIG is not set."""
        response = client.get("/api/health/resources")
        assert response.status_code == 500
        data = response.get_json()
        assert "error" in data
        assert "System config not available" in data["error"]

    def test_resources_with_valid_config(self, app, client, tmpdir_path):
        """Test that /resources returns resource snapshot with valid config."""
        # Create a mock config
        mock_config = MagicMock()
        mock_config.data_root = tmpdir_path

        app.config["SYSTEM_CONFIG"] = mock_config

        # Mock the ResourceMonitor to avoid actual system calls
        mock_snapshot = {
            "disk": {
                "total_gb": 100.0,
                "used_gb": 50.0,
                "free_gb": 50.0,
                "percent": 50.0,
            },
            "memory": {
                "total_mb": 16000,
                "available_mb": 8000,
                "percent": 50.0,
            },
        }

        with patch(
            "monitoring.resource_monitor.ResourceMonitor"
        ) as MockResourceMonitor:
            mock_monitor = MagicMock()
            mock_monitor.snapshot.return_value = mock_snapshot
            MockResourceMonitor.return_value = mock_monitor

            response = client.get("/api/health/resources")

            assert response.status_code == 200
            data = response.get_json()
            assert "disk" in data
            assert "memory" in data
            assert data["disk"]["total_gb"] == 100.0
            assert data["memory"]["total_mb"] == 16000

            # Verify ResourceMonitor was instantiated with correct path
            MockResourceMonitor.assert_called_once_with(tmpdir_path)

    def test_resources_raises_health_check_error_on_failure(
        self, app, client, tmpdir_path
    ):
        """Test that /resources raises HealthCheckError when monitor fails."""
        from benchmarks.exceptions import HealthCheckError

        mock_config = MagicMock()
        mock_config.data_root = tmpdir_path
        app.config["SYSTEM_CONFIG"] = mock_config

        # Mock ResourceMonitor to raise an exception
        with patch(
            "monitoring.resource_monitor.ResourceMonitor"
        ) as MockResourceMonitor:
            MockResourceMonitor.side_effect = Exception("Monitor initialization failed")

            with pytest.raises(HealthCheckError) as exc_info:
                response = client.get("/api/health/resources")

            assert "Resource monitor failed" in str(exc_info.value)
            assert "Monitor initialization failed" in str(exc_info.value)


class TestMetricsEndpoint:
    """Test suite for GET /api/health/metrics endpoint."""

    def test_metrics_returns_200_with_hint_when_no_collector(self, client):
        """Test that /metrics returns 200 with hint when METRICS_COLLECTOR is not set."""
        response = client.get("/api/health/metrics")
        assert response.status_code == 200
        data = response.get_json()
        assert "error" in data
        assert "Metrics collector not configured" in data["error"]
        assert "hint" in data
        assert "METRICS_COLLECTOR" in data["hint"]

    def test_metrics_with_valid_collector(self, app, client):
        """Test that /metrics returns scan stats and errors with valid collector."""
        # Create a mock metrics collector
        mock_collector = MagicMock()
        mock_collector.get_scan_stats.return_value = {
            "total_scans": 100,
            "avg_duration_ms": 250.5,
            "success_rate": 0.95,
        }
        mock_collector.read_events.return_value = [
            {
                "timestamp": "2026-02-16T12:00:00Z",
                "error": "API timeout",
                "endpoint": "/api/scan",
            },
            {
                "timestamp": "2026-02-16T11:30:00Z",
                "error": "Rate limit exceeded",
                "endpoint": "/api/patterns",
            },
        ]

        app.config["METRICS_COLLECTOR"] = mock_collector

        response = client.get("/api/health/metrics")

        assert response.status_code == 200
        data = response.get_json()

        assert "scan_stats" in data
        assert "recent_errors" in data

        assert data["scan_stats"]["total_scans"] == 100
        assert data["scan_stats"]["avg_duration_ms"] == 250.5
        assert data["scan_stats"]["success_rate"] == 0.95

        assert len(data["recent_errors"]) == 2
        assert data["recent_errors"][0]["error"] == "API timeout"
        assert data["recent_errors"][1]["error"] == "Rate limit exceeded"

        # Verify the collector methods were called with correct parameters
        mock_collector.get_scan_stats.assert_called_once_with(last_n=50)
        mock_collector.read_events.assert_called_once_with("api_error", limit=10)

    def test_metrics_with_empty_results(self, app, client):
        """Test that /metrics handles empty scan stats and errors gracefully."""
        # Create a mock collector that returns empty results
        mock_collector = MagicMock()
        mock_collector.get_scan_stats.return_value = {}
        mock_collector.read_events.return_value = []

        app.config["METRICS_COLLECTOR"] = mock_collector

        response = client.get("/api/health/metrics")

        assert response.status_code == 200
        data = response.get_json()

        assert data["scan_stats"] == {}
        assert data["recent_errors"] == []


class TestBlueprintIntegration:
    """Test suite for blueprint integration and configuration."""

    def test_blueprint_name_is_health_detailed(self):
        """Test that the blueprint has the correct name."""
        assert health_bp.name == "health_detailed"

    def test_blueprint_can_be_registered_with_custom_prefix(self):
        """Test that blueprint can be registered with a custom URL prefix."""
        app = Flask(__name__)
        app.register_blueprint(health_bp, url_prefix="/custom/health")
        client = app.test_client()

        # Test that endpoints are accessible at the custom prefix
        response = client.get("/custom/health/detailed")
        assert response.status_code == 200

        response = client.get("/custom/health/resources")
        assert response.status_code == 500  # No config, but route exists

    def test_all_endpoints_return_json_content_type(self, client):
        """Test that all endpoints return JSON content type."""
        endpoints = [
            "/api/health/detailed",
            "/api/health/resources",
            "/api/health/metrics",
        ]

        for endpoint in endpoints:
            response = client.get(endpoint)
            assert response.content_type == "application/json"

    def test_detailed_endpoint_with_full_configuration(self, app, client, tmpdir_path):
        """Test /detailed endpoint with all configuration objects set."""
        # Create all mock objects
        mock_store = MagicMock()
        mock_store.get_total_results.return_value = 100
        mock_store.get_last_scan_time.return_value = "2026-02-16T12:00:00Z"

        mock_config = MagicMock()
        mock_config.data_root = tmpdir_path

        mock_predictor = MagicMock()
        mock_predictor.is_loaded = True
        mock_predictor.models = ["model1", "model2"]

        mock_scanner = MagicMock()
        mock_scanner.predictor = mock_predictor

        mock_watchdog = MagicMock()
        mock_watchdog.status.return_value = {
            "worker-1": {"alive": True, "last_seen": "2026-02-16T12:00:00Z"},
        }

        # Set all config
        app.config["RESULT_STORE"] = mock_store
        app.config["SYSTEM_CONFIG"] = mock_config
        app.config["SCANNER_ENGINE"] = mock_scanner
        app.config["WATCHDOG"] = mock_watchdog

        response = client.get("/api/health/detailed")
        data = response.get_json()

        # Verify all checks are present and OK
        assert data["status"] == "ok"
        assert data["checks"]["database"]["status"] == "ok"
        assert data["checks"]["data_dir"]["status"] == "ok"
        assert data["checks"]["ml_models"]["status"] == "ok"
        assert data["checks"]["threads"]["status"] == "ok"
        assert "thread_count" in data["checks"]
