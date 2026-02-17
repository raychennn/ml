"""Enhanced Health Check Blueprint.

Provides detailed health endpoints as a separate Flask blueprint.
Leaves existing ``/api/health`` untouched.

Endpoints:
    GET /api/health/detailed
        Full system health with resource info.
    GET /api/health/resources
        Disk and memory resource snapshot.
    GET /api/health/metrics
        Recent scan metrics summary.

Usage::

    from monitoring.health_check import health_bp
    app.register_blueprint(health_bp, url_prefix="/api/health")
"""

import os
import sys
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from flask import Blueprint, Response, jsonify, current_app

from benchmarks.exceptions import HealthCheckError

health_bp: Blueprint = Blueprint("health_detailed", __name__)


@health_bp.route("/detailed")
def detailed_health() -> Response:
    """Return comprehensive system health information.

    Checks database connectivity, data directory existence, ML model
    availability, thread watchdog status, and active thread counts.

    Returns:
        JSON response containing ``"status"``, ``"timestamp"``, and
        ``"checks"`` fields.
    """
    config: Optional[object] = current_app.config.get("SYSTEM_CONFIG")
    store: Optional[object] = current_app.config.get("RESULT_STORE")
    scanner: Optional[object] = current_app.config.get("SCANNER_ENGINE")
    watchdog: Optional[object] = current_app.config.get("WATCHDOG")

    status: str = "ok"
    checks: Dict[str, Any] = {}

    # Database check
    if store:
        try:
            total: int = store.get_total_results()
            last_scan: Optional[str] = store.get_last_scan_time()
            checks["database"] = {
                "status": "ok",
                "total_results": total,
                "last_scan_time": last_scan,
            }
        except Exception as e:
            checks["database"] = {"status": "error", "error": str(e)}
            status = "degraded"
    else:
        checks["database"] = {"status": "unavailable"}

    # Data directory
    if config:
        data_exists: bool = os.path.isdir(config.data_root)
        checks["data_dir"] = {
            "status": "ok" if data_exists else "error",
            "path": config.data_root,
        }
        if not data_exists:
            status = "degraded"

    # ML models
    if scanner:
        ml_loaded: bool = scanner.predictor.is_loaded
        checks["ml_models"] = {
            "status": "ok" if ml_loaded else "fallback",
            "count": len(scanner.predictor.models),
        }
    else:
        checks["ml_models"] = {"status": "unavailable"}

    # Thread watchdog
    if watchdog:
        thread_status: Dict[str, Any] = watchdog.status()
        dead_threads: List[str] = [
            n for n, s in thread_status.items() if not s["alive"]
        ]
        checks["threads"] = {
            "status": "error" if dead_threads else "ok",
            "monitored": len(thread_status),
            "dead": dead_threads,
            "details": thread_status,
        }
        if dead_threads:
            status = "degraded"
    else:
        checks["threads"] = {"status": "not_configured"}

    # Active threads
    checks["thread_count"] = {
        "active": threading.active_count(),
        "names": [t.name for t in threading.enumerate()],
    }

    return jsonify(
        {
            "status": status,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "checks": checks,
        }
    )


@health_bp.route("/resources")
def resource_health() -> Tuple[Response, int] | Response:
    """Return a disk and memory resource snapshot.

    Instantiates a :class:`ResourceMonitor` using the system configuration
    and returns its snapshot as JSON.

    Returns:
        JSON response with resource usage data, or a 500 error if the
        system config is unavailable or the monitor raises an error.

    Raises:
        HealthCheckError: If the resource monitor fails to produce a
            snapshot (caught internally and returned as a 500 JSON
            response).
    """
    config: Optional[object] = current_app.config.get("SYSTEM_CONFIG")
    if not config:
        return jsonify({"error": "System config not available"}), 500

    try:
        from monitoring.resource_monitor import ResourceMonitor

        monitor: ResourceMonitor = ResourceMonitor(config.data_root)
        snap: Dict[str, Any] = monitor.snapshot()
    except Exception as exc:
        raise HealthCheckError(f"Resource monitor failed: {exc}") from exc

    return jsonify(snap)


@health_bp.route("/metrics")
def metrics_health() -> Tuple[Response, int] | Response:
    """Return recent scan metrics summary.

    Reads aggregated scan statistics and recent API errors from the
    configured :class:`MetricsCollector`.

    Returns:
        JSON response with ``"scan_stats"`` and ``"recent_errors"``
        fields, or a 200 response with an error hint if no collector
        is configured.
    """
    metrics: Optional[object] = current_app.config.get("METRICS_COLLECTOR")
    if not metrics:
        return (
            jsonify(
                {
                    "error": "Metrics collector not configured",
                    "hint": "Set app.config['METRICS_COLLECTOR'] to a MetricsCollector instance",
                }
            ),
            200,
        )

    scan_stats: Dict[str, Any] = metrics.get_scan_stats(last_n=50)
    recent_errors: List[Dict[str, Any]] = metrics.read_events("api_error", limit=10)

    return jsonify(
        {
            "scan_stats": scan_stats,
            "recent_errors": recent_errors,
        }
    )
