"""Metrics Collector -- JSONL-based operational metrics.

Append-only JSONL file for recording:
    - Scan durations
    - Alert counts
    - API request counts and errors
    - Resource snapshots

No contention with existing SQLite -- purely additive.

Usage::

    from monitoring.metrics_collector import MetricsCollector
    mc = MetricsCollector("./data/logs/metrics.jsonl")
    mc.record_scan(timeframe="4h", duration_s=12.3, alerts=5)
"""

import json
import os
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from benchmarks.exceptions import MetricsReadError, MetricsWriteError


class MetricsCollector:
    """Thread-safe JSONL metrics writer.

    Attributes:
        metrics_path: Absolute or relative path to the JSONL metrics file.
    """

    metrics_path: str
    _lock: threading.Lock

    def __init__(self, metrics_path: Optional[str] = None) -> None:
        """Initialise the metrics collector.

        Args:
            metrics_path: Path to the JSONL file where events are stored.
                Defaults to ``data/logs/metrics.jsonl``.
        """
        if metrics_path is None:
            metrics_path = os.path.join("data", "logs", "metrics.jsonl")
        self.metrics_path = metrics_path
        self._lock = threading.Lock()
        os.makedirs(os.path.dirname(self.metrics_path), exist_ok=True)

    def _write_event(self, event: Dict[str, Any]) -> None:
        """Serialise *event* as JSON and append it to the metrics file.

        A UTC ISO-8601 timestamp is injected into every event under the
        ``"ts"`` key before writing.

        Args:
            event: Dictionary representing the metric event.

        Raises:
            MetricsWriteError: If the file cannot be opened or written to.
        """
        event["ts"] = datetime.now(timezone.utc).isoformat()
        try:
            with self._lock:
                with open(self.metrics_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(event, ensure_ascii=False) + "\n")
        except OSError as exc:
            raise MetricsWriteError(
                f"Failed to write metric event to {self.metrics_path}: {exc}"
            ) from exc

    # ----------------------------------------------------------------
    # Scan metrics
    # ----------------------------------------------------------------

    def record_scan_start(
        self,
        timeframes: List[str],
        run_id: Optional[int] = None,
    ) -> None:
        """Record the beginning of a scan run.

        Args:
            timeframes: List of timeframe labels being scanned.
            run_id: Optional identifier for this scan run.
        """
        self._write_event(
            {
                "event": "scan_start",
                "timeframes": timeframes,
                "run_id": run_id,
            }
        )

    def record_scan_end(
        self,
        timeframes: List[str],
        duration_s: float,
        total_alerts: int,
        run_id: Optional[int] = None,
        status: str = "completed",
    ) -> None:
        """Record the completion of a scan run.

        Args:
            timeframes: List of timeframe labels that were scanned.
            duration_s: Wall-clock duration of the scan in seconds.
            total_alerts: Total number of alerts produced.
            run_id: Optional identifier for this scan run.
            status: Outcome status string (e.g. ``"completed"``,
                ``"failed"``).
        """
        self._write_event(
            {
                "event": "scan_end",
                "timeframes": timeframes,
                "duration_s": round(duration_s, 3),
                "total_alerts": total_alerts,
                "run_id": run_id,
                "status": status,
            }
        )

    def record_scan_timeframe(
        self,
        timeframe: str,
        duration_s: float,
        symbols_scanned: int,
        alerts: int,
    ) -> None:
        """Record metrics for a single timeframe within a scan.

        Args:
            timeframe: Timeframe label (e.g. ``"4h"``).
            duration_s: Duration of this timeframe scan in seconds.
            symbols_scanned: Number of symbols examined.
            alerts: Number of alerts generated.
        """
        self._write_event(
            {
                "event": "scan_timeframe",
                "timeframe": timeframe,
                "duration_s": round(duration_s, 3),
                "symbols_scanned": symbols_scanned,
                "alerts": alerts,
            }
        )

    # ----------------------------------------------------------------
    # API metrics
    # ----------------------------------------------------------------

    def record_api_request(
        self,
        endpoint: str,
        status_code: int = 200,
        duration_ms: float = 0,
    ) -> None:
        """Record a single API request.

        Args:
            endpoint: The HTTP endpoint path.
            status_code: HTTP response status code.
            duration_ms: Request processing time in milliseconds.
        """
        self._write_event(
            {
                "event": "api_request",
                "endpoint": endpoint,
                "status_code": status_code,
                "duration_ms": round(duration_ms, 1),
            }
        )

    def record_api_error(
        self,
        endpoint: str,
        error: str,
        status_code: Optional[int] = None,
    ) -> None:
        """Record an API error.

        Args:
            endpoint: The HTTP endpoint path where the error occurred.
            error: Human-readable error description.
            status_code: HTTP status code, if available.
        """
        self._write_event(
            {
                "event": "api_error",
                "endpoint": endpoint,
                "error": error,
                "status_code": status_code,
            }
        )

    # ----------------------------------------------------------------
    # Resource metrics
    # ----------------------------------------------------------------

    def record_resource_snapshot(self, snapshot: Dict[str, Any]) -> None:
        """Record a resource-usage snapshot.

        Args:
            snapshot: Dictionary produced by
                :meth:`ResourceMonitor.snapshot`.
        """
        self._write_event(
            {
                "event": "resource_snapshot",
                "data": snapshot,
            }
        )

    # ----------------------------------------------------------------
    # Generic
    # ----------------------------------------------------------------

    def record_event(self, event_name: str, **kwargs: Any) -> None:
        """Record an arbitrary named event.

        Args:
            event_name: The event type identifier.
            **kwargs: Additional key-value pairs to include in the event.
        """
        self._write_event({"event": event_name, **kwargs})

    # ----------------------------------------------------------------
    # Query
    # ----------------------------------------------------------------

    def read_events(
        self,
        event_type: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Read recent events from the JSONL file.

        Args:
            event_type: If provided, only return events whose ``"event"``
                field matches this value.
            limit: Maximum number of events to return (taken from the end
                of the file).

        Returns:
            A list of event dictionaries, most recent last.

        Raises:
            MetricsReadError: If the metrics file exists but cannot be
                read or contains no parseable lines due to an I/O error.
        """
        events: List[Dict[str, Any]] = []
        if not os.path.exists(self.metrics_path):
            return events

        try:
            with open(self.metrics_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        ev: Dict[str, Any] = json.loads(line)
                        if event_type is None or ev.get("event") == event_type:
                            events.append(ev)
                    except json.JSONDecodeError:
                        continue
        except OSError as exc:
            raise MetricsReadError(
                f"Failed to read metrics from {self.metrics_path}: {exc}"
            ) from exc

        return events[-limit:]

    def get_scan_stats(self, last_n: int = 50) -> Dict[str, Any]:
        """Aggregate statistics from recent ``scan_end`` events.

        Args:
            last_n: Number of most-recent scan events to aggregate over.

        Returns:
            Dictionary with keys ``"count"``, ``"avg_duration_s"``,
            ``"max_duration_s"``, ``"total_alerts"``, and
            ``"avg_alerts_per_scan"``.
        """
        scans: List[Dict[str, Any]] = self.read_events("scan_end", limit=last_n)
        if not scans:
            return {"count": 0}

        durations: List[float] = [s["duration_s"] for s in scans if "duration_s" in s]
        alerts: List[int] = [s["total_alerts"] for s in scans if "total_alerts" in s]

        return {
            "count": len(scans),
            "avg_duration_s": (
                round(sum(durations) / len(durations), 2) if durations else 0
            ),
            "max_duration_s": round(max(durations), 2) if durations else 0,
            "total_alerts": sum(alerts),
            "avg_alerts_per_scan": round(sum(alerts) / len(alerts), 1) if alerts else 0,
        }
