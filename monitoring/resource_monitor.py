"""Resource Monitor -- track system resource usage.

Uses stdlib only (``resource.getrusage``, ``os.walk``) -- no psutil dependency.

Tracks:
    - Peak memory usage (RSS)
    - Disk usage breakdown (parquet, images, SQLite, logs)
    - Image count and cleanup candidates
    - SQLite database size
    - Active thread count

Usage::

    python -m monitoring.resource_monitor
"""

import os
import sys
import resource
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from benchmarks.exceptions import MonitoringError

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def get_peak_memory_mb() -> float:
    """Return peak resident set size in megabytes.

    macOS reports ``ru_maxrss`` in bytes while Linux reports it in kilobytes,
    so the conversion factor is chosen based on ``sys.platform``.

    Returns:
        Peak RSS of the current process in MB.
    """
    usage: resource.struct_rusage = resource.getrusage(resource.RUSAGE_SELF)
    ru_maxrss: int = usage.ru_maxrss
    if sys.platform == "darwin":
        return ru_maxrss / (1024 * 1024)
    else:
        return ru_maxrss / 1024


def get_directory_size(path: str) -> int:
    """Compute the total size of all files under *path* in bytes.

    Args:
        path: Root directory to measure.

    Returns:
        Total size in bytes, or ``0`` if *path* is not a directory.
    """
    total: int = 0
    if not os.path.isdir(path):
        return 0
    for dirpath, _dirnames, filenames in os.walk(path):
        for f in filenames:
            fp: str = os.path.join(dirpath, f)
            try:
                total += os.path.getsize(fp)
            except OSError:
                pass
    return total


def get_file_count(path: str, extension: Optional[str] = None) -> int:
    """Count files in a directory tree, optionally filtered by extension.

    Args:
        path: Root directory to scan.
        extension: If provided, only count files whose name ends with this
            suffix (e.g. ``".parquet"``).

    Returns:
        Number of matching files, or ``0`` if *path* is not a directory.
    """
    count: int = 0
    if not os.path.isdir(path):
        return 0
    for _dirpath, _dirnames, filenames in os.walk(path):
        for f in filenames:
            if extension is None or f.endswith(extension):
                count += 1
    return count


def get_file_size(path: str) -> int:
    """Return the size of a single file in bytes.

    Args:
        path: File path to measure.

    Returns:
        Size in bytes, or ``0`` if the file cannot be stat-ed.
    """
    try:
        return os.path.getsize(path)
    except OSError:
        return 0


def _format_bytes(n: int) -> str:
    """Format a byte count as a human-readable string.

    Args:
        n: Number of bytes.

    Returns:
        A string such as ``"12.3 MB"`` or ``"1.0 GB"``.
    """
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


class ResourceMonitor:
    """Collects a snapshot of system resource usage.

    Attributes:
        data_root: Base directory containing data files to monitor.
    """

    data_root: str

    def __init__(self, data_root: str = "./data") -> None:
        """Initialise the resource monitor.

        Args:
            data_root: Root directory for data files (parquet, images,
                models, logs, and the SQLite database).
        """
        self.data_root = data_root

    def snapshot(self) -> Dict[str, Any]:
        """Collect a full resource snapshot.

        Returns:
            A dictionary with keys ``"timestamp"``, ``"memory"``,
            ``"disk"``, and ``"threads"``.
        """
        snap: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "memory": self._memory_info(),
            "disk": self._disk_info(),
            "threads": self._thread_info(),
        }
        return snap

    def _memory_info(self) -> Dict[str, float]:
        """Return memory usage information.

        Returns:
            Dictionary containing ``"peak_rss_mb"``.
        """
        return {
            "peak_rss_mb": round(get_peak_memory_mb(), 2),
        }

    def _disk_info(self) -> Dict[str, Any]:
        """Return disk usage breakdown across data sub-directories.

        Returns:
            Dictionary with per-category size and file-count information.
        """
        parquet_dir: str = os.path.join(self.data_root, "parquet")
        image_dir: str = os.path.join(self.data_root, "images")
        model_dir: str = os.path.join(self.data_root, "models")
        log_dir: str = os.path.join(self.data_root, "logs")
        db_path: str = os.path.join(self.data_root, "scan_results.db")

        parquet_size: int = get_directory_size(parquet_dir)
        image_size: int = get_directory_size(image_dir)
        model_size: int = get_directory_size(model_dir)
        log_size: int = get_directory_size(log_dir)
        db_size: int = get_file_size(db_path)

        # Per-timeframe breakdown
        tf_breakdown: Dict[str, Dict[str, Any]] = {}
        if os.path.isdir(parquet_dir):
            for entry in os.listdir(parquet_dir):
                tf_path: str = os.path.join(parquet_dir, entry)
                if os.path.isdir(tf_path) and entry.startswith("timeframe="):
                    tf_name: str = entry.replace("timeframe=", "")
                    tf_breakdown[tf_name] = {
                        "size": get_directory_size(tf_path),
                        "size_human": _format_bytes(get_directory_size(tf_path)),
                        "file_count": get_file_count(tf_path, ".parquet"),
                    }

        # Image breakdown
        alert_images: int = get_file_count(os.path.join(image_dir, "alerts"), ".png")
        ref_images: int = get_file_count(os.path.join(image_dir, "references"), ".png")
        hist_images: int = get_file_count(os.path.join(image_dir, "historical"), ".png")

        return {
            "parquet": {
                "size": parquet_size,
                "size_human": _format_bytes(parquet_size),
                "by_timeframe": tf_breakdown,
            },
            "images": {
                "size": image_size,
                "size_human": _format_bytes(image_size),
                "alert_count": alert_images,
                "reference_count": ref_images,
                "historical_count": hist_images,
                "total_count": alert_images + ref_images + hist_images,
            },
            "models": {"size": model_size, "size_human": _format_bytes(model_size)},
            "logs": {"size": log_size, "size_human": _format_bytes(log_size)},
            "sqlite": {
                "size": db_size,
                "size_human": _format_bytes(db_size),
                "path": os.path.join(self.data_root, "scan_results.db"),
            },
            "total": {
                "size": parquet_size + image_size + model_size + log_size + db_size,
                "size_human": _format_bytes(
                    parquet_size + image_size + model_size + log_size + db_size
                ),
            },
        }

    def _thread_info(self) -> Dict[str, Any]:
        """Return information about currently active threads.

        Returns:
            Dictionary with ``"active_count"`` and ``"thread_names"``.
        """
        threads: List[threading.Thread] = threading.enumerate()
        return {
            "active_count": threading.active_count(),
            "thread_names": [t.name for t in threads],
        }

    def print_report(self) -> None:
        """Print a human-readable resource report to stdout."""
        snap: Dict[str, Any] = self.snapshot()

        print("=" * 60)
        print("  RESOURCE MONITOR SNAPSHOT")
        print(f"  {snap['timestamp']}")
        print("=" * 60)

        # Memory
        mem: Dict[str, float] = snap["memory"]
        print(f"\n  Memory:")
        print(f"    Peak RSS:     {mem['peak_rss_mb']:.1f} MB")

        # Disk
        disk: Dict[str, Any] = snap["disk"]
        print(f"\n  Disk Usage (total: {disk['total']['size_human']}):")
        print(f"    Parquet:      {disk['parquet']['size_human']}")
        for tf, info in disk["parquet"].get("by_timeframe", {}).items():
            print(
                f"      {tf:6s}:     {info['size_human']} ({info['file_count']} files)"
            )
        print(
            f"    Images:       {disk['images']['size_human']} "
            f"({disk['images']['total_count']} files: "
            f"{disk['images']['alert_count']} alerts, "
            f"{disk['images']['reference_count']} refs, "
            f"{disk['images']['historical_count']} historical)"
        )
        print(f"    Models:       {disk['models']['size_human']}")
        print(f"    Logs:         {disk['logs']['size_human']}")
        print(f"    SQLite DB:    {disk['sqlite']['size_human']}")

        # Threads
        thr: Dict[str, Any] = snap["threads"]
        print(f"\n  Threads ({thr['active_count']} active):")
        for name in thr["thread_names"]:
            print(f"    - {name}")

        print()


def main() -> None:
    """Run a standalone resource report using the system configuration."""
    from config import SystemConfig

    config: object = SystemConfig()
    monitor: ResourceMonitor = ResourceMonitor(config.data_root)
    monitor.print_report()


if __name__ == "__main__":
    main()
