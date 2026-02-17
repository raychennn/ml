"""Thread Liveness Watchdog.

Monitors liveness of key system threads (bot, scheduler, scan).
Periodically checks that registered threads are alive and logs warnings.

Usage::

    from monitoring.watchdog import ThreadWatchdog
    wd = ThreadWatchdog()
    wd.register("bot_poller", bot_thread)
    wd.register("scheduler", scheduler_thread)
    wd.start(check_interval=30)
    ...
    wd.stop()
"""

import logging
import threading
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from benchmarks.exceptions import WatchdogError

logger: logging.Logger = logging.getLogger(__name__)


class ThreadWatchdog:
    """Monitors thread liveness and invokes callbacks on failure.

    Attributes:
        _threads: Mapping of logical names to the threads being monitored.
        _last_seen: Mapping of logical names to ISO-8601 timestamps of
            last confirmed liveness.
        _lock: Guard for concurrent access to internal state.
        _stop_event: Event used to signal the watchdog loop to terminate.
        _watchdog_thread: The background thread running the watchdog loop.
        _on_thread_dead: Callback invoked when a monitored thread is found
            dead.
        _check_interval: Seconds between liveness checks.
    """

    _threads: Dict[str, threading.Thread]
    _last_seen: Dict[str, str]
    _lock: threading.Lock
    _stop_event: threading.Event
    _watchdog_thread: Optional[threading.Thread]
    _on_thread_dead: Callable[[str, threading.Thread], None]
    _check_interval: int

    def __init__(
        self,
        on_thread_dead: Optional[Callable[[str, threading.Thread], None]] = None,
    ) -> None:
        """Initialise the thread watchdog.

        Args:
            on_thread_dead: Optional callback invoked as
                ``callback(thread_name, thread_obj)`` when a monitored
                thread is found dead.  Defaults to logging a warning.
        """
        self._threads = {}
        self._last_seen = {}
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._watchdog_thread = None
        self._on_thread_dead = on_thread_dead or self._default_on_dead
        self._check_interval = 30

    @staticmethod
    def _default_on_dead(name: str, thread: threading.Thread) -> None:
        """Log a warning when a monitored thread is no longer alive.

        Args:
            name: Logical name of the dead thread.
            thread: The ``threading.Thread`` object that died.
        """
        logger.warning(f"Watchdog: thread '{name}' is no longer alive (was {thread})")

    def register(self, name: str, thread: threading.Thread) -> None:
        """Register a thread to be monitored.

        Args:
            name: Logical name used to identify the thread in status
                reports and callbacks.
            thread: The ``threading.Thread`` instance to watch.
        """
        with self._lock:
            self._threads[name] = thread
            self._last_seen[name] = datetime.now(timezone.utc).isoformat()
        logger.info(f"Watchdog: registered thread '{name}'")

    def unregister(self, name: str) -> None:
        """Stop monitoring a thread.

        Args:
            name: Logical name of the thread to remove.
        """
        with self._lock:
            self._threads.pop(name, None)
            self._last_seen.pop(name, None)

    def start(self, check_interval: int = 30) -> None:
        """Start the watchdog background thread.

        Args:
            check_interval: Seconds between consecutive liveness checks.

        Raises:
            WatchdogError: If the watchdog thread cannot be started.
        """
        try:
            self._check_interval = check_interval
            self._stop_event.clear()
            self._watchdog_thread = threading.Thread(
                target=self._run, name="watchdog", daemon=True
            )
            self._watchdog_thread.start()
            logger.info(f"Watchdog started (check every {check_interval}s)")
        except Exception as exc:
            raise WatchdogError(f"Failed to start watchdog thread: {exc}") from exc

    def stop(self) -> None:
        """Stop the watchdog.

        Raises:
            WatchdogError: If the watchdog thread cannot be stopped cleanly.
        """
        try:
            self._stop_event.set()
            if self._watchdog_thread and self._watchdog_thread.is_alive():
                self._watchdog_thread.join(timeout=5)
            logger.info("Watchdog stopped")
        except Exception as exc:
            raise WatchdogError(f"Failed to stop watchdog thread: {exc}") from exc

    def _run(self) -> None:
        """Main loop executed by the watchdog background thread."""
        while not self._stop_event.is_set():
            self._check()
            self._stop_event.wait(self._check_interval)

    def _check(self) -> None:
        """Perform a single liveness check on all registered threads."""
        with self._lock:
            threads_copy: Dict[str, threading.Thread] = dict(self._threads)

        dead: List[str] = []
        for name, thread in threads_copy.items():
            if thread.is_alive():
                with self._lock:
                    self._last_seen[name] = datetime.now(timezone.utc).isoformat()
            else:
                dead.append(name)
                try:
                    self._on_thread_dead(name, thread)
                except Exception as e:
                    logger.error(f"Watchdog callback error for '{name}': {e}")

    def status(self) -> Dict[str, Dict[str, Any]]:
        """Get current status of all monitored threads.

        Returns:
            Dictionary keyed by thread name, each value containing
            ``"alive"``, ``"daemon"``, and ``"last_seen"`` fields.
        """
        with self._lock:
            result: Dict[str, Dict[str, Any]] = {}
            for name, thread in self._threads.items():
                result[name] = {
                    "alive": thread.is_alive(),
                    "daemon": thread.daemon,
                    "last_seen": self._last_seen.get(name, "unknown"),
                }
        return result

    @property
    def is_running(self) -> bool:
        """Whether the watchdog background thread is currently alive."""
        return self._watchdog_thread is not None and self._watchdog_thread.is_alive()
