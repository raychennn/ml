"""Tests for the thread watchdog module."""

import threading
import time
from unittest.mock import patch, MagicMock

import pytest

from monitoring.watchdog import ThreadWatchdog
from benchmarks.exceptions import WatchdogError


class TestThreadWatchdogInit:
    """Test ThreadWatchdog initialization."""

    def test_init_initializes_empty_state(self):
        """Initialization should create empty internal state."""
        watchdog = ThreadWatchdog()

        assert watchdog._threads == {}
        assert watchdog._last_seen == {}
        assert watchdog._watchdog_thread is None
        assert not watchdog._stop_event.is_set()
        assert watchdog._check_interval == 30

    def test_init_with_custom_callback(self):
        """Initialization should accept custom on_thread_dead callback."""
        callback = MagicMock()
        watchdog = ThreadWatchdog(on_thread_dead=callback)

        assert watchdog._on_thread_dead == callback


class TestRegisterUnregister:
    """Test thread registration and unregistration."""

    def test_register_adds_thread(self):
        """register() should add thread to internal dict."""
        watchdog = ThreadWatchdog()
        thread = threading.Thread(target=lambda: None, name="test_thread")

        watchdog.register("test", thread)

        assert "test" in watchdog._threads
        assert watchdog._threads["test"] == thread
        assert "test" in watchdog._last_seen
        assert watchdog._last_seen["test"] is not None

    def test_register_multiple_threads(self):
        """register() should handle multiple threads."""
        watchdog = ThreadWatchdog()
        thread1 = threading.Thread(target=lambda: None, name="thread1")
        thread2 = threading.Thread(target=lambda: None, name="thread2")

        watchdog.register("first", thread1)
        watchdog.register("second", thread2)

        assert len(watchdog._threads) == 2
        assert "first" in watchdog._threads
        assert "second" in watchdog._threads

    def test_unregister_removes_thread(self):
        """unregister() should remove thread from monitoring."""
        watchdog = ThreadWatchdog()
        thread = threading.Thread(target=lambda: None, name="test_thread")
        watchdog.register("test", thread)

        watchdog.unregister("test")

        assert "test" not in watchdog._threads
        assert "test" not in watchdog._last_seen

    def test_unregister_nonexistent_thread_is_safe(self):
        """unregister() should not raise error for nonexistent thread."""
        watchdog = ThreadWatchdog()

        # Should not raise
        watchdog.unregister("nonexistent")

        assert len(watchdog._threads) == 0


class TestStatus:
    """Test status() method."""

    def test_status_returns_correct_structure(self):
        """status() should return correct structure for registered threads."""
        watchdog = ThreadWatchdog()

        # Create and start a thread
        def worker():
            time.sleep(0.5)

        thread = threading.Thread(target=worker, name="worker", daemon=True)
        thread.start()
        watchdog.register("worker", thread)

        status = watchdog.status()

        assert "worker" in status
        assert "alive" in status["worker"]
        assert "daemon" in status["worker"]
        assert "last_seen" in status["worker"]
        assert status["worker"]["alive"] is True
        assert status["worker"]["daemon"] is True
        assert status["worker"]["last_seen"] != "unknown"

        thread.join()

    def test_status_reflects_dead_thread(self):
        """status() should correctly report dead threads."""
        watchdog = ThreadWatchdog()

        # Create a short-lived thread
        def short_task():
            pass

        thread = threading.Thread(target=short_task, name="short")
        thread.start()
        thread.join()  # Wait for it to finish

        watchdog.register("short", thread)
        status = watchdog.status()

        assert status["short"]["alive"] is False

    def test_status_empty_for_no_threads(self):
        """status() should return empty dict when no threads registered."""
        watchdog = ThreadWatchdog()
        status = watchdog.status()

        assert status == {}


class TestStartStop:
    """Test watchdog lifecycle."""

    def test_start_creates_watchdog_thread(self):
        """start() should create and start the watchdog thread."""
        watchdog = ThreadWatchdog()

        watchdog.start(check_interval=1)

        assert watchdog._watchdog_thread is not None
        assert watchdog._watchdog_thread.is_alive()
        assert watchdog._check_interval == 1

        watchdog.stop()

    def test_stop_terminates_watchdog_thread(self):
        """stop() should cleanly terminate the watchdog thread."""
        watchdog = ThreadWatchdog()
        watchdog.start(check_interval=1)

        watchdog.stop()

        time.sleep(0.2)  # Give it time to stop
        assert not watchdog._watchdog_thread.is_alive()
        assert watchdog._stop_event.is_set()

    def test_is_running_property_reflects_state(self):
        """is_running property should reflect watchdog state."""
        watchdog = ThreadWatchdog()

        assert not watchdog.is_running

        watchdog.start(check_interval=1)
        assert watchdog.is_running

        watchdog.stop()
        time.sleep(0.2)
        assert not watchdog.is_running


class TestDeadThreadDetection:
    """Test detection of dead threads."""

    def test_dead_thread_triggers_callback(self):
        """Dead thread should trigger the callback."""
        callback_called = threading.Event()
        dead_thread_name = None

        def custom_callback(name, thread):
            nonlocal dead_thread_name
            dead_thread_name = name
            callback_called.set()

        watchdog = ThreadWatchdog(on_thread_dead=custom_callback)

        # Create a very short-lived thread
        def quick_task():
            time.sleep(0.01)

        thread = threading.Thread(target=quick_task, name="quick")
        thread.start()
        thread.join()  # Ensure it's dead

        watchdog.register("quick", thread)
        watchdog.start(check_interval=0.1)

        # Wait for callback to be triggered
        assert callback_called.wait(timeout=2.0)
        assert dead_thread_name == "quick"

        watchdog.stop()

    def test_custom_on_thread_dead_callback_invoked(self):
        """Custom on_thread_dead callback should be invoked correctly."""
        call_count = [0]
        thread_names = []

        def counting_callback(name, thread):
            call_count[0] += 1
            thread_names.append(name)

        watchdog = ThreadWatchdog(on_thread_dead=counting_callback)

        # Create two short-lived threads
        def quick_task():
            pass

        thread1 = threading.Thread(target=quick_task, name="first")
        thread2 = threading.Thread(target=quick_task, name="second")

        thread1.start()
        thread2.start()
        thread1.join()
        thread2.join()

        watchdog.register("first", thread1)
        watchdog.register("second", thread2)
        watchdog.start(check_interval=0.1)

        # Wait for callbacks
        time.sleep(0.5)

        watchdog.stop()

        assert call_count[0] >= 2
        assert "first" in thread_names
        assert "second" in thread_names

    def test_callback_exception_doesnt_crash_watchdog(self):
        """Exception in callback should not crash the watchdog."""

        def bad_callback(name, thread):
            raise ValueError("Intentional error in callback")

        watchdog = ThreadWatchdog(on_thread_dead=bad_callback)

        def quick_task():
            pass

        thread = threading.Thread(target=quick_task)
        thread.start()
        thread.join()

        watchdog.register("test", thread)
        watchdog.start(check_interval=0.1)

        time.sleep(0.5)

        # Watchdog should still be running
        assert watchdog.is_running

        watchdog.stop()


class TestErrorHandling:
    """Test error handling scenarios."""

    def test_start_failure_raises_watchdog_error(self):
        """start() should raise WatchdogError on failure."""
        watchdog = ThreadWatchdog()

        # Mock threading.Thread to raise an exception
        with patch(
            "threading.Thread", side_effect=RuntimeError("Thread creation failed")
        ):
            with pytest.raises(WatchdogError) as exc_info:
                watchdog.start()

            assert "Failed to start watchdog thread" in str(exc_info.value)

    def test_stop_handles_timeout_gracefully(self):
        """stop() should handle thread join timeout gracefully."""
        watchdog = ThreadWatchdog()

        # Start watchdog
        watchdog.start(check_interval=0.1)

        # Mock join to simulate a thread that won't stop
        original_join = watchdog._watchdog_thread.join

        def slow_join(timeout=None):
            time.sleep(0.1)  # Simulate some delay
            original_join(timeout=0.01)  # Join with very short timeout

        watchdog._watchdog_thread.join = slow_join

        # This should not raise, even if join times out
        watchdog.stop()

    def test_alive_thread_updates_last_seen(self):
        """Alive threads should have their last_seen timestamp updated."""
        watchdog = ThreadWatchdog()

        # Create a long-running thread
        stop_event = threading.Event()

        def long_task():
            stop_event.wait()

        thread = threading.Thread(target=long_task, name="long", daemon=True)
        thread.start()

        watchdog.register("long", thread)

        # Get initial last_seen
        initial_last_seen = watchdog._last_seen["long"]

        watchdog.start(check_interval=0.1)
        time.sleep(0.3)  # Wait for at least one check cycle

        # Get updated last_seen
        updated_last_seen = watchdog._last_seen["long"]

        # Should be different (updated)
        assert updated_last_seen != initial_last_seen

        watchdog.stop()
        stop_event.set()
        thread.join(timeout=1.0)


class TestIntegration:
    """Integration tests for realistic scenarios."""

    def test_watchdog_monitors_multiple_threads(self):
        """Watchdog should correctly monitor multiple threads simultaneously."""
        watchdog = ThreadWatchdog()

        # Create mix of alive and dead threads
        stop_events = [threading.Event() for _ in range(3)]

        def long_task(event):
            event.wait()

        def short_task():
            pass

        alive1 = threading.Thread(target=long_task, args=(stop_events[0],), daemon=True)
        alive2 = threading.Thread(target=long_task, args=(stop_events[1],), daemon=True)
        alive3 = threading.Thread(target=long_task, args=(stop_events[2],), daemon=True)
        dead = threading.Thread(target=short_task)

        alive1.start()
        alive2.start()
        alive3.start()
        dead.start()
        dead.join()

        watchdog.register("alive1", alive1)
        watchdog.register("alive2", alive2)
        watchdog.register("alive3", alive3)
        watchdog.register("dead", dead)

        watchdog.start(check_interval=0.1)
        time.sleep(0.3)

        status = watchdog.status()

        assert status["alive1"]["alive"] is True
        assert status["alive2"]["alive"] is True
        assert status["alive3"]["alive"] is True
        assert status["dead"]["alive"] is False

        watchdog.stop()
        for event in stop_events:
            event.set()
        alive1.join(timeout=1.0)
        alive2.join(timeout=1.0)
        alive3.join(timeout=1.0)
