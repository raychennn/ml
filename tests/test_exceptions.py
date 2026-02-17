"""Tests for custom exceptions in benchmarks.exceptions module."""

import pytest

from benchmarks.exceptions import (
    BaselineLoadError,
    BenchmarkError,
    BenchmarkRunError,
    BenchmarkSetupError,
    HealthCheckError,
    LoadTestAssertionError,
    LoadTestError,
    LoadTestSetupError,
    MetricsReadError,
    MetricsWriteError,
    MonitoringError,
    PerfInfraError,
    ProfilerError,
    WatchdogError,
)


class TestInheritanceChains:
    """Test that each exception class inherits from the correct parent."""

    def test_perf_infra_error_inherits_from_exception(self):
        assert issubclass(PerfInfraError, Exception)

    def test_benchmark_error_inherits_from_perf_infra_error(self):
        assert issubclass(BenchmarkError, PerfInfraError)

    def test_benchmark_setup_error_inherits_from_benchmark_error(self):
        assert issubclass(BenchmarkSetupError, BenchmarkError)

    def test_benchmark_run_error_inherits_from_benchmark_error(self):
        assert issubclass(BenchmarkRunError, BenchmarkError)

    def test_baseline_load_error_inherits_from_benchmark_error(self):
        assert issubclass(BaselineLoadError, BenchmarkError)

    def test_profiler_error_inherits_from_benchmark_error(self):
        assert issubclass(ProfilerError, BenchmarkError)

    def test_load_test_error_inherits_from_perf_infra_error(self):
        assert issubclass(LoadTestError, PerfInfraError)

    def test_load_test_setup_error_inherits_from_load_test_error(self):
        assert issubclass(LoadTestSetupError, LoadTestError)

    def test_load_test_assertion_error_inherits_from_load_test_error(self):
        assert issubclass(LoadTestAssertionError, LoadTestError)

    def test_monitoring_error_inherits_from_perf_infra_error(self):
        assert issubclass(MonitoringError, PerfInfraError)

    def test_metrics_write_error_inherits_from_monitoring_error(self):
        assert issubclass(MetricsWriteError, MonitoringError)

    def test_metrics_read_error_inherits_from_monitoring_error(self):
        assert issubclass(MetricsReadError, MonitoringError)

    def test_watchdog_error_inherits_from_monitoring_error(self):
        assert issubclass(WatchdogError, MonitoringError)

    def test_health_check_error_inherits_from_monitoring_error(self):
        assert issubclass(HealthCheckError, MonitoringError)


class TestPerfInfraErrorInstances:
    """Test that all exceptions are instances of PerfInfraError."""

    def test_perf_infra_error_instance(self):
        exc = PerfInfraError("test")
        assert isinstance(exc, PerfInfraError)

    def test_benchmark_error_is_perf_infra_error(self):
        exc = BenchmarkError("test")
        assert isinstance(exc, PerfInfraError)

    def test_benchmark_setup_error_is_perf_infra_error(self):
        exc = BenchmarkSetupError("test")
        assert isinstance(exc, PerfInfraError)

    def test_benchmark_run_error_is_perf_infra_error(self):
        exc = BenchmarkRunError("test")
        assert isinstance(exc, PerfInfraError)

    def test_baseline_load_error_is_perf_infra_error(self):
        exc = BaselineLoadError("test")
        assert isinstance(exc, PerfInfraError)

    def test_profiler_error_is_perf_infra_error(self):
        exc = ProfilerError("test")
        assert isinstance(exc, PerfInfraError)

    def test_load_test_error_is_perf_infra_error(self):
        exc = LoadTestError("test")
        assert isinstance(exc, PerfInfraError)

    def test_load_test_setup_error_is_perf_infra_error(self):
        exc = LoadTestSetupError("test")
        assert isinstance(exc, PerfInfraError)

    def test_load_test_assertion_error_is_perf_infra_error(self):
        exc = LoadTestAssertionError("test")
        assert isinstance(exc, PerfInfraError)

    def test_monitoring_error_is_perf_infra_error(self):
        exc = MonitoringError("test")
        assert isinstance(exc, PerfInfraError)

    def test_metrics_write_error_is_perf_infra_error(self):
        exc = MetricsWriteError("test")
        assert isinstance(exc, PerfInfraError)

    def test_metrics_read_error_is_perf_infra_error(self):
        exc = MetricsReadError("test")
        assert isinstance(exc, PerfInfraError)

    def test_watchdog_error_is_perf_infra_error(self):
        exc = WatchdogError("test")
        assert isinstance(exc, PerfInfraError)

    def test_health_check_error_is_perf_infra_error(self):
        exc = HealthCheckError("test")
        assert isinstance(exc, PerfInfraError)


class TestLoadTestAssertionErrorAttributes:
    """Test LoadTestAssertionError stores metric, actual, and threshold attributes."""

    def test_default_attributes(self):
        exc = LoadTestAssertionError("test message")
        assert exc.metric == ""
        assert exc.actual is None
        assert exc.threshold is None
        assert str(exc) == "test message"

    def test_with_all_attributes(self):
        exc = LoadTestAssertionError(
            "Latency exceeded",
            metric="p95_latency",
            actual=1500.5,
            threshold=1000,
        )
        assert exc.metric == "p95_latency"
        assert exc.actual == 1500.5
        assert exc.threshold == 1000
        assert str(exc) == "Latency exceeded"

    def test_with_partial_attributes(self):
        exc = LoadTestAssertionError(
            "Test failed",
            metric="throughput",
            actual=50,
        )
        assert exc.metric == "throughput"
        assert exc.actual == 50
        assert exc.threshold is None


class TestExceptionsCanBeRaisedAndCaught:
    """Test that all exceptions can be raised and caught properly."""

    def test_raise_and_catch_perf_infra_error(self):
        with pytest.raises(PerfInfraError) as exc_info:
            raise PerfInfraError("base error")
        assert str(exc_info.value) == "base error"

    def test_raise_and_catch_benchmark_error(self):
        with pytest.raises(BenchmarkError) as exc_info:
            raise BenchmarkError("benchmark failed")
        assert str(exc_info.value) == "benchmark failed"

    def test_raise_and_catch_benchmark_setup_error(self):
        with pytest.raises(BenchmarkSetupError) as exc_info:
            raise BenchmarkSetupError("setup failed")
        assert str(exc_info.value) == "setup failed"

    def test_raise_and_catch_benchmark_run_error(self):
        with pytest.raises(BenchmarkRunError) as exc_info:
            raise BenchmarkRunError("run failed")
        assert str(exc_info.value) == "run failed"

    def test_raise_and_catch_baseline_load_error(self):
        with pytest.raises(BaselineLoadError) as exc_info:
            raise BaselineLoadError("baseline load failed")
        assert str(exc_info.value) == "baseline load failed"

    def test_raise_and_catch_profiler_error(self):
        with pytest.raises(ProfilerError) as exc_info:
            raise ProfilerError("profiler failed")
        assert str(exc_info.value) == "profiler failed"

    def test_raise_and_catch_load_test_error(self):
        with pytest.raises(LoadTestError) as exc_info:
            raise LoadTestError("load test failed")
        assert str(exc_info.value) == "load test failed"

    def test_raise_and_catch_load_test_setup_error(self):
        with pytest.raises(LoadTestSetupError) as exc_info:
            raise LoadTestSetupError("load test setup failed")
        assert str(exc_info.value) == "load test setup failed"

    def test_raise_and_catch_load_test_assertion_error(self):
        with pytest.raises(LoadTestAssertionError) as exc_info:
            raise LoadTestAssertionError(
                "assertion failed",
                metric="latency",
                actual=100,
                threshold=50,
            )
        assert str(exc_info.value) == "assertion failed"
        assert exc_info.value.metric == "latency"
        assert exc_info.value.actual == 100
        assert exc_info.value.threshold == 50

    def test_raise_and_catch_monitoring_error(self):
        with pytest.raises(MonitoringError) as exc_info:
            raise MonitoringError("monitoring failed")
        assert str(exc_info.value) == "monitoring failed"

    def test_raise_and_catch_metrics_write_error(self):
        with pytest.raises(MetricsWriteError) as exc_info:
            raise MetricsWriteError("write failed")
        assert str(exc_info.value) == "write failed"

    def test_raise_and_catch_metrics_read_error(self):
        with pytest.raises(MetricsReadError) as exc_info:
            raise MetricsReadError("read failed")
        assert str(exc_info.value) == "read failed"

    def test_raise_and_catch_watchdog_error(self):
        with pytest.raises(WatchdogError) as exc_info:
            raise WatchdogError("watchdog failed")
        assert str(exc_info.value) == "watchdog failed"

    def test_raise_and_catch_health_check_error(self):
        with pytest.raises(HealthCheckError) as exc_info:
            raise HealthCheckError("health check failed")
        assert str(exc_info.value) == "health check failed"


class TestCatchByBaseClass:
    """Test that child exceptions can be caught by their parent classes."""

    def test_catch_benchmark_error_as_perf_infra_error(self):
        with pytest.raises(PerfInfraError):
            raise BenchmarkSetupError("setup failed")

    def test_catch_load_test_error_as_perf_infra_error(self):
        with pytest.raises(PerfInfraError):
            raise LoadTestSetupError("setup failed")

    def test_catch_monitoring_error_as_perf_infra_error(self):
        with pytest.raises(PerfInfraError):
            raise MetricsWriteError("write failed")

    def test_catch_specific_benchmark_error_as_benchmark_error(self):
        with pytest.raises(BenchmarkError):
            raise ProfilerError("profiler failed")

    def test_catch_specific_load_test_error_as_load_test_error(self):
        with pytest.raises(LoadTestError):
            raise LoadTestAssertionError("assertion failed")

    def test_catch_specific_monitoring_error_as_monitoring_error(self):
        with pytest.raises(MonitoringError):
            raise HealthCheckError("health check failed")


class TestMessagePreservation:
    """Test that message strings are preserved correctly."""

    def test_perf_infra_error_message(self):
        msg = "Performance infrastructure encountered an error"
        exc = PerfInfraError(msg)
        assert str(exc) == msg
        assert exc.args[0] == msg

    def test_benchmark_error_message(self):
        msg = "Benchmark execution failed unexpectedly"
        exc = BenchmarkError(msg)
        assert str(exc) == msg
        assert exc.args[0] == msg

    def test_load_test_assertion_error_message(self):
        msg = "Load test assertion violated threshold"
        exc = LoadTestAssertionError(msg, metric="cpu", actual=95, threshold=80)
        assert str(exc) == msg
        assert exc.args[0] == msg

    def test_monitoring_error_message(self):
        msg = "Monitoring system is unavailable"
        exc = MonitoringError(msg)
        assert str(exc) == msg
        assert exc.args[0] == msg

    def test_empty_message(self):
        exc = PerfInfraError("")
        assert str(exc) == ""

    def test_multiline_message(self):
        msg = "Error occurred:\nLine 1\nLine 2"
        exc = PerfInfraError(msg)
        assert str(exc) == msg
