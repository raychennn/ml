"""Custom exceptions for the benchmarks and load-testing infrastructure.

All exceptions inherit from :class:`PerfInfraError` so callers can catch
the entire family with a single ``except PerfInfraError`` clause.
"""


class PerfInfraError(Exception):
    """Base exception for the performance-infrastructure packages."""


# ── benchmarks ──────────────────────────────────────────────────────


class BenchmarkError(PerfInfraError):
    """Base for benchmark-specific errors."""


class BenchmarkSetupError(BenchmarkError):
    """Raised when synthetic data or temp directories cannot be created."""


class BenchmarkRunError(BenchmarkError):
    """Raised when a benchmark function fails during execution."""


class BaselineLoadError(BenchmarkError):
    """Raised when ``baselines.json`` cannot be loaded or parsed."""


class ProfilerError(BenchmarkError):
    """Raised when profiling a target fails."""


# ── load tests ──────────────────────────────────────────────────────


class LoadTestError(PerfInfraError):
    """Base for load-test-specific errors."""


class LoadTestSetupError(LoadTestError):
    """Raised when the test environment cannot be prepared."""


class LoadTestAssertionError(LoadTestError):
    """Raised when a load-test assertion fails.

    Attributes:
        metric: The metric name that violated the threshold.
        actual: The actual observed value.
        threshold: The expected threshold value.
    """

    def __init__(
        self,
        message: str,
        metric: str = "",
        actual: object = None,
        threshold: object = None,
    ) -> None:
        super().__init__(message)
        self.metric = metric
        self.actual = actual
        self.threshold = threshold


# ── monitoring ──────────────────────────────────────────────────────


class MonitoringError(PerfInfraError):
    """Base for monitoring-specific errors."""


class MetricsWriteError(MonitoringError):
    """Raised when writing a metric event to the JSONL file fails."""


class MetricsReadError(MonitoringError):
    """Raised when reading/parsing the JSONL metrics file fails."""


class WatchdogError(MonitoringError):
    """Raised when the thread watchdog encounters an unrecoverable error."""


class HealthCheckError(MonitoringError):
    """Raised when a health-check probe cannot determine system status."""
