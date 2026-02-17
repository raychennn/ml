# Performance Engineering Guide

This document covers performance baselines, profiling, monitoring, load testing, and optimization for the crypto pattern recognition system.

---

## Table of Contents

- [Performance Baselines](#performance-baselines)
- [Running Benchmarks](#running-benchmarks)
- [Profiling Guide](#profiling-guide)
- [Load Testing](#load-testing)
- [Monitoring & Health Checks](#monitoring--health-checks)
- [Capacity Planning](#capacity-planning)
- [Optimization Reference](#optimization-reference)
- [Troubleshooting](#troubleshooting)

---

## Performance Baselines

Expected performance on a typical 2-4 core machine. Baselines are defined in `benchmarks/baselines.json`.

| Operation | Expected | WARN threshold | FAIL threshold |
|-----------|----------|---------------|----------------|
| DTW single pair (120pts) | 10ms | 15ms | 30ms |
| DTW batch (10 pairs) | 100ms | 150ms | 300ms |
| NumPy DTW fallback (120pts) | 50ms | 100ms | 250ms |
| Prescreening (100 symbols) | 500ms | 1s | 2.5s |
| Prescreening throughput | 200 sym/s | 100 sym/s | 40 sym/s |
| Parquet write (5000 rows) | 50ms | 100ms | 250ms |
| Parquet read (5000 rows) | 30ms | 60ms | 150ms |
| Bulk read (50 files) | 1s | 2s | 5s |
| ML inference (single) | 20ms | 40ms | 100ms |
| ML inference (batch 10) | 150ms | 300ms | 750ms |
| Full scan (20 symbols) | 5s | 10s | 25s |
| prepare_for_dtw (500 rows) | 5ms | 10ms | 25ms |

---

## Running Benchmarks

### Full Benchmark Suite

```bash
python -m benchmarks.run_all_benchmarks
```

Runs all benchmarks and compares against baselines. Exit code 0 if all PASS/WARN, 1 if any FAIL.

### Individual Benchmarks

```bash
python -m benchmarks.bench_dtw              # DTW computation speed
python -m benchmarks.bench_prescreening      # Stage 1 prescreening throughput
python -m benchmarks.bench_parquet           # Parquet I/O speed
python -m benchmarks.bench_ml_inference      # ML prediction latency
python -m benchmarks.bench_scan_cycle        # End-to-end scan cycle
```

### Optimization Comparisons (A/B)

```bash
python -m benchmarks.compare_optimizations
```

Compares:
- dtaidistance (C) vs numpy DTW
- With vs without prescreening
- Snappy vs uncompressed Parquet
- BTC cache hit vs miss

### Scaling Tests

```bash
python -m benchmarks.scaling_tests
```

Measures how performance scales with:
- Symbol count (10 → 200)
- Reference count (1 → 20)
- Data volume (100 → 10000 rows)
- Worker count (1 → 8)

---

## Profiling Guide

### Quick Profile

```bash
python -m benchmarks.profiler --target scan       # Profile scan_timeframe
python -m benchmarks.profiler --target dtw         # Profile compute_similarity
python -m benchmarks.profiler --target normalize   # Profile prepare_for_dtw
python -m benchmarks.profiler --target bulk_read   # Profile bulk_read_timeframe
python -m benchmarks.profiler --target all         # Profile everything
```

### Flamegraph Output

```bash
python -m benchmarks.profiler --target scan --output-dir ./profiles
# Generates ./profiles/profile_scan.txt in collapsed stack format
```

### Interpreting Results

The profiler outputs cProfile stats sorted by cumulative time. Key columns:

- **ncalls**: Number of calls (primitive calls / total)
- **tottime**: Total time in this function (excluding subcalls)
- **cumtime**: Cumulative time (including subcalls)
- **percall**: Per-call time

**Common bottlenecks:**
1. `dtw.distance()` — DTW computation (use dtaidistance C library)
2. `pq.read_table()` — Parquet I/O (use tail_bars to limit reads)
3. `prepare_for_dtw()` — Feature computation (scales with data volume)
4. `Pool.map()` — Multiprocessing overhead (only beneficial for 5+ candidates)

---

## Load Testing

### Concurrent Scans

```bash
python -m load_tests.test_concurrent_scans
```

Verifies scan_lock correctly prevents concurrent scans.

### Web Dashboard Load

```bash
python -m load_tests.test_web_load
```

Tests Flask endpoints under concurrent HTTP requests (20 threads × 10 requests).

### Large Symbol Count

```bash
python -m load_tests.test_large_symbol_count
```

Scans 250+ symbols and verifies memory stays under 2GB, time under 2 minutes.

### Sustained Scheduler

```bash
python -m load_tests.test_sustained_scheduler
```

Runs 10 simulated scheduler ticks and checks for thread leaks and timing degradation.

### SQLite Concurrent Access

```bash
python -m load_tests.test_sqlite_concurrent
```

Tests concurrent read/write to SQLite with WAL mode enabled.

---

## Monitoring & Health Checks

### Resource Monitor

```bash
python -m monitoring.resource_monitor
```

Prints a snapshot of:
- Peak RSS memory usage
- Disk usage breakdown (parquet, images, models, logs, SQLite)
- Per-timeframe Parquet size
- Active thread count

### Metrics Collector

JSONL-based metrics recording. Integrates into application code:

```python
from monitoring.metrics_collector import MetricsCollector

mc = MetricsCollector("./data/logs/metrics.jsonl")
mc.record_scan_end(timeframes=["4h"], duration_s=12.3, total_alerts=5)
mc.record_api_request("/api/health", status_code=200, duration_ms=15.2)
```

Query metrics:

```python
stats = mc.get_scan_stats(last_n=50)
# {'count': 50, 'avg_duration_s': 10.5, 'max_duration_s': 45.2, ...}
```

### Thread Watchdog

```python
from monitoring.watchdog import ThreadWatchdog

wd = ThreadWatchdog()
wd.register("bot_poller", bot_thread)
wd.register("scheduler", scheduler_thread)
wd.start(check_interval=30)
# Logs warnings when registered threads die
```

### Enhanced Health Endpoints

Register the health blueprint for detailed endpoints:

```python
from monitoring.health_check import health_bp
app.register_blueprint(health_bp, url_prefix="/api/health")
```

| Endpoint | Description |
|----------|-------------|
| `GET /api/health/detailed` | Full health check with thread status |
| `GET /api/health/resources` | Disk + memory resource snapshot |
| `GET /api/health/metrics` | Recent scan metrics summary |

The existing `GET /api/health` endpoint remains unchanged.

---

## Capacity Planning

### Per-Symbol Resource Usage

| Resource | Per Symbol (4h, 500 bars) |
|----------|--------------------------|
| Parquet file size | ~40 KB (snappy compressed) |
| Memory during scan | ~2 MB (DataFrame + features) |
| DTW computation | ~10ms (with dtaidistance C) |
| Prescreening | ~1ms |

### Scaling Estimates

| Symbols | Prescreening | DTW (top 20%) | Total Scan |
|---------|-------------|---------------|------------|
| 50 | ~0.3s | ~2s | ~3s |
| 100 | ~0.5s | ~4s | ~5s |
| 250 | ~1.2s | ~10s | ~15s |
| 500 | ~2.5s | ~20s | ~30s |

### Disk Space Planning

| Component | Size Estimate |
|-----------|--------------|
| Parquet (250 symbols × 4 timeframes) | 2-5 GB |
| Images (after 1 month of scanning) | 500 MB - 1 GB |
| SQLite (after 1 month of scanning) | 100 MB - 500 MB |
| ML models | ~10 MB |
| Logs | ~50 MB |
| **Total recommended** | **10 GB+** |

### Memory Planning

| Scenario | Estimated Peak RSS |
|----------|-------------------|
| Idle (web only) | 100-200 MB |
| Single timeframe scan | 300-500 MB |
| Full 4-timeframe scan | 500-800 MB |
| ML training | 1-2 GB |
| **Recommended minimum** | **2 GB** |

---

## Optimization Reference

### Key Optimizations Already In Place

1. **dtaidistance C library**: 10-50x faster than numpy DTW fallback
2. **Prescreening (Stage 1)**: Eliminates 80% of symbols before expensive DTW
3. **Snappy compression**: Fast Parquet read/write with decent compression
4. **Sakoe-Chiba band**: Constrains DTW search space (window_ratio=0.12)
5. **Multiprocessing**: Parallel DTW for 5+ candidates
6. **Per-file write locks**: Thread-safe Parquet without global lock
7. **SQLite WAL mode**: Concurrent reads during writes
8. **Incremental data updates**: Only fetch new klines from Binance

### Potential Future Optimizations

| Optimization | Expected Impact | Complexity |
|-------------|----------------|------------|
| In-memory BTC data cache | 2-5x for BTC reads | Low |
| Parquet column pruning (read only close+volume) | 20-30% faster reads | Low |
| NumPy vectorized prescreening | 2-3x for Stage 1 | Medium |
| DTW early abandoning | 10-30% for obvious non-matches | Medium |
| Arrow memory mapping | 30-50% for bulk reads | Medium |
| Process pool reuse across scans | Saves pool creation overhead | Low |

---

## Troubleshooting

### Benchmark Failures

**DTW benchmarks fail or produce no output:**
- Ensure `dtaidistance` is installed: `pip install dtaidistance`
- Check C extension: `python -c "from dtaidistance import dtw; print(dtw.distance([0,1,2], [0,1,2], use_c=True))"`

**Parquet benchmarks are slow:**
- Check disk I/O: `dd if=/dev/zero of=/tmp/test bs=1M count=100`
- Ensure SSD storage, not network-mounted volume

**ML inference benchmarks show "fallback" mode:**
- Train models first: `python scripts/train_model.py --timeframes 4h`
- Or accept fallback results (DTW-score-based heuristic)

### Memory Issues

**Peak RSS keeps growing:**
1. Check image accumulation: `python -m monitoring.resource_monitor`
2. Clean old alert images: keep only last N days
3. Check SQLite size: consider archiving old scan results

**OOM during large scans:**
1. Reduce `prescreening_top_ratio` (e.g., 0.10 instead of 0.20)
2. Limit parallel workers: scan with `num_workers=2`
3. Scan timeframes sequentially instead of in parallel

### Slow Scans

1. Run profiler: `python -m benchmarks.profiler --target scan`
2. Check if dtaidistance C extension is loaded (10-50x difference)
3. Verify prescreening is working (should filter 80%+ of symbols)
4. Check disk I/O during bulk_read_timeframe
5. Monitor with `python -m monitoring.resource_monitor` during scan
