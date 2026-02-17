#!/usr/bin/env python3
"""
Phase 2 Integration Test
=========================
用已下載的 4h 數據測試整個掃描 pipeline：
1. 建立一個 reference pattern
2. 跑 3-stage 掃描
3. 生成圖表
"""

import sys
import os
import time
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import SystemConfig
from data.data_manager import DataManager
from core.data_normalizer import DataNormalizer
from core.dtw_calculator import DTWCalculator
from core.scanner_engine import ScannerEngine
from references.reference_manager import ReferenceManager, ReferencePattern
from visualization.chart_generator import ChartGenerator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-5s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    config = SystemConfig()
    config.ensure_directories()

    dm = DataManager(config)
    normalizer = DataNormalizer(config)
    dtw_calc = DTWCalculator(config)
    ref_mgr = ReferenceManager(config)
    chart_gen = ChartGenerator(config)
    scanner = ScannerEngine(config, dm)

    print(f"\n{'='*60}")
    print(f"  PHASE 2 INTEGRATION TEST")
    print(f"  Data root: {config.data_root}")
    print(f"{'='*60}\n")

    # === Test 1: DataNormalizer ===
    print("── Test 1: DataNormalizer ──")
    df = dm.read_symbol_tail("BTCUSDT", "4h", 100)
    if df.empty:
        print("  ❌ No BTCUSDT 4h data. Run bulk_download first.")
        return

    prepared = normalizer.prepare_for_dtw(df)
    print(f"  ✅ close_znorm shape: {prepared['close_znorm'].shape}")
    print(f"  ✅ slope shape: {prepared['slope'].shape}")
    print(f"  ✅ SMA diffs: {list(prepared['sma_diffs_znorm'].keys())}")
    sma_cols = [c for c in prepared['df'].columns if c.startswith('SMA_')]
    print(f"  ✅ SMA columns in df: {sma_cols}")

    # === Test 2: Reference Manager ===
    print("\n── Test 2: Reference Manager ──")
    # 用 BTCUSDT 最近 50 根 K 棒作為 reference
    btc_df = dm.read_symbol("BTCUSDT", "4h")
    if len(btc_df) < 100:
        print("  ❌ Not enough BTC data")
        return

    # 取中間一段作為 reference（不是尾部，這樣掃描其他幣時有意義）
    mid = len(btc_df) // 2
    ref_start_ts = int(btc_df.iloc[mid]["timestamp"])
    ref_end_ts = int(btc_df.iloc[mid + 50]["timestamp"])

    try:
        ref = ref_mgr.add_reference(
            symbol="BTCUSDT",
            timeframe="4h",
            start_ts=ref_start_ts,
            end_ts=ref_end_ts,
            label="test_pattern",
            description="Phase 2 test reference",
        )
        print(f"  ✅ Added reference: {ref.id}")
    except ValueError:
        # 已存在
        ref = ref_mgr.get_reference("BTC_4h_test_pattern")
        print(f"  ✅ Reference already exists: {ref.id}")

    refs = ref_mgr.list_references()
    print(f"  ✅ Total references: {len(refs)}")

    # === Test 3: DTW Calculator (direct) ===
    print("\n── Test 3: DTW Calculator ──")
    # 用 BTC reference 對 ETH 做 DTW
    ref_df = dm.read_symbol("BTCUSDT", "4h", start_ts=ref_start_ts, end_ts=ref_end_ts)
    eth_df = dm.read_symbol_tail("ETHUSDT", "4h", 100)

    if not ref_df.empty and not eth_df.empty:
        ref_prep = normalizer.prepare_for_dtw(ref_df)
        eth_prep = normalizer.prepare_for_dtw(eth_df)

        result = dtw_calc.compute_similarity(
            ref_close_znorm=ref_prep["close_znorm"],
            ref_sma_diffs_znorm=ref_prep["sma_diffs_znorm"],
            ref_slope=ref_prep["slope"],
            target_close_znorm=eth_prep["close_znorm"],
            target_sma_diffs_znorm=eth_prep["sma_diffs_znorm"],
            target_slope=eth_prep["slope"],
            symbol="ETHUSDT",
            timeframe="4h",
            target_df=eth_prep["df"],
        )
        if result:
            print(f"  ✅ ETHUSDT vs BTC ref: score={result.score:.4f}, "
                  f"price_dist={result.price_distance:.4f}, "
                  f"diff_dist={result.diff_distance:.4f}, "
                  f"scale={result.best_scale_factor}")
        else:
            print(f"  ⚠️  ETHUSDT: below similarity threshold (expected for random ref)")
    else:
        print("  ❌ Missing data for DTW test")

    # === Test 4: Scanner Engine (limited to 20 symbols) ===
    print("\n── Test 4: Scanner Engine (20 symbols) ──")
    # 限制 symbols 以加速測試
    test_symbols = dm.list_symbols_for_timeframe("4h")[:20]
    start_time = time.time()

    # 暫時用一個小數據集測試
    alerts = scanner.scan_timeframe(
        timeframe="4h",
        references=refs,
        num_workers=2,
    )
    elapsed = time.time() - start_time

    print(f"  ✅ Scan complete in {elapsed:.1f}s")
    print(f"  ✅ Found {len(alerts)} alerts")
    if alerts:
        for a in alerts[:5]:
            print(f"     {a.symbol}: score={a.dtw_score:.4f}, ref={a.reference.id}")

    # === Test 5: Chart Generator ===
    print("\n── Test 5: Chart Generator ──")
    if alerts:
        alert = alerts[0]
        if alert.window_data is not None and not alert.window_data.empty:
            chart_path = chart_gen.generate_alert_chart(
                window_df=alert.window_data,
                symbol=alert.symbol,
                timeframe=alert.timeframe,
                ref_name=alert.reference.label,
                score=alert.dtw_score,
                ml_probs=alert.ml_probabilities,
            )
            if chart_path:
                size_kb = os.path.getsize(chart_path) / 1024
                print(f"  ✅ Alert chart: {chart_path} ({size_kb:.0f} KB)")
            else:
                print("  ❌ Chart generation returned empty path")
        else:
            print("  ⚠️  No window_data in alert, generating from raw data")
            # 用 ETH 的最後 50 根 K 棒生成
            eth_tail = dm.read_symbol_tail("ETHUSDT", "4h", 50)
            chart_path = chart_gen.generate_alert_chart(
                window_df=eth_tail,
                symbol="ETHUSDT",
                timeframe="4h",
                ref_name="test",
                score=0.5,
                ml_probs={0.05: 0.7, 0.10: 0.5, 0.15: 0.3, 0.20: 0.2},
            )
            if chart_path:
                size_kb = os.path.getsize(chart_path) / 1024
                print(f"  ✅ Fallback chart: {chart_path} ({size_kb:.0f} KB)")
    else:
        # 沒有 alert，直接測試圖表生成
        eth_tail = dm.read_symbol_tail("ETHUSDT", "4h", 50)
        chart_path = chart_gen.generate_alert_chart(
            window_df=eth_tail,
            symbol="ETHUSDT",
            timeframe="4h",
            ref_name="test",
            score=0.5,
            ml_probs={0.05: 0.7, 0.10: 0.5, 0.15: 0.3, 0.20: 0.2},
        )
        if chart_path and os.path.exists(chart_path):
            size_kb = os.path.getsize(chart_path) / 1024
            print(f"  ✅ Chart generated: {chart_path} ({size_kb:.0f} KB)")
        else:
            print(f"  ❌ Chart generation failed")

    # === Cleanup test reference ===
    ref_mgr.hard_delete_reference("BTC_4h_test_pattern")

    print(f"\n{'='*60}")
    print(f"  PHASE 2 TESTS COMPLETE")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
