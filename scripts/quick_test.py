#!/usr/bin/env python3
"""
Quick Test — 驗證 Phase 1 數據管線
===================================
下載 3 個 symbol 的 4h 數據，驗證 Parquet 讀寫和增量更新。

用法：
    python -m scripts.quick_test
"""

import sys
import os
import time
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import SystemConfig
from data.binance_client import BinanceClient
from data.data_manager import DataManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-5s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

TEST_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
TEST_TIMEFRAME = "4h"
# 只下載最近 30 天（快速測試）
TEST_START_TS = int(time.time()) - 30 * 24 * 3600


def main():
    config = SystemConfig()
    config.data_root = "./data_test"  # 測試用獨立目錄
    config.ensure_directories()

    binance = BinanceClient(config)
    dm = DataManager(config)

    print(f"\n{'='*50}")
    print(f"  PHASE 1 QUICK TEST")
    print(f"  Symbols: {TEST_SYMBOLS}")
    print(f"  Timeframe: {TEST_TIMEFRAME}")
    print(f"  Data root: {config.data_root}")
    print(f"{'='*50}\n")

    # === Test 1: 取得交易對列表 ===
    print("── Test 1: Fetch symbol list ──")
    try:
        all_symbols = binance.get_all_usdt_perpetual_symbols()
        print(f"  ✅ Found {len(all_symbols)} symbols")
        print(f"  First 10: {all_symbols[:10]}")
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        return

    # === Test 2: 下載 K 線 ===
    print("\n── Test 2: Download klines ──")
    for symbol in TEST_SYMBOLS:
        try:
            df = binance.fetch_klines(symbol, TEST_TIMEFRAME, TEST_START_TS)
            if df is not None and not df.empty:
                print(f"  ✅ {symbol}: {len(df)} rows, "
                      f"columns={list(df.columns)}, "
                      f"dtypes: timestamp={df['timestamp'].dtype}, close={df['close'].dtype}")
            else:
                print(f"  ❌ {symbol}: no data returned")
        except Exception as e:
            print(f"  ❌ {symbol}: {e}")

    # === Test 3: 寫入 Parquet ===
    print("\n── Test 3: Write to Parquet ──")
    for symbol in TEST_SYMBOLS:
        df = binance.fetch_klines(symbol, TEST_TIMEFRAME, TEST_START_TS)
        if df is not None and not df.empty:
            dm.write_symbol(symbol, TEST_TIMEFRAME, df)
            # 讀回驗證
            read_back = dm.read_symbol(symbol, TEST_TIMEFRAME)
            match = len(read_back) == len(df)
            status = "✅" if match else "❌"
            print(f"  {status} {symbol}: wrote {len(df)} rows, read back {len(read_back)} rows")

    # === Test 4: 增量更新 ===
    print("\n── Test 4: Incremental update ──")
    symbol = TEST_SYMBOLS[0]
    # 先讀現有數據
    before = dm.read_symbol(symbol, TEST_TIMEFRAME)
    before_count = len(before)

    # 模擬增量更新（重新抓取，應該會 append 0 或少量新行）
    last_ts = dm.get_last_timestamp(symbol, TEST_TIMEFRAME)
    new_df = binance.fetch_klines_incremental(symbol, TEST_TIMEFRAME, last_ts)
    if new_df is not None and not new_df.empty:
        added = dm.incremental_update(symbol, TEST_TIMEFRAME, new_df)
        after = dm.read_symbol(symbol, TEST_TIMEFRAME)
        print(f"  ✅ {symbol}: before={before_count}, new={added}, after={len(after)}")
    else:
        print(f"  ✅ {symbol}: already up-to-date (no new data)")

    # === Test 5: Cache meta ===
    print("\n── Test 5: Cache meta ──")
    for symbol in TEST_SYMBOLS:
        ts = dm.get_last_timestamp(symbol, TEST_TIMEFRAME)
        if ts:
            from datetime import datetime, timezone
            dt = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
            print(f"  ✅ {symbol}: last_ts={ts} ({dt} UTC)")
        else:
            print(f"  ❌ {symbol}: no meta found")

    # === Test 6: Bulk read ===
    print("\n── Test 6: Bulk read timeframe ──")
    bulk = dm.bulk_read_timeframe(TEST_TIMEFRAME)
    print(f"  ✅ Loaded {len(bulk)} symbols")
    for sym, df in bulk.items():
        print(f"     {sym}: {len(df)} rows")

    # === Test 7: Tail read ===
    print("\n── Test 7: Tail read (last 50 bars) ──")
    for symbol in TEST_SYMBOLS:
        tail = dm.read_symbol_tail(symbol, TEST_TIMEFRAME, 50)
        print(f"  ✅ {symbol}: {len(tail)} rows (requested 50)")

    # === Test 8: Integrity check ===
    print("\n── Test 8: Data integrity ──")
    for symbol in TEST_SYMBOLS:
        result = dm.validate_integrity(symbol, TEST_TIMEFRAME)
        status = "✅" if result["valid"] else "❌"
        issues = result.get("issues", [])
        print(f"  {status} {symbol}: {result['rows']} rows, "
              f"range={result.get('first_date', '?')} ~ {result.get('last_date', '?')}, "
              f"age={result.get('age_hours', '?')}h"
              + (f", issues={issues}" if issues else ""))

    # === Test 9: Storage stats ===
    print("\n── Test 9: Storage stats ──")
    stats = dm.get_storage_stats()
    print(f"  Total symbols: {stats['total_symbols']}")
    print(f"  Total rows: {stats['total_rows']:,}")
    print(f"  Total size: {stats['total_size_mb']:.1f} MB")
    for tf, tf_stats in stats["by_timeframe"].items():
        print(f"  {tf}: {tf_stats['symbols']} symbols, {tf_stats['size_mb']:.1f} MB")

    print(f"\n{'='*50}")
    print(f"  ALL TESTS COMPLETE ✅")
    print(f"{'='*50}")
    print(f"\n💡 Next step: run full download with:")
    print(f"   python -m scripts.bulk_download --timeframes 4h")
    print(f"   (takes ~2-3 hours for all symbols)\n")

    # 清理測試目錄提示
    print(f"🧹 Test data saved to: {os.path.abspath(config.data_root)}")
    print(f"   Delete with: rm -rf {config.data_root}\n")


if __name__ == "__main__":
    main()
