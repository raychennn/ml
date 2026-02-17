#!/usr/bin/env python3
"""
Bulk Historical Data Downloader
================================
一次性從 Binance Futures 下載所有 USDT 永續合約的歷史 K 線數據。

特色：
- 斷點續傳：已下載的 symbol 自動跳過（基於 cache_meta）
- 進度追蹤：即時顯示完成百分比、預估剩餘時間
- 可指定時框：只下載需要的時框（節省時間）
- 可指定 symbol：只下載特定幣（測試用）
- 批次休息：每批之間自動暫停，避免 rate limit

預估時間（~250 symbols, 全歷史）：
    4h  → ~2-3 小時
    2h  → ~3-4 小時
    1h  → ~5-7 小時
    30m → ~8-12 小時
    全部 → ~18-26 小時（建議先跑 4h，確認正常後再跑其他）

用法：
    # 先測試：只下載 3 個 symbol 的 4h 數據
    python -m scripts.bulk_download --timeframes 4h --symbols BTCUSDT ETHUSDT SOLUSDT

    # 正式：下載所有 symbol 的 4h 數據
    python -m scripts.bulk_download --timeframes 4h

    # 下載所有時框（建議用 tmux/screen 背景執行）
    python -m scripts.bulk_download

    # 只下載缺失的（斷點續傳）
    python -m scripts.bulk_download --resume

    # 查看目前下載狀態
    python -m scripts.bulk_download --stats-only
"""

import sys
import os
import time
import argparse
import logging
from datetime import datetime, timedelta, timezone

# 將專案根目錄加入 path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import SystemConfig
from data.binance_client import BinanceClient
from data.data_manager import DataManager

# ================================================================
# Logging 設定
# ================================================================

def setup_logging(config: SystemConfig, verbose: bool = False):
    """設定 logging：同時輸出到 console 和檔案"""
    config.ensure_directories()

    log_file = os.path.join(config.log_dir, "bulk_download.log")
    level = logging.DEBUG if verbose else logging.INFO

    # 格式
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)-5s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(fmt)

    # File handler
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)

    # Root logger
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.addHandler(console)
    root.addHandler(file_handler)

    return logging.getLogger(__name__)


# ================================================================
# 下載邏輯
# ================================================================

def download_timeframe(
    binance: BinanceClient,
    dm: DataManager,
    config: SystemConfig,
    timeframe: str,
    symbols: list,
    resume: bool = True,
    logger: logging.Logger = None,
) -> dict:
    """
    下載單一時框的所有 symbol

    Returns:
        {"downloaded": N, "skipped": N, "failed": N, "failed_symbols": [...]}
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    total = len(symbols)
    downloaded = 0
    skipped = 0
    failed = 0
    failed_symbols = []

    # 計時
    tf_start = time.time()
    batch_times = []  # 用於預估剩餘時間

    logger.info(f"{'='*60}")
    logger.info(f"TIMEFRAME: {timeframe} | {total} symbols to process")
    logger.info(f"{'='*60}")

    for i, symbol in enumerate(symbols):
        sym_start = time.time()
        progress = f"[{i+1}/{total}]"

        # === 斷點續傳 ===
        if resume:
            last_ts = dm.get_last_timestamp(symbol, timeframe)
            if last_ts is not None:
                # 檢查是否已經是最新（最後 K 棒在 2 小時以內）
                age_hours = (time.time() - last_ts) / 3600
                if age_hours < 2:
                    logger.debug(f"{progress} {symbol} {timeframe}: up-to-date, skip")
                    skipped += 1
                    continue

                # 增量更新
                logger.info(f"{progress} {symbol} {timeframe}: incremental update (last: {datetime.fromtimestamp(last_ts, tz=timezone.utc).strftime('%Y-%m-%d %H:%M')})")
                new_df = binance.fetch_klines_incremental(symbol, timeframe, last_ts)
                if new_df is not None and not new_df.empty:
                    added = dm.incremental_update(symbol, timeframe, new_df)
                    logger.info(f"  → +{added} new rows")
                    downloaded += 1
                else:
                    logger.info(f"  → no new data")
                    skipped += 1

                _log_eta(batch_times, sym_start, i, total, logger)
                continue

        # === 全量下載 ===
        logger.info(f"{progress} {symbol} {timeframe}: full download...")

        df = binance.fetch_klines(
            symbol=symbol,
            timeframe=timeframe,
            start_ts=1577836800,  # 2020-01-01 UTC
        )

        if df is not None and not df.empty:
            dm.write_symbol(symbol, timeframe, df)
            first_date = datetime.fromtimestamp(df["timestamp"].min(), tz=timezone.utc).strftime("%Y-%m-%d")
            last_date = datetime.fromtimestamp(df["timestamp"].max(), tz=timezone.utc).strftime("%Y-%m-%d")
            logger.info(f"  → {len(df)} rows ({first_date} ~ {last_date})")
            downloaded += 1
        else:
            logger.warning(f"  → FAILED (no data returned)")
            failed += 1
            failed_symbols.append(symbol)

        _log_eta(batch_times, sym_start, i, total, logger)

        # === 批次休息 ===
        if (i + 1) % config.download_batch_size == 0:
            logger.info(f"--- Batch pause ({config.download_batch_sleep}s) ---")
            time.sleep(config.download_batch_sleep)

    # 時框完成
    tf_elapsed = time.time() - tf_start
    logger.info(f"\n{'='*60}")
    logger.info(f"TIMEFRAME {timeframe} COMPLETE in {_fmt_duration(tf_elapsed)}")
    logger.info(f"  Downloaded: {downloaded}")
    logger.info(f"  Skipped:    {skipped}")
    logger.info(f"  Failed:     {failed}")
    if failed_symbols:
        logger.info(f"  Failed symbols: {', '.join(failed_symbols[:20])}")
    logger.info(f"{'='*60}\n")

    return {
        "downloaded": downloaded,
        "skipped": skipped,
        "failed": failed,
        "failed_symbols": failed_symbols,
    }


def _log_eta(batch_times, sym_start, current_idx, total, logger):
    """計算並 log 預估剩餘時間"""
    elapsed = time.time() - sym_start
    batch_times.append(elapsed)

    # 每 10 個 symbol 輸出一次 ETA
    if (current_idx + 1) % 10 == 0 and len(batch_times) >= 5:
        avg_time = sum(batch_times[-20:]) / len(batch_times[-20:])
        remaining = total - (current_idx + 1)
        eta_seconds = avg_time * remaining
        logger.info(
            f"  ⏱ Progress: {current_idx+1}/{total} "
            f"({(current_idx+1)/total*100:.1f}%) | "
            f"Avg: {avg_time:.1f}s/sym | "
            f"ETA: {_fmt_duration(eta_seconds)}"
        )


def _fmt_duration(seconds: float) -> str:
    """格式化時間長度"""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        return f"{h}h {m}m"


def print_stats(dm: DataManager, config: SystemConfig):
    """印出目前的下載統計"""
    stats = dm.get_storage_stats()
    print(f"\n{'='*50}")
    print(f"  DATA STORAGE STATISTICS")
    print(f"{'='*50}")
    print(f"  Total symbols: {stats['total_symbols']}")
    print(f"  Total rows:    {stats['total_rows']:,}")
    print(f"  Total size:    {stats['total_size_mb']:.1f} MB")
    print(f"  {'─'*46}")

    for tf in config.scan_timeframes:
        tf_stats = stats["by_timeframe"].get(tf, {})
        n = tf_stats.get("symbols", 0)
        size = tf_stats.get("size_mb", 0)
        print(f"  {tf:>4s}: {n:>4d} symbols | {size:>8.1f} MB")

    print(f"{'='*50}\n")


# ================================================================
# CLI 入口
# ================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Bulk download Binance Futures historical data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 測試：只下載 3 個 symbol
  python -m scripts.bulk_download --timeframes 4h --symbols BTCUSDT ETHUSDT SOLUSDT

  # 正式：下載所有 4h 數據
  python -m scripts.bulk_download --timeframes 4h

  # 全部時框（建議背景執行）
  python -m scripts.bulk_download

  # 查看統計
  python -m scripts.bulk_download --stats-only
        """,
    )
    parser.add_argument(
        "--timeframes", nargs="+", default=None,
        help="指定要下載的時框 (預設: 30m 1h 2h 4h)",
    )
    parser.add_argument(
        "--symbols", nargs="+", default=None,
        help="指定要下載的 symbol (預設: 全部 USDT 永續合約)",
    )
    parser.add_argument(
        "--resume", action="store_true", default=True,
        help="啟用斷點續傳 (預設: 啟用)",
    )
    parser.add_argument(
        "--no-resume", action="store_true",
        help="停用斷點續傳，全部重新下載",
    )
    parser.add_argument(
        "--stats-only", action="store_true",
        help="只顯示目前下載統計，不執行下載",
    )
    parser.add_argument(
        "--data-root", default=None,
        help="數據根目錄 (預設: ./data)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="顯示 debug 日誌",
    )

    args = parser.parse_args()

    # === 配置 ===
    config = SystemConfig()
    if args.data_root:
        config.data_root = args.data_root
    config.ensure_directories()

    logger = setup_logging(config, verbose=args.verbose)

    # === 初始化 ===
    dm = DataManager(config)
    binance = BinanceClient(config)

    # === 統計模式 ===
    if args.stats_only:
        print_stats(dm, config)
        return

    # === 時框 ===
    timeframes = args.timeframes or config.scan_timeframes
    for tf in timeframes:
        config.timeframe_to_binance_interval(tf)  # 驗證

    # === Symbol 列表 ===
    if args.symbols:
        symbols = args.symbols
        logger.info(f"Using specified symbols: {symbols}")
    else:
        logger.info("Fetching all USDT perpetual symbols from Binance...")
        symbols = binance.get_all_usdt_perpetual_symbols()
        logger.info(f"Found {len(symbols)} symbols")

    # === Resume 設定 ===
    resume = not args.no_resume
    if resume:
        logger.info("Resume mode: ON (will skip up-to-date symbols)")
    else:
        logger.info("Resume mode: OFF (will re-download everything)")

    # === 開始下載 ===
    overall_start = time.time()
    all_results = {}

    logger.info(f"\n{'#'*60}")
    logger.info(f"  BULK DOWNLOAD START")
    logger.info(f"  Timeframes: {timeframes}")
    logger.info(f"  Symbols: {len(symbols)}")
    logger.info(f"  Resume: {resume}")
    logger.info(f"  Data root: {config.data_root}")
    logger.info(f"{'#'*60}\n")

    for tf in timeframes:
        result = download_timeframe(
            binance=binance,
            dm=dm,
            config=config,
            timeframe=tf,
            symbols=symbols,
            resume=resume,
            logger=logger,
        )
        all_results[tf] = result

    # === 完成摘要 ===
    overall_elapsed = time.time() - overall_start
    logger.info(f"\n{'#'*60}")
    logger.info(f"  BULK DOWNLOAD COMPLETE")
    logger.info(f"  Total time: {_fmt_duration(overall_elapsed)}")
    logger.info(f"{'#'*60}")

    for tf, result in all_results.items():
        logger.info(
            f"  {tf}: downloaded={result['downloaded']}, "
            f"skipped={result['skipped']}, "
            f"failed={result['failed']}"
        )

    # 印出最終統計
    print_stats(dm, config)

    # 如果有失敗的，提示用戶
    total_failed = sum(r["failed"] for r in all_results.values())
    if total_failed > 0:
        logger.warning(
            f"\n⚠️  {total_failed} downloads failed. "
            f"Re-run with --resume to retry."
        )


if __name__ == "__main__":
    main()
