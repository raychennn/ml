#!/usr/bin/env python3
"""
Crypto Pattern Recognition System — Entry Point
=================================================

啟動系統的主入口。目前 Phase 1 提供以下 CLI 子命令：

    python main.py status          # 查看數據狀態
    python main.py download        # 開始/繼續下載歷史數據
    python main.py validate        # 驗證數據完整性
"""

import sys
import os
import argparse
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime, timezone

from config import SystemConfig
from data.binance_client import BinanceClient
from data.data_manager import DataManager
from data.symbol_manager import SymbolManager


def setup_logging(config: SystemConfig, verbose: bool = False):
    config.ensure_directories()
    level = logging.DEBUG if verbose else logging.INFO
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)-5s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    # avoid duplicate handlers on re-entry
    if root.handlers:
        return

    # Console handler
    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(fmt)
    root.addHandler(console)

    # Rotating file handler — 10 MB per file, keep 5 backups
    log_file = os.path.join(config.log_dir, "system.log")
    file_handler = RotatingFileHandler(
        log_file, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)


def cmd_status(args, config):
    """顯示系統狀態"""
    dm = DataManager(config)
    stats = dm.get_storage_stats()

    print(f"\n{'='*55}")
    print(f"  CRYPTO PATTERN SYSTEM — STATUS")
    print(f"  Data root: {os.path.abspath(config.data_root)}")
    print(f"{'='*55}")
    print(f"  Total symbols: {stats['total_symbols']}")
    print(f"  Total rows:    {stats['total_rows']:,}")
    print(f"  Total size:    {stats['total_size_mb']:.1f} MB")
    print(f"  {'─'*51}")

    for tf in config.scan_timeframes:
        tf_stats = stats["by_timeframe"].get(tf, {})
        n = tf_stats.get("symbols", 0)
        size = tf_stats.get("size_mb", 0)
        print(f"  {tf:>4s}: {n:>4d} symbols | {size:>8.1f} MB")

    print(f"{'='*55}\n")


def cmd_download(args, config):
    """啟動/繼續下載（委託給 bulk_download）"""
    from scripts.bulk_download import main as bulk_main
    # 將 args 轉為 bulk_download 的 sys.argv 格式
    argv = []
    if args.timeframes:
        argv += ["--timeframes"] + args.timeframes
    if args.symbols:
        argv += ["--symbols"] + args.symbols
    if args.verbose:
        argv += ["--verbose"]
    sys.argv = ["bulk_download"] + argv
    bulk_main()


def cmd_validate(args, config):
    """驗證數據完整性"""
    dm = DataManager(config)
    timeframes = args.timeframes or config.scan_timeframes

    total_issues = 0
    for tf in timeframes:
        symbols = dm.list_symbols_for_timeframe(tf)
        if not symbols:
            print(f"  {tf}: no data")
            continue

        issues = []
        for sym in symbols:
            result = dm.validate_integrity(sym, tf)
            if not result["valid"] or result.get("issues"):
                issues.append((sym, result))

        if issues:
            total_issues += len(issues)
            print(f"\n  {tf}: {len(issues)}/{len(symbols)} symbols have issues:")
            for sym, result in issues[:10]:
                print(f"    {sym}: {result.get('issues', [])}")
            if len(issues) > 10:
                print(f"    ... and {len(issues)-10} more")
        else:
            print(f"  {tf}: {len(symbols)} symbols OK")

    if total_issues == 0:
        print("\n  All data validated successfully.")


def cmd_serve(args, config):
    """啟動 Bot + Scheduler + Web 服務"""
    import signal
    from data.binance_client import BinanceClient
    from data.data_manager import DataManager
    from data.symbol_manager import SymbolManager
    from references.reference_manager import ReferenceManager
    from core.scanner_engine import ScannerEngine
    from core.scan_result_store import ScanResultStore
    from core.scan_orchestrator import ScanOrchestrator
    from telegram_bot.notifier import TelegramNotifier
    from telegram_bot.bot import TelegramBot
    from visualization.chart_generator import ChartGenerator
    from scheduler.scheduler import ScanScheduler
    from web.app import create_app

    binance = BinanceClient(config)
    dm = DataManager(config)
    sm = SymbolManager(config, binance_client=binance)
    notifier = TelegramNotifier(config)
    chart_gen = ChartGenerator(config)
    ref_mgr = ReferenceManager(config, data_manager=dm, chart_generator=chart_gen, notifier=notifier)

    # Seed initial references on first startup (silent, no Telegram notifications)
    ref_mgr.seed_initial_references()

    scanner = ScannerEngine(config, dm)
    result_store = ScanResultStore(config)

    # Shared orchestrator — single source of truth for scan pipeline
    orchestrator = ScanOrchestrator(
        config, scanner, ref_mgr, dm, binance, sm,
        notifier, result_store, chart_gen,
    )

    bot = TelegramBot(
        config, scanner, ref_mgr, notifier, result_store,
        data_manager=dm, binance_client=binance, symbol_manager=sm,
        scan_orchestrator=orchestrator,
    )

    sched = None
    components = args.components or "all"

    if components in ("all", "bot"):
        print("Starting Telegram bot...")
        bot.start()

        sched = ScanScheduler(
            config, orchestrator, notifier,
            data_manager=dm, symbol_manager=sm,
        )
        print("Starting scan scheduler...")
        sched.start()

    if components in ("all", "web"):
        app = create_app(
            config, result_store,
            scanner_engine=scanner, reference_manager=ref_mgr,
            data_manager=dm, chart_generator=chart_gen,
        )
        host = args.host or config.web_host
        port = args.port or config.web_port
        print(f"Starting web dashboard at http://{host}:{port}")
        app.run(host=host, port=port, debug=False, use_reloader=False)
    elif components == "bot":
        def shutdown(signum, frame):
            print("\nShutting down...")
            bot.stop()
            orchestrator.stop()
            if sched:
                sched.stop()
            sys.exit(0)
        signal.signal(signal.SIGINT, shutdown)
        signal.signal(signal.SIGTERM, shutdown)
        signal.pause()


def main():
    parser = argparse.ArgumentParser(
        description="Crypto Pattern Recognition System",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--data-root", default=None)
    sub = parser.add_subparsers(dest="command")

    # status
    sub.add_parser("status", help="Show data storage status")

    # download
    dl = sub.add_parser("download", help="Download historical data")
    dl.add_argument("--timeframes", nargs="+", default=None)
    dl.add_argument("--symbols", nargs="+", default=None)

    # validate
    val = sub.add_parser("validate", help="Validate data integrity")
    val.add_argument("--timeframes", nargs="+", default=None)

    # serve
    srv = sub.add_parser("serve", help="Start bot and/or web dashboard")
    srv.add_argument("--components", choices=["all", "bot", "web"], default="all",
                     help="Which components to start (default: all)")
    srv.add_argument("--host", default=None, help="Web server host")
    srv.add_argument("--port", type=int, default=None, help="Web server port")

    args = parser.parse_args()

    config = SystemConfig()
    if args.data_root:
        config.data_root = args.data_root
    config.ensure_directories()
    setup_logging(config, verbose=args.verbose)

    if args.command == "status":
        cmd_status(args, config)
    elif args.command == "download":
        cmd_download(args, config)
    elif args.command == "validate":
        cmd_validate(args, config)
    elif args.command == "serve":
        cmd_serve(args, config)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
