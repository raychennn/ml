#!/usr/bin/env python3
"""
Run Telegram Bot — Start the Telegram bot + scheduler (no web)
================================================================

Usage:
    python scripts/run_bot.py
    python scripts/run_bot.py --verbose
    python scripts/run_bot.py --no-scheduler
"""

import os
import sys
import signal
import logging
from logging.handlers import RotatingFileHandler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import SystemConfig
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


def main():
    no_scheduler = "--no-scheduler" in sys.argv
    verbose = "--verbose" in sys.argv or "-v" in sys.argv

    config = SystemConfig()
    config.ensure_directories()

    level = logging.DEBUG if verbose else logging.INFO
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(fmt)
    root.addHandler(console)
    file_handler = RotatingFileHandler(
        os.path.join(config.log_dir, "system.log"),
        maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)

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

    def shutdown(signum, frame):
        print("\nShutting down...")
        bot.stop()
        orchestrator.stop()
        if sched:
            sched.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    print("Starting Telegram bot...")
    print(f"Bot token configured: {'Yes' if config.telegram_bot_token else 'No'}")
    print(f"Chat ID configured: {'Yes' if config.telegram_chat_id else 'No'}")
    bot.start()

    if not no_scheduler:
        sched = ScanScheduler(
            config, orchestrator, notifier,
            data_manager=dm, symbol_manager=sm,
        )
        print("Starting scan scheduler...")
        sched.start()

    # Keep main thread alive
    signal.pause()


if __name__ == "__main__":
    main()
