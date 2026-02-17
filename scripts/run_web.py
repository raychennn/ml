#!/usr/bin/env python3
"""
Run Web Dashboard — Start the Flask web server
=================================================

Usage:
    python scripts/run_web.py
    python scripts/run_web.py --port 8080
    python scripts/run_web.py --verbose
"""

import os
import sys
import logging
from logging.handlers import RotatingFileHandler
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import SystemConfig
from data.data_manager import DataManager
from references.reference_manager import ReferenceManager
from core.scanner_engine import ScannerEngine
from core.scan_result_store import ScanResultStore
from visualization.chart_generator import ChartGenerator
from web.app import create_app


def main():
    parser = argparse.ArgumentParser(description="Run crypto pattern web dashboard")
    parser.add_argument("--host", default=None, help="Host to bind (default: from config)")
    parser.add_argument("--port", type=int, default=None, help="Port to bind (default: from config)")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--debug", action="store_true", help="Enable Flask debug mode")
    args = parser.parse_args()

    config = SystemConfig()
    config.ensure_directories()

    level = logging.DEBUG if args.verbose else logging.INFO
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

    dm = DataManager(config)
    chart_gen = ChartGenerator(config)
    ref_mgr = ReferenceManager(config, data_manager=dm, chart_generator=chart_gen)

    # Seed initial references on first startup (silent, no Telegram notifications)
    ref_mgr.seed_initial_references()

    scanner = ScannerEngine(config, dm)
    result_store = ScanResultStore(config)

    app = create_app(
        config, result_store,
        scanner_engine=scanner, reference_manager=ref_mgr,
        data_manager=dm, chart_generator=chart_gen,
    )

    host = args.host or config.web_host
    port = args.port or config.web_port

    print(f"Starting web dashboard at http://{host}:{port}")
    app.run(host=host, port=port, debug=args.debug)


if __name__ == "__main__":
    main()
