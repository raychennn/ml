"""
Add Sample References — Run this once to populate reference patterns
=====================================================================

Imports initial references from data.initial_references.
This script is for manual re-seeding; normally references are
auto-seeded on system startup via ReferenceManager.seed_initial_references().

Usage:
    python scripts/add_references.py
"""

import sys
import os
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import SystemConfig
from data.data_manager import DataManager
from visualization.chart_generator import ChartGenerator
from telegram_bot.notifier import TelegramNotifier
from references.reference_manager import ReferenceManager
from data.initial_references import INITIAL_REFERENCES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)


def main():
    config = SystemConfig()
    config.ensure_directories()

    dm = DataManager(config)
    cg = ChartGenerator(config)
    notifier = TelegramNotifier(config)
    ref_mgr = ReferenceManager(config, data_manager=dm, chart_generator=cg, notifier=notifier)

    print(f"Timezone: {config.reference_input_timezone}")
    print(f"Existing references: {ref_mgr.count()}\n")

    if not INITIAL_REFERENCES:
        print("No references defined in data/initial_references.py!")
        print("Add patterns there, then re-run this script.")
        return

    added = 0
    for ref_def in INITIAL_REFERENCES:
        try:
            ref = ref_mgr.add_reference_by_datetime(**ref_def)
            print(f"  Added: {ref.id}")
            added += 1
        except ValueError as e:
            print(f"  Skipped: {e}")
        except Exception as e:
            print(f"  Error: {ref_def.get('symbol', '?')} -- {e}")

    print(f"\nDone! Added {added} reference(s).")
    print(f"Total active references: {ref_mgr.count()}")
    print(f"Charts saved to: {config.image_dir}/references/")
    print(f"\nNote: References are auto-seeded on 'python main.py serve'.")
    print(f"      Use Telegram /add_ref for new references going forward.")
    print(f"\nNext step:")
    print(f"  python scripts/train_model.py --timeframes 4h --workers 2")


if __name__ == "__main__":
    main()
