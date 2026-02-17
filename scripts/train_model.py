"""
Train Model CLI — Historical scanning + model training pipeline
================================================================

Usage:
    # Full pipeline (scan history + train models)
    python scripts/train_model.py --timeframes 4h

    # Just scan and generate training CSV (no training)
    python scripts/train_model.py --scan-only --timeframes 4h

    # Just train from existing CSV
    python scripts/train_model.py --train-only

    # Custom params
    python scripts/train_model.py --timeframes 4h --stride 8 --workers 4
"""

import os
import sys
import argparse
import logging
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import SystemConfig
from references.reference_manager import ReferenceManager
from ml.historical_scanner import HistoricalScanner
from ml.trainer import ModelTrainer


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main():
    parser = argparse.ArgumentParser(
        description="Train ML models for crypto pattern recognition"
    )
    parser.add_argument(
        "--timeframes", nargs="+", default=["4h"],
        help="Timeframes to scan (default: 4h)"
    )
    parser.add_argument(
        "--stride", type=int, default=None,
        help="Sliding window stride in bars (default: ref_len // 4)"
    )
    parser.add_argument(
        "--workers", type=int, default=4,
        help="Number of parallel workers (default: 4)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=50,
        help="Symbols per batch (default: 50)"
    )
    parser.add_argument(
        "--scan-only", action="store_true",
        help="Only scan history and generate CSV, skip training"
    )
    parser.add_argument(
        "--train-only", action="store_true",
        help="Only train from existing CSV, skip scanning"
    )
    parser.add_argument(
        "--csv", type=str, default=None,
        help="Path to training CSV (default: data/models/training_data.csv)"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable verbose logging"
    )

    # Model hyperparameters
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--min-samples-leaf", type=int, default=20)
    parser.add_argument("--n-splits", type=int, default=5)

    args = parser.parse_args()
    setup_logging(args.verbose)

    config = SystemConfig()
    config.ensure_directories()

    csv_path = args.csv or os.path.join(config.model_dir, "training_data.csv")

    print("=" * 60)
    print("Crypto Pattern ML Training Pipeline")
    print("=" * 60)
    print(f"Timeframes: {args.timeframes}")
    print(f"Workers: {args.workers}")
    print(f"CSV path: {csv_path}")
    print()

    # === Phase A: Historical Scanning ===
    if not args.train_only:
        ref_mgr = ReferenceManager(config)
        references = ref_mgr.list_references(active_only=True)

        if not references:
            print("ERROR: No active reference patterns found.")
            print("Add references first using ReferenceManager.add_reference()")
            print("See README.md for details.")
            sys.exit(1)

        print(f"Found {len(references)} active reference(s):")
        for ref in references:
            print(f"  - {ref.id}: {ref.symbol} {ref.timeframe} ({ref.label})")
        print()

        scanner = HistoricalScanner(config)
        training_df = scanner.scan_all_references(
            references=references,
            timeframes=args.timeframes,
            stride=args.stride,
            num_workers=args.workers,
            batch_size=args.batch_size,
            output_csv=csv_path,
        )

        if training_df.empty:
            print("ERROR: No training samples generated. Check your references and data.")
            sys.exit(1)

        print(f"\nScanning complete: {len(training_df)} training samples")

        if args.scan_only:
            print(f"Training data saved to: {csv_path}")
            print("Run with --train-only to train models from this CSV.")
            return

    # === Phase B: Model Training ===
    if args.train_only and not os.path.exists(csv_path):
        print(f"ERROR: Training CSV not found: {csv_path}")
        print("Run without --train-only first to generate training data.")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("Training Models")
    print("=" * 60)

    trainer = ModelTrainer(config)

    train_kwargs = {
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "learning_rate": args.learning_rate,
        "subsample": args.subsample,
        "min_samples_leaf": args.min_samples_leaf,
        "n_splits": args.n_splits,
    }

    if args.train_only:
        train_kwargs["training_csv"] = csv_path
    else:
        train_kwargs["training_df"] = training_df

    report = trainer.train(**train_kwargs)

    print(f"\nModels saved to: {config.model_dir}/")
    print("Files:")
    for threshold in config.ml_thresholds:
        model_file = f"gbr_{threshold:.2f}.joblib"
        path = os.path.join(config.model_dir, model_file)
        if os.path.exists(path):
            size_kb = os.path.getsize(path) / 1024
            print(f"  {model_file} ({size_kb:.0f} KB)")
    print(f"  training_meta.json")
    print(f"  training_data.csv ({os.path.getsize(csv_path) / 1024:.0f} KB)")

    print("\nDone! The scanner will now use trained ML models for predictions.")


if __name__ == "__main__":
    main()
