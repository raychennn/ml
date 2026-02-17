"""
Model Trainer — GradientBoostingRegressor ensemble for gain prediction
======================================================================

Trains one GBR model per gain threshold (5%, 10%, 15%, 20%).
Uses TimeSeriesSplit cross-validation to avoid look-ahead bias.
"""

import os
import json
import logging
import time
from datetime import datetime, timezone
from typing import List, Optional, Dict

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error

from config import SystemConfig
from core.feature_extractor import FEATURE_DEFINITIONS

logger = logging.getLogger(__name__)

# Features used for training (must match FeatureExtractor output)
TRAINING_FEATURES = list(FEATURE_DEFINITIONS.keys())


class ModelTrainer:
    """Trains GBR models for gain probability prediction"""

    def __init__(self, config: SystemConfig):
        self.config = config
        self.model_dir = config.model_dir

    def train(
        self,
        training_csv: Optional[str] = None,
        training_df: Optional[pd.DataFrame] = None,
        n_estimators: int = 300,
        max_depth: int = 4,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        min_samples_leaf: int = 20,
        n_splits: int = 5,
    ) -> Dict:
        """
        Train models from training data.

        Args:
            training_csv: Path to training CSV (alternative to training_df)
            training_df: Training DataFrame (alternative to training_csv)
            n_estimators: Number of boosting stages
            max_depth: Max tree depth
            learning_rate: Learning rate
            subsample: Fraction of samples per tree
            min_samples_leaf: Min samples in leaf node
            n_splits: TimeSeriesSplit folds

        Returns:
            Training report dict
        """
        train_start = time.time()

        # Load data
        if training_df is not None:
            df = training_df
        elif training_csv:
            df = pd.read_csv(training_csv)
        else:
            raise ValueError("Either training_csv or training_df must be provided")

        if df.empty:
            raise ValueError("Training data is empty")

        # Sort by time
        if "match_end_ts" in df.columns:
            df = df.sort_values("match_end_ts").reset_index(drop=True)

        # Identify available features
        available_features = [f for f in TRAINING_FEATURES if f in df.columns]
        if len(available_features) < 5:
            raise ValueError(
                f"Too few features available ({len(available_features)}). "
                f"Expected features: {TRAINING_FEATURES}"
            )

        logger.info(f"Training with {len(df)} samples, {len(available_features)} features")
        logger.info(f"Features: {available_features}")

        # Prepare feature matrix
        X = df[available_features].fillna(0).values

        # Train one model per threshold
        os.makedirs(self.model_dir, exist_ok=True)
        thresholds = self.config.ml_thresholds
        report = {
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "n_samples": len(df),
            "n_features": len(available_features),
            "feature_names": available_features,
            "thresholds": {},
            "params": {
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "learning_rate": learning_rate,
                "subsample": subsample,
                "min_samples_leaf": min_samples_leaf,
                "n_splits": n_splits,
            },
        }

        for threshold in thresholds:
            label_col = f"label_{threshold}"
            if label_col not in df.columns:
                logger.warning(f"Label column {label_col} not found, skipping threshold {threshold}")
                continue

            y = df[label_col].fillna(0).values

            logger.info(f"\n--- Training model for threshold {threshold:.0%} ---")
            logger.info(f"  Label stats: mean={y.mean():.3f}, std={y.std():.3f}, "
                        f"zeros={np.sum(y == 0)}, ones={np.sum(y >= 1.0)}")

            # Cross-validation
            tscv = TimeSeriesSplit(n_splits=n_splits)
            cv_mae_scores = []
            cv_rmse_scores = []

            for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]

                model = GradientBoostingRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    learning_rate=learning_rate,
                    subsample=subsample,
                    min_samples_leaf=min_samples_leaf,
                    random_state=42,
                )
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
                y_pred = np.clip(y_pred, 0, 1)

                mae = mean_absolute_error(y_val, y_pred)
                rmse = np.sqrt(mean_squared_error(y_val, y_pred))
                cv_mae_scores.append(mae)
                cv_rmse_scores.append(rmse)
                logger.info(f"  Fold {fold+1}: MAE={mae:.4f}, RMSE={rmse:.4f}")

            # Train final model on all data
            final_model = GradientBoostingRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                subsample=subsample,
                min_samples_leaf=min_samples_leaf,
                random_state=42,
            )
            final_model.fit(X, y)

            # Feature importances
            importances = dict(zip(available_features, final_model.feature_importances_))
            sorted_imp = sorted(importances.items(), key=lambda x: x[1], reverse=True)

            # Save model
            model_path = os.path.join(self.model_dir, f"gbr_{threshold:.2f}.joblib")
            joblib.dump(final_model, model_path)

            threshold_report = {
                "cv_mae_mean": float(np.mean(cv_mae_scores)),
                "cv_mae_std": float(np.std(cv_mae_scores)),
                "cv_rmse_mean": float(np.mean(cv_rmse_scores)),
                "cv_rmse_std": float(np.std(cv_rmse_scores)),
                "label_mean": float(y.mean()),
                "label_std": float(y.std()),
                "feature_importances": {k: float(v) for k, v in sorted_imp},
                "model_path": model_path,
            }
            report["thresholds"][str(threshold)] = threshold_report

            logger.info(f"  CV MAE: {np.mean(cv_mae_scores):.4f} +/- {np.std(cv_mae_scores):.4f}")
            logger.info(f"  Top 5 features:")
            for fname, fimp in sorted_imp[:5]:
                logger.info(f"    {fname}: {fimp:.4f}")
            logger.info(f"  Model saved: {model_path}")

        # Save training metadata
        meta_path = os.path.join(self.model_dir, "training_meta.json")
        with open(meta_path, "w") as f:
            json.dump(report, f, indent=2)
        logger.info(f"\nTraining metadata saved: {meta_path}")

        elapsed = time.time() - train_start
        report["training_time_seconds"] = elapsed
        logger.info(f"Total training time: {elapsed:.1f}s")

        self._print_report(report)
        return report

    @staticmethod
    def _print_report(report: Dict):
        """Print human-readable training report"""
        print("\n" + "=" * 60)
        print("TRAINING REPORT")
        print("=" * 60)
        print(f"Samples: {report['n_samples']}")
        print(f"Features: {report['n_features']}")
        print(f"Time: {report.get('training_time_seconds', 0):.1f}s")
        print()

        for threshold_str, info in report.get("thresholds", {}).items():
            threshold = float(threshold_str)
            print(f"--- Threshold: {threshold:.0%} gain ---")
            print(f"  CV MAE:  {info['cv_mae_mean']:.4f} +/- {info['cv_mae_std']:.4f}")
            print(f"  CV RMSE: {info['cv_rmse_mean']:.4f} +/- {info['cv_rmse_std']:.4f}")
            print(f"  Label mean: {info['label_mean']:.3f}, std: {info['label_std']:.3f}")
            print(f"  Top features:")
            top_features = list(info["feature_importances"].items())[:5]
            for fname, fimp in top_features:
                print(f"    {fname}: {fimp:.4f}")
            print()

        print("=" * 60)
