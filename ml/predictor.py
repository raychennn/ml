"""
ML Predictor — Load trained models and predict gain probabilities
=================================================================

Loads GBR models from disk and provides inference for live scanning.
Falls back to DTW-score-based placeholder if no models are available.
"""

import os
import json
import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

from config import SystemConfig
from core.feature_extractor import FeatureExtractor, FEATURE_DEFINITIONS

logger = logging.getLogger(__name__)


class MLPredictor:
    """Loads trained models and predicts gain probabilities"""

    def __init__(self, config: SystemConfig):
        self.config = config
        self.model_dir = config.model_dir
        self.feature_extractor = FeatureExtractor(config)
        self.models = {}  # {threshold: model}
        self.feature_names = []
        self._loaded = False

    def load(self) -> bool:
        """
        Load trained models from disk.

        Returns:
            True if models loaded successfully, False otherwise (will use fallback)
        """
        meta_path = os.path.join(self.model_dir, "training_meta.json")
        if not os.path.exists(meta_path):
            logger.info("No training_meta.json found — using DTW-score fallback")
            return False

        try:
            import joblib

            with open(meta_path, "r") as f:
                meta = json.load(f)

            self.feature_names = meta.get("feature_names", list(FEATURE_DEFINITIONS.keys()))

            loaded_count = 0
            for threshold in self.config.ml_thresholds:
                model_path = os.path.join(self.model_dir, f"gbr_{threshold:.2f}.joblib")
                if os.path.exists(model_path):
                    self.models[threshold] = joblib.load(model_path)
                    loaded_count += 1
                else:
                    logger.warning(f"Model file not found: {model_path}")

            if loaded_count > 0:
                self._loaded = True
                logger.info(
                    f"Loaded {loaded_count}/{len(self.config.ml_thresholds)} ML models"
                )
                return True
            else:
                logger.warning("No model files found — using DTW-score fallback")
                return False

        except Exception as e:
            logger.warning(f"Failed to load ML models: {e} — using DTW-score fallback")
            return False

    def predict(
        self,
        match_result,
        window_df: pd.DataFrame,
        btc_df: Optional[pd.DataFrame] = None,
    ) -> Dict[float, float]:
        """
        Predict gain probabilities for a DTW match.

        Args:
            match_result: DTWMatchResult (or any object with score, price_distance, etc.)
            window_df: DataFrame of the matched window
            btc_df: BTC data for market context features

        Returns:
            {threshold: probability} e.g. {0.05: 0.82, 0.10: 0.65, ...}
        """
        if not self._loaded:
            return self._fallback(match_result.score)

        try:
            # Extract features
            features = self.feature_extractor.extract(match_result, window_df, btc_df)

            # Build feature vector in correct order
            feature_vector = np.array([
                features.get(fname, 0.0) for fname in self.feature_names
            ]).reshape(1, -1)

            # Predict with each model
            probabilities = {}
            for threshold in self.config.ml_thresholds:
                if threshold in self.models:
                    pred = self.models[threshold].predict(feature_vector)[0]
                    probabilities[threshold] = float(np.clip(pred, 0, 1))
                else:
                    probabilities[threshold] = self._fallback_single(
                        match_result.score, threshold
                    )

            return probabilities

        except Exception as e:
            logger.debug(f"ML prediction failed: {e}, using fallback")
            return self._fallback(match_result.score)

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @staticmethod
    def _fallback(dtw_score: float) -> Dict[float, float]:
        """DTW-score-based placeholder (same as Phase 2 _ml_placeholder)"""
        return {
            0.05: min(0.95, dtw_score * 1.1),
            0.10: min(0.90, dtw_score * 0.9),
            0.15: min(0.85, dtw_score * 0.7),
            0.20: min(0.80, dtw_score * 0.5),
        }

    @staticmethod
    def _fallback_single(dtw_score: float, threshold: float) -> float:
        """Fallback for a single threshold"""
        multiplier = max(0.3, 1.2 - threshold * 4)
        cap = max(0.5, 1.0 - threshold)
        return min(cap, dtw_score * multiplier)
