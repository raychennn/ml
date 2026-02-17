"""
Historical Scanner — Slide references across full history to generate training data
====================================================================================

For each reference pattern, slides it across ALL symbols' full historical data,
finds DTW matches, computes forward labels, and extracts features for ML training.
"""

import os
import logging
import time
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed

from config import SystemConfig
from data.data_manager import DataManager
from core.data_normalizer import DataNormalizer
from core.feature_extractor import FeatureExtractor
from references.reference_manager import ReferencePattern

logger = logging.getLogger(__name__)


@dataclass
class HistoricalMatch:
    """A single historical match with features and labels"""
    symbol: str
    timeframe: str
    ref_id: str
    match_end_ts: int
    dtw_score: float
    features: Dict[str, float]
    labels: Dict[str, float]  # {threshold: soft_label}


def _scan_symbol_worker(args: tuple) -> List[dict]:
    """
    Multiprocessing worker: scan one symbol's full history against one reference.

    args = (symbol_path, ref_close_znorm, ref_len, stride, dtw_threshold,
            euclidean_prescreen_ratio, config_dict, timeframe,
            ml_thresholds, ml_extension_factors)
    """
    (symbol, symbol_data_bytes, ref_close_znorm, ref_len, stride,
     euclidean_prescreen_ratio, config_dict, timeframe,
     ml_thresholds, ml_extension_factors, ref_id) = args

    try:
        # Deserialize symbol data
        import pickle
        symbol_df = pickle.loads(symbol_data_bytes)

        from config import SystemConfig
        config = SystemConfig()
        for k, v in config_dict.items():
            if hasattr(config, k):
                setattr(config, k, v)

        from core.data_normalizer import DataNormalizer
        normalizer = DataNormalizer(config)

        close_all = symbol_df["close"].values
        n = len(close_all)

        if n < ref_len * 2:
            return []

        matches = []
        # Maximum forward extension for labels
        max_forward = int(ref_len * max(ml_extension_factors))

        # Start after SMA warmup to ensure accurate SMA features
        sma_warmup = max(config_dict.get('sma_periods', [60]))
        start_from = max(sma_warmup, 0)

        # Slide window across history
        for start_idx in range(start_from, n - ref_len - max_forward, stride):
            end_idx = start_idx + ref_len
            window_close = close_all[start_idx:end_idx]

            # Z-normalize window
            std = np.std(window_close)
            if std < 1e-10:
                continue
            window_znorm = (window_close - np.mean(window_close)) / std

            # Quick Euclidean prescreen
            if len(window_znorm) != len(ref_close_znorm):
                window_znorm = np.interp(
                    np.linspace(0, 1, len(ref_close_znorm)),
                    np.linspace(0, 1, len(window_znorm)),
                    window_znorm,
                )

            euc_dist = np.sqrt(np.sum((ref_close_znorm - window_znorm) ** 2)) / ref_len
            if euc_dist > euclidean_prescreen_ratio:
                continue

            # Full DTW computation
            try:
                from dtaidistance import dtw as dtw_lib
                dtw_dist = dtw_lib.distance(
                    ref_close_znorm.astype(np.double),
                    window_znorm.astype(np.double),
                    window=max(2, int(ref_len * config.dtw_window_ratio)),
                    max_dist=config.dtw_max_point_distance * ref_len,
                )
            except Exception:
                # Fallback: simple normalized distance
                dtw_dist = euc_dist * ref_len

            if dtw_dist is None or np.isinf(dtw_dist):
                continue

            # Convert distance to score [0, 1]
            norm_dist = dtw_dist / ref_len
            score = max(0.0, 1.0 - norm_dist)

            if score < config.min_similarity_score:
                continue

            # Compute forward labels
            match_end_price = close_all[end_idx - 1]
            match_end_ts = int(symbol_df["timestamp"].iloc[end_idx - 1])

            labels = {}
            for threshold in ml_thresholds:
                horizons_met = 0
                horizons_total = 0
                for ext_factor in ml_extension_factors:
                    forward_len = max(1, int(ref_len * ext_factor))
                    forward_end = end_idx + forward_len
                    if forward_end > n:
                        continue
                    horizons_total += 1
                    forward_max = np.max(close_all[end_idx:forward_end])
                    gain = (forward_max / match_end_price - 1) if match_end_price > 0 else 0
                    if gain >= threshold:
                        horizons_met += 1

                if horizons_total > 0:
                    labels[threshold] = horizons_met / horizons_total
                else:
                    labels[threshold] = 0.0

            matches.append({
                "symbol": symbol,
                "timeframe": timeframe,
                "ref_id": ref_id,
                "match_start_idx": start_idx,
                "match_end_idx": end_idx,
                "match_end_ts": match_end_ts,
                "dtw_score": score,
                "euc_dist": euc_dist,
                "labels": labels,
            })

        # Deduplication: keep best score within ref_len//2 neighborhood
        if matches:
            matches.sort(key=lambda m: m["match_end_idx"])
            deduped = []
            for m in matches:
                if not deduped:
                    deduped.append(m)
                    continue
                if m["match_end_idx"] - deduped[-1]["match_end_idx"] < ref_len // 2:
                    if m["dtw_score"] > deduped[-1]["dtw_score"]:
                        deduped[-1] = m
                else:
                    deduped.append(m)
            matches = deduped

        return matches

    except Exception as e:
        logger.debug(f"Worker error for {symbol}: {e}")
        return []


class HistoricalScanner:
    """Scans historical data to generate ML training samples"""

    def __init__(self, config: SystemConfig):
        self.config = config
        self.data_manager = DataManager(config)
        self.normalizer = DataNormalizer(config)
        self.feature_extractor = FeatureExtractor(config)

    def scan_reference(
        self,
        ref: ReferencePattern,
        symbols: Optional[List[str]] = None,
        stride: Optional[int] = None,
        num_workers: int = 4,
        batch_size: int = 50,
    ) -> pd.DataFrame:
        """
        Scan one reference pattern across all symbols' full history.

        Args:
            ref: Reference pattern to scan
            symbols: Optional list of symbols (default: all available)
            stride: Sliding stride in bars (default: ref_len // 4)
            num_workers: Number of parallel workers
            batch_size: Symbols per batch (memory control)

        Returns:
            DataFrame with features + labels for all matches
        """
        scan_start = time.time()

        # Load reference data (with SMA padding for accurate SMA-diff features)
        sma_lookback = self.config.sma_lookback
        padding_seconds = sma_lookback * self.config.timeframe_to_seconds(ref.timeframe)
        padded_start_ts = ref.start_ts - padding_seconds

        ref_df_padded = self.data_manager.read_symbol(
            ref.symbol, ref.timeframe,
            start_ts=padded_start_ts, end_ts=ref.end_ts
        )
        if ref_df_padded.empty or len(ref_df_padded) < 10:
            logger.warning(f"Reference {ref.id}: insufficient data ({len(ref_df_padded)} bars)")
            return pd.DataFrame()

        # 找到 reference 實際起點在 padded df 中的 index
        ref_mask = ref_df_padded["timestamp"] >= ref.start_ts
        trim_start = int(ref_mask.values.argmax()) if ref_mask.any() else 0
        ref_len = len(ref_df_padded) - trim_start

        if ref_len < 10:
            logger.warning(f"Reference {ref.id}: insufficient ref data after trim ({ref_len} bars)")
            return pd.DataFrame()
        if stride is None:
            stride = max(1, ref_len // 4)

        # Prepare reference z-normalized close (from actual ref range, after padding trim)
        ref_close = ref_df_padded["close"].values[trim_start:]
        ref_std = np.std(ref_close)
        if ref_std < 1e-10:
            logger.warning(f"Reference {ref.id}: zero variance in close prices")
            return pd.DataFrame()
        ref_close_znorm = (ref_close - np.mean(ref_close)) / ref_std

        # Get symbol list
        if symbols is None:
            symbols = self.data_manager.list_symbols_for_timeframe(ref.timeframe)

        logger.info(
            f"Scanning ref={ref.id} ({ref_len} bars, stride={stride}) "
            f"across {len(symbols)} symbols with {num_workers} workers..."
        )

        # Load BTC data for market context features
        btc_df = self.data_manager.read_symbol("BTCUSDT", ref.timeframe)

        # Euclidean prescreen threshold (~10-15% pass rate)
        euclidean_prescreen_ratio = 0.8

        # Config dict for workers
        config_dict = {
            "dtw_window_ratio": self.config.dtw_window_ratio,
            "dtw_max_point_distance": self.config.dtw_max_point_distance,
            "min_similarity_score": self.config.min_similarity_score,
            "volume_spike_ratio": self.config.volume_spike_ratio,
            "sma_periods": self.config.sma_periods,
        }

        all_raw_matches = []
        total_symbols = len(symbols)

        # Process in batches to control memory
        for batch_start in range(0, total_symbols, batch_size):
            batch_symbols = symbols[batch_start:batch_start + batch_size]
            batch_num = batch_start // batch_size + 1
            total_batches = (total_symbols + batch_size - 1) // batch_size
            logger.info(
                f"  Batch {batch_num}/{total_batches}: "
                f"loading {len(batch_symbols)} symbols..."
            )

            # Prepare worker args
            import pickle
            worker_args = []
            for sym in batch_symbols:
                sym_df = self.data_manager.read_symbol(sym, ref.timeframe)
                if sym_df.empty or len(sym_df) < ref_len * 2:
                    continue
                sym_bytes = pickle.dumps(sym_df)
                worker_args.append((
                    sym, sym_bytes, ref_close_znorm, ref_len, stride,
                    euclidean_prescreen_ratio, config_dict, ref.timeframe,
                    self.config.ml_thresholds, self.config.ml_extension_factors,
                    ref.id,
                ))

            # Run workers
            batch_matches = []
            if num_workers <= 1 or len(worker_args) <= 3:
                for args in worker_args:
                    batch_matches.extend(_scan_symbol_worker(args))
            else:
                with ProcessPoolExecutor(max_workers=num_workers) as executor:
                    futures = {
                        executor.submit(_scan_symbol_worker, args): args[0]
                        for args in worker_args
                    }
                    for future in as_completed(futures):
                        try:
                            result = future.result(timeout=300)
                            batch_matches.extend(result)
                        except Exception as e:
                            sym = futures[future]
                            logger.debug(f"Worker failed for {sym}: {e}")

            all_raw_matches.extend(batch_matches)
            logger.info(
                f"  Batch {batch_num}: found {len(batch_matches)} matches "
                f"(total so far: {len(all_raw_matches)})"
            )

        if not all_raw_matches:
            logger.warning(f"No matches found for ref={ref.id}")
            return pd.DataFrame()

        # Extract full features for each match
        # Group by symbol to avoid re-reading parquet files
        from collections import defaultdict
        matches_by_symbol = defaultdict(list)
        for raw in all_raw_matches:
            matches_by_symbol[raw["symbol"]].append(raw)

        logger.info(
            f"Extracting features for {len(all_raw_matches)} matches "
            f"across {len(matches_by_symbol)} symbols..."
        )

        class _MatchProxy:
            """Minimal match result object for feature extraction"""
            pass

        rows = []
        processed = 0
        for sym, sym_matches in matches_by_symbol.items():
            sym_df = self.data_manager.read_symbol(sym, ref.timeframe)
            if sym_df.empty:
                continue

            # Pre-compute SMAs for the entire symbol once (accurate SMA)
            sym_with_sma = sym_df.copy()
            for period in self.config.sma_periods:
                sym_with_sma[f"SMA_{period}"] = (
                    sym_with_sma["close"].rolling(period, min_periods=period).mean()
                )

            for raw in sym_matches:
                window_df = sym_with_sma.iloc[
                    raw["match_start_idx"]:raw["match_end_idx"]
                ].copy()

                match_proxy = _MatchProxy()
                match_proxy.score = raw["dtw_score"]
                match_proxy.price_distance = raw.get("euc_dist", 0)
                match_proxy.diff_distance = 0.0
                match_proxy.best_scale_factor = 1.0

                features = self.feature_extractor.extract(
                    match_proxy, window_df, btc_df,
                    match_end_ts=raw["match_end_ts"]
                )

                row = {
                    "symbol": sym,
                    "timeframe": raw["timeframe"],
                    "ref_id": raw["ref_id"],
                    "match_end_ts": raw["match_end_ts"],
                    "dtw_score": raw["dtw_score"],
                }
                row.update(features)
                for threshold, label in raw["labels"].items():
                    row[f"label_{threshold}"] = label

                rows.append(row)

            processed += len(sym_matches)
            if processed % 50000 == 0:
                logger.info(f"  Feature extraction progress: {processed}/{len(all_raw_matches)}")

        df = pd.DataFrame(rows)
        elapsed = time.time() - scan_start
        logger.info(
            f"Ref {ref.id}: {len(df)} training samples extracted in {elapsed:.1f}s"
        )
        return df

    def scan_all_references(
        self,
        references: List[ReferencePattern],
        timeframes: Optional[List[str]] = None,
        stride: Optional[int] = None,
        num_workers: int = 4,
        batch_size: int = 50,
        output_csv: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Scan all references and combine into one training DataFrame.

        Args:
            references: List of reference patterns to scan
            timeframes: Optional filter for timeframes
            stride: Sliding stride (default: ref_len // 4)
            num_workers: Workers per reference scan
            batch_size: Symbols per batch
            output_csv: Path to save combined CSV

        Returns:
            Combined training DataFrame
        """
        all_dfs = []

        for ref in references:
            if timeframes and ref.timeframe not in timeframes:
                continue

            logger.info(f"\n{'='*60}")
            logger.info(f"Scanning reference: {ref.id}")
            logger.info(f"  Symbol: {ref.symbol}, Timeframe: {ref.timeframe}")
            logger.info(f"{'='*60}")

            df = self.scan_reference(
                ref, stride=stride, num_workers=num_workers,
                batch_size=batch_size
            )
            if not df.empty:
                all_dfs.append(df)

        if not all_dfs:
            logger.warning("No training samples generated from any reference")
            return pd.DataFrame()

        combined = pd.concat(all_dfs, ignore_index=True)

        # Sort by match_end_ts for time-series consistency
        combined = combined.sort_values("match_end_ts").reset_index(drop=True)

        if output_csv:
            os.makedirs(os.path.dirname(output_csv), exist_ok=True)
            combined.to_csv(output_csv, index=False)
            logger.info(f"Training data saved to {output_csv} ({len(combined)} rows)")

        return combined
