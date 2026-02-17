"""
Scanner Engine — 三階段掃描引擎
================================

Stage 1: 快速預篩 (z-normalized Euclidean distance)
    250 symbols → ~40-50 candidates（淘汰 80%+）

Stage 2: DTW + ShapeDTW 精確比對
    ~40 candidates → Top K matches（multiprocessing 並行）

Stage 3: ML 分類器評分（Phase 3 實現，目前為 placeholder）
    每個 match 輸出: P(5%漲), P(10%漲), P(15%漲), P(20%漲)
"""

import logging
import threading
import time
import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass, field
from multiprocessing import Pool, cpu_count

from core.data_normalizer import DataNormalizer
from core.dtw_calculator import DTWCalculator, DTWMatchResult
from references.reference_manager import ReferencePattern
from ml.predictor import MLPredictor

logger = logging.getLogger(__name__)


@dataclass
class AlertResult:
    """掃描結果：一個 match alert"""
    symbol: str
    timeframe: str
    reference: ReferencePattern
    dtw_score: float
    price_distance: float
    diff_distance: float
    best_scale_factor: float
    ml_probabilities: Dict[float, float] = field(default_factory=dict)
    window_data: Optional[object] = field(default=None, repr=False)
    match_start_idx: int = 0
    match_end_idx: int = 0
    context_bars: int = 0  # 前期上下文 K 棒數（用於圖表顯示 SMA 預熱段）


# ================================================================
# Worker function for multiprocessing (must be top-level)
# ================================================================

def _dtw_worker(args: tuple) -> Optional[DTWMatchResult]:
    """
    Multiprocessing worker: 對單一 symbol 做 DTW 比對

    args = (symbol, target_data_dict, ref_prepared, config_dict)
    target_data_dict 包含該 symbol 的 prepared data
    """
    symbol, target_prepared, ref_prepared, config_dict = args

    try:
        # 重建 config 和 calculator（每個 worker 獨立）
        from config import SystemConfig
        config = SystemConfig()
        # 覆寫 config 值
        for k, v in config_dict.items():
            if hasattr(config, k):
                setattr(config, k, v)

        calculator = DTWCalculator(config)

        result = calculator.compute_similarity(
            ref_close_znorm=ref_prepared["close_znorm"],
            ref_sma_diffs_znorm=ref_prepared["sma_diffs_znorm"],
            ref_slope=ref_prepared["slope"],
            target_close_znorm=target_prepared["close_znorm"],
            target_sma_diffs_znorm=target_prepared["sma_diffs_znorm"],
            target_slope=target_prepared["slope"],
            symbol=symbol,
            timeframe=ref_prepared.get("timeframe", ""),
            target_df=None,  # 不在 worker 中傳大 DataFrame
        )
        return result

    except Exception as e:
        logger.debug(f"DTW worker error for {symbol}: {e}")
        return None


class ScannerEngine:
    """三階段掃描引擎"""

    def __init__(self, config, data_manager):
        self.config = config
        self.data_manager = data_manager
        self.normalizer = DataNormalizer(config)
        self.dtw_calculator = DTWCalculator(config)
        self.predictor = MLPredictor(config)
        self.predictor.load()
        self._btc_cache = {}  # {timeframe: DataFrame}
        self._btc_cache_lock = threading.Lock()

    # ================================================================
    # Main scan entry
    # ================================================================

    def scan_timeframe(
        self,
        timeframe: str,
        references: List[ReferencePattern],
        num_workers: int = None,
    ) -> List[AlertResult]:
        """
        掃描單一時框的所有 symbol

        Args:
            timeframe: e.g. "4h"
            references: 要比對的 reference patterns
            num_workers: multiprocessing workers 數（預設 = CPU 數）

        Returns:
            按分數排序的 AlertResult 列表
        """
        if num_workers is None:
            num_workers = min(4, cpu_count())

        # 只取屬於此時框的 references
        tf_refs = [r for r in references if r.timeframe == timeframe]
        if not tf_refs:
            logger.info(f"No references for timeframe {timeframe}, skipping")
            return []

        # 載入該時框所有 symbol 數據
        # 計算所需的最大尾部長度（reference 長度 × 最大 scale × 安全邊際）
        max_ref_bars = 0
        for ref in tf_refs:
            ref_df = self.data_manager.read_symbol(ref.symbol, ref.timeframe,
                                                     start_ts=ref.start_ts,
                                                     end_ts=ref.end_ts)
            ref_bars = len(ref_df)
            max_ref_bars = max(max_ref_bars, ref_bars)

        # 需要的尾部長度：ref 長度 × 最大 scale + SMA 預熱期
        tail_bars = int(max_ref_bars * max(self.config.window_scale_factors) * 1.2)
        tail_bars = max(tail_bars, max(self.config.sma_periods) + max_ref_bars)

        logger.info(
            f"Loading data for {timeframe} (tail={tail_bars} bars)..."
        )
        data_dict = self.data_manager.bulk_read_timeframe(
            timeframe, tail_bars=tail_bars
        )

        if not data_dict:
            logger.warning(f"No data for timeframe {timeframe}")
            return []

        all_alerts = []

        for ref in tf_refs:
            ref_start = time.time()

            # 載入 reference 數據（含 SMA 預熱段）
            # 先載入精確區間以取得 bar count
            ref_df_exact = self.data_manager.read_symbol(
                ref.symbol, ref.timeframe,
                start_ts=ref.start_ts, end_ts=ref.end_ts
            )
            if ref_df_exact.empty or len(ref_df_exact) < 10:
                logger.warning(
                    f"Reference {ref.id}: insufficient data ({len(ref_df_exact)} bars)"
                )
                continue

            # 載入含 SMA 預熱段的數據（往前多取 sma_lookback 根）
            sma_lookback = self.config.sma_lookback
            padding_seconds = sma_lookback * self.config.timeframe_to_seconds(timeframe)
            padded_start_ts = ref.start_ts - padding_seconds
            ref_df_padded = self.data_manager.read_symbol(
                ref.symbol, ref.timeframe,
                start_ts=padded_start_ts, end_ts=ref.end_ts
            )

            # 計算實際 padding 長度（前期可能因數據不足而少於 sma_lookback）
            ref_mask = ref_df_padded["timestamp"] >= ref.start_ts
            if ref_mask.any():
                trim_start = ref_mask.values.argmax()  # 第一個 True 的 index
            else:
                trim_start = 0

            # 使用 padded 數據做 DTW 準備（SMA 在完整數據上計算，再 trim 回 reference 區間）
            ref_prepared = self.normalizer.prepare_for_dtw(ref_df_padded, trim_start=trim_start)
            ref_prepared["timeframe"] = timeframe

            logger.info(
                f"Scanning ref={ref.id} ({len(ref_df_exact)} bars, "
                f"SMA padding={trim_start}) "
                f"against {len(data_dict)} symbols..."
            )

            # === Stage 1: 快速預篩 ===
            stage1_start = time.time()
            candidates = self._prescreening(ref_prepared, data_dict)
            stage1_time = time.time() - stage1_start
            logger.info(
                f"  Stage 1: {len(data_dict)} → {len(candidates)} candidates "
                f"({stage1_time:.1f}s)"
            )

            if not candidates:
                continue

            # === Stage 2: DTW + ShapeDTW 精確比對 ===
            stage2_start = time.time()
            matches = self._dtw_matching(
                ref_prepared, candidates, timeframe,
                num_workers=num_workers
            )
            stage2_time = time.time() - stage2_start
            logger.info(
                f"  Stage 2: {len(candidates)} → {len(matches)} matches "
                f"({stage2_time:.1f}s)"
            )

            # === Stage 3: ML 評分 ===
            btc_df = self._get_btc_data(timeframe)
            for match in matches:
                context_bars = getattr(match, '_context_bars', 0)

                # ML predictor 只用匹配區間的數據（不含前期上下文）
                if match.window_data is not None and context_bars > 0:
                    match_only_df = match.window_data.iloc[context_bars:].copy()
                else:
                    match_only_df = match.window_data

                ml_probs = self.predictor.predict(
                    match, match_only_df, btc_df
                )
                alert = AlertResult(
                    symbol=match.symbol,
                    timeframe=timeframe,
                    reference=ref,
                    dtw_score=match.score,
                    price_distance=match.price_distance,
                    diff_distance=match.diff_distance,
                    best_scale_factor=match.best_scale_factor,
                    ml_probabilities=ml_probs,
                    window_data=match.window_data,  # 含前期上下文（用於圖表）
                    match_start_idx=match.match_start_idx,
                    match_end_idx=match.match_end_idx,
                    context_bars=context_bars,
                )
                all_alerts.append(alert)

            ref_elapsed = time.time() - ref_start
            logger.info(
                f"  Ref {ref.id} complete: {len(matches)} alerts ({ref_elapsed:.1f}s)"
            )

        # 按分數排序
        all_alerts.sort(key=lambda a: a.dtw_score, reverse=True)
        return all_alerts

    # ================================================================
    # Stage 1: Prescreening
    # ================================================================

    def _prescreening(
        self,
        ref_prepared: dict,
        data_dict: Dict[str, object],
    ) -> Dict[str, dict]:
        """
        Stage 1: z-normalized Euclidean distance 預篩

        只用 close 價格的 z-norm 版本，計算與 reference 尾部的歐氏距離。
        保留距離最小的前 N%。
        """
        ref_znorm = ref_prepared["close_znorm"]
        ref_len = len(ref_znorm)

        distances = {}

        for symbol, df in data_dict.items():
            if len(df) < ref_len:
                continue

            close = df["close"].values
            # 取尾部 ref_len 根
            tail = close[-ref_len:]
            std = np.std(tail)
            if std < 1e-10:
                continue
            tail_znorm = (tail - np.mean(tail)) / std

            # 如果長度不匹配（因為 scale），做線性插值
            if len(tail_znorm) != len(ref_znorm):
                tail_znorm = np.interp(
                    np.linspace(0, 1, len(ref_znorm)),
                    np.linspace(0, 1, len(tail_znorm)),
                    tail_znorm,
                )

            # 正規化歐氏距離
            dist = np.sqrt(np.sum((ref_znorm - tail_znorm) ** 2)) / ref_len
            distances[symbol] = dist

        if not distances:
            return {}

        # 取前 N%
        sorted_symbols = sorted(distances, key=distances.get)
        top_n = max(10, int(len(sorted_symbols) * self.config.prescreening_top_ratio))
        top_n = min(top_n, len(sorted_symbols))

        # 對 candidates 做 DTW 前的數據準備
        candidates = {}
        for sym in sorted_symbols[:top_n]:
            df = data_dict[sym]
            prepared = self.normalizer.prepare_for_dtw(df)
            candidates[sym] = prepared

        return candidates

    # ================================================================
    # Stage 2: DTW Matching
    # ================================================================

    def _dtw_matching(
        self,
        ref_prepared: dict,
        candidates: Dict[str, dict],
        timeframe: str,
        num_workers: int = 4,
    ) -> List[DTWMatchResult]:
        """
        Stage 2: 完整 DTW + ShapeDTW 比對

        對所有 candidate 做精確 DTW 計算。
        使用 multiprocessing 並行化（如果 candidates > 10）。
        """
        # 準備 config dict（用於 worker 重建 config）
        config_dict = {
            "dtw_window_ratio": self.config.dtw_window_ratio,
            "dtw_window_ratio_diff": self.config.dtw_window_ratio_diff,
            "dtw_max_point_distance": self.config.dtw_max_point_distance,
            "dtw_max_point_distance_diff": self.config.dtw_max_point_distance_diff,
            "window_scale_factors": self.config.window_scale_factors,
            "price_weight": self.config.price_weight,
            "diff_weight": self.config.diff_weight,
            "slope_window_size": self.config.slope_window_size,
            "min_similarity_score": self.config.min_similarity_score,
            "sma_periods": self.config.sma_periods,
        }

        # 準備 ref 序列（不包含 DataFrame，只傳 numpy）
        ref_serial = {
            "close_znorm": ref_prepared["close_znorm"],
            "sma_diffs_znorm": ref_prepared["sma_diffs_znorm"],
            "slope": ref_prepared["slope"],
            "timeframe": timeframe,
        }

        # 準備 worker args（只傳 numpy arrays，不傳 DataFrame）
        args_list = []
        for symbol, prepared in candidates.items():
            target_serial = {
                "close_znorm": prepared["close_znorm"],
                "sma_diffs_znorm": prepared["sma_diffs_znorm"],
                "slope": prepared["slope"],
            }
            args_list.append((symbol, target_serial, ref_serial, config_dict))

        # 決定是否用 multiprocessing
        if len(args_list) <= 5 or num_workers <= 1:
            # 少量 candidate，直接在主進程算
            results = [_dtw_worker(a) for a in args_list]
        else:
            try:
                with Pool(processes=num_workers) as pool:
                    results = pool.map(_dtw_worker, args_list)
            except Exception as e:
                logger.warning(f"Multiprocessing failed ({e}), falling back to sequential")
                results = [_dtw_worker(a) for a in args_list]

        # 過濾 None 和低分結果，補上 window_data（含前期上下文）
        sma_lookback = self.config.sma_lookback
        matches = []
        for r in results:
            if r is not None and r.score >= self.config.min_similarity_score:
                # 補上 window_data（從原始 candidates 的 df 中切）
                # 包含 sma_lookback 根前期 K 棒作為 SMA 預熱 + 圖表上下文
                if r.symbol in candidates:
                    cand_df = candidates[r.symbol].get("df")
                    if cand_df is not None:
                        padded_start = max(0, r.match_start_idx - sma_lookback)
                        actual_context = r.match_start_idx - padded_start
                        r.window_data = cand_df.iloc[
                            padded_start:r.match_end_idx
                        ].copy()
                        # 附加 context_bars 資訊到 match result
                        r._context_bars = actual_context
                    else:
                        r._context_bars = 0
                else:
                    r._context_bars = 0
                matches.append(r)

        matches.sort(key=lambda x: x.score, reverse=True)
        return matches

    # ================================================================
    # BTC data cache for market context features
    # ================================================================

    def clear_btc_cache(self):
        """Clear BTC data cache — call at start of each scan cycle"""
        with self._btc_cache_lock:
            self._btc_cache.clear()

    def _get_btc_data(self, timeframe: str):
        """Load and cache BTC data for market context features (thread-safe)"""
        with self._btc_cache_lock:
            if timeframe not in self._btc_cache:
                try:
                    self._btc_cache[timeframe] = self.data_manager.read_symbol(
                        "BTCUSDT", timeframe
                    )
                except Exception:
                    self._btc_cache[timeframe] = None
            return self._btc_cache[timeframe]
