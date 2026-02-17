"""
DTW Calculator — DTW + ShapeDTW 核心計算引擎
=============================================

實現兩階段相似度比對：
1. Price DTW: 對 z-normalized close 序列做 DTW
2. Diff DTW: 對 z-normalized SMA-diff 序列做 DTW
3. ShapeDTW: 加入斜率描述子的 shape-aware DTW

最終分數 = price_weight * price_score + diff_weight * diff_score

支援多窗口縮放（window_scale_factors）以捕捉不同速度的型態。
"""

import numpy as np
import logging
from typing import List, Optional, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class DTWMatchResult:
    """單一 DTW 比對結果"""
    symbol: str
    timeframe: str
    score: float                    # 綜合相似度分數 [0, 1]，越高越相似
    price_distance: float           # 價格通道 DTW 距離
    diff_distance: float            # SMA-diff 通道 DTW 距離
    best_scale_factor: float        # 最佳匹配的窗口縮放因子
    match_start_idx: int            # 匹配區間在 target 中的起始 index
    match_end_idx: int              # 匹配區間在 target 中的結束 index
    window_data: Optional[object] = field(default=None, repr=False)  # 匹配區間的 DataFrame


class DTWCalculator:
    """DTW + ShapeDTW 計算引擎"""

    def __init__(self, config):
        self.config = config
        self._dtw_available = False
        try:
            from dtaidistance import dtw as dtw_lib
            self._dtw_lib = dtw_lib
            self._dtw_available = True
        except ImportError:
            logger.warning(
                "dtaidistance not installed, falling back to numpy DTW. "
                "Install with: pip install dtaidistance"
            )

    # ================================================================
    # Public API
    # ================================================================

    def compute_similarity(
        self,
        ref_close_znorm: np.ndarray,
        ref_sma_diffs_znorm: dict,
        ref_slope: np.ndarray,
        target_close_znorm: np.ndarray,
        target_sma_diffs_znorm: dict,
        target_slope: np.ndarray,
        symbol: str = "",
        timeframe: str = "",
        target_df=None,
    ) -> Optional[DTWMatchResult]:
        """
        計算 reference 與 target 尾部的相似度

        嘗試多個 window_scale_factors，取最佳結果。

        Args:
            ref_*: reference pattern 的各序列
            target_*: target symbol 的各序列（完整歷史）
            symbol: target symbol name（用於結果標記）
            timeframe: 時框（用於結果標記）
            target_df: target 的完整 DataFrame（用於輸出 window_data）

        Returns:
            DTWMatchResult 或 None（如果不滿足最低門檻）
        """
        ref_len = len(ref_close_znorm)
        target_len = len(target_close_znorm)

        if target_len < ref_len:
            return None

        best_result = None
        best_score = -1.0

        for scale in self.config.window_scale_factors:
            scaled_len = max(10, int(ref_len * scale))
            if scaled_len > target_len:
                continue

            # 取 target 尾部的 scaled_len 根
            t_start = target_len - scaled_len
            t_end = target_len

            t_close = target_close_znorm[t_start:t_end]
            t_slope = target_slope[t_start:t_end]
            t_diffs = {
                k: v[t_start:t_end] for k, v in target_sma_diffs_znorm.items()
            }

            # === Price DTW ===
            price_dist = self._compute_dtw_distance(
                ref_close_znorm, t_close,
                window_ratio=self.config.dtw_window_ratio,
                max_point_dist=self.config.dtw_max_point_distance,
            )

            # === Diff DTW（對所有 SMA diff 取平均距離）===
            diff_dists = []
            for key in ref_sma_diffs_znorm:
                if key in t_diffs:
                    d = self._compute_dtw_distance(
                        ref_sma_diffs_znorm[key], t_diffs[key],
                        window_ratio=self.config.dtw_window_ratio_diff,
                        max_point_dist=self.config.dtw_max_point_distance_diff,
                    )
                    diff_dists.append(d)

            diff_dist = np.mean(diff_dists) if diff_dists else price_dist

            # === ShapeDTW 加成（斜率描述子距離）===
            shape_dist = self._compute_shape_distance(ref_slope, t_slope)

            # === 綜合分數 ===
            # 將距離轉為相似度 [0, 1]
            # 使用 balance ratio 平衡 price 和 diff
            combined_dist = (
                self.config.price_weight * price_dist
                + self.config.diff_weight * diff_dist
            )

            # 加入 shape 距離的微調
            shape_penalty = shape_dist * 0.1  # 輕微的 shape 懲罰
            combined_dist += shape_penalty

            # 距離轉相似度：score = exp(-dist)，確保 [0, 1]
            score = np.exp(-combined_dist)

            if score > best_score:
                best_score = score
                best_result = DTWMatchResult(
                    symbol=symbol,
                    timeframe=timeframe,
                    score=score,
                    price_distance=price_dist,
                    diff_distance=diff_dist,
                    best_scale_factor=scale,
                    match_start_idx=t_start,
                    match_end_idx=t_end,
                    window_data=(
                        target_df.iloc[t_start:t_end].copy()
                        if target_df is not None
                        else None
                    ),
                )

        if best_result is None:
            return None

        # 門檻過濾
        if best_result.score < self.config.min_similarity_score:
            return None

        return best_result

    # ================================================================
    # DTW 距離計算
    # ================================================================

    def _compute_dtw_distance(
        self,
        s1: np.ndarray,
        s2: np.ndarray,
        window_ratio: float = 0.12,
        max_point_dist: float = 0.6,
    ) -> float:
        """
        計算兩個序列的 DTW 距離

        優先使用 dtaidistance C 加速版，fallback 到 numpy 實現。

        Args:
            s1, s2: 兩個一維序列
            window_ratio: Sakoe-Chiba band 寬度比例
            max_point_dist: 點對點最大允許距離

        Returns:
            正規化後的 DTW 距離
        """
        window = max(1, int(max(len(s1), len(s2)) * window_ratio))

        if self._dtw_available:
            try:
                dist = self._dtw_lib.distance(
                    s1.astype(np.float64),
                    s2.astype(np.float64),
                    window=window,
                    max_dist=max_point_dist * max(len(s1), len(s2)),
                    use_pruning=True,
                )
                if dist is None or np.isinf(dist):
                    return float("inf")
                # 正規化
                return dist / max(len(s1), len(s2))
            except Exception:
                pass

        # Numpy fallback
        return self._dtw_numpy(s1, s2, window, max_point_dist)

    def _dtw_numpy(
        self,
        s1: np.ndarray,
        s2: np.ndarray,
        window: int,
        max_point_dist: float,
    ) -> float:
        """
        純 numpy 的 DTW 實現（帶 Sakoe-Chiba band）

        速度比 dtaidistance 慢約 10-50x，但作為 fallback 夠用。
        """
        n, m = len(s1), len(s2)
        # 初始化 cost matrix
        dtw_matrix = np.full((n + 1, m + 1), np.inf)
        dtw_matrix[0, 0] = 0.0

        for i in range(1, n + 1):
            # Sakoe-Chiba band
            j_start = max(1, i - window)
            j_end = min(m, i + window)
            for j in range(j_start, j_end + 1):
                cost = abs(s1[i - 1] - s2[j - 1])
                if cost > max_point_dist:
                    cost = max_point_dist  # clip

                dtw_matrix[i, j] = cost + min(
                    dtw_matrix[i - 1, j],      # insertion
                    dtw_matrix[i, j - 1],      # deletion
                    dtw_matrix[i - 1, j - 1],  # match
                )

        total_dist = dtw_matrix[n, m]
        if np.isinf(total_dist):
            return float("inf")

        # 正規化
        return total_dist / max(n, m)

    # ================================================================
    # Shape 距離（斜率描述子）
    # ================================================================

    def _compute_shape_distance(
        self,
        slope1: np.ndarray,
        slope2: np.ndarray,
    ) -> float:
        """
        計算兩個斜率序列的形態距離

        用歐氏距離的正規化版本，因為斜率序列已經是 shape descriptor。
        """
        if len(slope1) == 0 or len(slope2) == 0:
            return 0.0

        # 如果長度不同，用線性插值對齊
        if len(slope1) != len(slope2):
            target_len = max(len(slope1), len(slope2))
            slope1 = np.interp(
                np.linspace(0, 1, target_len),
                np.linspace(0, 1, len(slope1)),
                slope1,
            )
            slope2 = np.interp(
                np.linspace(0, 1, target_len),
                np.linspace(0, 1, len(slope2)),
                slope2,
            )

        dist = np.sqrt(np.mean((slope1 - slope2) ** 2))
        return dist
