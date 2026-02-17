"""
Feature Extractor — ML 特徵提取（Phase 3 interface）
=====================================================

定義從 DTW match 中提取的特徵向量。
Phase 2 只定義介面，Phase 3 實現完整特徵工程。
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional


# 特徵定義（文檔用途 + Phase 3 實現時的 checklist）
FEATURE_DEFINITIONS = {
    # === DTW 匹配品質特徵 ===
    "dtw_score": "整體相似度分數",
    "price_distance": "價格 ShapeDTW 距離",
    "diff_distance": "SMA 差值 ShapeDTW 距離",
    "window_scale_factor": "最佳匹配的窗口縮放因子",

    # === 型態結構特徵 ===
    "pattern_length": "匹配型態的 K 棒數量",
    "pattern_return": "型態期間的漲跌幅 (%)",
    "pattern_volatility": "型態期間的收益率標準差",
    "pattern_max_drawdown": "型態期間的最大回撤 (%)",
    "sma30_slope_end": "SMA30 在型態末端的斜率",
    "sma_alignment_end": "型態末端 SMA 排列狀態 (多頭=1/空頭=-1/糾纏=0)",
    "close_vs_sma30_end": "收盤價相對 SMA30 的距離 (%)",
    "close_vs_sma60_end": "收盤價相對 SMA60 的距離 (%)",

    # === 量能特徵 ===
    "volume_trend": "型態期間成交量的線性趨勢斜率",
    "spike_count": "突兀量出現次數",
    "spike_ratio": "突兀量佔總 K 棒比例",
    "last_5bar_avg_volume_ratio": "最後 5 根平均量 / 型態平均量",

    # === 市場環境特徵 ===
    "btc_return_20bar": "同期 BTC 20 根 K 棒收益率",
    "btc_volatility_20bar": "同期 BTC 20 根 K 棒波動率",
    "altcoin_vs_btc": "該幣 20bar 收益 - BTC 20bar 收益",
}


class FeatureExtractor:
    """從 DTW match 提取 ML 特徵"""

    def __init__(self, config):
        self.config = config

    def extract(
        self,
        match_result,
        window_df: pd.DataFrame,
        btc_df: Optional[pd.DataFrame] = None,
        match_end_ts: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        提取特徵向量

        Args:
            match_result: DTWMatchResult
            window_df: 匹配區間的 DataFrame
            btc_df: 同期 BTC 數據（用於市場環境特徵）
            match_end_ts: 匹配結束時間戳（歷史訓練模式用，避免未來洩漏）

        Returns:
            特徵字典 {feature_name: value}
        """
        features = {}

        # DTW 匹配品質
        features["dtw_score"] = match_result.score
        features["price_distance"] = match_result.price_distance
        features["diff_distance"] = match_result.diff_distance
        features["window_scale_factor"] = match_result.best_scale_factor

        if window_df is not None and not window_df.empty:
            close = window_df["close"].values
            volume = window_df["volume"].values

            # 型態結構
            features["pattern_length"] = len(window_df)
            features["pattern_return"] = (close[-1] / close[0] - 1) if close[0] > 0 else 0
            returns = np.diff(close) / (close[:-1] + 1e-10)
            features["pattern_volatility"] = np.std(returns) if len(returns) > 0 else 0

            # 最大回撤
            cummax = np.maximum.accumulate(close)
            drawdowns = (close - cummax) / (cummax + 1e-10)
            features["pattern_max_drawdown"] = float(np.min(drawdowns))

            # SMA 排列
            if "SMA_30" in window_df.columns and "SMA_60" in window_df.columns:
                sma30_end = window_df["SMA_30"].iloc[-1]
                sma60_end = window_df["SMA_60"].iloc[-1]
                close_end = close[-1]
                features["close_vs_sma30_end"] = (close_end / sma30_end - 1) if sma30_end > 0 else 0
                features["close_vs_sma60_end"] = (close_end / sma60_end - 1) if sma60_end > 0 else 0

                if close_end > sma30_end > sma60_end:
                    features["sma_alignment_end"] = 1.0  # 多頭排列
                elif close_end < sma30_end < sma60_end:
                    features["sma_alignment_end"] = -1.0  # 空頭排列
                else:
                    features["sma_alignment_end"] = 0.0  # 糾纏

            # SMA30 末端斜率
            if "SMA_30" in window_df.columns:
                sma30_vals = window_df["SMA_30"].values
                tail_len = min(5, len(sma30_vals))
                features["sma30_slope_end"] = self._linear_slope(sma30_vals[-tail_len:])
            else:
                features["sma30_slope_end"] = 0.0

            # 量能
            avg_vol = np.mean(volume) if len(volume) > 0 else 1.0
            features["volume_trend"] = self._linear_slope(volume)
            spikes = sum(
                1 for i in range(1, len(volume))
                if volume[i] > volume[i-1] * self.config.volume_spike_ratio
            )
            features["spike_count"] = spikes
            features["spike_ratio"] = spikes / max(1, len(volume))
            last5_avg = np.mean(volume[-5:]) if len(volume) >= 5 else avg_vol
            features["last_5bar_avg_volume_ratio"] = last5_avg / (avg_vol + 1e-10)

        # BTC 市場環境特徵
        if btc_df is not None and not btc_df.empty:
            btc_filtered = btc_df
            if match_end_ts is not None:
                btc_filtered = btc_df[btc_df["timestamp"] <= match_end_ts]

            if len(btc_filtered) >= 20:
                btc_tail = btc_filtered.tail(20)
                btc_close = btc_tail["close"].values
                btc_ret = (btc_close[-1] / btc_close[0] - 1) if btc_close[0] > 0 else 0
                btc_returns = np.diff(btc_close) / (btc_close[:-1] + 1e-10)
                btc_vol = np.std(btc_returns) if len(btc_returns) > 0 else 0

                features["btc_return_20bar"] = btc_ret
                features["btc_volatility_20bar"] = btc_vol

                # altcoin vs BTC relative strength
                alt_ret = features.get("pattern_return", 0)
                features["altcoin_vs_btc"] = alt_ret - btc_ret
            else:
                features["btc_return_20bar"] = 0.0
                features["btc_volatility_20bar"] = 0.0
                features["altcoin_vs_btc"] = 0.0
        else:
            features["btc_return_20bar"] = 0.0
            features["btc_volatility_20bar"] = 0.0
            features["altcoin_vs_btc"] = 0.0

        return features

    @staticmethod
    def _linear_slope(series: np.ndarray) -> float:
        """計算序列的線性回歸斜率"""
        if len(series) < 2:
            return 0.0
        x = np.arange(len(series))
        x_mean = x.mean()
        y_mean = series.mean()
        denom = np.sum((x - x_mean) ** 2)
        if denom < 1e-10:
            return 0.0
        return float(np.sum((x - x_mean) * (series - y_mean)) / denom)

    @staticmethod
    def feature_names() -> list:
        """返回特徵名稱列表（與向量順序一致）"""
        return list(FEATURE_DEFINITIONS.keys())
