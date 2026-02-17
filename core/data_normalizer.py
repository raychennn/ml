"""
Data Normalizer — 數據標準化
============================

提供掃描引擎所需的數據預處理：
- Z-normalization（標準化到 mean=0, std=1）
- SMA 計算（30, 45, 60）
- SMA 差值序列（close - SMA）
- 斜率序列（用於 ShapeDTW 的形態描述子）
- PAA（Piecewise Aggregate Approximation）降維
"""

import numpy as np
import pandas as pd
from typing import List, Optional


class DataNormalizer:
    """數據標準化與特徵計算"""

    def __init__(self, config):
        self.config = config

    # ================================================================
    # Z-Normalization
    # ================================================================

    @staticmethod
    def z_normalize(series: np.ndarray) -> np.ndarray:
        """
        Z-normalization: (x - mean) / std

        用於 DTW 比對前的標準化，消除絕對價格差異。
        """
        std = np.std(series)
        if std < 1e-10:
            return np.zeros_like(series)
        return (series - np.mean(series)) / std

    @staticmethod
    def z_normalize_df(df: pd.DataFrame, columns: List[str] = None) -> pd.DataFrame:
        """對 DataFrame 的指定欄位做 z-normalization"""
        result = df.copy()
        if columns is None:
            columns = ["open", "high", "low", "close"]
        for col in columns:
            if col in result.columns:
                result[col] = DataNormalizer.z_normalize(result[col].values)
        return result

    # ================================================================
    # SMA 計算
    # ================================================================

    def compute_smas(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        計算 SMA 並附加到 DataFrame

        新增欄位: SMA_30, SMA_45, SMA_60

        注意：使用 min_periods=period 以確保 SMA 值準確，
        前期不足 period 根的 row 會是 NaN。
        若需要完整的 SMA，請確保輸入的 df 包含足夠的前期數據
        （至少 max(sma_periods) 根作為預熱期）。
        """
        result = df.copy()
        for period in self.config.sma_periods:
            col_name = f"SMA_{period}"
            result[col_name] = result["close"].rolling(
                window=period, min_periods=period
            ).mean()
        return result

    def compute_sma_diffs(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        計算 close 與各 SMA 的差值序列（用於 ShapeDTW 的 diff 通道）

        新增欄位: diff_SMA_30, diff_SMA_45, diff_SMA_60
        差值以 close 的百分比表示（標準化）
        """
        result = df.copy()
        for period in self.config.sma_periods:
            sma_col = f"SMA_{period}"
            diff_col = f"diff_SMA_{period}"
            if sma_col not in result.columns:
                result[sma_col] = result["close"].rolling(
                    window=period, min_periods=period
                ).mean()
            # 百分比差值，避免除零
            result[diff_col] = (result["close"] - result[sma_col]) / (
                result[sma_col].abs() + 1e-10
            )
        return result

    # ================================================================
    # 斜率序列（ShapeDTW descriptor）
    # ================================================================

    def compute_slope_descriptor(
        self, series: np.ndarray, window: int = None
    ) -> np.ndarray:
        """
        計算局部斜率描述子

        對序列中每個點，取前後 window 個點的線性回歸斜率。
        用於 ShapeDTW 的形態特徵。

        Args:
            series: 一維數值序列
            window: 斜率計算窗口（預設使用 config.slope_window_size）

        Returns:
            與 series 等長的斜率序列
        """
        if window is None:
            window = self.config.slope_window_size

        n = len(series)
        slopes = np.zeros(n)
        half_w = window // 2

        for i in range(n):
            start = max(0, i - half_w)
            end = min(n, i + half_w + 1)
            segment = series[start:end]
            if len(segment) < 2:
                slopes[i] = 0.0
            else:
                x = np.arange(len(segment))
                # 最小二乘法斜率: slope = cov(x,y) / var(x)
                x_mean = x.mean()
                y_mean = segment.mean()
                denominator = np.sum((x - x_mean) ** 2)
                if denominator < 1e-10:
                    slopes[i] = 0.0
                else:
                    slopes[i] = np.sum((x - x_mean) * (segment - y_mean)) / denominator

        return slopes

    # ================================================================
    # PAA（Piecewise Aggregate Approximation）
    # ================================================================

    def paa_transform(
        self, series: np.ndarray, window: int = None
    ) -> np.ndarray:
        """
        PAA 降維：將序列分段取平均

        用於 ShapeDTW 的降維加速。每 window 個點取一個平均值。

        Args:
            series: 一維數值序列
            window: PAA 窗口大小（預設使用 config.paa_window_size）

        Returns:
            降維後的序列（長度 = ceil(len(series) / window)）
        """
        if window is None:
            window = self.config.paa_window_size

        n = len(series)
        n_segments = int(np.ceil(n / window))
        result = np.zeros(n_segments)

        for i in range(n_segments):
            start = i * window
            end = min(start + window, n)
            result[i] = np.mean(series[start:end])

        return result

    # ================================================================
    # 複合特徵準備
    # ================================================================

    def prepare_for_dtw(self, df: pd.DataFrame, trim_start: int = 0) -> dict:
        """
        為 DTW 比對準備所有需要的序列

        Args:
            df: 包含 OHLCV 的 DataFrame（可包含 SMA 預熱用的前期數據）
            trim_start: 計算完 SMA 等指標後，從此 index 開始截取
                        （用於去除 SMA 預熱段，只保留目標區間）
                        設為 0 表示不截取。

        Returns:
            {
                "close_znorm": z-normalized close 序列,
                "close_raw": 原始 close 序列,
                "sma_diffs_znorm": {
                    "diff_SMA_30": z-normalized diff 序列,
                    "diff_SMA_45": ...,
                    "diff_SMA_60": ...,
                },
                "slope": 斜率描述子,
                "volume": volume 序列,
                "df": 包含所有計算欄位的完整 DataFrame,
            }
        """
        # 在完整 df 上計算 SMA 和 diff（含前期預熱段）
        enriched = self.compute_smas(df)
        enriched = self.compute_sma_diffs(enriched)

        # 截取目標區間（去除 SMA 預熱段）
        if trim_start > 0 and trim_start < len(enriched):
            trimmed = enriched.iloc[trim_start:].reset_index(drop=True)
        else:
            trimmed = enriched

        close = trimmed["close"].values
        close_znorm = self.z_normalize(close)

        # Z-normalize 各 SMA diff
        sma_diffs_znorm = {}
        for period in self.config.sma_periods:
            diff_col = f"diff_SMA_{period}"
            diff_values = trimmed[diff_col].values
            # 填充 NaN（在 trim 後理論上不應存在，但做防守）
            diff_values = np.nan_to_num(diff_values, nan=0.0)
            sma_diffs_znorm[diff_col] = self.z_normalize(diff_values)

        # 斜率描述子
        slope = self.compute_slope_descriptor(close_znorm)

        return {
            "close_znorm": close_znorm,
            "close_raw": close,
            "sma_diffs_znorm": sma_diffs_znorm,
            "slope": slope,
            "volume": trimmed["volume"].values,
            "df": trimmed,
        }
