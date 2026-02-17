"""
Chart Generator — K 棒 + SMA + 突兀量 圖片生成
================================================

生成 Telegram 友好的 K 棒圖：
- OHLC K 棒（紅綠色）
- SMA 30/45/60 三條均線
- 突兀量標記：volume > prev_volume × 2.5
- 1200 × 800 px, 150 DPI
"""

import os
import time
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # 非互動後端
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from matplotlib.patches import FancyBboxPatch
from datetime import datetime, timezone
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# 全局樣式設定
plt.rcParams.update({
    "figure.facecolor": "#1a1a2e",
    "axes.facecolor": "#16213e",
    "axes.edgecolor": "#333333",
    "axes.labelcolor": "#cccccc",
    "text.color": "#cccccc",
    "xtick.color": "#999999",
    "ytick.color": "#999999",
    "grid.color": "#2a2a4a",
    "grid.alpha": 0.5,
    "font.size": 9,
})

# 顏色定義
COLOR_UP = "#26A69A"       # 漲（綠）
COLOR_DOWN = "#EF5350"     # 跌（紅）
COLOR_SPIKE = "#FFD600"    # 突兀量（黃）
COLOR_SMA = {
    30: "#2196F3",         # SMA30 藍
    45: "#FF9800",         # SMA45 橘
    60: "#9C27B0",         # SMA60 紫
}


class ChartGenerator:
    """K 棒圖片生成器"""

    def __init__(self, config):
        self.config = config

    # ================================================================
    # Alert Chart（即時掃描結果用）
    # ================================================================

    def generate_alert_chart(
        self,
        window_df: pd.DataFrame,
        symbol: str,
        timeframe: str,
        ref_name: str,
        score: float,
        ml_probs: Dict[float, float],
        context_bars: int = 0,
    ) -> str:
        """
        生成掃描 alert 圖片

        Args:
            window_df: 匹配區間的 OHLCV DataFrame（可包含前期上下文 K 棒）
            symbol: e.g. "SOLUSDT"
            timeframe: e.g. "4h"
            ref_name: reference 標籤
            score: DTW 相似度分數
            ml_probs: {0.05: 0.78, 0.10: 0.52, ...}
            context_bars: 前期上下文 K 棒數量（用於 SMA 預熱，圖表中以
                         垂直虛線標記匹配起點）

        Returns:
            儲存的圖片檔案路徑
        """
        df = self._prepare_dataframe(window_df)
        if df.empty:
            return ""

        fig, (ax_price, ax_vol) = plt.subplots(
            2, 1,
            figsize=(self.config.chart_width / self.config.chart_dpi,
                     self.config.chart_height / self.config.chart_dpi),
            dpi=self.config.chart_dpi,
            gridspec_kw={"height_ratios": [3, 1]},
            sharex=True,
        )
        fig.subplots_adjust(hspace=0.05)

        # K 棒
        self._plot_candlesticks(ax_price, df)

        # SMA 線
        self._plot_sma_lines(ax_price, df)

        # Volume + 突兀量
        spike_indices = self._plot_volume_with_spikes(ax_vol, df)

        # 在價格圖上也標記突兀量位置
        self._mark_spikes_on_price(ax_price, df, spike_indices)

        # 匹配起點標記（前期上下文與匹配區間的分界線）
        if context_bars > 0 and context_bars < len(df):
            self._draw_match_boundary(ax_price, ax_vol, df, context_bars)

        # 對數座標
        self._apply_log_scale(ax_price)

        # 標題
        title = f"{symbol}  |  {timeframe}  |  Ref: {ref_name}  |  Score: {score:.3f}"
        fig.suptitle(title, fontsize=12, fontweight="bold", color="white", y=0.98)

        # ML 機率副標題
        if ml_probs:
            ml_text = "  |  ".join([
                f"{int(t*100)}%↑: {p:.0%}" for t, p in sorted(ml_probs.items())
            ])
            ax_price.set_title(ml_text, fontsize=9, color="#aaaaaa", pad=8)

        # 美化
        ax_price.set_ylabel("Price", fontsize=9)
        ax_vol.set_ylabel("Volume", fontsize=9)
        ax_price.legend(loc="upper left", fontsize=8, framealpha=0.3,
                        edgecolor="none", facecolor="#1a1a2e")
        self._format_xaxis(ax_vol, df)

        # 儲存
        path = os.path.join(
            self.config.image_dir, "alerts",
            f"{symbol}_{timeframe}_{int(time.time())}.png",
        )
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)

        logger.debug(f"Chart saved: {path}")
        return path

    # ================================================================
    # Reference Preview Chart（新增 reference 時的確認圖）
    # ================================================================

    def generate_reference_chart(
        self,
        window_df: pd.DataFrame,
        symbol: str,
        timeframe: str,
        label: str,
        description: str = "",
        stats: Optional[Dict] = None,
        context_bars: int = 0,
    ) -> str:
        """
        Generate a preview chart for a newly added reference pattern.

        Shows candlesticks + SMAs + volume with pattern stats overlay,
        so the user can visually verify the reference makes sense.

        Args:
            window_df: The reference pattern OHLCV DataFrame（可包含前期上下文 K 棒）
            symbol: e.g. "AVAXUSDT"
            timeframe: e.g. "4h"
            label: Reference label
            description: Optional description
            stats: Optional dict of pattern stats to display
            context_bars: 前期上下文 K 棒數量（SMA 預熱段）

        Returns:
            Saved image file path
        """
        df = self._prepare_dataframe(window_df)
        if df.empty:
            return ""

        fig, (ax_price, ax_vol) = plt.subplots(
            2, 1,
            figsize=(self.config.chart_width / self.config.chart_dpi,
                     self.config.chart_height / self.config.chart_dpi),
            dpi=self.config.chart_dpi,
            gridspec_kw={"height_ratios": [3, 1]},
            sharex=True,
        )
        fig.subplots_adjust(hspace=0.05)

        # K bars
        self._plot_candlesticks(ax_price, df)

        # SMA lines
        self._plot_sma_lines(ax_price, df)

        # Volume + spikes
        spike_indices = self._plot_volume_with_spikes(ax_vol, df)
        self._mark_spikes_on_price(ax_price, df, spike_indices)

        # 參考型態起點標記
        if context_bars > 0 and context_bars < len(df):
            self._draw_match_boundary(ax_price, ax_vol, df, context_bars)

        # 對數座標
        self._apply_log_scale(ax_price)

        # Title
        title = f"NEW REFERENCE  |  {symbol}  |  {timeframe}  |  {label}"
        fig.suptitle(title, fontsize=12, fontweight="bold", color="#FFD600", y=0.98)

        # Date range subtitle（顯示 reference 的實際時間範圍，排除前期上下文）
        if "datetime" in df.columns:
            ref_start_idx = context_bars if context_bars > 0 else 0
            dt_start = pd.Timestamp(df["datetime"].iloc[ref_start_idx]).strftime("%Y-%m-%d %H:%M")
            dt_end = pd.Timestamp(df["datetime"].iloc[-1]).strftime("%Y-%m-%d %H:%M")
            ref_bars = len(df) - ref_start_idx
            ctx_text = f" (+{context_bars} context)" if context_bars > 0 else ""
            date_text = f"{dt_start}  →  {dt_end}  |  {ref_bars} bars{ctx_text}"
            ax_price.set_title(date_text, fontsize=9, color="#aaaaaa", pad=8)

        # Stats overlay box
        if stats:
            stats_lines = []
            if "pattern_return" in stats:
                ret = stats["pattern_return"]
                ret_color = "#26A69A" if ret >= 0 else "#EF5350"
                stats_lines.append(f"Return: {ret:+.2%}")
            if "pattern_volatility" in stats:
                stats_lines.append(f"Volatility: {stats['pattern_volatility']:.4f}")
            if "pattern_max_drawdown" in stats:
                stats_lines.append(f"Max DD: {stats['pattern_max_drawdown']:.2%}")
            if "sma_alignment" in stats:
                alignment_map = {1.0: "Bullish", -1.0: "Bearish", 0.0: "Tangled"}
                stats_lines.append(
                    f"SMA Align: {alignment_map.get(stats['sma_alignment'], 'N/A')}"
                )
            if "volume_spikes" in stats:
                stats_lines.append(f"Vol Spikes: {stats['volume_spikes']}")

            if stats_lines:
                stats_text = "\n".join(stats_lines)
                ax_price.text(
                    0.98, 0.97, stats_text,
                    transform=ax_price.transAxes,
                    fontsize=8, color="#ffffff",
                    verticalalignment="top", horizontalalignment="right",
                    bbox=dict(
                        boxstyle="round,pad=0.5",
                        facecolor="#1a1a2e", edgecolor="#FFD600",
                        alpha=0.85,
                    ),
                    family="monospace",
                )

        # Description at bottom
        if description:
            fig.text(
                0.5, 0.01, description,
                ha="center", fontsize=8, color="#888888", style="italic",
            )

        ax_price.set_ylabel("Price", fontsize=9)
        ax_vol.set_ylabel("Volume", fontsize=9)
        ax_price.legend(loc="upper left", fontsize=8, framealpha=0.3,
                        edgecolor="none", facecolor="#1a1a2e")
        self._format_xaxis(ax_vol, df)

        # Save
        safe_label = label.replace("/", "_").replace(" ", "_")
        path = os.path.join(
            self.config.image_dir, "references",
            f"{symbol}_{timeframe}_{safe_label}.png",
        )
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)

        logger.info(f"Reference chart saved: {path}")
        return path

    # ================================================================
    # Reference SMA-Only Chart (pure SMA lines, no candlesticks)
    # ================================================================

    def generate_reference_sma_chart(
        self,
        window_df: pd.DataFrame,
        symbol: str,
        timeframe: str,
        label: str,
        context_bars: int = 0,
    ) -> str:
        """
        Generate a pure SMA chart for a reference pattern.
        Shows only SMA 30/45/60 lines on a clean background (no candlesticks, no volume).

        Args:
            context_bars: 前期上下文 K 棒數量（SMA 預熱段）

        Returns:
            Saved image file path
        """
        df = self._prepare_dataframe(window_df)
        if df.empty:
            return ""

        fig, ax = plt.subplots(
            figsize=(self.config.chart_width / self.config.chart_dpi,
                     (self.config.chart_height * 0.6) / self.config.chart_dpi),
            dpi=self.config.chart_dpi,
        )

        dates = pd.to_datetime(df["datetime"].values)

        # Plot SMA lines
        for period in self.config.sma_periods:
            col = f"SMA_{period}"
            sma = df[col] if col in df.columns else df["close"].rolling(
                window=period, min_periods=period
            ).mean()
            color = COLOR_SMA.get(period, "#ffffff")
            valid_mask = sma.notna().values
            if valid_mask.any():
                ax.plot(dates[valid_mask], sma.values[valid_mask],
                        color=color, linewidth=2.0,
                        alpha=0.9, label=f"SMA {period}")

        # 參考型態起點標記
        if context_bars > 0 and context_bars < len(df):
            self._draw_match_boundary(ax, None, df, context_bars)

        # 對數座標
        self._apply_log_scale(ax)

        # Title
        title = f"SMA ONLY  |  {symbol}  |  {timeframe}  |  {label}"
        fig.suptitle(title, fontsize=12, fontweight="bold", color="white", y=0.98)

        # Date range（顯示 reference 實際範圍）
        if len(df) >= 2:
            ref_start_idx = context_bars if context_bars > 0 else 0
            dt_start = pd.Timestamp(df["datetime"].iloc[ref_start_idx]).strftime("%Y-%m-%d %H:%M")
            dt_end = pd.Timestamp(df["datetime"].iloc[-1]).strftime("%Y-%m-%d %H:%M")
            ref_bars = len(df) - ref_start_idx
            ctx_text = f" (+{context_bars} context)" if context_bars > 0 else ""
            ax.set_title(f"{dt_start}  →  {dt_end}  |  {ref_bars} bars{ctx_text}",
                        fontsize=9, color="#aaaaaa", pad=8)

        ax.set_ylabel("SMA Value", fontsize=9)
        ax.legend(loc="upper left", fontsize=9, framealpha=0.3,
                  edgecolor="none", facecolor="#1a1a2e")
        ax.grid(True, alpha=0.2)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
        ax.tick_params(axis="x", rotation=30)

        # Save
        safe_label = label.replace("/", "_").replace(" ", "_")
        path = os.path.join(
            self.config.image_dir, "references",
            f"{symbol}_{timeframe}_{safe_label}_sma.png",
        )
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)

        logger.info(f"Reference SMA chart saved: {path}")
        return path

    # ================================================================
    # Historical Match Chart（離線訓練結果輸出用）
    # ================================================================

    def generate_historical_match_chart(
        self,
        ref_df: pd.DataFrame,
        match_df: pd.DataFrame,
        symbol: str,
        ref_name: str,
        score: float,
        future_df: Optional[pd.DataFrame] = None,
        trend_result: str = "",
    ) -> str:
        """
        歷史 match 圖片（對比 reference vs match）

        包含: reference 型態 + match 型態 + 後續走勢
        """
        fig, axes = plt.subplots(
            2, 1,
            figsize=(14, 10),
            dpi=self.config.chart_dpi,
        )
        fig.subplots_adjust(hspace=0.3)

        # 上圖: Reference pattern
        ref_prep = self._prepare_dataframe(ref_df)
        if not ref_prep.empty:
            self._plot_candlesticks(axes[0], ref_prep)
            self._plot_sma_lines(axes[0], ref_prep)
            self._apply_log_scale(axes[0])
            axes[0].set_title(f"Reference: {ref_name}", fontsize=11,
                             fontweight="bold", color="white")
            axes[0].set_ylabel("Price", fontsize=9)

        # 下圖: Match + 後續走勢
        match_prep = self._prepare_dataframe(match_df)
        if not match_prep.empty:
            if future_df is not None and not future_df.empty:
                combined = pd.concat([match_df, future_df])
                combined_prep = self._prepare_dataframe(combined)
                self._plot_candlesticks(axes[1], combined_prep)
                self._plot_sma_lines(axes[1], combined_prep)

                # 在 match 結束位置畫垂直虛線
                match_end_dt = match_prep["datetime"].iloc[-1]
                axes[1].axvline(
                    x=match_end_dt, color="#FF6F00",
                    linestyle="--", linewidth=1.5, alpha=0.8,
                    label="Pattern End",
                )
                axes[1].legend(loc="upper left", fontsize=8, framealpha=0.3,
                              edgecolor="none", facecolor="#1a1a2e")
            else:
                self._plot_candlesticks(axes[1], match_prep)
                self._plot_sma_lines(axes[1], match_prep)

            self._apply_log_scale(axes[1])
            subtitle = f"{symbol}  |  Score: {score:.3f}"
            if trend_result:
                subtitle += f"  |  After: {trend_result}"
            axes[1].set_title(subtitle, fontsize=11, fontweight="bold", color="white")
            axes[1].set_ylabel("Price", fontsize=9)

        # 儲存
        safe_ref = ref_name.replace("/", "_").replace(" ", "_")
        path = os.path.join(
            self.config.image_dir, "historical", safe_ref,
            f"{symbol}_{score:.3f}.png",
        )
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        return path

    # ================================================================
    # Internal: K 棒繪製
    # ================================================================

    def _plot_candlesticks(self, ax, df: pd.DataFrame):
        """繪製 OHLC K 棒"""
        if df.empty:
            return

        dates = df["datetime"].values
        opens = df["open"].values
        highs = df["high"].values
        lows = df["low"].values
        closes = df["close"].values

        # 計算 K 棒寬度（基於時間間隔）
        if len(dates) >= 2:
            time_diff = (pd.Timestamp(dates[1]) - pd.Timestamp(dates[0]))
            width = time_diff * 0.7
        else:
            width = pd.Timedelta(hours=1)

        for i in range(len(df)):
            is_up = closes[i] >= opens[i]
            color = COLOR_UP if is_up else COLOR_DOWN
            dt = pd.Timestamp(dates[i])

            # 影線（上下影線）
            ax.plot(
                [dt, dt],
                [lows[i], highs[i]],
                color=color, linewidth=0.8,
            )

            # K 棒實體
            body_bottom = min(opens[i], closes[i])
            body_height = abs(closes[i] - opens[i])
            if body_height < 1e-10:
                body_height = (highs[i] - lows[i]) * 0.01  # 十字線

            rect = plt.Rectangle(
                (mdates.date2num(dt) - mdates.date2num(dt + width) / 2
                 + mdates.date2num(dt) / 2,
                 body_bottom),
                width=mdates.date2num(dt + width) - mdates.date2num(dt),
                height=body_height,
                facecolor=color if is_up else color,
                edgecolor=color,
                alpha=0.9,
            )

        # 用更簡單的方式繪製（直接用 bar）
        ax.clear()

        up_mask = closes >= opens
        down_mask = ~up_mask

        # 影線
        for i in range(len(df)):
            dt = pd.Timestamp(dates[i])
            color = COLOR_UP if closes[i] >= opens[i] else COLOR_DOWN
            ax.vlines(dt, lows[i], highs[i], color=color, linewidth=0.6)

        # 實體（用 bar）
        if len(dates) >= 2:
            bar_width = (pd.Timestamp(dates[1]) - pd.Timestamp(dates[0])) * 0.6
        else:
            bar_width = pd.Timedelta(hours=1)

        # 漲
        if up_mask.any():
            up_dates = pd.to_datetime(dates[up_mask])
            ax.bar(up_dates, closes[up_mask] - opens[up_mask],
                   bottom=opens[up_mask], width=bar_width,
                   color=COLOR_UP, edgecolor=COLOR_UP, linewidth=0.5)

        # 跌
        if down_mask.any():
            dn_dates = pd.to_datetime(dates[down_mask])
            ax.bar(dn_dates, opens[down_mask] - closes[down_mask],
                   bottom=closes[down_mask], width=bar_width,
                   color=COLOR_DOWN, edgecolor=COLOR_DOWN, linewidth=0.5)

        ax.grid(True, alpha=0.2)

    # ================================================================
    # Internal: SMA 線
    # ================================================================

    def _plot_sma_lines(self, ax, df: pd.DataFrame):
        """繪製 SMA 均線（自動跳過 NaN 區段）"""
        dates = pd.to_datetime(df["datetime"].values)
        for period in self.config.sma_periods:
            col = f"SMA_{period}"
            if col not in df.columns:
                sma = df["close"].rolling(window=period, min_periods=period).mean()
            else:
                sma = df[col]

            color = COLOR_SMA.get(period, "#ffffff")
            # 使用 dropna 避免 NaN 斷線（前期預熱段自然留白）
            valid_mask = sma.notna().values
            if valid_mask.any():
                ax.plot(dates[valid_mask], sma.values[valid_mask],
                        color=color, linewidth=1.2,
                        alpha=0.85, label=f"SMA {period}")

    # ================================================================
    # Internal: Volume + 突兀量
    # ================================================================

    def _plot_volume_with_spikes(self, ax, df: pd.DataFrame) -> list:
        """
        繪製 volume 並標記突兀量

        Returns:
            spike_indices: 突兀量的 index 列表
        """
        dates = pd.to_datetime(df["datetime"].values)
        volumes = df["volume"].values
        closes = df["close"].values
        opens = df["open"].values

        colors = []
        spike_indices = []

        for i in range(len(volumes)):
            is_up = closes[i] >= opens[i]
            base_color = COLOR_UP if is_up else COLOR_DOWN

            # 突兀量判斷
            if (i > 0
                and volumes[i - 1] > 0
                and volumes[i] > volumes[i - 1] * self.config.volume_spike_ratio):
                colors.append(COLOR_SPIKE)
                spike_indices.append(i)
            else:
                colors.append(base_color)

        # 計算寬度
        if len(dates) >= 2:
            bar_width = (dates[1] - dates[0]) * 0.7
        else:
            bar_width = pd.Timedelta(hours=1)

        ax.bar(dates, volumes, width=bar_width, color=colors, alpha=0.8)

        # 突兀量標記
        for idx in spike_indices:
            ax.annotate(
                "\u2605",  # ★
                xy=(dates[idx], volumes[idx]),
                fontsize=12, color="#FF6F00",
                ha="center", va="bottom",
                fontweight="bold",
            )
            if idx > 0 and volumes[idx - 1] > 0:
                ratio = volumes[idx] / volumes[idx - 1]
                ax.annotate(
                    f"{ratio:.1f}x",
                    xy=(dates[idx], volumes[idx]),
                    xytext=(0, 10), textcoords="offset points",
                    fontsize=7, color="#FF6F00", ha="center",
                )

        ax.grid(True, alpha=0.2)
        return spike_indices

    # ================================================================
    # Internal: 突兀量在價格圖上的標記
    # ================================================================

    def _mark_spikes_on_price(self, ax, df: pd.DataFrame, spike_indices: list):
        """在價格圖的對應位置畫垂直虛線標記突兀量"""
        if not spike_indices:
            return
        dates = pd.to_datetime(df["datetime"].values)
        for idx in spike_indices:
            ax.axvline(
                x=dates[idx], color=COLOR_SPIKE,
                linestyle=":", linewidth=0.5, alpha=0.4,
            )

    # ================================================================
    # Internal: 數據準備
    # ================================================================

    def _prepare_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """確保 DataFrame 有 datetime 欄位，並計算準確的 SMA"""
        result = df.copy()
        if "datetime" not in result.columns and "timestamp" in result.columns:
            result["datetime"] = pd.to_datetime(result["timestamp"], unit="s", utc=True)
        if "datetime" not in result.columns:
            logger.warning("DataFrame has no timestamp or datetime column")
            return pd.DataFrame()

        # 計算 SMA（使用 min_periods=period 確保準確性）
        # 如果 df 已包含 SMA 預熱段，前期 NaN 是正常的
        for period in self.config.sma_periods:
            col = f"SMA_{period}"
            if col not in result.columns:
                result[col] = result["close"].rolling(
                    window=period, min_periods=period
                ).mean()

        return result

    def _format_xaxis(self, ax, df: pd.DataFrame):
        """格式化 X 軸時間顯示"""
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
        ax.tick_params(axis="x", rotation=30)

    # ================================================================
    # Internal: 對數座標
    # ================================================================

    @staticmethod
    def _apply_log_scale(ax):
        """
        將價格軸設為對數座標（log scale）

        使用 ScalarFormatter 使 Y 軸標籤顯示實際價格
        （而非 10^n 科學記號），更貼近 TradingView 的顯示方式。
        """
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.yaxis.get_major_formatter().set_scientific(False)
        ax.yaxis.get_major_formatter().set_useOffset(False)
        # 自動選擇合理的 tick 數量
        ax.yaxis.set_minor_formatter(mticker.NullFormatter())

    # ================================================================
    # Internal: 匹配區間分界線
    # ================================================================

    def _draw_match_boundary(self, ax_price, ax_vol, df: pd.DataFrame, context_bars: int):
        """
        在前期上下文與匹配區間的分界處畫垂直虛線

        Args:
            ax_price: 價格圖的 axes
            ax_vol: 成交量圖的 axes（可為 None）
            df: 完整 DataFrame（含前期上下文）
            context_bars: 前期上下文 K 棒數量
        """
        if context_bars <= 0 or context_bars >= len(df):
            return

        boundary_dt = pd.Timestamp(df["datetime"].iloc[context_bars])

        for ax in [ax_price, ax_vol]:
            if ax is not None:
                ax.axvline(
                    x=boundary_dt, color="#00E5FF",
                    linestyle="--", linewidth=1.5, alpha=0.7,
                )

        # 在價格圖上標註文字
        ax_price.annotate(
            "Match Start ▸",
            xy=(boundary_dt, ax_price.get_ylim()[1]),
            xytext=(-8, -15), textcoords="offset points",
            fontsize=7, color="#00E5FF",
            ha="right", va="top",
            fontweight="bold",
        )
