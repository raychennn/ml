"""
Reference Manager — Reference Pattern CRUD
============================================

管理 reference trends 的新增/刪除/列出。
每個 reference 定義一段歷史型態（symbol + 時間範圍 + 時框 + 標籤）。

存儲格式: JSON
    {
        "references": [
            {
                "id": "AVAX_1h_standard",
                "symbol": "AVAXUSDT",
                "timeframe": "1h",
                "start_ts": 1699531200,
                "end_ts": 1699920000,
                "label": "standard",
                "description": "AVAX 2023-11 standard uptrend",
                "created_at": "2025-01-15T10:00:00Z",
                "active": true
            },
            ...
        ]
    }
"""

import os
import json
import logging
import numpy as np
import pytz
from datetime import datetime, timezone
from typing import List, Optional, Dict
from dataclasses import dataclass, field, asdict

logger = logging.getLogger(__name__)


@dataclass
class ReferencePattern:
    """單一 reference pattern 的定義"""
    id: str                     # 唯一識別碼，e.g. "AVAX_1h_standard"
    symbol: str                 # e.g. "AVAXUSDT"
    timeframe: str              # e.g. "1h"
    start_ts: int               # 型態起始 Unix timestamp (秒)
    end_ts: int                 # 型態結束 Unix timestamp (秒)
    label: str                  # 型態標籤，e.g. "standard", "uptrend"
    description: str = ""       # 可選描述
    created_at: str = ""        # ISO format
    active: bool = True         # 是否啟用（False = 已歸檔）

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()
        if not self.id:
            base = self.symbol.replace("USDT", "")
            self.id = f"{base}_{self.timeframe}_{self.label}"


class ReferenceManager:
    """Reference pattern JSON CRUD with optional Telegram notifications"""

    def __init__(self, config, data_manager=None, chart_generator=None, notifier=None):
        """
        Args:
            config: SystemConfig
            data_manager: Optional DataManager (needed for chart generation on add)
            chart_generator: Optional ChartGenerator (generates reference preview)
            notifier: Optional TelegramNotifier (sends chart/messages to Telegram)

        If data_manager/chart_generator/notifier are not provided,
        notifications are silently skipped. Existing code is unaffected.
        """
        self.config = config
        self.ref_file = config.reference_file
        self._references: Optional[List[ReferencePattern]] = None
        self._data_manager = data_manager
        self._chart_generator = chart_generator
        self._notifier = notifier

    # ================================================================
    # CRUD
    # ================================================================

    def list_references(self, active_only: bool = True) -> List[ReferencePattern]:
        """列出所有 reference patterns"""
        refs = self._load()
        if active_only:
            return [r for r in refs if r.active]
        return refs

    def get_reference(self, ref_id: str) -> Optional[ReferencePattern]:
        """取得單一 reference"""
        refs = self._load()
        for r in refs:
            if r.id == ref_id:
                return r
        return None

    def add_reference(
        self,
        symbol: str,
        timeframe: str,
        start_ts: int,
        end_ts: int,
        label: str,
        description: str = "",
    ) -> ReferencePattern:
        """
        新增 reference pattern

        Args:
            symbol: e.g. "AVAXUSDT"
            timeframe: e.g. "1h"
            start_ts: 型態起始 timestamp（秒）
            end_ts: 型態結束 timestamp（秒）
            label: 型態標籤
            description: 可選描述

        Returns:
            新建的 ReferencePattern

        Raises:
            ValueError: 如果 id 已存在
        """
        ref = ReferencePattern(
            id="",  # 會在 __post_init__ 中自動生成
            symbol=symbol,
            timeframe=timeframe,
            start_ts=start_ts,
            end_ts=end_ts,
            label=label,
            description=description,
        )

        refs = self._load()

        # 檢查重複
        if any(r.id == ref.id for r in refs):
            raise ValueError(f"Reference '{ref.id}' already exists")

        refs.append(ref)
        self._save(refs)
        logger.info(f"Added reference: {ref.id}")

        self._notify_reference_added(ref)
        return ref

    def add_reference_by_datetime(
        self,
        symbol: str,
        timeframe: str,
        start_dt: str,
        end_dt: str,
        label: str,
        description: str = "",
    ) -> "ReferencePattern":
        """
        Add reference using human-readable datetime strings in configured timezone.

        Datetimes are interpreted as config.reference_input_timezone (default: Asia/Taipei GMT+8).

        Args:
            symbol: e.g. "AVAXUSDT"
            timeframe: e.g. "4h"
            start_dt: Start datetime, e.g. "2023-11-09 12:00"
            end_dt: End datetime, e.g. "2023-11-14 00:00"
            label: Pattern label
            description: Optional description

        Returns:
            New ReferencePattern
        """
        tz = pytz.timezone(self.config.reference_input_timezone)

        start_local = datetime.strptime(start_dt, "%Y-%m-%d %H:%M")
        start_local = tz.localize(start_local)
        start_ts = int(start_local.timestamp())

        end_local = datetime.strptime(end_dt, "%Y-%m-%d %H:%M")
        end_local = tz.localize(end_local)
        end_ts = int(end_local.timestamp())

        # Validate: end must be after start
        if end_ts <= start_ts:
            raise ValueError(
                f"end_dt ({end_dt}) must be after start_dt ({start_dt}). "
                f"Check your dates — possibly a typo in the year?"
            )

        logger.info(
            f"Converting {start_dt} → {end_dt} "
            f"({self.config.reference_input_timezone}) to UTC timestamps: "
            f"{start_ts} → {end_ts}"
        )

        return self.add_reference(
            symbol=symbol,
            timeframe=timeframe,
            start_ts=start_ts,
            end_ts=end_ts,
            label=label,
            description=description,
        )

    def delete_reference(self, ref_id: str) -> bool:
        """
        刪除 reference（軟刪除：設 active=False）

        Returns:
            True if found and deleted, False if not found
        """
        refs = self._load()
        for r in refs:
            if r.id == ref_id:
                r.active = False
                self._save(refs)
                logger.info(f"Deactivated reference: {ref_id}")
                self._notify_reference_deleted(r)
                return True
        return False

    def hard_delete_reference(self, ref_id: str) -> bool:
        """硬刪除 reference"""
        refs = self._load()
        before = len(refs)
        refs = [r for r in refs if r.id != ref_id]
        if len(refs) < before:
            self._save(refs)
            logger.info(f"Hard deleted reference: {ref_id}")
            return True
        return False

    def reactivate_reference(self, ref_id: str) -> bool:
        """重新啟用已歸檔的 reference"""
        refs = self._load()
        for r in refs:
            if r.id == ref_id:
                r.active = True
                self._save(refs)
                logger.info(f"Reactivated reference: {ref_id}")
                return True
        return False

    def get_references_for_timeframe(self, timeframe: str) -> List[ReferencePattern]:
        """取得指定時框的所有活躍 reference"""
        return [
            r for r in self.list_references(active_only=True)
            if r.timeframe == timeframe
        ]

    def count(self, active_only: bool = True) -> int:
        """取得 reference 數量"""
        return len(self.list_references(active_only=active_only))

    # ================================================================
    # Initial Reference Seeding
    # ================================================================

    def seed_initial_references(self) -> int:
        """
        Seed initial reference patterns on first startup.

        Imports references from data.initial_references without sending
        Telegram notifications (silent seeding). Skips any references
        whose ID already exists.

        Returns:
            Number of newly added references.
        """
        try:
            from data.initial_references import INITIAL_REFERENCES
        except ImportError:
            logger.debug("No initial_references module found, skipping seed")
            return 0

        if not INITIAL_REFERENCES:
            return 0

        existing_refs = self._load()
        existing_ids = {r.id for r in existing_refs}

        if existing_ids:
            logger.info(
                f"Reference store has {len(existing_ids)} existing references, "
                f"checking for missing initial references..."
            )

        added = 0
        for ref_def in INITIAL_REFERENCES:
            try:
                tz = pytz.timezone(self.config.reference_input_timezone)

                start_local = datetime.strptime(ref_def["start_dt"], "%Y-%m-%d %H:%M")
                start_local = tz.localize(start_local)
                start_ts = int(start_local.timestamp())

                end_local = datetime.strptime(ref_def["end_dt"], "%Y-%m-%d %H:%M")
                end_local = tz.localize(end_local)
                end_ts = int(end_local.timestamp())

                if end_ts <= start_ts:
                    continue

                ref = ReferencePattern(
                    id="",
                    symbol=ref_def["symbol"],
                    timeframe=ref_def["timeframe"],
                    start_ts=start_ts,
                    end_ts=end_ts,
                    label=ref_def["label"],
                    description=ref_def.get("description", ""),
                )

                if ref.id in existing_ids:
                    continue

                existing_refs.append(ref)
                existing_ids.add(ref.id)
                added += 1
                logger.debug(f"Seeded initial reference: {ref.id}")

            except Exception as e:
                logger.warning(
                    f"Failed to seed reference "
                    f"{ref_def.get('symbol', '?')}: {e}"
                )

        if added > 0:
            self._save(existing_refs)
            logger.info(f"Seeded {added} initial reference(s)")
        else:
            logger.info("All initial references already present")

        return added

    # ================================================================
    # Notifications
    # ================================================================

    def _notify_reference_added(self, ref: ReferencePattern) -> None:
        """Generate chart and send Telegram notification for a new reference"""
        chart_path = None
        stats = {}

        # Generate chart if data_manager + chart_generator are available
        if self._data_manager and self._chart_generator:
            try:
                # 載入精確的 reference 區間（用於統計計算）
                ref_df = self._data_manager.read_symbol(
                    ref.symbol, ref.timeframe,
                    start_ts=ref.start_ts, end_ts=ref.end_ts,
                )

                # 載入含 SMA 預熱段的數據（用於圖表顯示完整 SMA）
                sma_lookback = self.config.sma_lookback
                padding_seconds = sma_lookback * self.config.timeframe_to_seconds(ref.timeframe)
                padded_start_ts = ref.start_ts - padding_seconds
                ref_df_padded = self._data_manager.read_symbol(
                    ref.symbol, ref.timeframe,
                    start_ts=padded_start_ts, end_ts=ref.end_ts,
                )

                # 計算實際 context bars
                if not ref_df_padded.empty:
                    ref_mask = ref_df_padded["timestamp"] >= ref.start_ts
                    context_bars = int(ref_mask.values.argmax()) if ref_mask.any() else 0
                else:
                    context_bars = 0

                if not ref_df.empty and len(ref_df) >= 3:
                    # Compute pattern stats for the chart overlay
                    close = ref_df["close"].values
                    volume = ref_df["volume"].values

                    pattern_return = (close[-1] / close[0] - 1) if close[0] > 0 else 0
                    returns = np.diff(close) / (close[:-1] + 1e-10)
                    volatility = float(np.std(returns)) if len(returns) > 0 else 0
                    cummax = np.maximum.accumulate(close)
                    drawdowns = (close - cummax) / (cummax + 1e-10)
                    max_dd = float(np.min(drawdowns))

                    # SMA alignment check（使用 padded 數據計算準確的 SMA）
                    padded_close = ref_df_padded["close"]
                    sma30_full = padded_close.rolling(30, min_periods=30).mean()
                    sma60_full = padded_close.rolling(60, min_periods=60).mean()
                    sma30_end = sma30_full.iloc[-1]
                    sma60_end = sma60_full.iloc[-1]
                    close_end = close[-1]

                    if pd.notna(sma30_end) and pd.notna(sma60_end):
                        if close_end > sma30_end > sma60_end:
                            sma_alignment = 1.0
                        elif close_end < sma30_end < sma60_end:
                            sma_alignment = -1.0
                        else:
                            sma_alignment = 0.0
                    else:
                        sma_alignment = 0.0

                    # Volume spikes
                    spike_ratio = self.config.volume_spike_ratio
                    vol_spikes = sum(
                        1 for i in range(1, len(volume))
                        if volume[i] > volume[i - 1] * spike_ratio
                    )

                    stats = {
                        "pattern_return": pattern_return,
                        "pattern_volatility": volatility,
                        "pattern_max_drawdown": max_dd,
                        "sma_alignment": sma_alignment,
                        "volume_spikes": vol_spikes,
                    }

                    chart_path = self._chart_generator.generate_reference_chart(
                        window_df=ref_df_padded,
                        symbol=ref.symbol,
                        timeframe=ref.timeframe,
                        label=ref.label,
                        description=ref.description,
                        stats=stats,
                        context_bars=context_bars,
                    )
                    if chart_path:
                        logger.info(f"Reference preview chart: {chart_path}")
                else:
                    logger.warning(
                        f"Not enough data to generate chart for {ref.id} "
                        f"({len(ref_df)} bars)"
                    )
            except Exception as e:
                logger.warning(f"Failed to generate reference chart: {e}")

        # Send to Telegram
        if self._notifier:
            try:
                # Build caption
                display_tz = pytz.timezone(self.config.reference_input_timezone)
                dt_start = datetime.fromtimestamp(ref.start_ts, tz=display_tz)
                dt_end = datetime.fromtimestamp(ref.end_ts, tz=display_tz)
                caption = (
                    f"<b>New Reference Added</b>\n"
                    f"<b>ID:</b> <code>{ref.id}</code>\n"
                    f"<b>Symbol:</b> {ref.symbol}\n"
                    f"<b>Timeframe:</b> {ref.timeframe}\n"
                    f"<b>Period:</b> {dt_start:%Y-%m-%d %H:%M} → {dt_end:%Y-%m-%d %H:%M}\n"
                    f"<b>Label:</b> {ref.label}\n"
                )
                if ref.description:
                    caption += f"<b>Desc:</b> {ref.description}\n"
                if stats:
                    ret = stats.get("pattern_return", 0)
                    caption += (
                        f"\n<b>Pattern Stats:</b>\n"
                        f"  Return: {ret:+.2%}\n"
                        f"  Volatility: {stats.get('pattern_volatility', 0):.4f}\n"
                        f"  Max DD: {stats.get('pattern_max_drawdown', 0):.2%}\n"
                        f"  Vol Spikes: {stats.get('volume_spikes', 0)}\n"
                    )

                if chart_path and os.path.exists(chart_path):
                    self._notifier.send_photo(chart_path, caption=caption)
                else:
                    self._notifier.send_message(caption)
            except Exception as e:
                logger.warning(f"Failed to send Telegram notification: {e}")

    def _notify_reference_deleted(self, ref: ReferencePattern) -> None:
        """Send Telegram notification when a reference is deactivated"""
        if not self._notifier:
            return

        try:
            active_count = len([r for r in self._load() if r.active])
            message = (
                f"<b>Reference Deactivated</b>\n"
                f"<b>ID:</b> <code>{ref.id}</code>\n"
                f"<b>Symbol:</b> {ref.symbol}  |  {ref.timeframe}\n"
                f"<b>Label:</b> {ref.label}\n"
                f"\n<i>Active references remaining: {active_count}</i>\n"
                f"<i>Re-run train_model.py to update models.</i>"
            )
            self._notifier.send_message(message)
        except Exception as e:
            logger.warning(f"Failed to send delete notification: {e}")

    # ================================================================
    # Persistence
    # ================================================================

    def _load(self) -> List[ReferencePattern]:
        """從 JSON 載入"""
        if self._references is not None:
            return self._references

        if not os.path.exists(self.ref_file):
            self._references = []
            return self._references

        try:
            with open(self.ref_file, "r") as f:
                data = json.load(f)

            self._references = [
                ReferencePattern(**item)
                for item in data.get("references", [])
            ]
        except (json.JSONDecodeError, IOError, TypeError) as e:
            logger.warning(f"Failed to load references: {e}")
            self._references = []

        return self._references

    def _save(self, refs: List[ReferencePattern]) -> None:
        """儲存到 JSON（使用原子寫入防止損壞）"""
        self._references = refs
        os.makedirs(os.path.dirname(self.ref_file), exist_ok=True)

        data = {
            "references": [asdict(r) for r in refs],
            "count": len(refs),
            "active_count": len([r for r in refs if r.active]),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

        tmp_path = self.ref_file + ".tmp"
        try:
            with open(tmp_path, "w") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            os.replace(tmp_path, self.ref_file)
        except Exception:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
            raise

    def reload(self) -> None:
        """強制重新載入（清除快取）"""
        self._references = None
