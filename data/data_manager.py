"""
Data Manager — Parquet 分區存儲 + 增量更新
==========================================

存儲結構:
    {parquet_dir}/
        timeframe=30m/
            BTCUSDT.parquet
            ETHUSDT.parquet
        timeframe=1h/
            ...
        timeframe=2h/
            ...
        timeframe=4h/
            ...

    {cache_meta_file}:
        {
            "BTCUSDT_4h": {"last_ts": 1718000000, "rows": 6543, "updated_at": "2025-..."},
            "ETHUSDT_4h": {...},
            ...
        }

設計原則:
    - 每個 symbol × timeframe = 一個 parquet 檔（小檔策略，方便增量更新）
    - 用 snappy 壓縮，讀寫速度最快
    - cache_meta.json 追蹤每個檔案的最後 timestamp，啟動時自動恢復
"""

import os
import json
import time
import logging
import threading
from typing import Optional, List, Dict
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)


# Parquet schema 定義（確保型別一致性）
PARQUET_SCHEMA = pa.schema([
    ("timestamp", pa.int64()),
    ("open", pa.float64()),
    ("high", pa.float64()),
    ("low", pa.float64()),
    ("close", pa.float64()),
    ("volume", pa.float64()),
])


class DataManager:
    """Parquet-based data storage with incremental update support"""

    def __init__(self, config):
        self.config = config
        self.parquet_dir = config.parquet_dir
        self.cache_meta_file = config.cache_meta_file
        self._cache_meta = None  # lazy load
        self._meta_lock = threading.Lock()  # protects _cache_meta and meta file
        self._write_locks: Dict[str, threading.Lock] = {}  # per-file write locks
        self._write_locks_guard = threading.Lock()  # protects _write_locks dict

    # ================================================================
    # Parquet 檔案路徑
    # ================================================================

    def _parquet_path(self, symbol: str, timeframe: str) -> str:
        """單一 symbol 的 parquet 檔路徑"""
        return os.path.join(
            self.parquet_dir, f"timeframe={timeframe}", f"{symbol}.parquet"
        )

    def _timeframe_dir(self, timeframe: str) -> str:
        """時框目錄路徑"""
        return os.path.join(self.parquet_dir, f"timeframe={timeframe}")

    def _get_write_lock(self, symbol: str, timeframe: str) -> threading.Lock:
        """Get or create a per-file write lock"""
        key = f"{symbol}_{timeframe}"
        with self._write_locks_guard:
            if key not in self._write_locks:
                self._write_locks[key] = threading.Lock()
            return self._write_locks[key]

    # ================================================================
    # 讀取
    # ================================================================

    def read_symbol(
        self,
        symbol: str,
        timeframe: str,
        start_ts: Optional[int] = None,
        end_ts: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        讀取單一 symbol 的數據

        Args:
            symbol: e.g. "BTCUSDT"
            timeframe: e.g. "4h"
            start_ts: 起始 timestamp（可選）
            end_ts: 結束 timestamp（可選）

        Returns:
            DataFrame [timestamp, open, high, low, close, volume]
            如果檔案不存在返回空 DataFrame
        """
        path = self._parquet_path(symbol, timeframe)
        if not os.path.exists(path):
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

        try:
            table = pq.read_table(path)
            df = table.to_pandas()

            # 時間範圍過濾
            if start_ts is not None:
                df = df[df["timestamp"] >= start_ts]
            if end_ts is not None:
                df = df[df["timestamp"] <= end_ts]

            return df.reset_index(drop=True)

        except Exception as e:
            logger.error(f"Failed to read {path}: {e}")
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

    def read_symbol_tail(
        self,
        symbol: str,
        timeframe: str,
        n_bars: int,
    ) -> pd.DataFrame:
        """
        讀取 symbol 的最後 N 根 K 棒（掃描時用，避免載入全歷史）

        比讀全部再切片更高效：先讀 metadata 算 row count，
        但由於單檔不大，直接讀全部再 tail 也可接受。
        """
        df = self.read_symbol(symbol, timeframe)
        if len(df) <= n_bars:
            return df
        return df.tail(n_bars).reset_index(drop=True)

    def bulk_read_timeframe(
        self,
        timeframe: str,
        symbols: Optional[List[str]] = None,
        tail_bars: Optional[int] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        批次讀取整個時框的所有（或指定）symbol

        Args:
            timeframe: e.g. "4h"
            symbols: 限定讀取的 symbol 列表（None = 全部）
            tail_bars: 只讀最後 N 根（節省記憶體）

        Returns:
            {symbol: DataFrame}
        """
        tf_dir = self._timeframe_dir(timeframe)
        if not os.path.exists(tf_dir):
            return {}

        result = {}
        files = [f for f in os.listdir(tf_dir) if f.endswith(".parquet")]

        for fname in files:
            sym = fname.replace(".parquet", "")
            if symbols is not None and sym not in symbols:
                continue

            if tail_bars is not None:
                df = self.read_symbol_tail(sym, timeframe, tail_bars)
            else:
                df = self.read_symbol(sym, timeframe)

            if not df.empty:
                result[sym] = df

        logger.info(
            f"Loaded {len(result)} symbols for timeframe={timeframe}"
            + (f" (tail {tail_bars} bars)" if tail_bars else "")
        )
        return result

    # ================================================================
    # 寫入 + 增量更新
    # ================================================================

    def write_symbol(
        self,
        symbol: str,
        timeframe: str,
        df: pd.DataFrame,
    ) -> None:
        """
        完整寫入（覆寫）一個 symbol 的 parquet 檔

        用於初次下載或全量重建。Thread-safe via per-file lock.
        使用原子寫入（先寫暫存檔再 rename）防止寫入中途斷電導致損壞。
        """
        if df.empty:
            return

        lock = self._get_write_lock(symbol, timeframe)
        with lock:
            self._write_parquet_locked(symbol, timeframe, df)

    def _write_parquet_locked(
        self,
        symbol: str,
        timeframe: str,
        df: pd.DataFrame,
    ) -> None:
        """
        內部寫入方法（呼叫者必須已持有寫鎖）。
        使用原子寫入：先寫 .tmp 暫存檔，完成後 rename 覆蓋目標檔。
        """
        path = self._parquet_path(symbol, timeframe)
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # 確保排序 + 去重
        df = df.drop_duplicates(subset=["timestamp"], keep="first")
        df = df.sort_values("timestamp").reset_index(drop=True)

        # 原子寫入：先寫暫存檔，完成後 rename
        tmp_path = path + ".tmp"
        try:
            table = pa.Table.from_pandas(df, schema=PARQUET_SCHEMA, preserve_index=False)
            pq.write_table(table, tmp_path, compression="snappy")
            os.replace(tmp_path, path)  # 原子操作（同一 filesystem）
        except Exception:
            # 清理暫存檔
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
            raise

        # 更新 meta
        self._update_meta(symbol, timeframe, df)

    def incremental_update(
        self,
        symbol: str,
        timeframe: str,
        new_df: pd.DataFrame,
    ) -> int:
        """
        增量追加新 K 棒到現有 parquet

        整個 讀取→合併→寫入 流程在同一把鎖內完成，
        避免多線程並發導致數據遺失或 parquet 損壞。

        Args:
            symbol, timeframe: 標的
            new_df: 新數據（可能與舊數據有重疊，會自動去重）

        Returns:
            實際新增的行數
        """
        if new_df.empty:
            return 0

        lock = self._get_write_lock(symbol, timeframe)
        with lock:
            path = self._parquet_path(symbol, timeframe)

            if os.path.exists(path):
                try:
                    table = pq.read_table(path)
                    existing_df = table.to_pandas()
                except Exception as e:
                    logger.error(f"Failed to read {path} during update: {e}")
                    existing_df = pd.DataFrame(
                        columns=["timestamp", "open", "high", "low", "close", "volume"]
                    )

                if not existing_df.empty:
                    last_ts = existing_df["timestamp"].max()
                    # 只取真正新的行
                    new_rows = new_df[new_df["timestamp"] > last_ts]
                    if new_rows.empty:
                        return 0
                    combined = pd.concat([existing_df, new_rows], ignore_index=True)
                else:
                    combined = new_df
                    new_rows = new_df
            else:
                combined = new_df
                new_rows = new_df

            # 在鎖內寫入（不經過 write_symbol，避免雙重鎖）
            self._write_parquet_locked(symbol, timeframe, combined)

            added = len(new_rows)
            if added > 0:
                logger.debug(f"{symbol} {timeframe}: +{added} rows (total {len(combined)})")
            return added

    # ================================================================
    # Cache Meta 管理
    # ================================================================

    def _load_meta(self) -> dict:
        """載入 cache_meta.json (thread-safe)"""
        with self._meta_lock:
            if self._cache_meta is not None:
                return self._cache_meta

            if os.path.exists(self.cache_meta_file):
                try:
                    with open(self.cache_meta_file, "r") as f:
                        self._cache_meta = json.load(f)
                except (json.JSONDecodeError, IOError) as e:
                    logger.warning(f"Cache meta corrupted, resetting: {e}")
                    self._cache_meta = {}
            else:
                self._cache_meta = {}

            return self._cache_meta

    def _save_meta(self) -> None:
        """儲存 cache_meta.json (must hold _meta_lock)。使用原子寫入。"""
        if self._cache_meta is None:
            return

        os.makedirs(os.path.dirname(self.cache_meta_file), exist_ok=True)
        tmp_path = self.cache_meta_file + ".tmp"
        try:
            with open(tmp_path, "w") as f:
                json.dump(self._cache_meta, f, indent=2)
            os.replace(tmp_path, self.cache_meta_file)
        except Exception:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
            raise

    def _update_meta(self, symbol: str, timeframe: str, df: pd.DataFrame) -> None:
        """更新單一 symbol 的 meta (thread-safe)"""
        with self._meta_lock:
            if self._cache_meta is None:
                self._cache_meta = {}
            key = f"{symbol}_{timeframe}"
            self._cache_meta[key] = {
                "last_ts": int(df["timestamp"].max()),
                "first_ts": int(df["timestamp"].min()),
                "rows": len(df),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
            self._save_meta()

    def get_last_timestamp(self, symbol: str, timeframe: str) -> Optional[int]:
        """
        取得該 symbol 最後已知的 timestamp

        優先從 meta 讀取（快），如果沒有則從 parquet 讀取（慢但準）。
        Returns None 如果沒有任何數據。
        """
        meta = self._load_meta()
        key = f"{symbol}_{timeframe}"

        if key in meta:
            return meta[key]["last_ts"]

        # Fallback: 直接讀 parquet
        df = self.read_symbol(symbol, timeframe)
        if not df.empty:
            last_ts = int(df["timestamp"].max())
            # 順便更新 meta
            self._update_meta(symbol, timeframe, df)
            return last_ts

        return None

    # ================================================================
    # 統計 & 工具
    # ================================================================

    def get_storage_stats(self) -> dict:
        """取得存儲統計資訊"""
        meta = self._load_meta()
        stats = {
            "total_symbols": 0,
            "by_timeframe": {},
            "total_rows": 0,
            "total_size_mb": 0.0,
        }

        for tf in self.config.scan_timeframes:
            tf_dir = self._timeframe_dir(tf)
            if not os.path.exists(tf_dir):
                stats["by_timeframe"][tf] = {"symbols": 0, "size_mb": 0.0}
                continue

            files = [f for f in os.listdir(tf_dir) if f.endswith(".parquet")]
            total_size = sum(
                os.path.getsize(os.path.join(tf_dir, f)) for f in files
            )
            stats["by_timeframe"][tf] = {
                "symbols": len(files),
                "size_mb": round(total_size / 1024 / 1024, 1),
            }
            stats["total_symbols"] = max(stats["total_symbols"], len(files))
            stats["total_size_mb"] += stats["by_timeframe"][tf]["size_mb"]

        for v in meta.values():
            stats["total_rows"] += v.get("rows", 0)

        stats["total_size_mb"] = round(stats["total_size_mb"], 1)
        return stats

    def rebuild_meta_from_files(self) -> int:
        """
        從 parquet 檔案重建 cache_meta.json

        用於 meta 損壞或首次初始化時。
        """
        logger.info("Rebuilding cache meta from parquet files...")
        with self._meta_lock:
            self._cache_meta = {}

        count = 0
        for tf in self.config.scan_timeframes:
            tf_dir = self._timeframe_dir(tf)
            if not os.path.exists(tf_dir):
                continue

            for fname in os.listdir(tf_dir):
                if not fname.endswith(".parquet"):
                    continue
                sym = fname.replace(".parquet", "")
                df = self.read_symbol(sym, tf)
                if not df.empty:
                    self._update_meta(sym, tf, df)
                    count += 1

        logger.info(f"Rebuilt meta for {count} symbol-timeframe pairs")
        return count

    def list_symbols_for_timeframe(self, timeframe: str) -> List[str]:
        """列出某個時框下已有數據的所有 symbol"""
        tf_dir = self._timeframe_dir(timeframe)
        if not os.path.exists(tf_dir):
            return []
        return sorted([
            f.replace(".parquet", "")
            for f in os.listdir(tf_dir)
            if f.endswith(".parquet")
        ])

    def validate_integrity(self, symbol: str, timeframe: str) -> dict:
        """
        驗證單一 symbol 的數據完整性

        檢查:
        - 時間戳連續性（是否有缺失的 K 棒）
        - 價格合理性（是否有 0 或負值）
        - 數據新鮮度
        """
        df = self.read_symbol(symbol, timeframe)
        if df.empty:
            return {"valid": False, "error": "no_data"}

        result = {"valid": True, "rows": len(df), "issues": []}

        # 檢查時間戳連續性
        interval_sec = self.config.timeframe_to_seconds(timeframe)
        diffs = np.diff(df["timestamp"].values)
        expected_diff = interval_sec

        # 允許少量缺失（交易所維護等），但標記
        gaps = np.where(diffs > expected_diff * 1.5)[0]
        if len(gaps) > 0:
            gap_ratio = len(gaps) / len(diffs)
            if gap_ratio > 0.05:  # 超過 5% 缺失
                result["issues"].append(f"many_gaps: {len(gaps)} ({gap_ratio:.1%})")
            else:
                result["issues"].append(f"minor_gaps: {len(gaps)}")

        # 檢查價格合理性
        for col in ["open", "high", "low", "close"]:
            if (df[col] <= 0).any():
                result["valid"] = False
                result["issues"].append(f"{col}_has_zero_or_negative")

        # 檢查 volume 非負
        if (df["volume"] < 0).any():
            result["issues"].append("negative_volume")

        # 數據新鮮度
        last_ts = df["timestamp"].max()
        age_hours = (time.time() - last_ts) / 3600
        result["age_hours"] = round(age_hours, 1)
        result["first_date"] = datetime.fromtimestamp(df["timestamp"].min(), tz=timezone.utc).isoformat()
        result["last_date"] = datetime.fromtimestamp(df["timestamp"].max(), tz=timezone.utc).isoformat()

        return result
