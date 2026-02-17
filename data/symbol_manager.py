"""
Symbol Manager — 交易對列表管理
================================

管理活躍 USDT 永續合約交易對列表：
- 從 Binance 拉取完整列表並快取到本地 JSON
- 可過濾排除特定 symbol（如低流動性、新上線）
- 掃描時提供快速存取（不需每次都打 API）
"""

import os
import json
import time
import logging
from typing import List, Optional, Set

logger = logging.getLogger(__name__)

# 常見的異常 symbol（流動性極低、即將下架等），可手動維護
DEFAULT_EXCLUDE = set()


class SymbolManager:
    """管理活躍交易對列表"""

    CACHE_FILENAME = "active_symbols.json"

    def __init__(self, config, binance_client=None):
        self.config = config
        self.binance_client = binance_client
        self._cache_path = os.path.join(
            os.path.dirname(config.cache_meta_file), self.CACHE_FILENAME
        )
        self._symbols: Optional[List[str]] = None
        self._exclude: Set[str] = set(DEFAULT_EXCLUDE)

    def get_active_symbols(self, force_refresh: bool = False) -> List[str]:
        """
        取得活躍交易對列表

        優先從本地快取讀取。若快取不存在或 force_refresh，
        則從 Binance API 拉取。
        快取有效期預設 24 小時。

        Returns:
            排序後的 symbol 列表，e.g. ["BTCUSDT", "ETHUSDT", ...]
        """
        if self._symbols is not None and not force_refresh:
            return self._symbols

        # 嘗試從快取讀取
        if not force_refresh:
            cached = self._load_cache()
            if cached is not None:
                self._symbols = cached
                return self._symbols

        # 從 API 拉取
        if self.binance_client is None:
            raise RuntimeError(
                "No cached symbols and no binance_client provided. "
                "Either provide binance_client or ensure cache exists."
            )

        all_symbols = self.binance_client.get_all_usdt_perpetual_symbols()

        # 過濾排除列表
        filtered = [s for s in all_symbols if s not in self._exclude]

        self._symbols = filtered
        self._save_cache(filtered)

        logger.info(
            f"Symbol list refreshed: {len(filtered)} active "
            f"({len(all_symbols) - len(filtered)} excluded)"
        )
        return self._symbols

    def add_exclude(self, symbols: List[str]) -> None:
        """新增排除的 symbol"""
        self._exclude.update(symbols)
        if self._symbols is not None:
            self._symbols = [s for s in self._symbols if s not in self._exclude]

    def remove_exclude(self, symbols: List[str]) -> None:
        """移除排除的 symbol"""
        self._exclude -= set(symbols)

    def get_exclude_list(self) -> List[str]:
        """取得排除列表"""
        return sorted(self._exclude)

    def symbol_count(self) -> int:
        """取得活躍 symbol 數量"""
        symbols = self.get_active_symbols()
        return len(symbols)

    def _load_cache(self) -> Optional[List[str]]:
        """從本地 JSON 讀取快取的 symbol 列表"""
        if not os.path.exists(self._cache_path):
            return None

        try:
            with open(self._cache_path, "r") as f:
                data = json.load(f)

            # 檢查快取有效期（24 小時）
            cached_at = data.get("cached_at", 0)
            age_hours = (time.time() - cached_at) / 3600
            if age_hours > 24:
                logger.info(f"Symbol cache expired ({age_hours:.1f}h old), will refresh")
                return None

            symbols = data.get("symbols", [])
            logger.info(f"Loaded {len(symbols)} symbols from cache ({age_hours:.1f}h old)")
            return symbols

        except (json.JSONDecodeError, IOError, KeyError) as e:
            logger.warning(f"Failed to load symbol cache: {e}")
            return None

    def _save_cache(self, symbols: List[str]) -> None:
        """將 symbol 列表快取到本地 JSON"""
        os.makedirs(os.path.dirname(self._cache_path), exist_ok=True)
        data = {
            "symbols": symbols,
            "count": len(symbols),
            "cached_at": time.time(),
        }
        with open(self._cache_path, "w") as f:
            json.dump(data, f, indent=2)