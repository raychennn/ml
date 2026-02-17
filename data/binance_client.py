"""
Binance API Client
==================
封裝 Binance Futures API 的 K 線數據取得邏輯。

核心能力：
- 增量抓取：只拿 last_timestamp 之後的新 K 棒
- 自動分頁：超過 1500 根時自動多次請求
- Rate limit 保護：請求間自動 sleep
- Retry with exponential backoff
- 取得所有 USDT 永續合約交易對列表
"""

import time
import logging
import numpy as np
import pandas as pd
from typing import Optional, List, Tuple
from binance.client import Client
from binance.exceptions import BinanceAPIException

logger = logging.getLogger(__name__)


class BinanceClient:
    """Binance Futures K-line data client"""

    # Binance Futures K 線欄位名稱
    RAW_COLUMNS = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades",
        "taker_buy_base", "taker_buy_quote", "ignore",
    ]

    # 最終輸出的欄位（與你現有系統一致）
    OUTPUT_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]

    def __init__(self, config):
        self.config = config
        self.client = Client(
            requests_params={"timeout": config.api_request_timeout}
        )
        self._request_count = 0
        self._last_request_time = 0.0

    # ================================================================
    # Public API
    # ================================================================

    def get_all_usdt_perpetual_symbols(self) -> List[str]:
        """
        取得所有 Binance USDT 永續合約交易對

        Returns:
            sorted list, e.g. ["BTCUSDT", "ETHUSDT", "SOLUSDT", ...]
        """
        try:
            info = self.client.futures_exchange_info()
        except Exception as e:
            logger.error(f"Failed to fetch exchange info: {e}")
            raise

        symbols = set()
        for item in info["symbols"]:
            # 只取 USDT 保證金的永續合約（排除交割合約）
            if (
                item.get("contractType") == "PERPETUAL"
                and item.get("quoteAsset") == "USDT"
                and item.get("status") == "TRADING"
            ):
                symbols.add(item["symbol"])

        result = sorted(symbols)
        logger.info(f"Found {len(result)} active USDT perpetual symbols")
        return result

    def fetch_klines(
        self,
        symbol: str,
        timeframe: str,
        start_ts: int,
        end_ts: Optional[int] = None,
    ) -> Optional[pd.DataFrame]:
        """
        抓取 K 線數據（自動分頁）

        Args:
            symbol: e.g. "BTCUSDT"
            timeframe: e.g. "4h"
            start_ts: 起始 Unix timestamp（秒）
            end_ts: 結束 Unix timestamp（秒），預設 now

        Returns:
            DataFrame with columns: [timestamp, open, high, low, close, volume]
            timestamp 為 Unix seconds (int64)
            如果失敗返回 None
        """
        if end_ts is None:
            end_ts = int(time.time())

        interval = self.config.timeframe_to_binance_interval(timeframe)
        start_ms = int(start_ts) * 1000
        end_ms = int(end_ts) * 1000

        all_rows = []
        current_ms = start_ms
        page = 0

        while current_ms < end_ms:
            page += 1
            rows = self._fetch_one_page(symbol, interval, current_ms, end_ms)

            if rows is None:
                # API 錯誤，已在內部 log
                return None

            if len(rows) == 0:
                break

            all_rows.extend(rows)

            # 移動到最後一根 K 棒的 close_time + 1ms
            last_close_time_ms = int(rows[-1][6])
            current_ms = last_close_time_ms + 1

            # 如果返回不足 1500 根，代表已到最新
            if len(rows) < self.config.max_klines_per_request:
                break

            # 進度 log（每 5 頁）
            if page % 5 == 0:
                pct = min(100, (current_ms - start_ms) / max(1, end_ms - start_ms) * 100)
                logger.debug(f"  {symbol} {timeframe}: page {page}, {len(all_rows)} rows, {pct:.0f}%")

        if not all_rows:
            logger.warning(f"{symbol} {timeframe}: no data returned")
            return None

        # 轉 DataFrame
        df = pd.DataFrame(all_rows, columns=self.RAW_COLUMNS)

        # 轉換型別
        df["timestamp"] = (df["open_time"].astype(np.int64) // 1000).astype(np.int64)
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(np.float64)

        # 去重 + 排序
        df = df.drop_duplicates(subset=["timestamp"], keep="first")
        df = df.sort_values("timestamp").reset_index(drop=True)

        # 過濾到請求的時間範圍
        df = df[(df["timestamp"] >= start_ts) & (df["timestamp"] <= end_ts)]

        # 只保留輸出欄位
        df = df[self.OUTPUT_COLUMNS].reset_index(drop=True)

        return df

    def fetch_klines_incremental(
        self,
        symbol: str,
        timeframe: str,
        last_known_ts: Optional[int],
    ) -> Optional[pd.DataFrame]:
        """
        增量抓取：從 last_known_ts 之後開始

        如果 last_known_ts 為 None，代表全新 symbol，
        會從很早（2020-01-01）開始抓。

        Args:
            symbol: e.g. "BTCUSDT"
            timeframe: e.g. "4h"
            last_known_ts: 最後已知 K 棒的 open timestamp（秒）

        Returns:
            只包含新 K 棒的 DataFrame，或 None
        """
        if last_known_ts is not None:
            # 從最後已知 K 棒的下一根開始
            interval_sec = self.config.timeframe_to_seconds(timeframe)
            start_ts = last_known_ts + interval_sec
        else:
            # 全新 symbol：從 2020-01-01 開始（覆蓋大部分幣種上線日）
            start_ts = 1577836800  # 2020-01-01 00:00:00 UTC

        end_ts = int(time.time())

        # 如果 start > end，不需要更新
        if start_ts >= end_ts:
            return None

        return self.fetch_klines(symbol, timeframe, start_ts, end_ts)

    # ================================================================
    # Internal
    # ================================================================

    def _fetch_one_page(
        self, symbol: str, interval: str, start_ms: int, end_ms: int
    ) -> Optional[list]:
        """抓取單頁 K 線（最多 1500 根），帶 retry"""
        for attempt in range(1, self.config.api_max_retries + 1):
            self._rate_limit_sleep()
            try:
                rows = self.client.futures_klines(
                    symbol=symbol,
                    interval=interval,
                    startTime=int(start_ms),
                    endTime=int(end_ms),
                    limit=self.config.max_klines_per_request,
                )
                return rows

            except BinanceAPIException as e:
                if e.code == -1121:
                    # Invalid symbol — 不需要重試
                    logger.warning(f"{symbol}: invalid symbol, skipping")
                    return None
                elif e.code == -1003:
                    # Rate limit exceeded
                    wait = self.config.api_retry_backoff ** attempt * 10
                    logger.warning(
                        f"{symbol}: rate limit hit, waiting {wait:.0f}s "
                        f"(attempt {attempt}/{self.config.api_max_retries})"
                    )
                    time.sleep(wait)
                else:
                    wait = self.config.api_retry_backoff ** attempt
                    logger.warning(
                        f"{symbol}: API error {e.code} '{e.message}', "
                        f"retry in {wait:.1f}s (attempt {attempt}/{self.config.api_max_retries})"
                    )
                    time.sleep(wait)

            except Exception as e:
                wait = self.config.api_retry_backoff ** attempt
                logger.warning(
                    f"{symbol}: unexpected error '{e}', "
                    f"retry in {wait:.1f}s (attempt {attempt}/{self.config.api_max_retries})"
                )
                time.sleep(wait)

        logger.error(f"{symbol}: all {self.config.api_max_retries} retries failed")
        return None

    def _rate_limit_sleep(self):
        """確保請求間隔不低於 api_sleep_seconds"""
        now = time.time()
        elapsed = now - self._last_request_time
        if elapsed < self.config.api_sleep_seconds:
            time.sleep(self.config.api_sleep_seconds - elapsed)
        self._last_request_time = time.time()
        self._request_count += 1
