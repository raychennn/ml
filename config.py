"""
Crypto Pattern Recognition System — Central Configuration
==========================================================
所有可調參數集中在此，不再散落在各檔案頂部。
環境變數優先於預設值（用於 Zeabur 部署時注入 secrets）。
"""

import os
from dataclasses import dataclass, field
from typing import List


@dataclass
class SystemConfig:
    """系統全局配置"""

    # ================================================================
    # 路徑配置（Zeabur persistent volume 掛載在 /data）
    # 本地開發時可改為 ./data
    # ================================================================
    data_root: str = field(
        default_factory=lambda: os.environ.get("DATA_ROOT", "./data")
    )

    @property
    def parquet_dir(self) -> str:
        return os.path.join(self.data_root, "parquet")

    @property
    def model_dir(self) -> str:
        return os.path.join(self.data_root, "models")

    @property
    def image_dir(self) -> str:
        return os.path.join(self.data_root, "images")

    @property
    def reference_file(self) -> str:
        return os.path.join(self.data_root, "references", "references.json")

    @property
    def cache_meta_file(self) -> str:
        return os.path.join(self.data_root, "cache", "cache_meta.json")

    @property
    def log_dir(self) -> str:
        return os.path.join(self.data_root, "logs")

    # ================================================================
    # 時區
    # ================================================================
    timezone: str = "Asia/Taipei"
    reference_input_timezone: str = "Asia/Taipei"  # 使用者輸入 reference 時間時的預設時區 (GMT+8)

    # ================================================================
    # Binance API
    # ================================================================
    api_sleep_seconds: float = 0.35         # 每次 API 請求間隔（秒）
    max_klines_per_request: int = 1500      # Binance 單次最大 K 線數
    api_request_timeout: int = 30           # 單次請求超時（秒）
    api_max_retries: int = 3                # 最大重試次數
    api_retry_backoff: float = 2.0          # 重試退避倍率

    # ================================================================
    # 掃描時框
    # ================================================================
    scan_timeframes: List[str] = field(
        default_factory=lambda: ["30m", "1h", "2h", "4h"]
    )

    # ================================================================
    # 下載配置
    # ================================================================
    download_batch_size: int = 20           # 每批下載的 symbol 數量
    download_batch_sleep: float = 5.0       # 批次間休息秒數
    download_progress_interval: int = 10    # 每 N 個 symbol 印一次進度

    # ================================================================
    # DTW 參數
    # ================================================================
    dtw_window_ratio: float = 0.12
    dtw_window_ratio_diff: float = 0.1
    dtw_max_point_distance: float = 0.6
    dtw_max_point_distance_diff: float = 0.5
    window_scale_factors: List[float] = field(
        default_factory=lambda: [0.9, 0.95, 1.0, 1.05, 1.1]
    )

    # ================================================================
    # ShapeDTW
    # ================================================================
    shapedtw_balance_pd_ratio: float = 4.0
    price_weight: float = 0.6
    diff_weight: float = 0.4
    slope_window_size: int = 5
    paa_window_size: int = 5

    # ================================================================
    # SMA
    # ================================================================
    sma_periods: List[int] = field(default_factory=lambda: [30, 45, 60])

    @property
    def sma_lookback(self) -> int:
        """最大 SMA 週期，用作 SMA 預熱所需的前期 K 棒數量"""
        return max(self.sma_periods) if self.sma_periods else 60

    # ================================================================
    # 預篩
    # ================================================================
    prescreening_top_ratio: float = 0.20    # Stage 1 保留前 20%
    min_similarity_score: float = 0.25      # 最低相似度門檻

    # ================================================================
    # ML
    # ================================================================
    ml_thresholds: List[float] = field(
        default_factory=lambda: [0.05, 0.10, 0.15, 0.20]
    )
    ml_extension_factors: List[float] = field(
        default_factory=lambda: [0.5, 1.0, 1.5, 2.0]
    )

    # ================================================================
    # 突兀量
    # ================================================================
    volume_spike_ratio: float = 2.5

    # ================================================================
    # 圖片
    # ================================================================
    chart_width: int = 1200
    chart_height: int = 800
    chart_dpi: int = 150

    # ================================================================
    # Telegram
    # ================================================================
    telegram_bot_token: str = field(
        default_factory=lambda: os.environ.get("TELEGRAM_BOT_TOKEN", "")
    )
    telegram_chat_id: str = field(
        default_factory=lambda: os.environ.get("TELEGRAM_CHAT_ID", "")
    )

    # ================================================================
    # Web Dashboard
    # ================================================================
    web_host: str = "0.0.0.0"
    web_port: int = field(
        default_factory=lambda: int(os.environ.get("PORT", "8080"))
    )
    web_public_url: str = field(
        default_factory=lambda: os.environ.get("WEB_PUBLIC_URL", "")
    )

    @property
    def web_base_url(self) -> str:
        """Get the public-facing web URL for links in Telegram messages"""
        if self.web_public_url:
            return self.web_public_url.rstrip("/")
        return f"http://localhost:{self.web_port}"

    # ================================================================
    # Alert Deduplication
    # ================================================================
    alert_cooldown_hours: int = 24          # 同一 (symbol, tf, ref) 的告警冷卻時間

    # ================================================================
    # 排程
    # ================================================================
    scan_minute_offset: int = 2             # HH:02 開始掃描
    scan_timeout_seconds: int = 3300        # 單次掃描超時 55 分鐘

    def ensure_directories(self):
        """建立所有必要的目錄"""
        dirs = [
            self.parquet_dir,
            self.model_dir,
            self.image_dir,
            os.path.join(self.image_dir, "alerts"),
            os.path.join(self.image_dir, "references"),
            os.path.join(self.image_dir, "historical"),
            os.path.dirname(self.reference_file),
            os.path.dirname(self.cache_meta_file),
            self.log_dir,
        ]
        # 為每個時框建立 parquet 子目錄
        for tf in self.scan_timeframes:
            dirs.append(os.path.join(self.parquet_dir, f"timeframe={tf}"))

        for d in dirs:
            os.makedirs(d, exist_ok=True)

    def timeframe_to_seconds(self, timeframe: str) -> int:
        """將時框字串轉為秒數"""
        if timeframe.endswith("m"):
            return int(timeframe[:-1]) * 60
        elif timeframe.endswith("h"):
            return int(timeframe[:-1]) * 3600
        elif timeframe.endswith("d"):
            return int(timeframe[:-1]) * 86400
        raise ValueError(f"Unknown timeframe: {timeframe}")

    def timeframe_to_binance_interval(self, timeframe: str) -> str:
        """將時框字串轉為 Binance API interval 格式（其實一樣，但做一層保險）"""
        valid = {"1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d"}
        if timeframe in valid:
            return timeframe
        raise ValueError(f"Invalid Binance interval: {timeframe}")
