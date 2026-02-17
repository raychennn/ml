# Crypto Pattern Recognition System

# 加密貨幣形態辨識系統

DTW-based pattern matching system for cryptocurrency markets with ML-powered gain prediction, Telegram bot control, and interactive web dashboard.

基於 DTW 的加密貨幣形態比對系統，搭配 ML 漲幅預測、Telegram 機器人控制、互動式網頁儀表板。

---

## Architecture / 系統架構

```
┌──────────────────────────────────────────────────────────────┐
│                      Entry Points                            │
│  main.py serve │ run_all.py │ run_bot.py │ run_web.py       │
└────────┬─────────────┬──────────────┬──────────────┬────────┘
         │             │              │              │
    ┌────▼────┐  ┌─────▼─────┐  ┌────▼────┐  ┌─────▼─────┐
    │Telegram │  │   Scan    │  │   Scan  │  │   Flask   │
    │  Bot    │  │ Scheduler │  │Orchestr.│  │    Web    │
    └────┬────┘  └─────┬─────┘  └────┬────┘  └─────┬─────┘
         │             │              │              │
         └─────────────┴──────┬───────┘              │
                              │                      │
                    ┌─────────▼──────────┐           │
                    │  ScanOrchestrator  │           │
                    │  (single pipeline) │           │
                    └─────────┬──────────┘           │
                              │                      │
         ┌────────┬───────────┼───────────┬──────────┘
         │        │           │           │
    ┌────▼───┐ ┌──▼───┐ ┌────▼────┐ ┌────▼─────┐
    │Scanner │ │ Data │ │Reference│ │  Result  │
    │Engine  │ │Mgr   │ │ Manager │ │  Store   │
    └────┬───┘ └──┬───┘ └─────────┘ │ (SQLite) │
         │        │                  └──────────┘
    ┌────▼───┐ ┌──▼──────┐
    │  ML    │ │ Parquet │
    │Predict │ │ Files   │
    └────────┘ └─────────┘
```

## Key Components / 核心元件

| Component / 元件 | File / 檔案 | Description / 說明 |
|-----------|------|-------------|
| **ScanOrchestrator** | `core/scan_orchestrator.py` | Single source of truth for scan pipeline: data update → scan → cooldown → chart → save → notify / 掃描管線唯一入口：資料更新 → 掃描 → 冷卻過濾 → 圖表 → 儲存 → 通知 |
| **ScannerEngine** | `core/scanner_engine.py` | 3-stage scanner: prescreening → DTW matching → ML scoring / 三階段掃描引擎：預篩 → DTW 比對 → ML 評分 |
| **DataManager** | `data/data_manager.py` | Thread-safe Parquet storage with per-file write locks / 執行緒安全的 Parquet 存儲，含逐檔寫入鎖 |
| **ReferenceManager** | `references/reference_manager.py` | CRUD for reference patterns with chart generation / 參考形態 CRUD 管理與圖表生成 |
| **ScanResultStore** | `core/scan_result_store.py` | SQLite persistence for scan results and run history / SQLite 掃描結果持久化 |
| **TelegramBot** | `telegram_bot/bot.py` | Long-polling command handler with runtime config / 長輪詢指令處理器，支援執行期設定 |
| **TelegramNotifier** | `telegram_bot/notifier.py` | Message/photo sending with retry + exponential backoff / 訊息/圖片發送，含重試與指數退避 |
| **ScanScheduler** | `scheduler/scheduler.py` | Cron-style automatic scanning at HH:02 / 類 Cron 自動掃描排程（每小時 :02 觸發） |
| **ChartGenerator** | `visualization/chart_generator.py` | Matplotlib alert charts + reference SMA/OHLC charts / Matplotlib 告警圖表 + 參考形態 SMA/OHLC 圖表 |
| **Flask Web App** | `web/app.py` | Dashboard with interactive Plotly.js charts / 互動式 Plotly.js 圖表儀表板 |

---

## Quick Start / 快速開始

### 1. Install Dependencies / 安裝依賴

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure Environment / 設定環境變數

```bash
export TELEGRAM_BOT_TOKEN="your-bot-token"       # Telegram 機器人 Token
export TELEGRAM_CHAT_ID="your-chat-id"            # Telegram 聊天室 ID
export WEB_PUBLIC_URL="https://your-domain.com"   # (optional / 可選) 用於 Telegram 訊息中的網頁連結
export DATA_ROOT="./data"                          # (default / 預設: ./data) 資料根目錄
```

### 3. Download Historical Data / 下載歷史數據

```bash
python main.py download --timeframes 30m 1h 2h 4h
```

Or let the system auto-download on first start (the scheduler detects missing data and initiates download automatically).

或讓系統首次啟動時自動下載（排程器偵測到無本地資料時會自動觸發下載）。

### 4. Train ML Models / 訓練 ML 模型

```bash
python scripts/train_model.py --timeframes 4h
```

### 5. Run Everything / 啟動所有服務

```bash
# All components (bot + scheduler + web dashboard)
# 全部元件（機器人 + 排程器 + 網頁儀表板）
python main.py serve

# Or via standalone script / 或透過獨立腳本
python scripts/run_all.py

# Individual components / 個別元件
python scripts/run_bot.py       # Bot + scheduler only / 僅機器人 + 排程器
python scripts/run_web.py       # Web dashboard only / 僅網頁儀表板
```

### 6. Docker

```bash
docker build -t crypto-pattern .
docker run -d \
  -e TELEGRAM_BOT_TOKEN=xxx \
  -e TELEGRAM_CHAT_ID=xxx \
  -v /path/to/data:/data \
  -p 5000:5000 \
  crypto-pattern
```

### 7. Deploy on Zeabur / 部署到 Zeabur

#### Step 1: Push code to GitHub / 將程式碼推送到 GitHub

```bash
git init
git add .
git commit -m "initial commit"
git remote add origin https://github.com/YOUR_USER/crypto-pattern-system.git
git push -u origin main
```

#### Step 2: Create project on Zeabur / 在 Zeabur 建立專案

1. Go to [Zeabur Dashboard](https://dash.zeabur.com) and sign in.

   前往 [Zeabur Dashboard](https://dash.zeabur.com) 並登入。

2. Click **"New Project"** → select a region (recommend **Asia — Tokyo** or **Asia — Hong Kong** for lower latency to Binance API).

   點擊 **「New Project」** → 選擇地區（建議選 **Asia — Tokyo** 或 **Asia — Hong Kong** 以降低 Binance API 延遲）。

3. Click **"Add Service"** → **"Git"** → select your GitHub repository.

   點擊 **「Add Service」** → **「Git」** → 選擇你的 GitHub 倉庫。

4. Zeabur will auto-detect the `Dockerfile` and start building.

   Zeabur 會自動偵測 `Dockerfile` 並開始建置。

#### Step 3: Add persistent volume / 掛載持久化磁碟

> **CRITICAL / 關鍵步驟**: Without a persistent volume, all data (Parquet files, ML models, SQLite database, reference patterns) will be **lost on every redeploy**.
>
> 若不掛載持久化磁碟，所有資料（Parquet 檔案、ML 模型、SQLite 資料庫、參考形態）將在**每次重新部署時遺失**。

1. In your service page, go to **"Storage"** tab.

   在服務頁面中，進入 **「Storage」** 分頁。

2. Click **"Add Volume"**.

   點擊 **「Add Volume」**。

3. Set **Mount Path** to `/data`.

   將 **Mount Path** 設為 `/data`。

4. Set the volume size based on your needs (recommended: **10 GB+** for full data across 4 timeframes with ~250 symbols).

   根據需求設定磁碟大小（建議：完整 4 時框 ~250 幣種資料需要 **10 GB 以上**）。

#### Step 4: Configure environment variables / 設定環境變數

Go to **"Variables"** tab and add:

進入 **「Variables」** 分頁，新增以下變數：

| Variable / 變數 | Value / 值 | Required / 必要 |
|---------|-------|----------|
| `TELEGRAM_BOT_TOKEN` | Your Telegram bot token from [@BotFather](https://t.me/BotFather) / 從 @BotFather 取得的 Token | Yes |
| `TELEGRAM_CHAT_ID` | Your Telegram chat ID (use [@userinfobot](https://t.me/userinfobot) to find it) / 你的聊天室 ID | Yes |
| `WEB_PUBLIC_URL` | Your Zeabur-assigned domain, e.g. `https://crypto-pattern-xxx.zeabur.app` / Zeabur 指派的網域 | Yes |
| `DATA_ROOT` | `/data` (should match volume mount path) / 應與磁碟掛載路徑一致 | Already set in Dockerfile |
| `FLASK_SECRET_KEY` | A random secret string for Flask sessions / Flask 隨機密鑰 | Recommended for production |

#### Step 5: Enable networking / 啟用網路

1. Go to **"Networking"** tab.

   進入 **「Networking」** 分頁。

2. Click **"Add Domain"** → choose a Zeabur subdomain (e.g. `crypto-pattern.zeabur.app`) or bind your custom domain.

   點擊 **「Add Domain」** → 選擇 Zeabur 子域名或綁定自訂域名。

3. Set **Port** to `5000`.

   將 **Port** 設為 `5000`。

4. Copy the full URL (e.g. `https://crypto-pattern-xxx.zeabur.app`) and paste it into the `WEB_PUBLIC_URL` environment variable.

   複製完整 URL 並貼到 `WEB_PUBLIC_URL` 環境變數。

#### Step 6: Deploy and verify / 部署並驗證

1. Zeabur will automatically deploy after setting variables. Wait for the build to complete.

   設定完變數後 Zeabur 會自動部署。等待建置完成。

2. Check the health endpoint: visit `https://your-domain.zeabur.app/api/health` — should return `{"status": "ok", ...}`.

   檢查健康端點：瀏覽 `https://your-domain.zeabur.app/api/health` — 應回傳 `{"status": "ok", ...}`。

3. Send `/status` to your Telegram bot to verify it's running.

   向 Telegram 機器人發送 `/status` 確認運作中。

4. On first start, the scheduler will automatically detect no local data and begin downloading (~30-60 minutes for all timeframes). You'll receive a Telegram notification when complete.

   首次啟動時，排程器會自動偵測無本地資料並開始下載（所有時框約需 30-60 分鐘）。下載完成後會收到 Telegram 通知。

#### Redeployment notes / 重新部署注意事項

- **Code changes**: Push to GitHub → Zeabur auto-redeploys. Data on the persistent volume (`/data`) is preserved.

  **程式碼更新**：推送到 GitHub → Zeabur 自動重新部署。持久化磁碟（`/data`）上的資料會保留。

- **Retrain models**: Use `/retrain` via Telegram — no need to redeploy. Models are saved to `/data/models/` on the persistent volume.

  **重新訓練模型**：透過 Telegram 使用 `/retrain` — 無需重新部署。模型儲存在持久化磁碟的 `/data/models/`。

- **Add references**: Use `/add_ref` via Telegram — references are saved to `/data/references/references.json` on the persistent volume.

  **新增參考形態**：透過 Telegram 使用 `/add_ref` — 參考形態儲存在持久化磁碟的 `/data/references/references.json`。

---

## Important Notices & Cautions / 重要注意事項與警告

### Binance API / 幣安 API

- **No API key required**: This system uses Binance's **public** market data endpoints (klines/candlestick data). No API key or secret is needed.

  **不需要 API 密鑰**：本系統使用幣安的**公開**市場數據端點（K 線數據），不需要 API key 或 secret。

- **Rate limiting**: The system sleeps `0.35s` between API requests (`api_sleep_seconds` in config). If you encounter `HTTP 429` errors, increase this value.

  **速率限制**：系統在每次 API 請求間休眠 `0.35s`（設定中的 `api_sleep_seconds`）。若遇到 `HTTP 429` 錯誤，請增加此值。

- **IP restrictions**: Some cloud providers' IP ranges may be blocked by Binance. If data download consistently fails, check if your server region has access to `api.binance.com`.

  **IP 限制**：部分雲端服務商的 IP 段可能被幣安封鎖。若資料下載持續失敗，請確認伺服器所在區域能存取 `api.binance.com`。

### Data & Storage / 資料與儲存

- **First download takes time**: Initial data download for all 4 timeframes (~250 symbols each) can take **30-60 minutes**. The bot will send a Telegram notification when complete. Do not restart the service during download.

  **首次下載耗時**：初始下載所有 4 個時框（每個約 250 個幣種）可能需要 **30-60 分鐘**。下載完成後機器人會發送 Telegram 通知。下載期間請勿重啟服務。

- **Disk space**: Full data for 4 timeframes requires approximately **2-5 GB**. With images and ML models, plan for at least **10 GB** of persistent storage.

  **磁碟空間**：4 個時框的完整資料約需 **2-5 GB**。加上圖片和 ML 模型，建議至少 **10 GB** 的持久化儲存空間。

- **Scan results accumulate**: SQLite scan results (`scan_results.db`) grow over time. With frequent scans, this can reach several GB over months. Monitor disk usage periodically.

  **掃描結果持續累積**：SQLite 掃描結果（`scan_results.db`）會隨時間增長。頻繁掃描下，數月後可能達到數 GB。請定期監控磁碟使用量。

- **Backup references**: Your reference patterns (`data/references/references.json`) are the most important user-created data. Back up this file regularly.

  **備份參考形態**：你的參考形態（`data/references/references.json`）是最重要的使用者資料。請定期備份此檔案。

### ML Models / ML 模型

- **Train before scanning**: Without trained models, the system uses a DTW-score-based heuristic fallback. ML probability predictions will show `0%` for all thresholds. Train models with `python scripts/train_model.py` or `/retrain` via Telegram for accurate predictions.

  **先訓練再掃描**：若無已訓練模型，系統使用基於 DTW 分數的啟發式回退方案，ML 機率預測會顯示 `0%`。請先用 `python scripts/train_model.py` 或 Telegram `/retrain` 訓練模型以獲得準確預測。

- **Retrain periodically**: Market conditions change. Retrain models every few weeks to months for best accuracy. The `/retrain` command handles this without downtime.

  **定期重新訓練**：市場狀況會變化。建議每幾週到幾個月重新訓練一次模型以維持準確度。`/retrain` 指令可在不停機的情況下完成。

- **Retraining is CPU-intensive**: The `/retrain` process scans historical data and trains GBR models. On a small server (1-2 CPU), this can take **1-2 hours**. The system has a 2-hour timeout.

  **重新訓練耗費 CPU**：`/retrain` 過程會掃描歷史資料並訓練 GBR 模型。在小型伺服器（1-2 CPU）上可能需要 **1-2 小時**。系統設有 2 小時超時限制。

### Telegram Bot / Telegram 機器人

- **One bot instance only**: Do not run multiple instances of the bot with the same bot token. Telegram's `getUpdates` long polling will cause conflicts and missed commands.

  **僅運行一個機器人實例**：勿使用同一個 bot token 運行多個實例。Telegram 的 `getUpdates` 長輪詢會造成衝突和遺漏指令。

- **Chat ID security**: The bot only responds to commands from the configured `TELEGRAM_CHAT_ID`. Messages from other chats are silently ignored.

  **聊天室 ID 安全性**：機器人僅回應來自設定的 `TELEGRAM_CHAT_ID` 的指令，其他聊天室的訊息會被靜默忽略。

- **Bot token is a secret**: Never commit your `TELEGRAM_BOT_TOKEN` to Git. Always use environment variables.

  **Bot token 是機密**：切勿將 `TELEGRAM_BOT_TOKEN` 提交到 Git。務必使用環境變數。

### Web Dashboard / 網頁儀表板

- **No authentication**: The web dashboard has **no login system**. Anyone with the URL can view scan results. If you need privacy, restrict access via Zeabur's networking settings, a reverse proxy, or keep the URL private.

  **無身份驗證**：網頁儀表板**沒有登入系統**。任何人取得 URL 即可查看掃描結果。若需隱私保護，請透過 Zeabur 網路設定、反向代理限制存取，或不公開 URL。

- **Flask secret key**: The default `FLASK_SECRET_KEY` is a development placeholder. For production, set a random string via environment variable to secure session cookies.

  **Flask 密鑰**：預設的 `FLASK_SECRET_KEY` 是開發用佔位值。正式環境請透過環境變數設定隨機字串以保護 session cookie。

### Scanning & Alerts / 掃描與告警

- **Not financial advice**: This system is a technical analysis tool. Pattern matches and ML predictions do **not** guarantee future price movements. Always do your own research and manage risk appropriately.

  **非投資建議**：本系統為技術分析工具。形態匹配和 ML 預測**不保證**未來價格走勢。請務必自行研究並適當管理風險。

- **Alert cooldown**: By default, the same `(symbol, timeframe, reference)` combination will not trigger another alert for 24 hours. Adjust via `/config alert_cooldown_hours N`.

  **告警冷卻**：預設同一 `(幣種, 時框, 參考)` 組合在 24 小時內不會重複告警。可透過 `/config alert_cooldown_hours N` 調整。

- **Concurrent scans blocked**: Only one scan can run at a time. If a scheduled scan is running when you send `/scan`, the manual scan will be rejected. Wait for the current scan to finish.

  **並行掃描阻擋**：同一時間只能運行一次掃描。若排程掃描正在執行時發送 `/scan`，手動掃描會被拒絕。請等待當前掃描完成。

### Timezone / 時區

- **System timezone**: The system uses `Asia/Taipei` (GMT+8) by default for scheduling and reference date parsing. This is configured in `config.py` (`timezone` field) and the Dockerfile (`TZ=Asia/Taipei`).

  **系統時區**：系統預設使用 `Asia/Taipei`（GMT+8）作為排程和參考形態日期解析的時區。設定於 `config.py`（`timezone` 欄位）和 Dockerfile（`TZ=Asia/Taipei`）。

- **Reference dates**: When adding references via `/add_ref`, dates are interpreted in `Asia/Taipei` timezone. Example: `/add_ref SOLUSDT 4h 2024-01-01T00:00 ...` means midnight Taiwan time, which is `2023-12-31T16:00 UTC`.

  **參考形態日期**：透過 `/add_ref` 新增參考形態時，日期以 `Asia/Taipei` 時區解讀。例如：`2024-01-01T00:00` 表示台灣時間午夜，即 `2023-12-31T16:00 UTC`。

---

## Telegram Bot Commands / Telegram 機器人指令

| Command / 指令 | Description / 說明 |
|---------|-------------|
| `/scan` | Run pattern scanner for all timeframes / 執行所有時框的形態掃描 |
| `/add_ref SYMBOL TF START END LABEL [desc]` | Add a new reference pattern / 新增參考形態 |
| `/del_ref REF_ID` | Delete (deactivate) a reference / 刪除（停用）參考形態 |
| `/list_ref` | List all active references / 列出所有啟用中的參考形態 |
| `/retrain` | Launch ML model retraining / 啟動 ML 模型重新訓練 |
| `/status` | Show system status (models, refs, last scan) / 顯示系統狀態 |
| `/config` | View runtime settings / 檢視執行期設定 |
| `/config KEY VALUE` | Change a runtime setting / 修改執行期設定 |
| `/help` | Show available commands / 顯示可用指令 |

### Runtime Config Keys / 可調整參數

| Key / 參數名 | Default / 預設值 | Description / 說明 |
|-----|---------|-------------|
| `min_similarity_score` | 0.25 | Minimum DTW similarity threshold / DTW 最低相似度門檻 |
| `volume_spike_ratio` | 2.5 | Volume spike detection multiplier / 突兀量偵測倍率 |
| `prescreening_top_ratio` | 0.20 | Stage 1 prescreening retention ratio / 預篩保留比例（前 20%） |
| `alert_cooldown_hours` | 24 | Hours before re-alerting same (symbol, tf, ref) / 相同告警冷卻時間（小時） |

### Adding References via Telegram / 透過 Telegram 新增參考形態

```
/add_ref SOLUSDT 4h 2024-01-01T00:00 2024-01-03T00:00 perfect_trend nice accumulation pattern
```

Date format / 日期格式: `YYYY-MM-DDTHH:MM` (interpreted as Asia/Taipei timezone / 以台北時區解讀)

---

## Web Dashboard / 網頁儀表板

Access at / 存取網址: `http://localhost:5000` (or configured `WEB_PUBLIC_URL` / 或設定的 `WEB_PUBLIC_URL`)

### Pages / 頁面

| Page / 頁面 | Route / 路由 | Description / 說明 |
|------------|-------------|-------------|
| **Dashboard / 儀表板** | `/` | Scan results table with timeframe filter, DTW scores, ML probabilities / 掃描結果表格，可依時框過濾，顯示 DTW 分數與 ML 機率 |
| **Detail / 詳情** | `/result/<id>` | Interactive Plotly.js candlestick chart with SMA overlays and volume spikes / 互動式 K 線圖含 SMA 疊加與突兀量標記 |
| **References / 參考形態** | `/references` | Browse reference patterns with SMA and full OHLC charts / 瀏覽參考形態，含純 SMA 圖與完整 OHLC 圖 |
| **Reports / 報告** | `/reports` | Per-label performance metrics (sample count, DTW distribution, ML probability stats, forward gains) / 按標籤分類的績效指標 |
| **Status / 狀態** | `/status` | System health overview / 系統健康狀態總覽 |

### API Endpoints / API 端點

| Endpoint / 端點 | Description / 說明 |
|----------|-------------|
| `GET /api/results` | Paginated scan results (filter: `timeframe`, `page`, `per_page`) / 分頁掃描結果 |
| `GET /api/result/<id>/chart-data` | OHLC + SMA + volume JSON for Plotly rendering / 圖表資料 JSON |
| `GET /api/reports/label-metrics` | Per-label performance metrics / 按標籤績效指標 |
| `GET /api/health` | Health check (used by Docker HEALTHCHECK) / 健康檢查端點 |

---

## Scanner Pipeline / 掃描管線

### 3-Stage Scanning / 三階段掃描

1. **Stage 1 — Prescreening / 預篩**: Z-normalized Euclidean distance filters ~250 symbols down to top 20% (~50 candidates) / Z 正規化歐氏距離將 ~250 個幣種篩選至前 20%（~50 個候選）
2. **Stage 2 — DTW Matching / DTW 比對**: Full DTW + ShapeDTW comparison with multi-scale window matching (multiprocessing) / 完整 DTW + ShapeDTW 多尺度視窗比對（多進程並行）
3. **Stage 3 — ML Scoring / ML 評分**: GradientBoostingRegressor predicts P(5%/10%/15%/20% gain) / 梯度提升回歸預測各漲幅機率

### Alert Deduplication / 告警去重

The orchestrator tracks `(symbol, timeframe, ref_id)` tuples and suppresses repeat alerts within the cooldown window (default 24 hours). Configurable via `/config alert_cooldown_hours N`.

排程器追蹤 `(幣種, 時框, 參考ID)` 組合，在冷卻期內（預設 24 小時）抑制重複告警。可透過 `/config alert_cooldown_hours N` 調整。

### Scheduling Rules / 排程規則

All scans trigger at HH:02 (configurable via `scan_minute_offset`):

所有掃描於每小時 :02 觸發（可透過 `scan_minute_offset` 設定）:

| Timeframe / 時框 | Schedule / 排程 |
|-----------|----------|
| 30m, 1h | Every hour / 每小時 |
| 2h | Even hours (0, 2, 4, ..., 22) / 偶數小時 |
| 4h | Every 4 hours (0, 4, 8, 12, 16, 20) / 每 4 小時 |

---

## ML Model Training / ML 模型訓練

### Full Pipeline / 完整流程

```bash
python scripts/train_model.py --timeframes 4h
```

### Scan Only (generate training CSV) / 僅掃描（產生訓練 CSV）

```bash
python scripts/train_model.py --scan-only --timeframes 4h
```

### Train Only (from existing CSV) / 僅訓練（使用現有 CSV）

```bash
python scripts/train_model.py --train-only
```

### Custom Parameters / 自訂參數

```bash
python scripts/train_model.py --timeframes 4h --stride 8 --workers 4 \
    --n-estimators 300 --max-depth 4 --learning-rate 0.05
```

### Output / 輸出

Models are saved to `data/models/`:

模型儲存於 `data/models/`:

| File / 檔案 | Description / 說明 |
|------|-------------|
| `gbr_0.05.joblib` | P(5% gain) model / P(漲 5%) 模型 |
| `gbr_0.10.joblib` | P(10% gain) model / P(漲 10%) 模型 |
| `gbr_0.15.joblib` | P(15% gain) model / P(漲 15%) 模型 |
| `gbr_0.20.joblib` | P(20% gain) model / P(漲 20%) 模型 |
| `training_meta.json` | CV scores, feature importances, training params / 交叉驗證分數、特徵重要性、訓練參數 |
| `training_data.csv` | Generated training samples / 產生的訓練樣本 |

### Features (19 total) / 特徵（共 19 個）

- **DTW quality / DTW 品質**: dtw_score, price_distance, diff_distance, window_scale_factor
- **Pattern structure / 形態結構**: pattern_length, pattern_return, pattern_volatility, pattern_max_drawdown, sma30_slope_end, sma_alignment_end, close_vs_sma30_end, close_vs_sma60_end
- **Volume / 量能**: volume_trend, spike_count, spike_ratio, last_5bar_avg_volume_ratio
- **Market context / 市場環境**: btc_return_20bar, btc_volatility_20bar, altcoin_vs_btc

If no trained models exist, the scanner uses a DTW-score-based heuristic fallback.

若無已訓練模型，掃描器會使用基於 DTW 分數的啟發式回退方案。

---

## Reference Pattern Management / 參考形態管理

### Via Python / 透過 Python

```python
from config import SystemConfig
from references.reference_manager import ReferenceManager

config = SystemConfig()
ref_mgr = ReferenceManager(config)

# Add a reference / 新增參考形態
ref_mgr.add_reference(
    symbol="AVAXUSDT",
    timeframe="4h",
    start_ts=1699531200,
    end_ts=1699920000,
    label="standard_uptrend",
    description="AVAX classic accumulation breakout"
)

# List all / 列出全部
for r in ref_mgr.list_references():
    print(f"{r.id}: {r.symbol} {r.timeframe} {r.label}")

# Delete / 刪除
ref_mgr.delete_reference("AVAX_4h_standard_uptrend")
```

### Via Telegram / 透過 Telegram

```
/add_ref AVAXUSDT 4h 2023-11-09T20:00 2023-11-14T08:00 standard_uptrend classic accumulation
/list_ref
/del_ref AVAXUSDT_4h_standard_uptrend
```

---

## Configuration / 設定

All parameters are centralized in `config.py` (`SystemConfig` dataclass). Environment variables override defaults:

所有參數集中在 `config.py`（`SystemConfig` 資料類別）。環境變數優先於預設值：

| Env Variable / 環境變數 | Config Field / 設定欄位 | Default / 預設值 |
|-------------|-------------|---------|
| `DATA_ROOT` | `data_root` | `./data` |
| `TELEGRAM_BOT_TOKEN` | `telegram_bot_token` | (empty / 空) |
| `TELEGRAM_CHAT_ID` | `telegram_chat_id` | (empty / 空) |
| `WEB_PUBLIC_URL` | `web_public_url` | (empty / 空) |

---

## Data Storage / 資料儲存結構

```
data/
├── parquet/
│   ├── timeframe=30m/     # ~250 symbol Parquet files per timeframe / 每時框約 250 個幣種
│   ├── timeframe=1h/
│   ├── timeframe=2h/
│   └── timeframe=4h/
├── models/                # Trained ML models + training data / ML 模型 + 訓練資料
├── images/
│   ├── alerts/            # Scan alert charts / 掃描告警圖表
│   ├── references/        # Reference pattern charts / 參考形態圖表
│   └── historical/        # Historical scan charts / 歷史掃描圖表
├── references/
│   └── references.json    # Reference pattern definitions / 參考形態定義
├── cache/
│   └── cache_meta.json    # Incremental update tracking / 增量更新追蹤
├── logs/
│   └── system.log         # Rotating log (10MB x 5 backups) / 輪替日誌
└── scan_results.db        # SQLite scan result database / SQLite 掃描結果資料庫
```

---

## Reliability Features / 可靠性特性

| Feature / 特性 | Description / 說明 |
|---------|-------------|
| **Retry with backoff / 重試退避** | Telegram API calls retry 3 times with exponential backoff (1s → 2s → 4s) / Telegram API 呼叫失敗時指數退避重試 3 次 |
| **Alert cooldown / 告警冷卻** | Deduplicates alerts per (symbol, timeframe, reference) with configurable cooldown / 按（幣種、時框、參考）去重，可設定冷卻期 |
| **Thread safety / 執行緒安全** | Per-file write locks for Parquet, lock-protected metadata cache, thread-safe BTC data cache / Parquet 逐檔寫入鎖、元資料快取鎖、BTC 資料快取鎖 |
| **Log rotation / 日誌輪替** | 10MB per file, 5 backup files, UTF-8 encoding / 每檔 10MB，保留 5 份備份 |
| **Graceful shutdown / 優雅關閉** | Signal handlers clean up background threads and subprocesses / 信號處理器清理背景執行緒與子進程 |
| **Health check / 健康檢查** | `/api/health` endpoint for container orchestration (Docker HEALTHCHECK configured) / 容器編排健康檢查端點 |
| **Auto-download / 自動下載** | First-start detection triggers automatic data download / 首次啟動偵測，自動觸發資料下載 |
| **Non-blocking retrain / 非阻塞重訓** | Subprocess polling with stop event support, kills subprocess on shutdown / 子進程非阻塞輪詢，關閉時自動終止 |

---

## Project Structure / 專案結構

```
crypto_pattern_system/
├── config.py                          # Central configuration / 全局設定
├── main.py                            # CLI entry point / CLI 入口
├── Dockerfile                         # Container deployment / 容器部署
├── requirements.txt                   # Python dependencies / Python 依賴
├── core/
│   ├── scan_orchestrator.py           # Unified scan pipeline / 統一掃描管線
│   ├── scanner_engine.py              # 3-stage DTW scanner / 三階段 DTW 掃描引擎
│   ├── scan_result_store.py           # SQLite result persistence / SQLite 結果持久化
│   ├── data_normalizer.py             # Z-normalization + SMA prep / Z 正規化 + SMA 準備
│   └── dtw_calculator.py              # DTW + ShapeDTW computation / DTW + ShapeDTW 計算
├── data/
│   ├── binance_client.py              # Binance API client / 幣安 API 客戶端
│   ├── data_manager.py                # Parquet read/write (thread-safe) / Parquet 讀寫（執行緒安全）
│   └── symbol_manager.py              # Active symbol management / 活躍幣種管理
├── ml/
│   ├── predictor.py                   # ML inference (GBR models) / ML 推論
│   ├── historical_scanner.py          # Training data generation / 訓練資料生成
│   └── feature_extractor.py           # 19-feature extraction / 19 特徵擷取
├── references/
│   └── reference_manager.py           # Reference CRUD + charts / 參考形態管理 + 圖表
├── scheduler/
│   └── scheduler.py                   # Cron-style auto-scanning / 類 Cron 自動掃描
├── telegram_bot/
│   ├── bot.py                         # Long-polling command handler / 長輪詢指令處理器
│   ├── notifier.py                    # Send messages/photos (retry) / 發送訊息/圖片（含重試）
│   └── command_parser.py              # Parse /command arguments / 解析指令參數
├── visualization/
│   └── chart_generator.py             # Matplotlib chart rendering / Matplotlib 圖表繪製
├── web/
│   ├── app.py                         # Flask app factory / Flask 應用工廠
│   ├── routes/
│   │   ├── dashboard.py               # Dashboard + detail pages / 儀表板 + 詳情頁
│   │   ├── api.py                     # JSON API + health check / JSON API + 健康檢查
│   │   ├── references.py              # Reference browsing / 參考形態瀏覽
│   │   └── reports.py                 # Per-label metrics / 按標籤績效報告
│   ├── templates/                     # Jinja2 HTML templates / Jinja2 模板
│   └── static/                        # CSS + JS (Plotly charts) / 靜態資源
└── scripts/
    ├── bulk_download.py               # Batch data download / 批次資料下載
    ├── train_model.py                 # ML training pipeline / ML 訓練流程
    ├── run_all.py                     # Bot + scheduler + web / 全部啟動
    ├── run_bot.py                     # Bot + scheduler only / 僅機器人 + 排程
    └── run_web.py                     # Web dashboard only / 僅網頁儀表板
├── benchmarks/                        # Performance benchmarks / 效能基準測試
│   ├── bench_dtw.py                   # DTW speed benchmarks / DTW 速度基準
│   ├── bench_prescreening.py          # Prescreening throughput / 預篩吞吐量
│   ├── bench_parquet.py               # Parquet I/O benchmarks / Parquet 讀寫基準
│   ├── bench_ml_inference.py          # ML inference latency / ML 推論延遲
│   ├── bench_scan_cycle.py            # End-to-end scan cycle / 端到端掃描週期
│   ├── run_all_benchmarks.py          # Master runner + baseline comparison / 主執行器
│   ├── compare_optimizations.py       # A/B optimization comparisons / 優化比較
│   ├── scaling_tests.py               # Scaling curve tests / 擴展性測試
│   ├── profiler.py                    # cProfile profiling / cProfile 效能分析
│   └── baselines.json                 # Expected performance baselines / 預期基準
├── load_tests/                        # Load & stress tests / 負載壓力測試
│   ├── test_concurrent_scans.py       # Scan lock verification / 掃描鎖驗證
│   ├── test_web_load.py               # Web dashboard load test / 網頁負載測試
│   ├── test_large_symbol_count.py     # 250+ symbol scan test / 大量幣種測試
│   ├── test_sustained_scheduler.py    # Sustained operation test / 持續運行測試
│   └── test_sqlite_concurrent.py      # SQLite concurrent R/W / SQLite 並行讀寫
├── monitoring/                        # System monitoring / 系統監控
│   ├── resource_monitor.py            # Disk + memory monitoring / 磁碟記憶體監控
│   ├── metrics_collector.py           # JSONL metrics recording / JSONL 指標記錄
│   ├── watchdog.py                    # Thread liveness watchdog / 執行緒存活監控
│   └── health_check.py               # Enhanced health endpoints / 進階健康端點
└── PERFORMANCE.md                     # Performance engineering docs / 效能工程文件
```

---

## Performance / 效能

For comprehensive performance engineering documentation, see **[PERFORMANCE.md](PERFORMANCE.md)**.

詳細的效能工程文件請參閱 **[PERFORMANCE.md](PERFORMANCE.md)**。

### Quick Start / 快速開始

```bash
# Run all benchmarks and compare against baselines
# 執行所有基準測試並與預期值比較
python -m benchmarks.run_all_benchmarks

# Profile the scan pipeline
# 分析掃描管線效能
python -m benchmarks.profiler --target scan

# Check system resource usage
# 檢查系統資源使用量
python -m monitoring.resource_monitor

# Run load tests
# 執行負載測試
python -m load_tests.test_concurrent_scans
python -m load_tests.test_sqlite_concurrent
```

### Key Tools / 主要工具

| Tool / 工具 | Command / 指令 | Purpose / 用途 |
|-------------|----------------|----------------|
| Benchmark Suite / 基準測試 | `python -m benchmarks.run_all_benchmarks` | PASS/WARN/FAIL against baselines / 與基準值比較 |
| Profiler / 效能分析 | `python -m benchmarks.profiler --target all` | cProfile hotspot analysis / cProfile 熱點分析 |
| Scaling Tests / 擴展測試 | `python -m benchmarks.scaling_tests` | Performance vs scale curves / 效能隨規模變化曲線 |
| A/B Comparisons / A/B 比較 | `python -m benchmarks.compare_optimizations` | Quantify optimization impact / 量化優化效果 |
| Resource Monitor / 資源監控 | `python -m monitoring.resource_monitor` | Disk, memory, threads snapshot / 磁碟、記憶體、執行緒快照 |
| Load Tests / 負載測試 | `python -m load_tests.test_web_load` | Concurrent HTTP stress test / 並行 HTTP 壓力測試 |
