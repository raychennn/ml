"""
Scan Result Store — SQLite persistence for scan results
=========================================================

Stores scan results (per-match) and scan runs (per-session) in SQLite.
Provides retrieval with pagination, filtering, and chart data deserialization.
"""

import os
import json
import sqlite3
import logging
import pandas as pd
from datetime import datetime, timezone
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class ScanResultStore:
    """SQLite-backed storage for scan results"""

    def __init__(self, config):
        self.config = config
        self.db_path = os.path.join(config.data_root, "scan_results.db")
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_db(self):
        """Create tables if they don't exist"""
        conn = self._get_conn()
        try:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS scan_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    started_at TEXT NOT NULL,
                    finished_at TEXT,
                    timeframes TEXT,
                    total_alerts INTEGER DEFAULT 0,
                    status TEXT DEFAULT 'running'
                );

                CREATE TABLE IF NOT EXISTS scan_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id INTEGER,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    ref_id TEXT NOT NULL,
                    ref_label TEXT,
                    dtw_score REAL NOT NULL,
                    price_distance REAL,
                    diff_distance REAL,
                    best_scale_factor REAL,
                    ml_probs_json TEXT,
                    window_data_json TEXT,
                    match_start_idx INTEGER,
                    match_end_idx INTEGER,
                    chart_path TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (run_id) REFERENCES scan_runs(id)
                );

                CREATE INDEX IF NOT EXISTS idx_results_timeframe
                    ON scan_results(timeframe);
                CREATE INDEX IF NOT EXISTS idx_results_created
                    ON scan_results(created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_results_run
                    ON scan_results(run_id);
            """)
            conn.commit()
        finally:
            conn.close()

    # ================================================================
    # Write
    # ================================================================

    def start_run(self, timeframes: List[str]) -> int:
        """Start a new scan run, returns run_id"""
        conn = self._get_conn()
        try:
            cursor = conn.execute(
                "INSERT INTO scan_runs (started_at, timeframes, status) VALUES (?, ?, ?)",
                (datetime.now(timezone.utc).isoformat(),
                 json.dumps(timeframes), "running"),
            )
            conn.commit()
            return cursor.lastrowid
        finally:
            conn.close()

    def finish_run(self, run_id: int, total_alerts: int):
        """Mark a scan run as finished"""
        conn = self._get_conn()
        try:
            conn.execute(
                "UPDATE scan_runs SET finished_at=?, total_alerts=?, status=? WHERE id=?",
                (datetime.now(timezone.utc).isoformat(), total_alerts, "completed", run_id),
            )
            conn.commit()
        finally:
            conn.close()

    def save_results(self, alerts, run_id: int = None, chart_paths: Dict[int, str] = None):
        """
        Save a list of AlertResult objects to SQLite.

        Args:
            alerts: List[AlertResult]
            run_id: Optional scan run ID
            chart_paths: Optional {alert_index: chart_path} mapping
        """
        if not alerts:
            return

        conn = self._get_conn()
        try:
            now = datetime.now(timezone.utc).isoformat()
            for i, alert in enumerate(alerts):
                # Serialize window_data DataFrame to JSON
                window_json = None
                if alert.window_data is not None:
                    try:
                        df = alert.window_data
                        window_json = json.dumps({
                            "timestamp": df["timestamp"].tolist() if "timestamp" in df.columns else [],
                            "open": df["open"].tolist(),
                            "high": df["high"].tolist(),
                            "low": df["low"].tolist(),
                            "close": df["close"].tolist(),
                            "volume": df["volume"].tolist(),
                        })
                    except Exception as e:
                        logger.debug(f"Failed to serialize window_data: {e}")

                ml_json = json.dumps(
                    {str(k): v for k, v in alert.ml_probabilities.items()}
                ) if alert.ml_probabilities else None

                chart_path = (chart_paths or {}).get(i, None)

                conn.execute(
                    """INSERT INTO scan_results
                    (run_id, symbol, timeframe, ref_id, ref_label,
                     dtw_score, price_distance, diff_distance, best_scale_factor,
                     ml_probs_json, window_data_json, match_start_idx, match_end_idx,
                     chart_path, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        run_id,
                        alert.symbol,
                        alert.timeframe,
                        alert.reference.id,
                        alert.reference.label,
                        alert.dtw_score,
                        alert.price_distance,
                        alert.diff_distance,
                        alert.best_scale_factor,
                        ml_json,
                        window_json,
                        alert.match_start_idx,
                        alert.match_end_idx,
                        chart_path,
                        now,
                    ),
                )
            conn.commit()
            logger.info(f"Saved {len(alerts)} scan results to SQLite")
        finally:
            conn.close()

    # ================================================================
    # Read
    # ================================================================

    def get_results(
        self,
        page: int = 1,
        per_page: int = 20,
        timeframe: str = None,
        sort_by: str = "created_at",
        sort_dir: str = "DESC",
    ) -> Tuple[List[dict], int]:
        """
        Get paginated scan results.

        Returns:
            (results_list, total_count)
        """
        conn = self._get_conn()
        try:
            where = []
            params = []
            if timeframe:
                where.append("timeframe = ?")
                params.append(timeframe)

            where_sql = "WHERE " + " AND ".join(where) if where else ""

            # Count
            count_row = conn.execute(
                f"SELECT COUNT(*) as cnt FROM scan_results {where_sql}", params
            ).fetchone()
            total = count_row["cnt"]

            # Validate sort
            valid_sorts = {"created_at", "dtw_score", "symbol", "timeframe"}
            if sort_by not in valid_sorts:
                sort_by = "created_at"
            if sort_dir.upper() not in ("ASC", "DESC"):
                sort_dir = "DESC"

            offset = (page - 1) * per_page
            rows = conn.execute(
                f"""SELECT id, run_id, symbol, timeframe, ref_id, ref_label,
                           dtw_score, ml_probs_json, chart_path, created_at
                    FROM scan_results {where_sql}
                    ORDER BY {sort_by} {sort_dir}
                    LIMIT ? OFFSET ?""",
                params + [per_page, offset],
            ).fetchall()

            results = []
            for row in rows:
                r = dict(row)
                if r.get("ml_probs_json"):
                    r["ml_probs"] = json.loads(r["ml_probs_json"])
                else:
                    r["ml_probs"] = {}
                del r["ml_probs_json"]
                results.append(r)

            return results, total
        finally:
            conn.close()

    def get_result(self, result_id: int) -> Optional[dict]:
        """Get a single scan result by ID"""
        conn = self._get_conn()
        try:
            row = conn.execute(
                """SELECT id, run_id, symbol, timeframe, ref_id, ref_label,
                          dtw_score, price_distance, diff_distance, best_scale_factor,
                          ml_probs_json, chart_path, match_start_idx, match_end_idx,
                          created_at
                   FROM scan_results WHERE id = ?""",
                (result_id,),
            ).fetchone()
            if not row:
                return None
            r = dict(row)
            if r.get("ml_probs_json"):
                r["ml_probs"] = json.loads(r["ml_probs_json"])
            else:
                r["ml_probs"] = {}
            del r["ml_probs_json"]
            return r
        finally:
            conn.close()

    def get_chart_data(self, result_id: int) -> Optional[dict]:
        """
        Get chart data for a scan result: OHLCV + SMAs + volume spike flags + ML probs.

        Returns dict ready for Plotly.js rendering.
        """
        conn = self._get_conn()
        try:
            row = conn.execute(
                "SELECT window_data_json, ml_probs_json FROM scan_results WHERE id = ?",
                (result_id,),
            ).fetchone()
            if not row or not row["window_data_json"]:
                return None

            raw = json.loads(row["window_data_json"])
            ml_probs = json.loads(row["ml_probs_json"]) if row["ml_probs_json"] else {}

            df = pd.DataFrame(raw)
            if df.empty:
                return None

            # Convert timestamps to ISO strings
            dates = []
            if "timestamp" in df.columns and len(df["timestamp"]) > 0:
                dates = pd.to_datetime(df["timestamp"], unit="s", utc=True).dt.strftime(
                    "%Y-%m-%dT%H:%M:%SZ"
                ).tolist()
            else:
                dates = list(range(len(df)))

            # Compute SMAs
            sma_data = {}
            for period in self.config.sma_periods:
                sma = df["close"].rolling(window=period, min_periods=1).mean()
                sma_data[f"sma_{period}"] = [round(v, 6) for v in sma.tolist()]

            # Detect volume spikes
            volumes = df["volume"].values
            is_spike = [False] * len(volumes)
            for i in range(1, len(volumes)):
                if volumes[i - 1] > 0 and volumes[i] > volumes[i - 1] * self.config.volume_spike_ratio:
                    is_spike[i] = True

            return {
                "dates": dates,
                "open": [round(v, 8) for v in df["open"].tolist()],
                "high": [round(v, 8) for v in df["high"].tolist()],
                "low": [round(v, 8) for v in df["low"].tolist()],
                "close": [round(v, 8) for v in df["close"].tolist()],
                "volume": df["volume"].tolist(),
                "is_spike": is_spike,
                "sma": sma_data,
                "ml_probs": ml_probs,
            }
        finally:
            conn.close()

    def get_last_scan_time(self) -> Optional[str]:
        """Get the timestamp of the most recent scan run"""
        conn = self._get_conn()
        try:
            row = conn.execute(
                "SELECT finished_at FROM scan_runs WHERE status='completed' ORDER BY id DESC LIMIT 1"
            ).fetchone()
            return row["finished_at"] if row else None
        finally:
            conn.close()

    def get_total_results(self) -> int:
        """Get total number of scan results"""
        conn = self._get_conn()
        try:
            row = conn.execute("SELECT COUNT(*) as cnt FROM scan_results").fetchone()
            return row["cnt"]
        finally:
            conn.close()
