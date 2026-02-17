"""
Scan Orchestrator — Unified scan pipeline used by bot and scheduler
=====================================================================

Consolidates the duplicated scan logic from bot.py and scheduler.py
into a single, reusable class.

Pipeline:
    1. Incremental data update from Binance
    2. Pattern scanning via ScannerEngine
    3. Chart generation
    4. SQLite storage
    5. Telegram notifications with web links
    6. Alert deduplication (cooldown-based)
"""

import logging
import threading
import time
from datetime import datetime, timezone
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


class ScanOrchestrator:
    """
    Single source of truth for the scan-notify pipeline.

    Both TelegramBot._cmd_scan() and ScanScheduler._run_scheduled_scan()
    delegate to this class instead of reimplementing the same logic.
    """

    def __init__(
        self,
        config,
        scanner_engine,
        reference_manager,
        data_manager,
        binance_client,
        symbol_manager,
        notifier,
        result_store,
        chart_generator,
    ):
        self.config = config
        self.scanner = scanner_engine
        self.ref_mgr = reference_manager
        self.data_manager = data_manager
        self.binance_client = binance_client
        self.symbol_manager = symbol_manager
        self.notifier = notifier
        self.result_store = result_store
        self.chart_gen = chart_generator

        self._scan_lock = threading.Lock()
        self._stop_event = threading.Event()

        # Alert cooldown: {(symbol, timeframe, ref_id): last_alert_ts}
        self._alert_cooldown: Dict[tuple, float] = {}
        self._cooldown_seconds: float = getattr(
            config, "alert_cooldown_hours", 24
        ) * 3600

    # ================================================================
    # Public API
    # ================================================================

    def run_scan(
        self,
        timeframes: Optional[List[str]] = None,
        send_notifications: bool = True,
    ) -> int:
        """
        Execute a full scan cycle: update data → scan → chart → save → notify.

        Args:
            timeframes: Which timeframes to scan (default: all from config)
            send_notifications: Whether to send Telegram alerts

        Returns:
            Total number of alerts found.

        Raises:
            RuntimeError: If another scan is already running.
        """
        if not self._scan_lock.acquire(blocking=False):
            raise RuntimeError("A scan is already running")

        try:
            return self._run_scan_impl(timeframes, send_notifications)
        finally:
            self._scan_lock.release()

    @property
    def is_scanning(self) -> bool:
        return self._scan_lock.locked()

    def stop(self):
        """Signal the orchestrator to stop ongoing work."""
        self._stop_event.set()

    def reset(self):
        """Clear the stop signal for reuse."""
        self._stop_event.clear()

    # ================================================================
    # Internal pipeline
    # ================================================================

    def _run_scan_impl(self, timeframes, send_notifications) -> int:
        timeframes = timeframes or self.config.scan_timeframes

        # Clear BTC cache at the start of each scan cycle for fresh data
        self.scanner.clear_btc_cache()

        references = self.ref_mgr.list_references(active_only=True)
        if not references:
            logger.info("No active references, skipping scan")
            return 0

        run_id = self.result_store.start_run(timeframes)
        total_alerts = 0

        for tf in timeframes:
            if self._stop_event.is_set():
                break

            tf_refs = [r for r in references if r.timeframe == tf]
            if not tf_refs:
                continue

            try:
                # Step 1: Incremental data update
                updated = self.update_data(tf)
                logger.info(f"[{tf}] Data updated: {updated} symbols refreshed")

                # Step 2: Scan
                alerts = self.scanner.scan_timeframe(tf, references)
                logger.info(f"[{tf}] Scan complete: {len(alerts)} raw alerts")

                if not alerts:
                    continue

                # Step 3: Deduplicate (cooldown)
                alerts = self._apply_cooldown(alerts, tf)
                if not alerts:
                    logger.info(f"[{tf}] All alerts filtered by cooldown")
                    continue

                # Step 4: Generate charts
                chart_paths = self._generate_charts(alerts)

                # Step 5: Save to SQLite
                self.result_store.save_results(
                    alerts, run_id=run_id, chart_paths=chart_paths
                )

                # Step 6: Telegram notifications
                if send_notifications:
                    self._send_notifications(alerts, chart_paths)

                total_alerts += len(alerts)

            except Exception as e:
                logger.error(f"Scan failed for {tf}: {e}", exc_info=True)
                self.notifier.send_message(
                    f"Scan error for <b>{tf}</b>: {e}"
                )

        self.result_store.finish_run(run_id, total_alerts)
        return total_alerts

    # ================================================================
    # Data update
    # ================================================================

    def update_data(self, timeframe: str) -> int:
        """
        Incrementally update all symbols for a timeframe.

        Returns:
            Number of symbols that received new data.
        """
        if not self.data_manager or not self.binance_client:
            return 0

        try:
            existing = set(self.data_manager.list_symbols_for_timeframe(timeframe) or [])
            if self.symbol_manager:
                all_symbols = self.symbol_manager.get_active_symbols()
                # Merge: existing symbols + any new active symbols
                symbols = list(all_symbols) if all_symbols else list(existing)
            else:
                symbols = list(existing)
        except Exception:
            return 0

        updated = 0
        errors = 0

        for i, symbol in enumerate(symbols):
            if self._stop_event.is_set():
                break

            try:
                last_ts = self.data_manager.get_last_timestamp(symbol, timeframe)
                new_df = self.binance_client.fetch_klines_incremental(
                    symbol, timeframe, last_ts
                )
                if new_df is not None and not new_df.empty:
                    added = self.data_manager.incremental_update(
                        symbol, timeframe, new_df
                    )
                    if added > 0:
                        updated += 1
            except Exception as e:
                errors += 1
                if errors <= 3:
                    logger.warning(f"Update failed for {symbol} {timeframe}: {e}")

            # Progress log every 50 symbols
            if (i + 1) % 50 == 0:
                logger.info(
                    f"  [{timeframe}] Updated {i+1}/{len(symbols)} symbols..."
                )

        if errors > 3:
            logger.warning(f"  [{timeframe}] {errors} total update errors")

        return updated

    # ================================================================
    # Alert cooldown / deduplication
    # ================================================================

    def _apply_cooldown(self, alerts, timeframe: str) -> list:
        """
        Filter out alerts that were already sent within the cooldown window.

        Key = (symbol, timeframe, ref_id).
        Default cooldown = 24 hours (configurable via config.alert_cooldown_hours).
        """
        now = time.time()
        filtered = []

        for alert in alerts:
            key = (alert.symbol, timeframe, alert.reference.id)

            last_sent = self._alert_cooldown.get(key, 0)
            if now - last_sent < self._cooldown_seconds:
                logger.debug(
                    f"Cooldown active for {alert.symbol}/{timeframe}/{alert.reference.id}, "
                    f"skipping"
                )
                continue

            self._alert_cooldown[key] = now
            filtered.append(alert)

        if len(alerts) != len(filtered):
            logger.info(
                f"[{timeframe}] Cooldown filtered: "
                f"{len(alerts)} → {len(filtered)} alerts"
            )

        # Periodic cleanup of old cooldown entries
        if len(self._alert_cooldown) > 10000:
            cutoff = now - self._cooldown_seconds
            self._alert_cooldown = {
                k: v for k, v in self._alert_cooldown.items() if v > cutoff
            }

        return filtered

    # ================================================================
    # Chart generation
    # ================================================================

    def _generate_charts(self, alerts) -> Dict[int, str]:
        """Generate alert charts, returning {alert_index: path}."""
        chart_paths = {}
        for i, alert in enumerate(alerts):
            try:
                chart_path = self.chart_gen.generate_alert_chart(
                    window_df=alert.window_data,
                    symbol=alert.symbol,
                    timeframe=alert.timeframe,
                    ref_name=alert.reference.label,
                    score=alert.dtw_score,
                    ml_probs=alert.ml_probabilities,
                    context_bars=alert.context_bars,
                )
                if chart_path:
                    chart_paths[i] = chart_path
            except Exception as e:
                logger.warning(f"Chart generation error for {alert.symbol}: {e}")
        return chart_paths

    # ================================================================
    # Notifications
    # ================================================================

    def _send_notifications(self, alerts, chart_paths: Dict[int, str]):
        """Send Telegram photo + caption for each alert."""
        base_url = self.config.web_base_url

        for i, alert in enumerate(alerts):
            chart_path = chart_paths.get(i)
            if not chart_path:
                continue

            ml_text = " | ".join([
                f"{int(float(t)*100)}%↑:{p:.0%}"
                for t, p in sorted(alert.ml_probabilities.items())
            ])

            caption = (
                f"<b>{alert.symbol}</b> | {alert.timeframe}\n"
                f"Ref: {alert.reference.label} | Score: {alert.dtw_score:.3f}\n"
                f"{ml_text}"
            )

            if base_url:
                caption += f"\n\n{base_url}"

            try:
                self.notifier.send_photo(chart_path, caption=caption)
            except Exception as e:
                logger.warning(f"Failed to send alert for {alert.symbol}: {e}")
