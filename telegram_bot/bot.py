"""
Telegram Bot — Long-polling command handler
==============================================

Runs as a background daemon thread, polling Telegram for commands.
Uses raw urllib (consistent with existing notifier).

Commands:
    /scan       — Run scanner for all timeframes
    /add_ref    — Add a new reference pattern
    /del_ref    — Delete (deactivate) a reference
    /list_ref   — List all active references
    /retrain    — Launch model retraining subprocess
    /status     — Show system status
    /config     — View/change runtime settings
"""

import os
import sys
import json
import logging
import subprocess
import threading
import time
import traceback
import urllib.request
import urllib.error
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

TELEGRAM_API = "https://api.telegram.org"


class TelegramBot:
    """Long-polling Telegram bot for crypto pattern system"""

    def __init__(self, config, scanner_engine, reference_manager, notifier, result_store,
                 data_manager=None, binance_client=None, symbol_manager=None,
                 scan_orchestrator=None):
        self.config = config
        self.scanner = scanner_engine
        self.ref_mgr = reference_manager
        self.notifier = notifier
        self.result_store = result_store
        self.data_manager = data_manager
        self.binance_client = binance_client
        self.symbol_manager = symbol_manager
        self.orchestrator = scan_orchestrator

        self.bot_token = config.telegram_bot_token
        self.chat_id = str(config.telegram_chat_id)
        self._enabled = bool(self.bot_token and self.chat_id)

        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._last_update_id = 0
        self._retrain_lock = threading.Lock()
        self._retrain_process: Optional[subprocess.Popen] = None
        self._retrain_running = False
        self._last_scan_time: Optional[str] = None

        # Configurable settings (modifiable at runtime via /config)
        self._runtime_config = {
            "min_similarity_score": config.min_similarity_score,
            "volume_spike_ratio": config.volume_spike_ratio,
            "prescreening_top_ratio": config.prescreening_top_ratio,
            "alert_cooldown_hours": getattr(config, "alert_cooldown_hours", 24),
        }

    # ================================================================
    # Lifecycle
    # ================================================================

    def start(self):
        """Start the bot polling loop in a background thread"""
        if not self._enabled:
            logger.warning("Telegram bot disabled — missing bot token or chat ID")
            return

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._poll_loop, daemon=True, name="telegram-bot")
        self._thread.start()
        logger.info("Telegram bot started (long polling)")

    def stop(self):
        """Stop the bot polling loop and clean up resources"""
        self._stop_event.set()

        # Kill any running retrain subprocess
        with self._retrain_lock:
            if self._retrain_process and self._retrain_process.poll() is None:
                logger.info("Killing running retrain subprocess...")
                self._retrain_process.kill()
                self._retrain_process = None
                self._retrain_running = False

        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None
        logger.info("Telegram bot stopped")

    # ================================================================
    # Polling Loop
    # ================================================================

    def _poll_loop(self):
        """Main polling loop — runs in background thread"""
        logger.info("Bot polling loop started")
        while not self._stop_event.is_set():
            try:
                updates = self._get_updates(timeout=30)
                for update in updates:
                    self._handle_update(update)
            except Exception as e:
                logger.error(f"Bot polling error: {e}")
                if not self._stop_event.is_set():
                    time.sleep(5)

    def _get_updates(self, timeout: int = 30) -> list:
        """Fetch updates from Telegram using long polling"""
        url = (
            f"{TELEGRAM_API}/bot{self.bot_token}/getUpdates"
            f"?offset={self._last_update_id + 1}&timeout={timeout}"
        )
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=timeout + 10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                if data.get("ok"):
                    updates = data.get("result", [])
                    if updates:
                        self._last_update_id = updates[-1]["update_id"]
                    return updates
        except urllib.error.URLError:
            pass
        except Exception as e:
            logger.debug(f"getUpdates error: {e}")
        return []

    def _handle_update(self, update: dict):
        """Route an update to the appropriate command handler"""
        message = update.get("message")
        if not message:
            return

        chat_id = str(message.get("chat", {}).get("id", ""))
        if chat_id != self.chat_id:
            logger.debug(f"Ignoring message from unauthorized chat: {chat_id}")
            return

        text = message.get("text", "").strip()
        if not text.startswith("/"):
            return

        # Parse command
        cmd = text.split()[0].lower()
        # Strip @botname suffix if present
        if "@" in cmd:
            cmd = cmd.split("@")[0]

        try:
            if cmd == "/scan":
                self._cmd_scan(text)
            elif cmd == "/add_ref":
                self._cmd_add_ref(text)
            elif cmd == "/del_ref":
                self._cmd_del_ref(text)
            elif cmd == "/list_ref":
                self._cmd_list_ref()
            elif cmd == "/retrain":
                self._cmd_retrain()
            elif cmd == "/status":
                self._cmd_status()
            elif cmd == "/config":
                self._cmd_config(text)
            elif cmd == "/start" or cmd == "/help":
                self._cmd_help()
            else:
                self._send(f"Unknown command: <code>{cmd}</code>\nUse /help for available commands.")
        except Exception as e:
            logger.error(f"Command handler error: {traceback.format_exc()}")
            self._send(f"Error: {e}")

    # ================================================================
    # Command Handlers
    # ================================================================

    def _cmd_scan(self, text: str):
        """Run scanner via ScanOrchestrator"""
        if self.orchestrator and self.orchestrator.is_scanning:
            self._send("A scan is already running. Please wait.")
            return

        self._send("Scan started... This may take several minutes.")
        thread = threading.Thread(target=self._run_scan, daemon=True, name="bot-scan")
        thread.start()

    def _run_scan(self):
        """Worker thread for scanning — delegates to orchestrator"""
        try:
            if self.orchestrator:
                total = self.orchestrator.run_scan()
            else:
                self._send("Scan orchestrator not available.")
                return

            self._last_scan_time = datetime.now(timezone.utc).isoformat()
            base_url = self.config.web_base_url
            self._send(
                f"Scan complete: <b>{total}</b> alerts found.\n"
                f"Dashboard: {base_url}"
            )
        except RuntimeError as e:
            self._send(str(e))
        except Exception as e:
            logger.error(f"Scan error: {traceback.format_exc()}")
            self._send(f"Scan failed: {e}")

    def _cmd_add_ref(self, text: str):
        """Add a new reference pattern"""
        from telegram_bot.command_parser import parse_add_ref

        parsed = parse_add_ref(text)
        if not parsed:
            self._send(
                "Usage: /add_ref SYMBOL TF START END LABEL [description]\n"
                "Example:\n"
                "<code>/add_ref SOLUSDT 4h 2024-01-01T00:00 2024-01-03T00:00 perfect_trend nice pattern</code>\n\n"
                "Date format: YYYY-MM-DDTHH:MM"
            )
            return

        try:
            ref = self.ref_mgr.add_reference_by_datetime(
                symbol=parsed["symbol"],
                timeframe=parsed["timeframe"],
                start_dt=parsed["start_dt"],
                end_dt=parsed["end_dt"],
                label=parsed["label"],
                description=parsed["description"],
            )
            # Notification is handled by ReferenceManager._notify_reference_added
            if not self.ref_mgr._notifier:
                self._send(f"Reference added: <code>{ref.id}</code>")
        except ValueError as e:
            self._send(f"Failed to add reference: {e}")
        except Exception as e:
            logger.error(f"add_ref error: {e}")
            self._send(f"Error: {e}")

    def _cmd_del_ref(self, text: str):
        """Delete (deactivate) a reference"""
        from telegram_bot.command_parser import parse_del_ref

        ref_id = parse_del_ref(text)
        if not ref_id:
            self._send(
                "Usage: /del_ref REF_ID\n"
                "Use /list_ref to see available reference IDs."
            )
            return

        success = self.ref_mgr.delete_reference(ref_id)
        if not success:
            self._send(f"Reference not found: <code>{ref_id}</code>")

    def _cmd_list_ref(self):
        """List all active references"""
        refs = self.ref_mgr.list_references(active_only=True)
        if not refs:
            self._send("No active references.")
            return

        lines = ["<b>Active References</b>\n"]
        for r in refs:
            lines.append(
                f"<code>{r.id}</code>\n"
                f"  {r.symbol} | {r.timeframe} | {r.label}\n"
            )
        lines.append(f"\nTotal: {len(refs)}")
        self._send("\n".join(lines))

    def _cmd_retrain(self):
        """Launch model retraining as a subprocess"""
        with self._retrain_lock:
            if self._retrain_running:
                self._send("Retraining is already in progress.")
                return
            self._retrain_running = True

        self._send("Retraining started... This may take a long time.")
        thread = threading.Thread(target=self._run_retrain, daemon=True, name="bot-retrain")
        thread.start()

    def _run_retrain(self):
        """Worker thread for retraining — uses non-blocking polling"""
        try:
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            script_path = os.path.join(project_root, "scripts", "train_model.py")

            with self._retrain_lock:
                self._retrain_process = subprocess.Popen(
                    [sys.executable, script_path],
                    cwd=project_root,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                )

            # Non-blocking poll instead of communicate(timeout=7200)
            timeout_seconds = 7200
            start_time = time.time()
            output_lines = []

            while True:
                if self._stop_event.is_set():
                    with self._retrain_lock:
                        if self._retrain_process and self._retrain_process.poll() is None:
                            self._retrain_process.kill()
                    self._send("Retraining cancelled (bot stopping).")
                    return

                with self._retrain_lock:
                    proc = self._retrain_process
                    if proc is None:
                        return

                retcode = proc.poll()
                if retcode is not None:
                    # Process finished
                    remaining = proc.stdout.read() if proc.stdout else ""
                    if remaining:
                        output_lines.append(remaining)
                    break

                # Check timeout
                if time.time() - start_time > timeout_seconds:
                    with self._retrain_lock:
                        if self._retrain_process:
                            self._retrain_process.kill()
                    self._send("Retraining timed out (2h limit).")
                    return

                # Read available output (non-blocking)
                try:
                    line = proc.stdout.readline()
                    if line:
                        output_lines.append(line)
                except Exception:
                    pass

                time.sleep(1)

            # Check result
            if retcode == 0:
                try:
                    self.scanner.predictor.load()
                    self._send("Retraining completed successfully. Models reloaded.")
                except Exception as e:
                    self._send(f"Retraining completed but model reload failed: {e}")
            else:
                tail = "".join(output_lines)[-500:] if output_lines else "(no output)"
                self._send(f"Retraining failed (exit code {retcode}).\n<pre>{tail}</pre>")

        except Exception as e:
            logger.error(f"Retrain error: {traceback.format_exc()}")
            self._send(f"Retraining error: {e}")
        finally:
            with self._retrain_lock:
                self._retrain_running = False
                self._retrain_process = None

    def _cmd_status(self):
        """Show system status"""
        models_loaded = self.scanner.predictor.is_loaded
        model_count = len(self.scanner.predictor.models)
        ref_count = self.ref_mgr.count(active_only=True)
        total_results = self.result_store.get_total_results()
        last_scan = self._last_scan_time or self.result_store.get_last_scan_time() or "Never"
        scanning = self.orchestrator.is_scanning if self.orchestrator else False

        status_lines = [
            "<b>System Status</b>\n",
            f"<b>ML Models:</b> {'Loaded' if models_loaded else 'Fallback'} ({model_count} models)",
            f"<b>References:</b> {ref_count} active",
            f"<b>Scan Results:</b> {total_results} total",
            f"<b>Last Scan:</b> {last_scan}",
            f"<b>Scan Running:</b> {'Yes' if scanning else 'No'}",
            f"<b>Retrain Running:</b> {'Yes' if self._retrain_running else 'No'}",
            f"<b>Timeframes:</b> {', '.join(self.config.scan_timeframes)}",
            f"<b>Web Dashboard:</b> {self.config.web_base_url}",
        ]
        self._send("\n".join(status_lines))

    def _cmd_config(self, text: str):
        """View or change runtime configuration"""
        parts = text.strip().split()

        # /config — show current settings
        if len(parts) == 1:
            lines = ["<b>Runtime Configuration</b>\n"]
            for key, value in self._runtime_config.items():
                lines.append(f"  <code>{key}</code> = {value}")
            lines.append(
                "\nTo change: <code>/config KEY VALUE</code>\n"
                "Example: <code>/config min_similarity_score 0.30</code>"
            )
            self._send("\n".join(lines))
            return

        # /config KEY VALUE — set a value
        if len(parts) < 3:
            self._send("Usage: <code>/config KEY VALUE</code>\nUse <code>/config</code> to see available keys.")
            return

        key = parts[1]
        value_str = parts[2]

        if key not in self._runtime_config:
            self._send(
                f"Unknown config key: <code>{key}</code>\n"
                f"Available: {', '.join(self._runtime_config.keys())}"
            )
            return

        try:
            # Parse value as same type as current
            current = self._runtime_config[key]
            if isinstance(current, float):
                new_value = float(value_str)
            elif isinstance(current, int):
                new_value = int(value_str)
            else:
                new_value = value_str

            old_value = self._runtime_config[key]
            self._runtime_config[key] = new_value

            # Apply to actual config
            if hasattr(self.config, key):
                setattr(self.config, key, new_value)

            # Special handling for orchestrator cooldown
            if key == "alert_cooldown_hours" and self.orchestrator:
                self.orchestrator._cooldown_seconds = new_value * 3600

            self._send(
                f"Config updated:\n"
                f"  <code>{key}</code>: {old_value} → <b>{new_value}</b>"
            )
        except ValueError:
            self._send(f"Invalid value: <code>{value_str}</code> (expected number)")

    def _cmd_help(self):
        """Show available commands"""
        self._send(
            "<b>Available Commands</b>\n\n"
            "/scan — Run pattern scanner\n"
            "/add_ref — Add reference pattern\n"
            "/del_ref — Delete reference\n"
            "/list_ref — List references\n"
            "/retrain — Retrain ML models\n"
            "/status — System status\n"
            "/config — View/change settings\n"
            "/help — This message"
        )

    # ================================================================
    # Helpers
    # ================================================================

    def _send(self, text: str):
        """Send a message to the configured chat"""
        self.notifier.send_message(text)
