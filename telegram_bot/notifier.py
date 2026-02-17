"""
Telegram Notifier — Lightweight notification via urllib
========================================================

Uses raw HTTP (urllib) to call Telegram Bot API.
No dependency on python-telegram-bot — just sendPhoto and sendMessage.
Gracefully no-ops if bot token or chat ID is not configured.

Features:
    - Retry with exponential backoff (3 attempts)
    - Graceful degradation when Telegram is unreachable
"""

import os
import json
import logging
import time
import urllib.request
import urllib.error
from typing import Optional

logger = logging.getLogger(__name__)

TELEGRAM_API = "https://api.telegram.org"

# Retry configuration
MAX_RETRIES = 3
INITIAL_BACKOFF = 1.0  # seconds
BACKOFF_MULTIPLIER = 2.0


class TelegramNotifier:
    """Lightweight Telegram Bot API wrapper using urllib"""

    def __init__(self, config):
        self.bot_token = config.telegram_bot_token
        self.chat_id = config.telegram_chat_id
        self._enabled = bool(self.bot_token and self.chat_id)

        if not self._enabled:
            logger.debug(
                "Telegram notifier disabled — "
                "set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID to enable"
            )

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    def send_message(self, text: str, parse_mode: str = "HTML") -> bool:
        """
        Send a text message to the configured chat.

        Returns:
            True if sent successfully (or if Telegram is not configured),
            False on error.
        """
        if not self._enabled:
            logger.debug(f"[Telegram disabled] Would send: {text[:100]}...")
            return True

        url = f"{TELEGRAM_API}/bot{self.bot_token}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": parse_mode,
        }

        return self._post_json_with_retry(url, payload)

    def send_photo(
        self,
        photo_path: str,
        caption: str = "",
        parse_mode: str = "HTML",
    ) -> bool:
        """
        Send a photo with optional caption to the configured chat.

        Returns:
            True if sent successfully (or if Telegram is not configured),
            False on error.
        """
        if not self._enabled:
            logger.debug(
                f"[Telegram disabled] Would send photo: {photo_path} "
                f"caption: {caption[:80]}..."
            )
            return True

        if not os.path.exists(photo_path):
            logger.warning(f"Photo file not found: {photo_path}")
            return False

        url = f"{TELEGRAM_API}/bot{self.bot_token}/sendPhoto"
        return self._post_multipart_with_retry(url, photo_path, caption, parse_mode)

    # ================================================================
    # Retry wrappers
    # ================================================================

    def _post_json_with_retry(self, url: str, payload: dict) -> bool:
        """POST JSON with retry and exponential backoff"""
        backoff = INITIAL_BACKOFF
        for attempt in range(1, MAX_RETRIES + 1):
            success = self._post_json(url, payload)
            if success:
                return True
            if attempt < MAX_RETRIES:
                logger.info(f"Telegram send retry {attempt}/{MAX_RETRIES} in {backoff:.1f}s...")
                time.sleep(backoff)
                backoff *= BACKOFF_MULTIPLIER
        logger.error(f"Telegram send failed after {MAX_RETRIES} attempts")
        return False

    def _post_multipart_with_retry(self, url: str, photo_path: str,
                                    caption: str, parse_mode: str) -> bool:
        """POST multipart with retry and exponential backoff"""
        backoff = INITIAL_BACKOFF
        for attempt in range(1, MAX_RETRIES + 1):
            success = self._post_multipart(url, photo_path, caption, parse_mode)
            if success:
                return True
            if attempt < MAX_RETRIES:
                logger.info(f"Telegram sendPhoto retry {attempt}/{MAX_RETRIES} in {backoff:.1f}s...")
                time.sleep(backoff)
                backoff *= BACKOFF_MULTIPLIER
        logger.error(f"Telegram sendPhoto failed after {MAX_RETRIES} attempts")
        return False

    # ================================================================
    # Low-level HTTP methods
    # ================================================================

    def _post_json(self, url: str, payload: dict) -> bool:
        """POST JSON payload to Telegram API (single attempt)"""
        try:
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                url, data=data,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                if not result.get("ok"):
                    logger.warning(f"Telegram API error: {result}")
                    return False
                return True
        except urllib.error.URLError as e:
            logger.warning(f"Telegram send failed: {e}")
            return False
        except Exception as e:
            logger.warning(f"Telegram send error: {e}")
            return False

    def _post_multipart(
        self,
        url: str,
        photo_path: str,
        caption: str,
        parse_mode: str,
    ) -> bool:
        """POST multipart/form-data for sendPhoto (single attempt)"""
        try:
            boundary = "----TelegramBotBoundary"
            filename = os.path.basename(photo_path)

            with open(photo_path, "rb") as f:
                photo_data = f.read()

            body = bytearray()

            # chat_id field
            body.extend(f"--{boundary}\r\n".encode())
            body.extend(
                f'Content-Disposition: form-data; name="chat_id"\r\n\r\n'
                f"{self.chat_id}\r\n".encode()
            )

            # caption field
            if caption:
                body.extend(f"--{boundary}\r\n".encode())
                body.extend(
                    f'Content-Disposition: form-data; name="caption"\r\n\r\n'
                    f"{caption}\r\n".encode("utf-8")
                )

            # parse_mode field
            body.extend(f"--{boundary}\r\n".encode())
            body.extend(
                f'Content-Disposition: form-data; name="parse_mode"\r\n\r\n'
                f"{parse_mode}\r\n".encode()
            )

            # photo file field
            body.extend(f"--{boundary}\r\n".encode())
            body.extend(
                f'Content-Disposition: form-data; name="photo"; '
                f'filename="{filename}"\r\n'
                f"Content-Type: image/png\r\n\r\n".encode()
            )
            body.extend(photo_data)
            body.extend(f"\r\n--{boundary}--\r\n".encode())

            req = urllib.request.Request(
                url,
                data=bytes(body),
                headers={
                    "Content-Type": f"multipart/form-data; boundary={boundary}",
                },
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                if not result.get("ok"):
                    logger.warning(f"Telegram sendPhoto error: {result}")
                    return False
                return True

        except urllib.error.URLError as e:
            logger.warning(f"Telegram sendPhoto failed: {e}")
            return False
        except Exception as e:
            logger.warning(f"Telegram sendPhoto error: {e}")
            return False
