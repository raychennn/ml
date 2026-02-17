"""
Command Parser — Parse Telegram bot command arguments
=======================================================

Parses /command text into structured data.
Date format: YYYY-MM-DDTHH:MM (T separator, converted to space for ReferenceManager).
"""

import re
import logging
from typing import Optional, Dict

logger = logging.getLogger(__name__)


def parse_add_ref(text: str) -> Optional[Dict]:
    """
    Parse /add_ref command arguments.

    Format: /add_ref SYMBOL TIMEFRAME START_DT END_DT LABEL [DESCRIPTION...]

    Date format: YYYY-MM-DDTHH:MM (T separator)
    Example: /add_ref SOLUSDT 4h 2024-01-01T00:00 2024-01-03T00:00 perfect_trend optional description

    Returns:
        {symbol, timeframe, start_dt, end_dt, label, description} or None on parse failure
    """
    parts = text.strip().split()

    # Remove the /add_ref command itself
    if parts and parts[0].startswith("/"):
        parts = parts[1:]

    if len(parts) < 5:
        return None

    symbol = parts[0].upper()
    timeframe = parts[1].lower()
    start_raw = parts[2]
    end_raw = parts[3]
    label = parts[4]
    description = " ".join(parts[5:]) if len(parts) > 5 else ""

    # Validate timeframe
    valid_tfs = {"1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d"}
    if timeframe not in valid_tfs:
        return None

    # Validate and convert datetime format (T separator → space)
    dt_pattern = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}$")
    if not dt_pattern.match(start_raw) or not dt_pattern.match(end_raw):
        return None

    start_dt = start_raw.replace("T", " ")
    end_dt = end_raw.replace("T", " ")

    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "start_dt": start_dt,
        "end_dt": end_dt,
        "label": label,
        "description": description,
    }


def parse_del_ref(text: str) -> Optional[str]:
    """
    Parse /del_ref command arguments.

    Format: /del_ref REF_ID

    Returns:
        ref_id string or None
    """
    parts = text.strip().split()
    if parts and parts[0].startswith("/"):
        parts = parts[1:]

    if len(parts) != 1:
        return None

    return parts[0]
