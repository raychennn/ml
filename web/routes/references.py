"""
Reference Routes — View reference patterns with chart images
==============================================================
"""

import os
import io
import logging
from flask import Blueprint, render_template, request, current_app, send_file, abort

logger = logging.getLogger(__name__)

references_bp = Blueprint("references", __name__)


@references_bp.route("/references")
def list_references():
    """List all reference patterns with chart links"""
    ref_mgr = current_app.config.get("REFERENCE_MANAGER")
    config = current_app.config["SYSTEM_CONFIG"]

    if not ref_mgr:
        return render_template("references.html", references=[], error="Reference manager not available")

    show_inactive = request.args.get("show_inactive", "0") == "1"
    refs = ref_mgr.list_references(active_only=not show_inactive)

    # Enrich with chart availability info
    enriched = []
    for ref in refs:
        safe_label = ref.label.replace("/", "_").replace(" ", "_")
        sma_chart = os.path.join(
            config.image_dir, "references",
            f"{ref.symbol}_{ref.timeframe}_{safe_label}_sma.png",
        )
        full_chart = os.path.join(
            config.image_dir, "references",
            f"{ref.symbol}_{ref.timeframe}_{safe_label}.png",
        )
        enriched.append({
            "ref": ref,
            "has_sma_chart": os.path.exists(sma_chart),
            "has_full_chart": os.path.exists(full_chart),
        })

    return render_template(
        "references.html",
        references=enriched,
        show_inactive=show_inactive,
    )


@references_bp.route("/reference/<ref_id>")
def reference_detail(ref_id):
    """Detail page for a single reference with both chart types"""
    ref_mgr = current_app.config.get("REFERENCE_MANAGER")
    config = current_app.config["SYSTEM_CONFIG"]

    if not ref_mgr:
        abort(404)

    ref = ref_mgr.get_reference(ref_id)
    if not ref:
        abort(404)

    safe_label = ref.label.replace("/", "_").replace(" ", "_")
    sma_chart_name = f"{ref.symbol}_{ref.timeframe}_{safe_label}_sma.png"
    full_chart_name = f"{ref.symbol}_{ref.timeframe}_{safe_label}.png"

    sma_path = os.path.join(config.image_dir, "references", sma_chart_name)
    full_path = os.path.join(config.image_dir, "references", full_chart_name)

    # Generate charts if they don't exist yet
    chart_gen = current_app.config.get("CHART_GENERATOR")
    data_mgr = current_app.config.get("DATA_MANAGER")

    if chart_gen and data_mgr:
        if not os.path.exists(sma_path) or not os.path.exists(full_path):
            try:
                _generate_reference_charts(ref, config, data_mgr, chart_gen)
            except Exception as e:
                logger.warning(f"Failed to generate reference charts: {e}")

    import pytz
    display_tz = pytz.timezone(config.reference_input_timezone)
    from datetime import datetime
    dt_start = datetime.fromtimestamp(ref.start_ts, tz=display_tz)
    dt_end = datetime.fromtimestamp(ref.end_ts, tz=display_tz)

    return render_template(
        "reference_detail.html",
        ref=ref,
        dt_start=dt_start.strftime("%Y-%m-%d %H:%M"),
        dt_end=dt_end.strftime("%Y-%m-%d %H:%M"),
        has_sma_chart=os.path.exists(sma_path),
        has_full_chart=os.path.exists(full_path),
    )


@references_bp.route("/reference/<ref_id>/chart/<chart_type>")
def reference_chart_image(ref_id, chart_type):
    """Serve reference chart images (sma or full)"""
    ref_mgr = current_app.config.get("REFERENCE_MANAGER")
    config = current_app.config["SYSTEM_CONFIG"]

    if not ref_mgr or chart_type not in ("sma", "full"):
        abort(404)

    ref = ref_mgr.get_reference(ref_id)
    if not ref:
        abort(404)

    safe_label = ref.label.replace("/", "_").replace(" ", "_")
    suffix = "_sma" if chart_type == "sma" else ""
    filename = f"{ref.symbol}_{ref.timeframe}_{safe_label}{suffix}.png"
    path = os.path.join(config.image_dir, "references", filename)

    if not os.path.exists(path):
        # Try to generate on the fly
        chart_gen = current_app.config.get("CHART_GENERATOR")
        data_mgr = current_app.config.get("DATA_MANAGER")
        if chart_gen and data_mgr:
            try:
                _generate_reference_charts(ref, config, data_mgr, chart_gen)
            except Exception as e:
                logger.warning(f"Failed to generate chart: {e}")

    if not os.path.exists(path):
        abort(404)

    return send_file(path, mimetype="image/png")


def _generate_reference_charts(ref, config, data_manager, chart_generator):
    """Generate both SMA-only and full candlestick charts for a reference"""
    # 載入含 SMA 預熱段的數據
    sma_lookback = config.sma_lookback
    padding_seconds = sma_lookback * config.timeframe_to_seconds(ref.timeframe)
    padded_start_ts = ref.start_ts - padding_seconds

    ref_df_padded = data_manager.read_symbol(
        ref.symbol, ref.timeframe,
        start_ts=padded_start_ts, end_ts=ref.end_ts,
    )
    if ref_df_padded.empty or len(ref_df_padded) < 3:
        logger.warning(f"Not enough data for reference charts: {ref.id}")
        return

    # 計算 context bars
    ref_mask = ref_df_padded["timestamp"] >= ref.start_ts
    context_bars = int(ref_mask.values.argmax()) if ref_mask.any() else 0

    # Generate full chart (candle + SMA + volume + spikes)
    chart_generator.generate_reference_chart(
        window_df=ref_df_padded,
        symbol=ref.symbol,
        timeframe=ref.timeframe,
        label=ref.label,
        description=ref.description,
        context_bars=context_bars,
    )

    # Generate SMA-only chart (pure SMA lines, no candlesticks)
    chart_generator.generate_reference_sma_chart(
        window_df=ref_df_padded,
        symbol=ref.symbol,
        timeframe=ref.timeframe,
        label=ref.label,
        context_bars=context_bars,
    )
