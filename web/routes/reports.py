"""
Report Routes — Per-label performance metrics and analysis
=============================================================
"""

import os
import logging
import numpy as np
import pandas as pd
from flask import Blueprint, render_template, current_app, jsonify

logger = logging.getLogger(__name__)

reports_bp = Blueprint("reports", __name__)


@reports_bp.route("/reports")
def report_page():
    """Report page showing per-label performance metrics"""
    config = current_app.config["SYSTEM_CONFIG"]
    ref_mgr = current_app.config.get("REFERENCE_MANAGER")

    labels = []
    if ref_mgr:
        refs = ref_mgr.list_references(active_only=False)
        labels = sorted(set(r.label for r in refs))

    # Try to load training data CSV for metrics
    training_csv = os.path.join(config.data_root, "models", "training_data.csv")
    has_training_data = os.path.exists(training_csv)

    return render_template(
        "reports.html",
        labels=labels,
        has_training_data=has_training_data,
    )


@reports_bp.route("/api/reports/label-metrics")
def label_metrics_api():
    """
    GET /api/reports/label-metrics — Compute per-label performance metrics.

    Returns JSON with metrics computed from historical scan data:
    - For each label: sample count, avg DTW score, soft label distributions,
      and forward gain statistics computed from raw Parquet data.
    """
    config = current_app.config["SYSTEM_CONFIG"]
    ref_mgr = current_app.config.get("REFERENCE_MANAGER")
    data_mgr = current_app.config.get("DATA_MANAGER")

    if not ref_mgr:
        return jsonify({"error": "Reference manager not available", "labels": {}})

    training_csv = os.path.join(config.data_root, "models", "training_data.csv")

    # Strategy 1: Use training_data.csv if available (faster)
    if os.path.exists(training_csv):
        try:
            metrics = _compute_metrics_from_csv(training_csv, config)
            return jsonify({"source": "training_data.csv", "labels": metrics})
        except Exception as e:
            logger.warning(f"Failed to compute metrics from CSV: {e}")

    # Strategy 2: Compute from scan results SQLite + raw Parquet
    store = current_app.config.get("RESULT_STORE")
    if store and data_mgr:
        try:
            metrics = _compute_metrics_from_sqlite(store, data_mgr, config)
            return jsonify({"source": "scan_results", "labels": metrics})
        except Exception as e:
            logger.warning(f"Failed to compute metrics from SQLite: {e}")

    return jsonify({"error": "No data available for metrics", "labels": {}})


def _compute_metrics_from_csv(csv_path: str, config) -> dict:
    """
    Compute per-label metrics from training_data.csv.

    The CSV has columns: symbol, timeframe, ref_id, match_end_ts, dtw_score,
    19 features, and label_0.05/0.1/0.15/0.2 (soft labels: 0.0-1.0).
    """
    df = pd.read_csv(csv_path)

    # Extract label from ref_id (format: SYMBOL_TF_LABEL)
    if "ref_id" not in df.columns:
        return {}

    # Parse label from ref_id
    def extract_label(ref_id):
        parts = str(ref_id).split("_")
        if len(parts) >= 3:
            return "_".join(parts[2:])  # everything after SYMBOL_TF
        return ref_id

    df["label"] = df["ref_id"].apply(extract_label)

    label_cols = [c for c in df.columns if c.startswith("label_")]
    metrics = {}

    for label_name, group in df.groupby("label"):
        n = len(group)

        label_metrics = {
            "sample_count": n,
            "avg_dtw_score": round(float(group["dtw_score"].mean()), 4),
            "min_dtw_score": round(float(group["dtw_score"].min()), 4),
            "max_dtw_score": round(float(group["dtw_score"].max()), 4),
            "symbols_count": int(group["symbol"].nunique()),
            "timeframes": sorted(group["timeframe"].unique().tolist()),
        }

        # Per-threshold metrics from soft labels
        thresholds = {}
        for col in label_cols:
            threshold_str = col.replace("label_", "")
            vals = group[col].dropna()
            if len(vals) == 0:
                continue

            threshold_pct = f"{float(threshold_str)*100:.0f}%"
            thresholds[threshold_pct] = {
                "avg_probability": round(float(vals.mean()), 4),
                "median_probability": round(float(vals.median()), 4),
                "pct_above_50": round(float((vals > 0.5).mean() * 100), 1),
                "pct_above_75": round(float((vals > 0.75).mean() * 100), 1),
                "pct_zero": round(float((vals == 0).mean() * 100), 1),
                "pct_perfect": round(float((vals == 1.0).mean() * 100), 1),
            }

        label_metrics["thresholds"] = thresholds

        # Feature importance indicators
        if "pattern_return" in group.columns:
            label_metrics["avg_pattern_return"] = round(
                float(group["pattern_return"].mean()), 4
            )
        if "pattern_volatility" in group.columns:
            label_metrics["avg_pattern_volatility"] = round(
                float(group["pattern_volatility"].mean()), 4
            )
        if "max_drawdown" in group.columns:
            label_metrics["avg_max_drawdown"] = round(
                float(group["max_drawdown"].mean()), 4
            )

        metrics[str(label_name)] = label_metrics

    return metrics


def _compute_metrics_from_sqlite(store, data_manager, config) -> dict:
    """
    Compute per-label metrics from SQLite scan results + forward gain from Parquet.

    For each stored scan result, computes actual forward price movement
    to derive real P/L metrics.
    """
    import json

    conn = store._get_conn()
    try:
        rows = conn.execute(
            """SELECT symbol, timeframe, ref_id, ref_label, dtw_score,
                      ml_probs_json, window_data_json, created_at
               FROM scan_results
               ORDER BY created_at DESC
               LIMIT 5000"""
        ).fetchall()
    finally:
        conn.close()

    if not rows:
        return {}

    # Group by label
    from collections import defaultdict
    label_data = defaultdict(list)

    for row in rows:
        row = dict(row)
        label = row.get("ref_label") or row.get("ref_id", "unknown")
        ml_probs = json.loads(row["ml_probs_json"]) if row.get("ml_probs_json") else {}

        entry = {
            "symbol": row["symbol"],
            "timeframe": row["timeframe"],
            "dtw_score": row["dtw_score"],
            "ml_probs": ml_probs,
        }

        # Try to compute forward gains from window data
        if row.get("window_data_json"):
            try:
                window = json.loads(row["window_data_json"])
                close = window.get("close", [])
                if len(close) >= 2:
                    entry["pattern_return"] = (close[-1] / close[0] - 1) if close[0] > 0 else 0
                    # Max drawdown within the pattern window
                    close_arr = np.array(close)
                    cummax = np.maximum.accumulate(close_arr)
                    drawdowns = (close_arr - cummax) / (cummax + 1e-10)
                    entry["max_drawdown"] = float(np.min(drawdowns))
                    # Max gain within pattern
                    entry["max_gain"] = float((np.max(close_arr) / close_arr[0] - 1)) if close_arr[0] > 0 else 0
                    # Max loss within pattern
                    entry["max_loss"] = float((np.min(close_arr) / close_arr[0] - 1)) if close_arr[0] > 0 else 0
            except Exception:
                pass

        label_data[label].append(entry)

    # Aggregate metrics per label
    metrics = {}
    for label, entries in label_data.items():
        n = len(entries)
        scores = [e["dtw_score"] for e in entries]

        label_metrics = {
            "sample_count": n,
            "avg_dtw_score": round(np.mean(scores), 4),
            "min_dtw_score": round(np.min(scores), 4),
            "max_dtw_score": round(np.max(scores), 4),
            "symbols_count": len(set(e["symbol"] for e in entries)),
            "timeframes": sorted(set(e["timeframe"] for e in entries)),
        }

        # Aggregate ML probs
        thresholds = {}
        for thresh_key in ["0.05", "0.1", "0.15", "0.2"]:
            probs = [e["ml_probs"].get(thresh_key, 0) for e in entries if e.get("ml_probs")]
            if probs:
                threshold_pct = f"{float(thresh_key)*100:.0f}%"
                probs_arr = np.array(probs)
                thresholds[threshold_pct] = {
                    "avg_probability": round(float(np.mean(probs_arr)), 4),
                    "median_probability": round(float(np.median(probs_arr)), 4),
                    "pct_above_50": round(float((probs_arr > 0.5).mean() * 100), 1),
                    "pct_above_75": round(float((probs_arr > 0.75).mean() * 100), 1),
                }
        label_metrics["thresholds"] = thresholds

        # Forward gain stats
        returns = [e["pattern_return"] for e in entries if "pattern_return" in e]
        if returns:
            returns_arr = np.array(returns)
            label_metrics["avg_pattern_return"] = round(float(np.mean(returns_arr)), 4)
            label_metrics["max_pattern_gain"] = round(float(np.max(returns_arr)), 4)
            label_metrics["max_pattern_loss"] = round(float(np.min(returns_arr)), 4)

        drawdowns = [e["max_drawdown"] for e in entries if "max_drawdown" in e]
        if drawdowns:
            label_metrics["avg_max_drawdown"] = round(float(np.mean(drawdowns)), 4)
            label_metrics["worst_drawdown"] = round(float(np.min(drawdowns)), 4)

        max_gains = [e["max_gain"] for e in entries if "max_gain" in e]
        if max_gains:
            label_metrics["avg_max_gain"] = round(float(np.mean(max_gains)), 4)
            label_metrics["best_max_gain"] = round(float(np.max(max_gains)), 4)

        max_losses = [e["max_loss"] for e in entries if "max_loss" in e]
        if max_losses:
            label_metrics["avg_max_loss"] = round(float(np.mean(max_losses)), 4)
            label_metrics["worst_max_loss"] = round(float(np.min(max_losses)), 4)

        metrics[label] = label_metrics

    return metrics
