"""
API Routes — JSON endpoints for chart data, scan results, and health check
============================================================================
"""

import os
from flask import Blueprint, jsonify, request, current_app

api_bp = Blueprint("api", __name__)


@api_bp.route("/health")
def health():
    """GET /api/health — Health check endpoint for Zeabur / monitoring"""
    config = current_app.config["SYSTEM_CONFIG"]
    store = current_app.config["RESULT_STORE"]
    scanner = current_app.config.get("SCANNER_ENGINE")

    status = "ok"
    checks = {}

    # Check SQLite is accessible
    try:
        total = store.get_total_results()
        checks["database"] = {"status": "ok", "total_results": total}
    except Exception as e:
        checks["database"] = {"status": "error", "error": str(e)}
        status = "degraded"

    # Check data directory exists
    data_exists = os.path.isdir(config.data_root)
    checks["data_dir"] = {"status": "ok" if data_exists else "error", "path": config.data_root}
    if not data_exists:
        status = "degraded"

    # Check ML models
    if scanner:
        ml_loaded = scanner.predictor.is_loaded
        checks["ml_models"] = {
            "status": "ok" if ml_loaded else "fallback",
            "count": len(scanner.predictor.models),
        }
    else:
        checks["ml_models"] = {"status": "unavailable"}

    return jsonify({"status": status, "checks": checks}), 200 if status == "ok" else 200


@api_bp.route("/results")
def list_results():
    """GET /api/results — paginated scan results as JSON"""
    store = current_app.config["RESULT_STORE"]

    page = request.args.get("page", 1, type=int)
    per_page = request.args.get("per_page", 20, type=int)
    timeframe = request.args.get("timeframe", None)
    sort_by = request.args.get("sort_by", "created_at")
    sort_dir = request.args.get("sort_dir", "DESC")

    results, total = store.get_results(
        page=page, per_page=per_page,
        timeframe=timeframe, sort_by=sort_by, sort_dir=sort_dir,
    )

    return jsonify({
        "results": results,
        "total": total,
        "page": page,
        "per_page": per_page,
        "total_pages": (total + per_page - 1) // per_page,
    })


@api_bp.route("/result/<int:result_id>/chart-data")
def chart_data(result_id):
    """GET /api/result/<id>/chart-data — OHLCV + SMAs + spikes + ML probs for Plotly"""
    store = current_app.config["RESULT_STORE"]

    data = store.get_chart_data(result_id)
    if not data:
        return jsonify({"error": "No chart data available"}), 404

    return jsonify(data)
