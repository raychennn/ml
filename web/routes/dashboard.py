"""
Dashboard Routes — HTML pages for scan results browsing
=========================================================
"""

from flask import Blueprint, render_template, request, current_app

dashboard_bp = Blueprint("dashboard", __name__)


@dashboard_bp.route("/")
def index():
    """Main dashboard — paginated scan results table"""
    store = current_app.config["RESULT_STORE"]
    config = current_app.config["SYSTEM_CONFIG"]

    page = request.args.get("page", 1, type=int)
    per_page = request.args.get("per_page", 20, type=int)
    timeframe = request.args.get("timeframe", None)
    sort_by = request.args.get("sort_by", "created_at")
    sort_dir = request.args.get("sort_dir", "DESC")

    results, total = store.get_results(
        page=page, per_page=per_page,
        timeframe=timeframe, sort_by=sort_by, sort_dir=sort_dir,
    )

    total_pages = (total + per_page - 1) // per_page

    return render_template(
        "dashboard.html",
        results=results,
        page=page,
        total_pages=total_pages,
        total=total,
        per_page=per_page,
        timeframe=timeframe,
        sort_by=sort_by,
        sort_dir=sort_dir,
        timeframes=config.scan_timeframes,
    )


@dashboard_bp.route("/result/<int:result_id>")
def detail(result_id):
    """Detail page — interactive Plotly chart + ML probabilities"""
    store = current_app.config["RESULT_STORE"]

    result = store.get_result(result_id)
    if not result:
        return render_template("base.html", content="Result not found"), 404

    return render_template("detail.html", result=result)


@dashboard_bp.route("/status")
def status():
    """System status page"""
    config = current_app.config["SYSTEM_CONFIG"]
    store = current_app.config["RESULT_STORE"]
    scanner = current_app.config.get("SCANNER_ENGINE")
    ref_mgr = current_app.config.get("REFERENCE_MANAGER")

    info = {
        "ml_loaded": scanner.predictor.is_loaded if scanner else False,
        "model_count": len(scanner.predictor.models) if scanner else 0,
        "ref_count": ref_mgr.count(active_only=True) if ref_mgr else 0,
        "total_results": store.get_total_results(),
        "last_scan": store.get_last_scan_time() or "Never",
        "timeframes": config.scan_timeframes,
        "data_root": config.data_root,
        "web_host": config.web_host,
        "web_port": config.web_port,
    }

    return render_template("status.html", info=info)
