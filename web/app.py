"""
Flask Web App — App factory for the crypto pattern dashboard
==============================================================

Modular design with separate blueprints:
  - dashboard: Scan results browsing
  - api: JSON endpoints + health check
  - references: Reference pattern viewing with chart images
  - reports: Per-label performance metrics
"""

from flask import Flask


def create_app(
    config,
    result_store,
    scanner_engine=None,
    reference_manager=None,
    data_manager=None,
    chart_generator=None,
):
    """
    Create and configure the Flask application.

    Args:
        config: SystemConfig
        result_store: ScanResultStore
        scanner_engine: Optional ScannerEngine (for triggering scans from web)
        reference_manager: Optional ReferenceManager (for references + status)
        data_manager: Optional DataManager (for reading Parquet data)
        chart_generator: Optional ChartGenerator (for generating reference charts)

    Returns:
        Flask app instance
    """
    import os
    app = Flask(
        __name__,
        template_folder=os.path.join(os.path.dirname(__file__), "templates"),
        static_folder=os.path.join(os.path.dirname(__file__), "static"),
    )
    app.config["SECRET_KEY"] = os.environ.get("FLASK_SECRET_KEY", "crypto-pattern-dev-key")

    # Store references for use in routes
    app.config["SYSTEM_CONFIG"] = config
    app.config["RESULT_STORE"] = result_store
    app.config["SCANNER_ENGINE"] = scanner_engine
    app.config["REFERENCE_MANAGER"] = reference_manager
    app.config["DATA_MANAGER"] = data_manager
    app.config["CHART_GENERATOR"] = chart_generator

    # Register blueprints
    from web.routes.dashboard import dashboard_bp
    from web.routes.api import api_bp
    from web.routes.references import references_bp
    from web.routes.reports import reports_bp

    app.register_blueprint(dashboard_bp)
    app.register_blueprint(api_bp, url_prefix="/api")
    app.register_blueprint(references_bp)
    app.register_blueprint(reports_bp)

    return app
