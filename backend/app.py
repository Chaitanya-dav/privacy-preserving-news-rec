"""Privacy-Preserving Multimodal News Recommender — Flask Backend API.

Serves live recommendations from trained centralized/federated checkpoints
via a JSON REST API consumed by the frontend dashboard.
"""

import logging
import os
from pathlib import Path

from flask import Flask, jsonify
from flask_cors import CORS

from backend.routes.recommend import recommend_bp
from backend.routes.metrics import metrics_bp
from backend.routes.news import news_bp

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


def create_app() -> Flask:
    """Application factory."""
    app = Flask(__name__)
    CORS(app)  # Allow requests from frontend dev server

    # Register route blueprints
    app.register_blueprint(recommend_bp, url_prefix="/api/recommend")
    app.register_blueprint(metrics_bp,   url_prefix="/api/metrics")
    app.register_blueprint(news_bp,      url_prefix="/api/news")

    @app.route("/api/health")
    def health():
        return jsonify({"status": "ok", "service": "news-rec-api"})

    @app.errorhandler(404)
    def not_found(e):
        return jsonify({"error": "Not found", "code": 404}), 404

    @app.errorhandler(500)
    def internal_error(e):
        return jsonify({"error": "Internal server error", "code": 500}), 500

    logger.info("Flask app created — blueprints: recommend, metrics, news.")
    return app


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app = create_app()
    app.run(host="0.0.0.0", port=port, debug=True)
