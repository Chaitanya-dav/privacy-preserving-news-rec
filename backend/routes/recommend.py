# backend/routes/recommend.py

import logging
from flask import Blueprint, request, jsonify

from backend.services.demo_news import DEMO_NEWS

logger = logging.getLogger(__name__)
recommend_bp = Blueprint("recommend", __name__)


def demo_recommendations(history, top_k=10):
    """Transparent category-based fallback, not federated inference."""

    history_ids = {
        str(item.get("news_id", ""))
        if isinstance(item, dict) else str(item)
        for item in history
    }

    categories = {}

    # Build interests from articles in the demo catalogue.
    for article in DEMO_NEWS:
        if article["news_id"] in history_ids:
            category = article["category"]
            categories[category] = categories.get(category, 0) + 1

    candidates = [
        article for article in DEMO_NEWS
        if article["news_id"] not in history_ids
    ]

    candidates.sort(
        key=lambda article: (
            -categories.get(article["category"], 0),
            article["news_id"]
        )
    )

    recommendations = []

    for rank, article in enumerate(candidates[:top_k], start=1):
        result = dict(article)
        result["rank"] = rank
        result["score"] = categories.get(article["category"], 0)
        result["recommendation_mode"] = "demo_category_fallback"
        recommendations.append(result)

    return recommendations


@recommend_bp.route("/", methods=["POST"])
def recommend():
    data = request.get_json(silent=True) or {}
    history = data.get("history", [])
    top_k = data.get("top_k", 10)

    if not isinstance(history, list):
        return jsonify({
            "error": "history must be a list of article IDs"
        }), 400

    try:
        top_k = min(50, max(1, int(top_k)))
    except (TypeError, ValueError):
        top_k = 10

    # Try the existing trained recommender first.
    try:
        from backend.services.recommender_service import (
            get_top_k_recommendations
        )

        results = get_top_k_recommendations(
            history, None, top_k=top_k
        )

        if results:
            return jsonify({
                "mode": "model",
                "recommendations": results
            })

    except Exception:
        logger.exception(
            "Existing model unavailable; using demo recommendations."
        )

    return jsonify({
        "mode": "demo_category_fallback",
        "notice": (
            "These recommendations use category preferences, "
            "not a trained federated model."
        ),
        "recommendations": demo_recommendations(history, top_k)
    })


@recommend_bp.route("/demo", methods=["GET"])
def recommend_demo():
    history = ["DEMO001", "DEMO005"]

    return jsonify({
        "mode": "demo_category_fallback",
        "history_seed": history,
        "recommendations": demo_recommendations(history)
    })