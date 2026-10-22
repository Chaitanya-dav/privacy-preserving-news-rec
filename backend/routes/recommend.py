"""Recommend route — POST /api/recommend/"""

import logging
from flask import Blueprint, request, jsonify
from backend.services.recommender_service import get_top_k_recommendations

logger = logging.getLogger(__name__)
recommend_bp = Blueprint("recommend", __name__)


@recommend_bp.route("/", methods=["POST"])
def recommend():
    """Return top-K personalised news recommendations.

    Request JSON:
        {
          "history": ["N12345", "N67890"],   // clicked news IDs (newest last)
          "candidates": ["N111", ...],        // optional explicit candidate pool
          "top_k": 10
        }

    Response JSON:
        {
          "recommendations": [
            {"rank":1, "news_id":"N111", "title":"...", "category":"...",
             "subcategory":"...", "abstract":"...", "score":0.987},
            ...
          ]
        }
    """
    data = request.get_json(silent=True) or {}
    history    = data.get("history", [])
    candidates = data.get("candidates", None)
    top_k      = min(int(data.get("top_k", 10)), 50)

    if not isinstance(history, list):
        return jsonify({"error": "'history' must be a list of news IDs."}), 400

    try:
        recs = get_top_k_recommendations(history, candidates, top_k=top_k)
        return jsonify({"recommendations": recs})
    except Exception as exc:
        logger.exception("Recommendation error: %s", exc)
        return jsonify({"error": str(exc)}), 500


@recommend_bp.route("/demo", methods=["GET"])
def demo():
    """Demo endpoint using a hard-coded history seed — useful for the frontend."""
    from backend.services.model_loader import get_parser
    parser = get_parser()
    # Pick first 5 known news IDs as dummy history
    sample_ids = [nid for nid in list(parser.news2idx.keys())[1:6]]
    recs = get_top_k_recommendations(history_news_ids=sample_ids, candidate_news_ids=None, top_k=10)
    return jsonify({"history_seed": sample_ids, "recommendations": recs})
