"""News catalogue route — GET /api/news/ — browse categories and articles."""

import logging
from flask import Blueprint, request, jsonify
from backend.services.model_loader import get_news_df, get_parser

logger = logging.getLogger(__name__)
news_bp = Blueprint("news", __name__)


@news_bp.route("/", methods=["GET"])
def list_news():
    """Return a paginated list of news articles.

    Query params:
        page      (int, default 1)
        per_page  (int, default 20, max 100)
        category  (str, optional filter)
        q         (str, optional title keyword filter)
    """
    news_df = get_news_df()
    if news_df is None:
        return jsonify({"error": "News data not available"}), 503

    page     = max(1, int(request.args.get("page", 1)))
    per_page = min(100, max(1, int(request.args.get("per_page", 20))))
    category = request.args.get("category", "").strip().lower()
    query    = request.args.get("q", "").strip().lower()

    df = news_df.copy()
    if category:
        df = df[df["category"].str.lower() == category]
    if query:
        df = df[df["title"].str.lower().str.contains(query, na=False)]

    total   = len(df)
    start   = (page - 1) * per_page
    end     = start + per_page
    subset  = df.iloc[start:end]

    articles = subset[["news_id", "title", "category", "subcategory", "abstract"]].to_dict("records")
    return jsonify({
        "total": total,
        "page": page,
        "per_page": per_page,
        "articles": articles,
    })


@news_bp.route("/categories", methods=["GET"])
def categories():
    """Return all unique categories and their article counts."""
    news_df = get_news_df()
    if news_df is None:
        return jsonify({"error": "News data not available"}), 503

    counts = (
        news_df.groupby("category")
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .to_dict("records")
    )
    return jsonify({"categories": counts})


@news_bp.route("/<news_id>", methods=["GET"])
def get_article(news_id: str):
    """Return metadata for a single news article by ID."""
    news_df = get_news_df()
    if news_df is None:
        return jsonify({"error": "News data not available"}), 503

    rows = news_df[news_df["news_id"] == news_id]
    if rows.empty:
        return jsonify({"error": f"Article '{news_id}' not found."}), 404

    article = rows.iloc[0][["news_id", "title", "category", "subcategory", "abstract", "url"]].to_dict()
    return jsonify(article)
