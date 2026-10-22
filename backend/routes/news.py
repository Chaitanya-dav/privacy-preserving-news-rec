# backend/routes/news.py

import logging
from flask import Blueprint, request, jsonify

from backend.services.demo_news import DEMO_NEWS

logger = logging.getLogger(__name__)
news_bp = Blueprint("news", __name__)


def get_catalogue():
    """Try loading MIND data; otherwise return demo articles."""
    try:
        from backend.services.model_loader import get_news_df

        df = get_news_df()

        if df is not None and not df.empty:
            articles = []

            for row in df.fillna("").to_dict("records"):
                news_id = str(
                    row.get("news_id", row.get("newsId", ""))
                )

                if not news_id:
                    continue

                articles.append({
                    "news_id": news_id,
                    "title": str(row.get("title", news_id)),
                    "category": str(row.get("category", "news")),
                    "subcategory": str(row.get("subcategory", "")),
                    "abstract": str(row.get("abstract", "")),
                    "image_url": str(
                        row.get("image_url", row.get("image", ""))
                    ),
                    "url": str(row.get("url", "")),
                    "source": str(row.get("source", "MIND dataset"))
                })

            if articles:
                return articles, "mind"

    except Exception:
        logger.exception("MIND catalogue unavailable; using demo data.")

    return DEMO_NEWS, "demo"


@news_bp.route("/", methods=["GET"])
def list_news():
    articles, source = get_catalogue()

    category = request.args.get("category", "").strip().lower()
    query = request.args.get("q", "").strip().lower()
    page = max(1, request.args.get("page", 1, type=int) or 1)
    per_page = min(
        100, max(1, request.args.get("per_page", 20, type=int) or 20)
    )

    if category:
        articles = [
            a for a in articles
            if a.get("category", "").lower() == category
        ]

    if query:
        articles = [
            a for a in articles
            if query in a.get("title", "").lower()
            or query in a.get("abstract", "").lower()
        ]

    total = len(articles)
    start = (page - 1) * per_page

    return jsonify({
        "source": source,
        "total": total,
        "page": page,
        "per_page": per_page,
        "articles": articles[start:start + per_page]
    })


@news_bp.route("/categories", methods=["GET"])
def get_categories():
    articles, source = get_catalogue()
    counts = {}

    for article in articles:
        category = article.get("category", "other")
        counts[category] = counts.get(category, 0) + 1

    return jsonify({
        "source": source,
        "categories": [
            {"category": category, "count": count}
            for category, count in sorted(counts.items())
        ]
    })


@news_bp.route("/<news_id>", methods=["GET"])
def get_article(news_id):
    articles, _ = get_catalogue()

    for article in articles:
        if article["news_id"] == news_id:
            return jsonify(article)

    return jsonify({"error": "Article not found"}), 404