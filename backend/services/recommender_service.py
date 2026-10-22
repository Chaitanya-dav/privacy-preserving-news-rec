"""Recommendation service — scores candidates for a given user history."""

import logging
from typing import List, Optional

import torch

logger = logging.getLogger(__name__)


def get_top_k_recommendations(
    history_news_ids: List[str],
    candidate_news_ids: Optional[List[str]],
    top_k: int = 10,
) -> List[dict]:
    """Return top-K recommendations ranked by click probability.

    Args:
        history_news_ids:   List of news IDs the user already clicked.
        candidate_news_ids: Optional explicit candidate pool. If None, defaults
                            to a random sample of 200 news from the vocabulary.
        top_k:              Number of recommendations to return.

    Returns:
        List of dicts with keys: news_id, title, category, score, rank.
    """
    from backend.services.model_loader import (
        get_model, get_embedding_cache, get_parser, get_news_df,
    )

    model  = get_model()
    cache  = get_embedding_cache()
    parser = get_parser()
    news_df = get_news_df()

    if model is None:
        logger.warning("Model not loaded — returning empty recommendations.")
        return []

    # ── Build history tensor ────────────────────────────────────────────────
    history_indices = []
    for nid in history_news_ids[-50:]:               # cap at max_history_len
        history_indices.append(parser.news2idx.get(nid, 0))
    # Pad / truncate to exactly 50
    max_h = parser.max_history_len
    if len(history_indices) < max_h:
        history_indices = [0] * (max_h - len(history_indices)) + history_indices
    else:
        history_indices = history_indices[-max_h:]

    # ── Build candidate pool ────────────────────────────────────────────────
    if not candidate_news_ids:
        import random
        all_ids = [nid for nid in parser.news2idx if nid not in ("<PAD>",)]
        candidate_news_ids = random.sample(all_ids, min(200, len(all_ids)))

    candidate_indices = [parser.news2idx.get(nid, 0) for nid in candidate_news_ids]
    # Filter out unknowns (index 0 = PAD)
    valid = [(nid, idx) for nid, idx in zip(candidate_news_ids, candidate_indices) if idx != 0]
    if not valid:
        return []
    valid_nids, valid_idxs = zip(*valid)

    # ── Inference ───────────────────────────────────────────────────────────
    with torch.no_grad():
        hist_t  = torch.tensor([history_indices], dtype=torch.long)   # (1, H)
        cand_t  = torch.tensor([list(valid_idxs)], dtype=torch.long)  # (1, C)
        scores  = model.session_forward(cand_t, hist_t, cache)        # (1, C)
        scores  = scores[0].cpu().tolist()                             # (C,)

    # ── Rank and enrich ─────────────────────────────────────────────────────
    ranked = sorted(zip(valid_nids, scores), key=lambda x: -x[1])[:top_k]

    results = []
    news_map = {}
    if news_df is not None:
        news_map = news_df.set_index("news_id")[["title", "category", "subcategory", "abstract"]].to_dict("index")

    for rank, (nid, score) in enumerate(ranked, 1):
        meta = news_map.get(nid, {})
        results.append({
            "rank": rank,
            "news_id": nid,
            "title": meta.get("title", nid),
            "category": meta.get("category", ""),
            "subcategory": meta.get("subcategory", ""),
            "abstract": meta.get("abstract", "")[:200],
            "score": round(float(score), 6),
        })

    return results
