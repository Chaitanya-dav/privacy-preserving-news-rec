"""Ranking Evaluation Metrics for News Recommendation.

Implements standard metrics used in MIND benchmark evaluation:
  - AUC  (Area Under the ROC Curve)
  - MRR  (Mean Reciprocal Rank)
  - nDCG@5, nDCG@10 (Normalized Discounted Cumulative Gain)

All metrics are computed per-impression and then macro-averaged.
"""

import logging
import math
from typing import Dict, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def roc_auc_score_simple(labels: List[float], scores: List[float]) -> float:
    """Compute AUC without scikit-learn via Wilcoxon-Mann-Whitney statistic."""
    pos_scores = [s for l, s in zip(labels, scores) if l == 1.0]
    neg_scores = [s for l, s in zip(labels, scores) if l == 0.0]

    if not pos_scores or not neg_scores:
        return 0.5  # Degenerate case

    wins = sum(1 for p in pos_scores for n in neg_scores if p > n)
    ties = sum(0.5 for p in pos_scores for n in neg_scores if p == n)
    total = len(pos_scores) * len(neg_scores)
    return (wins + ties) / total


def mrr_score(labels: List[float], scores: List[float]) -> float:
    """Mean Reciprocal Rank (first relevant item rank)."""
    paired = sorted(zip(scores, labels), key=lambda x: -x[0])
    for rank, (_, label) in enumerate(paired, start=1):
        if label == 1.0:
            return 1.0 / rank
    return 0.0


def ndcg_score(labels: List[float], scores: List[float], k: int) -> float:
    """nDCG@k — normalized discounted cumulative gain at cutoff k."""
    paired = sorted(zip(scores, labels), key=lambda x: -x[0])
    dcg = sum(
        label / math.log2(rank + 1)
        for rank, (_, label) in enumerate(paired[:k], start=1)
    )
    # Ideal DCG: top-k ground truth sorted descending
    ideal = sorted(labels, reverse=True)[:k]
    idcg = sum(l / math.log2(rank + 1) for rank, l in enumerate(ideal, start=1))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_impressions(
    impression_scores: List[List[float]],
    impression_labels: List[List[float]],
    ndcg_ks: Tuple[int, ...] = (5, 10),
) -> Dict[str, float]:
    """Macro-average evaluation metrics over all dev impression sessions.

    Args:
        impression_scores: List of per-session score lists.
        impression_labels: List of per-session binary label lists.
        ndcg_ks: Tuple of K cutoffs for nDCG.

    Returns:
        Dictionary with keys: 'AUC', 'MRR', 'nDCG@5', 'nDCG@10', ...
    """
    if len(impression_scores) != len(impression_labels):
        raise ValueError("impression_scores and impression_labels must have same length.")

    auc_list, mrr_list = [], []
    ndcg_lists: Dict[int, List[float]] = {k: [] for k in ndcg_ks}

    for scores, labels in zip(impression_scores, impression_labels):
        if sum(labels) == 0:
            # Skip sessions with no positive samples (can't evaluate)
            continue
        auc_list.append(roc_auc_score_simple(labels, scores))
        mrr_list.append(mrr_score(labels, scores))
        for k in ndcg_ks:
            ndcg_lists[k].append(ndcg_score(labels, scores, k))

    results = {
        "AUC": float(np.mean(auc_list)) if auc_list else 0.0,
        "MRR": float(np.mean(mrr_list)) if mrr_list else 0.0,
    }
    for k in ndcg_ks:
        results[f"nDCG@{k}"] = float(np.mean(ndcg_lists[k])) if ndcg_lists[k] else 0.0

    return results
