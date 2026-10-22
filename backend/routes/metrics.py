"""Metrics route — GET /api/metrics/ — serves training history JSON for dashboard charts."""

import logging
from flask import Blueprint, jsonify
from backend.services.model_loader import get_training_history

logger = logging.getLogger(__name__)
metrics_bp = Blueprint("metrics", __name__)


@metrics_bp.route("/", methods=["GET"])
def all_metrics():
    """Return all training metrics for centralized and federated runs."""
    history = get_training_history()
    return jsonify(history)


@metrics_bp.route("/centralized", methods=["GET"])
def centralized_metrics():
    history = get_training_history()
    return jsonify({"centralized": history.get("centralized", [])})


@metrics_bp.route("/federated", methods=["GET"])
def federated_metrics():
    history = get_training_history()
    return jsonify({"federated": history.get("federated", [])})


@metrics_bp.route("/summary", methods=["GET"])
def summary():
    """Return the final-epoch best metrics for both modes."""
    history = get_training_history()
    result = {}
    for mode in ["centralized", "federated"]:
        rows = history.get(mode, [])
        if rows:
            last = rows[-1]
            result[mode] = {
                "auc":      last.get("AUC", 0),
                "mrr":      last.get("MRR", 0),
                "ndcg5":    last.get("nDCG@5", 0),
                "ndcg10":   last.get("nDCG@10", 0),
                "loss":     last.get("train_loss", 0),
                "rounds":   last.get("round", last.get("epoch", len(rows))),
            }
    return jsonify(result)
