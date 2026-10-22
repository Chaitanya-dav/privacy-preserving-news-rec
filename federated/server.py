"""Federated Server & FedAvg Aggregation (Phase 11 & Phase 12).

Orchestrates multi-round federated learning with FedAvg aggregation,
tracking client convergence, communication rounds, and ranking evaluation.
"""

import json
import logging
import math
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from models.recommender import MultimodalNewsRecommender
from preprocessing.embedding_cache import MultimodalEmbeddingCache
from evaluation.metrics import evaluate_impressions
from .client import FlowerNewsClient, get_model_parameters, set_model_parameters

logger = logging.getLogger(__name__)


def fedavg_aggregate(results: List[Tuple[List[np.ndarray], int]]) -> List[np.ndarray]:
    """Perform FedAvg weighted aggregation over client parameters.

    Args:
        results: List of (parameters_list, num_samples) from participating clients.

    Returns:
        List of aggregated NumPy arrays with the same layer shapes.
    """
    if not results:
        raise ValueError("Cannot aggregate empty client results.")

    total_samples = sum(num_samples for _, num_samples in results)
    if total_samples == 0:
        # Fallback to simple unweighted average if samples are zero
        weights = [1.0 / len(results)] * len(results)
    else:
        weights = [num_samples / total_samples for _, num_samples in results]

    # Initialize accumulator with zeros matching the first client's parameter shapes
    num_layers = len(results[0][0])
    aggregated = [np.zeros_like(results[0][0][layer]) for layer in range(num_layers)]

    for (client_params, _), weight in zip(results, weights):
        for layer in range(num_layers):
            aggregated[layer] += client_params[layer] * weight

    return aggregated


def compute_param_delta_norm(w_old: List[np.ndarray], w_new: List[np.ndarray]) -> float:
    """Compute Frobenius/L2 norm of weight difference across rounds."""
    diff_norm_sq = 0.0
    for o, n in zip(w_old, w_new):
        diff_norm_sq += float(np.sum((n - o) ** 2))
    return float(math.sqrt(diff_norm_sq))


class FederatedServer:
    """Federated Learning coordinator running FedAvg across edge clients (Phase 12)."""

    def __init__(
        self,
        global_model: MultimodalNewsRecommender,
        clients: List[FlowerNewsClient],
        dev_sessions: List[Dict[str, Any]],
        embedding_cache: MultimodalEmbeddingCache,
        device: torch.device,
        checkpoint_dir: str = "results/checkpoints",
        fraction_fit: float = 1.0,
    ):
        self.global_model = global_model
        self.clients = clients
        self.dev_sessions = dev_sessions
        self.embedding_cache = embedding_cache
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.fraction_fit = fraction_fit

        self.history: List[Dict[str, Any]] = []

    def evaluate_global_model(self) -> Dict[str, float]:
        """Benchmark global model on complete evaluation sessions."""
        if not self.dev_sessions:
            return {"AUC": 0.5, "MRR": 0.0, "nDCG@5": 0.0, "nDCG@10": 0.0}

        self.global_model.eval()
        all_scores: List[List[float]] = []
        all_labels: List[List[float]] = []

        with torch.no_grad():
            for session in self.dev_sessions:
                hist_list = session["history_indices"]
                cand_list = session["candidate_indices"]
                lbl_list = session["labels"]

                if not cand_list:
                    continue

                hist_t = torch.tensor([hist_list], dtype=torch.long, device=self.device)
                cand_t = torch.tensor([cand_list], dtype=torch.long, device=self.device)

                scores_t = self.global_model.session_forward(cand_t, hist_t, self.embedding_cache)
                scores = scores_t.squeeze(0).cpu().tolist()

                all_scores.append(scores)
                all_labels.append(lbl_list)

        return evaluate_impressions(all_scores, all_labels)

    def fit_rounds(
        self,
        num_rounds: int = 5,
        local_epochs: int = 2,
        lr: float = 1e-3,
        privacy_pipeline: Optional[Callable[[List[Tuple[List[np.ndarray], int]]], List[Tuple[List[np.ndarray], int]]]] = None,
    ) -> List[Dict[str, Any]]:
        """Execute multi-round federated training and log convergence metrics.

        Args:
            num_rounds: Total communication rounds.
            local_epochs: Local epochs executed by each edge client.
            lr: Edge client learning rate.
            privacy_pipeline: Optional hook for Phase 13/14 (Secure Aggregation / DP).

        Returns:
            List of per-round metric dicts.
        """
        best_auc = -1.0
        print(f"\n[Federated Learning] Starting {num_rounds} rounds across {len(self.clients)} clients...")

        for r in range(1, num_rounds + 1):
            t0 = time.time()
            current_weights = get_model_parameters(self.global_model)

            # 1. Select participating clients
            num_selected = max(1, int(round(len(self.clients) * self.fraction_fit)))
            selected_clients = self.clients[:num_selected]

            # 2. Local client training
            fit_results: List[Tuple[List[np.ndarray], int]] = []
            client_losses = []

            for client in selected_clients:
                new_weights, num_samples, metrics = client.fit(
                    current_weights,
                    config={"local_epochs": local_epochs, "lr": lr},
                )
                if num_samples > 0:
                    fit_results.append((new_weights, num_samples))
                    client_losses.append(metrics.get("train_loss", 0.0))

            if not fit_results:
                logger.warning("No clients returned updates in round %d.", r)
                continue

            # 3. Privacy Transform (Phases 13 & 14 hook: SecAgg / DP)
            if privacy_pipeline is not None:
                fit_results = privacy_pipeline(fit_results)

            # 4. FedAvg Aggregation
            aggregated_weights = fedavg_aggregate(fit_results)
            delta_norm = compute_param_delta_norm(current_weights, aggregated_weights)

            # 5. Update global model
            set_model_parameters(self.global_model, aggregated_weights)
            round_time = time.time() - t0

            # 6. Global evaluation
            eval_metrics = self.evaluate_global_model()
            avg_train_loss = float(np.mean(client_losses)) if client_losses else 0.0

            round_summary = {
                "round": r,
                "train_loss": round(avg_train_loss, 4),
                "param_delta_norm": round(delta_norm, 5),
                "round_time_s": round(round_time, 2),
                **{k: round(v, 4) for k, v in eval_metrics.items()},
            }
            self.history.append(round_summary)

            print(
                f"  Round {r:02d}/{num_rounds:02d} | "
                f"Loss: {avg_train_loss:.4f} | "
                f"Delta Norm: {delta_norm:.4f} | "
                f"AUC: {eval_metrics.get('AUC', 0):.4f} | "
                f"MRR: {eval_metrics.get('MRR', 0):.4f} | "
                f"nDCG@5: {eval_metrics.get('nDCG@5', 0):.4f} | "
                f"Time: {round_time:.1f}s"
            )

            # Best model checkpointing
            if eval_metrics.get("AUC", 0) > best_auc:
                best_auc = eval_metrics["AUC"]
                torch.save(
                    {
                        "round": r,
                        "model_state_dict": self.global_model.state_dict(),
                        "metrics": eval_metrics,
                    },
                    self.checkpoint_dir / "best_federated_model.pt",
                )

        # Save last checkpoint and training history
        torch.save(
            {
                "round": num_rounds,
                "model_state_dict": self.global_model.state_dict(),
                "history": self.history,
            },
            self.checkpoint_dir / "last_federated_model.pt",
        )

        with open(self.checkpoint_dir / "federated_training_history.json", "w") as f:
            json.dump(self.history, f, indent=2)

        return self.history
