"""Flower Edge Client Implementation (Phase 11).

Implements a Flower NumPyClient that trains the MultimodalNewsRecommender
locally on an edge partition's private interaction history.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from models.recommender import MultimodalNewsRecommender
from preprocessing.dataset import MINDRecommendationDataset, collate_recommendation_batch
from preprocessing.embedding_cache import MultimodalEmbeddingCache
from evaluation.metrics import evaluate_impressions

# Safe Flower import with graceful fallback
try:
    import flwr as fl
    NumPyClientBase = fl.client.NumPyClient
except ImportError:
    fl = None
    class NumPyClientBase:
        """Fallback base when Flower is running in offline standalone mode."""
        pass

logger = logging.getLogger(__name__)


def get_model_parameters(model: nn.Module) -> List[np.ndarray]:
    """Extract model trainable parameters as a list of NumPy arrays."""
    return [val.detach().cpu().numpy() for _, val in model.state_dict().items()]


def set_model_parameters(model: nn.Module, parameters: List[np.ndarray]) -> None:
    """Set model parameters from a list of NumPy arrays."""
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = {k: torch.tensor(v, dtype=model.state_dict()[k].dtype) for k, v in params_dict}
    model.load_state_dict(state_dict, strict=True)


class FlowerNewsClient(NumPyClientBase):
    """Flower edge client for Federated Multimodal News Recommendation.

    Each client simulates an edge device (e.g. mobile news app) holding private
    reading histories. Training occurs locally with only model weight deltas
    shared with the central server.
    """

    def __init__(
        self,
        client_id: int,
        model: MultimodalNewsRecommender,
        train_samples: List[Dict[str, Any]],
        dev_sessions: Optional[List[Dict[str, Any]]] = None,
        embedding_cache: Optional[MultimodalEmbeddingCache] = None,
        device: Optional[torch.device] = None,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        local_epochs: int = 2,
    ):
        self.client_id = client_id
        self.model = model
        self.train_samples = train_samples
        self.dev_sessions = dev_sessions or []
        self.embedding_cache = embedding_cache
        self.device = device or torch.device("cpu")
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.local_epochs = local_epochs

        self.model.to(self.device)

        # Build local DataLoader
        if len(self.train_samples) > 0:
            dataset = MINDRecommendationDataset(self.train_samples)
            self.train_loader = DataLoader(
                dataset,
                batch_size=min(self.batch_size, len(dataset)),
                shuffle=True,
                collate_fn=collate_recommendation_batch,
            )
        else:
            self.train_loader = None

    def get_parameters(self, config: Optional[Dict[str, Any]] = None) -> List[np.ndarray]:
        """Return the current local model parameters."""
        return get_model_parameters(self.model)

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """Update local model with parameters received from the server."""
        set_model_parameters(self.model, parameters)

    def fit(
        self,
        parameters: List[np.ndarray],
        config: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[np.ndarray], int, Dict[str, Any]]:
        """Train model locally for local_epochs on edge partition."""
        config = config or {}
        local_epochs = config.get("local_epochs", self.local_epochs)
        lr = config.get("lr", self.learning_rate)

        # 1. Update local model with global parameters
        self.set_parameters(parameters)

        if self.train_loader is None or len(self.train_samples) == 0:
            return self.get_parameters(), 0, {"train_loss": 0.0}

        self.model.train()
        optimizer = AdamW(self.model.parameters(), lr=lr, weight_decay=self.weight_decay)
        criterion = nn.BCEWithLogitsLoss()

        total_loss = 0.0
        num_batches = 0

        for epoch in range(local_epochs):
            for batch in self.train_loader:
                candidate_idx = batch["candidate_idx"].to(self.device)
                history_indices = batch["history_indices"].to(self.device)
                labels = batch["label"].to(self.device)

                optimizer.zero_grad()
                logits = self.model.pointwise_forward(
                    candidate_idx,
                    history_indices,
                    self.embedding_cache,
                )
                loss = criterion(logits, labels)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

                total_loss += float(loss.item())
                num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)

        metrics = {
            "client_id": self.client_id,
            "train_loss": float(avg_loss),
            "num_samples": len(self.train_samples),
        }

        return self.get_parameters(), len(self.train_samples), metrics

    def evaluate(
        self,
        parameters: List[np.ndarray],
        config: Optional[Dict[str, Any]] = None,
    ) -> Tuple[float, int, Dict[str, Any]]:
        """Evaluate local model on dev sessions."""
        self.set_parameters(parameters)

        if not self.dev_sessions or self.embedding_cache is None:
            return 0.0, 0, {"AUC": 0.5, "MRR": 0.0, "nDCG@5": 0.0, "nDCG@10": 0.0}

        self.model.eval()
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

                scores_t = self.model.session_forward(cand_t, hist_t, self.embedding_cache)
                scores = scores_t.squeeze(0).cpu().tolist()

                all_scores.append(scores)
                all_labels.append(lbl_list)

        metrics = evaluate_impressions(all_scores, all_labels)
        # Approximate eval loss from inverted AUC
        eval_loss = float(1.0 - metrics.get("AUC", 0.5))

        return eval_loss, len(self.dev_sessions), metrics
