"""Centralized Trainer for Multimodal News Recommendation Baseline.

Runs supervised training over all users' data in one place (no federation),
serving as the upper-bound performance baseline for comparison with federated
and privacy-preserving variants in later phases.
"""

import json
import logging
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from models.recommender import MultimodalNewsRecommender
from preprocessing.embedding_cache import MultimodalEmbeddingCache
from evaluation.metrics import evaluate_impressions

logger = logging.getLogger(__name__)


class CentralizedTrainer:
    """Full-data centralized training baseline with periodic ranking evaluation.

    Features:
      - AdamW optimizer with weight decay.
      - Cosine annealing LR scheduler.
      - Gradient clipping (max_norm=1.0).
      - Binary Cross-Entropy training loss.
      - Per-epoch dev evaluation (AUC, MRR, nDCG@5, nDCG@10).
      - Best model checkpointing to results/checkpoints/.
    """

    def __init__(
        self,
        model: MultimodalNewsRecommender,
        embedding_cache: MultimodalEmbeddingCache,
        device: torch.device,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        grad_clip_norm: float = 1.0,
        checkpoint_dir: str = "results/checkpoints",
    ):
        self.model = model.to(device)
        self.embedding_cache = embedding_cache
        self.device = device
        self.grad_clip_norm = grad_clip_norm
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.optimizer = AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        self.criterion = nn.BCEWithLogitsLoss()
        self.history: List[Dict[str, Any]] = []
        self.best_auc = 0.0

    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Run one epoch of training.

        Returns:
            Dict with 'train_loss' and 'train_time_s'.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        t0 = time.time()

        for batch in train_loader:
            candidate_idx = batch["candidate_idx"].to(self.device)    # (B,)
            history_indices = batch["history_indices"].to(self.device) # (B, H)
            labels = batch["label"].to(self.device)                    # (B,)

            self.optimizer.zero_grad()

            logits = self.model.pointwise_forward(
                candidate_indices=candidate_idx,
                history_indices=history_indices,
                embedding_cache=self.embedding_cache,
            )

            loss = self.criterion(logits, labels)
            loss.backward()

            if self.grad_clip_norm > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)

            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        return {
            "train_loss": avg_loss,
            "train_time_s": time.time() - t0,
        }

    @torch.no_grad()
    def evaluate(self, dev_sessions: List[Dict[str, Any]]) -> Dict[str, float]:
        """Evaluate on dev impression sessions.

        Args:
            dev_sessions: List of session dicts with 'history_indices',
                          'candidate_indices', and 'labels'.

        Returns:
            Dict with AUC, MRR, nDCG@5, nDCG@10.
        """
        self.model.eval()
        all_scores: List[List[float]] = []
        all_labels: List[List[float]] = []

        for session in dev_sessions:
            history = torch.tensor(
                session["history_indices"], dtype=torch.long
            ).unsqueeze(0).to(self.device)           # (1, H)

            candidates = torch.tensor(
                session["candidate_indices"], dtype=torch.long
            ).unsqueeze(0).to(self.device)           # (1, C)

            scores = self.model.session_forward(
                candidate_indices=candidates,
                history_indices=history,
                embedding_cache=self.embedding_cache,
            ).squeeze(0).cpu().tolist()              # (C,)

            all_scores.append(scores)
            all_labels.append(session["labels"])

        return evaluate_impressions(all_scores, all_labels)

    def fit(
        self,
        train_loader: DataLoader,
        dev_sessions: List[Dict[str, Any]],
        num_epochs: int = 5,
        scheduler: Optional[CosineAnnealingLR] = None,
        eval_every: int = 1,
    ) -> List[Dict[str, Any]]:
        """Full training loop.

        Args:
            train_loader:  DataLoader for training batches.
            dev_sessions:  Dev evaluation sessions.
            num_epochs:    Total training epochs.
            scheduler:     Optional LR scheduler.
            eval_every:    Evaluate dev metrics every N epochs.

        Returns:
            Training history list of per-epoch metric dicts.
        """
        print(f"\n{'='*60}")
        print(f"  Centralized Training Baseline")
        print(f"  Epochs: {num_epochs}  |  Device: {self.device}")
        print(f"{'='*60}")

        for epoch in range(1, num_epochs + 1):
            train_metrics = self.train_epoch(train_loader)

            log = {"epoch": epoch, **train_metrics}

            if epoch % eval_every == 0:
                dev_metrics = self.evaluate(dev_sessions)
                log.update(dev_metrics)

                auc = dev_metrics.get("AUC", 0.0)
                if auc > self.best_auc:
                    self.best_auc = auc
                    self._save_checkpoint("best_centralized_model.pt")
                    improved = " << NEW BEST"
                else:
                    improved = ""

                print(
                    f"  Epoch {epoch:>3}/{num_epochs}  |  "
                    f"Loss: {train_metrics['train_loss']:.4f}  |  "
                    f"AUC: {dev_metrics['AUC']:.4f}  "
                    f"MRR: {dev_metrics['MRR']:.4f}  "
                    f"nDCG@5: {dev_metrics['nDCG@5']:.4f}  "
                    f"nDCG@10: {dev_metrics['nDCG@10']:.4f}"
                    f"{improved}"
                )
            else:
                print(
                    f"  Epoch {epoch:>3}/{num_epochs}  |  "
                    f"Loss: {train_metrics['train_loss']:.4f}  |  "
                    f"Time: {train_metrics['train_time_s']:.1f}s"
                )

            if scheduler is not None:
                scheduler.step()

            self.history.append(log)

        # Save final model
        self._save_checkpoint("last_centralized_model.pt")
        self._save_history()

        print(f"\n  Best Dev AUC: {self.best_auc:.4f}")
        print(f"  Checkpoints saved to: {self.checkpoint_dir}")
        return self.history

    def _save_checkpoint(self, filename: str) -> None:
        """Save model state dict."""
        path = self.checkpoint_dir / filename
        torch.save(self.model.state_dict(), path)
        logger.info("Saved checkpoint: %s", path)

    def _save_history(self) -> None:
        """Persist training history as JSON."""
        path = self.checkpoint_dir / "centralized_training_history.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2)
