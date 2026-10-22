"""Top-Level Multimodal News Recommender Model.

Integrates MultimodalNewsEncoder + UserEncoder + Click Predictor into
a unified nn.Module that supports both pointwise (single candidate) and
listwise (all session candidates) forward passes for training and inference.
"""

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .news_encoder import MultimodalNewsEncoder
from .user_encoder import AttentiveUserEncoder, MeanUserEncoder

logger = logging.getLogger(__name__)


class MultimodalNewsRecommender(nn.Module):
    """End-to-end multimodal news recommendation model.

    Forward modes:
      - pointwise_forward: (B,) candidate + (B, H) history → (B,) click probability.
      - session_forward:   (B, C) candidates + (B, H) history → (B, C) scores.

    Both modes work from precomputed embedding cache for maximum speed.
    """

    def __init__(
        self,
        # News encoder
        text_input_dim: int = 768,
        image_input_dim: int = 512,
        fused_dim: int = 256,
        fusion_type: str = "concat_projection",
        # User encoder
        user_pooling: str = "attention",      # "attention" or "mean"
        attention_dim: int = 128,
        # Scoring
        scorer_type: str = "dot",             # "dot" or "mlp"
        scorer_hidden_dims: Tuple[int, ...] = (256, 128),
        # Regularization
        news_dropout: float = 0.2,
        user_dropout: float = 0.1,
    ):
        super().__init__()
        self.fused_dim = fused_dim

        # News encoder (shared for history and candidates)
        self.news_encoder = MultimodalNewsEncoder(
            text_input_dim=text_input_dim,
            image_input_dim=image_input_dim,
            fused_dim=fused_dim,
            fusion_type=fusion_type,
            dropout=news_dropout,
        )

        # User encoder
        if user_pooling == "attention":
            self.user_encoder = AttentiveUserEncoder(
                news_dim=fused_dim,
                attention_dim=attention_dim,
                dropout=user_dropout,
            )
        elif user_pooling == "mean":
            self.user_encoder = MeanUserEncoder(
                news_dim=fused_dim,
                dropout=user_dropout,
            )
        else:
            raise ValueError(f"Unknown user_pooling '{user_pooling}'. Use 'attention' or 'mean'.")

        # Scorer
        self.scorer_type = scorer_type
        if scorer_type == "mlp":
            layers = []
            in_dim = fused_dim * 2
            for h in scorer_hidden_dims:
                layers += [nn.Linear(in_dim, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(news_dropout)]
                in_dim = h
            layers.append(nn.Linear(in_dim, 1))
            self.mlp_scorer = nn.Sequential(*layers)

    def _encode_news_from_cache(
        self,
        news_indices: torch.LongTensor,
        embedding_cache,
    ) -> torch.Tensor:
        """Lookup embeddings and encode news. news_indices shape: (*)."""
        return self.news_encoder.encode_from_cache(news_indices, embedding_cache)

    def _score(self, user_vec: torch.Tensor, news_vec: torch.Tensor) -> torch.Tensor:
        """Compute click score(s).

        Args:
            user_vec: (B, D)
            news_vec: (B, D) or (B, C, D)
        Returns:
            (B,) or (B, C) raw logits.
        """
        if self.scorer_type == "dot":
            if news_vec.dim() == 3:
                # (B, C, D) × (B, D, 1) → (B, C)
                return torch.bmm(news_vec, user_vec.unsqueeze(-1)).squeeze(-1)
            else:
                return (user_vec * news_vec).sum(dim=-1)

        else:  # mlp
            if news_vec.dim() == 3:
                B, C, D = news_vec.shape
                u_exp = user_vec.unsqueeze(1).expand(B, C, D)  # (B, C, D)
                cat = torch.cat([u_exp, news_vec], dim=-1).view(B * C, -1)
                out = self.mlp_scorer(cat).view(B, C)
                return out
            else:
                cat = torch.cat([user_vec, news_vec], dim=-1)
                return self.mlp_scorer(cat).squeeze(-1)

    def encode_history(
        self,
        history_indices: torch.LongTensor,
        embedding_cache,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode clicked history articles and build user vector.

        Args:
            history_indices: (B, H) article indices, 0 = PAD.
            embedding_cache: MultimodalEmbeddingCache.

        Returns:
            user_vec:           (B, D)
            history_news_vecs:  (B, H, D)
        """
        B, H = history_indices.shape
        flat_ids = history_indices.view(-1)  # (B*H,)
        flat_vecs = self._encode_news_from_cache(flat_ids, embedding_cache)  # (B*H, D)
        history_news_vecs = flat_vecs.view(B, H, self.fused_dim)             # (B, H, D)

        # PAD mask: position where news_idx == 0
        history_mask = (history_indices == 0)  # (B, H)

        user_vec = self.user_encoder(history_news_vecs, history_mask)  # (B, D)
        return user_vec, history_news_vecs

    def pointwise_forward(
        self,
        candidate_indices: torch.LongTensor,
        history_indices: torch.LongTensor,
        embedding_cache,
    ) -> torch.Tensor:
        """Training forward pass: one candidate per sample.

        Args:
            candidate_indices: (B,) candidate news indices.
            history_indices:   (B, H) history news indices (0 = PAD).
            embedding_cache:   MultimodalEmbeddingCache.

        Returns:
            logits: (B,) raw click scores.
        """
        user_vec, _ = self.encode_history(history_indices, embedding_cache)
        candidate_vecs = self._encode_news_from_cache(candidate_indices, embedding_cache)
        return self._score(user_vec, candidate_vecs)

    def session_forward(
        self,
        candidate_indices: torch.LongTensor,
        history_indices: torch.LongTensor,
        embedding_cache,
    ) -> torch.Tensor:
        """Evaluation forward pass: all session candidates at once.

        Args:
            candidate_indices: (B, C) all candidate news indices.
            history_indices:   (B, H) history news indices (0 = PAD).
            embedding_cache:   MultimodalEmbeddingCache.

        Returns:
            scores: (B, C) click logits.
        """
        user_vec, _ = self.encode_history(history_indices, embedding_cache)

        B, C = candidate_indices.shape
        flat_cand = candidate_indices.view(-1)
        flat_vecs = self._encode_news_from_cache(flat_cand, embedding_cache)
        candidate_vecs = flat_vecs.view(B, C, self.fused_dim)

        return self._score(user_vec, candidate_vecs)

    def forward(
        self,
        candidate_indices: torch.LongTensor,
        history_indices: torch.LongTensor,
        embedding_cache,
    ) -> torch.Tensor:
        """Unified forward: auto-detects pointwise vs session based on candidate dims."""
        if candidate_indices.dim() == 1:
            return self.pointwise_forward(candidate_indices, history_indices, embedding_cache)
        else:
            return self.session_forward(candidate_indices, history_indices, embedding_cache)
