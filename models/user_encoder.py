"""User Representation Models with Attentive Pooling over Click History.

Implements additive self-attention user modeling that pools over a user's
clicked news history, applying strict PAD masking so padding articles
contribute exactly zero attention weight to the final user vector.
"""

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class AdditiveAttention(nn.Module):
    """Word-level / news-level additive (Bahdanau-style) attention.

    Computes:
        e_i = v^T tanh(W h_i + b)
        α   = softmax(e, mask)
        out = Σ α_i h_i
    """

    def __init__(self, input_dim: int, attention_dim: int = 128):
        super().__init__()
        self.W = nn.Linear(input_dim, attention_dim, bias=True)
        self.v = nn.Linear(attention_dim, 1, bias=False)

    def forward(
        self,
        hidden: torch.Tensor,
        key_padding_mask: Optional[torch.BoolTensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden:            (B, L, D) sequence of news vectors.
            key_padding_mask:  (B, L) bool tensor, True where position is PAD.

        Returns:
            output:  (B, D) weighted context vector.
            weights: (B, L) attention weights.
        """
        # (B, L, attention_dim)
        score = self.v(torch.tanh(self.W(hidden))).squeeze(-1)  # (B, L)

        if key_padding_mask is not None:
            score = score.masked_fill(key_padding_mask, float("-inf"))

        weights = torch.softmax(score, dim=-1)               # (B, L)
        # Guard against all-masked rows (all -inf → NaN after softmax)
        weights = torch.nan_to_num(weights, nan=0.0)

        output = torch.bmm(weights.unsqueeze(1), hidden).squeeze(1)  # (B, D)
        return output, weights


class AttentiveUserEncoder(nn.Module):
    """Models a user as an attention-weighted combination of their clicked news.

    Architecture:
        - For each news in user history, encode into (B, H, D) via news_encoder.
        - Apply AdditiveAttention with PAD masking to obtain (B, D) user vector.
        - Optional feedforward projection after pooling.

    PAD Token: index 0 (``<PAD>``) → masked out in attention.
    """

    def __init__(
        self,
        news_dim: int = 256,
        attention_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.news_dim = news_dim
        self.attention = AdditiveAttention(news_dim, attention_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(news_dim)

    def forward(
        self,
        history_news_vecs: torch.Tensor,
        history_mask: Optional[torch.BoolTensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            history_news_vecs: (B, H, D) encoded clicked news vectors.
                               H = max_history_len; PAD positions should be zeros.
            history_mask:      (B, H) bool tensor — True means PAD position to ignore.
                               If None, derived from all-zero rows.

        Returns:
            user_vec: (B, D) user embedding.
        """
        if history_mask is None:
            # Auto-detect padding: rows where all embedding dims are zero
            history_mask = (history_news_vecs.abs().sum(dim=-1) == 0)  # (B, H)

        history_news_vecs = self.dropout(history_news_vecs)
        user_vec, _ = self.attention(history_news_vecs, key_padding_mask=history_mask)
        user_vec = self.norm(user_vec)
        return user_vec


class MeanUserEncoder(nn.Module):
    """Simple masked mean pooling baseline over user click history."""

    def __init__(self, news_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.news_dim = news_dim
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(news_dim)

    def forward(
        self,
        history_news_vecs: torch.Tensor,
        history_mask: Optional[torch.BoolTensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            history_news_vecs: (B, H, D)
            history_mask:      (B, H) bool, True = PAD

        Returns:
            user_vec: (B, D)
        """
        if history_mask is None:
            history_mask = (history_news_vecs.abs().sum(dim=-1) == 0)

        history_news_vecs = self.dropout(history_news_vecs)

        # Zero out PAD positions before mean
        valid_mask = (~history_mask).float().unsqueeze(-1)     # (B, H, 1)
        masked = history_news_vecs * valid_mask
        num_valid = valid_mask.sum(dim=1).clamp(min=1.0)       # (B, 1)
        user_vec = masked.sum(dim=1) / num_valid               # (B, D)
        user_vec = self.norm(user_vec)
        return user_vec
