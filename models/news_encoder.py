"""Multimodal News Encoders: Text, Visual, and Fused Representations.

Provides modular encoder components that project DistilBERT text embeddings
and ResNet-18 visual features into a shared 256-dimensional latent space,
then fuse them via concatenation-projection or cross-modal attention.

Supports two operational modes:
  1. Raw mode: accepts token IDs + image tensors directly.
  2. Cache mode: accepts news indices + MultimodalEmbeddingCache for fast lookup.
"""

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class TextEncoder(nn.Module):
    """Projects pre-extracted DistilBERT CLS embeddings (768D) into a fused_dim space."""

    def __init__(
        self,
        input_dim: int = 768,
        output_dim: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, text_embedding: torch.Tensor) -> torch.Tensor:
        """
        Args:
            text_embedding: (*, 768) pre-extracted text embeddings.
        Returns:
            (*, output_dim) projected text representation.
        """
        return self.projection(text_embedding)


class VisualEncoder(nn.Module):
    """Projects pre-extracted ResNet-18 visual embeddings (512D) into a fused_dim space."""

    def __init__(
        self,
        input_dim: int = 512,
        output_dim: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, image_embedding: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image_embedding: (*, 512) pre-extracted visual embeddings.
        Returns:
            (*, output_dim) projected visual representation.
        """
        return self.projection(image_embedding)


class CrossModalAttentionFusion(nn.Module):
    """Cross-modal attention to adaptively fuse text and visual representations.

    Uses text as query to attend over visual, then combines residually.
    """

    def __init__(self, dim: int = 256, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
        )
        self.gate = nn.Sequential(nn.Linear(dim * 2, 1), nn.Sigmoid())
        self.proj_out = nn.Linear(dim * 2, dim)

    def forward(
        self, text_feat: torch.Tensor, visual_feat: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            text_feat:   (B, D) or (B, 1, D)
            visual_feat: (B, D) or (B, 1, D)
        Returns:
            (B, D) fused representation.
        """
        # Expand to sequence dimension for MHA
        squeeze = text_feat.dim() == 2
        if squeeze:
            text_feat = text_feat.unsqueeze(1)    # (B, 1, D)
            visual_feat = visual_feat.unsqueeze(1) # (B, 1, D)

        # Text attends to visual
        attended, _ = self.attn(query=text_feat, key=visual_feat, value=visual_feat)
        text_feat = self.norm1(text_feat + attended)
        text_feat = self.norm2(text_feat + self.ffn(text_feat))

        if squeeze:
            text_feat = text_feat.squeeze(1)
            visual_feat = visual_feat.squeeze(1)

        # Adaptive gating fusion
        gate_w = self.gate(torch.cat([text_feat, visual_feat], dim=-1))  # (B, 1)
        fused = gate_w * text_feat + (1 - gate_w) * visual_feat
        return fused


class MultimodalNewsEncoder(nn.Module):
    """Encodes news articles from pre-extracted embeddings into unified 256-dim vectors.

    Supports two fusion strategies:
      - 'concat_projection': Concat [text_proj; visual_proj] -> Linear(512, fused_dim)
      - 'attention_fusion': Cross-modal attention between text and visual projections.
    """

    def __init__(
        self,
        text_input_dim: int = 768,
        image_input_dim: int = 512,
        fused_dim: int = 256,
        fusion_type: str = "concat_projection",
        dropout: float = 0.2,
    ):
        super().__init__()
        self.fused_dim = fused_dim
        self.fusion_type = fusion_type

        # Individual modality projectors
        self.text_encoder = TextEncoder(text_input_dim, fused_dim, dropout)
        self.visual_encoder = VisualEncoder(image_input_dim, fused_dim, dropout)

        # Fusion layer
        if fusion_type == "concat_projection":
            self.fusion = nn.Sequential(
                nn.Linear(fused_dim * 2, fused_dim),
                nn.LayerNorm(fused_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
        elif fusion_type == "attention_fusion":
            self.fusion = CrossModalAttentionFusion(dim=fused_dim, dropout=dropout)
        else:
            raise ValueError(f"Unknown fusion_type '{fusion_type}'. Use 'concat_projection' or 'attention_fusion'.")

    def forward(
        self,
        text_embeddings: torch.Tensor,
        image_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Encode articles from pre-extracted embedding tensors.

        Args:
            text_embeddings:  (*, 768) DistilBERT CLS embeddings.
            image_embeddings: (*, 512) ResNet-18 pooled embeddings.

        Returns:
            (*, fused_dim) fused multimodal news representations.
        """
        text_feat = self.text_encoder(text_embeddings)
        visual_feat = self.visual_encoder(image_embeddings)

        if self.fusion_type == "concat_projection":
            fused = self.fusion(torch.cat([text_feat, visual_feat], dim=-1))
        else:
            # attention_fusion
            orig_shape = text_feat.shape
            # Flatten all batch dims except last
            if text_feat.dim() > 2:
                B_flat = text_feat.view(-1, text_feat.shape[-1])
                V_flat = visual_feat.view(-1, visual_feat.shape[-1])
                fused = self.fusion(B_flat, V_flat)
                fused = fused.view(*orig_shape)
            else:
                fused = self.fusion(text_feat, visual_feat)

        return fused

    def encode_from_cache(
        self,
        news_indices: torch.LongTensor,
        embedding_cache,
    ) -> torch.Tensor:
        """Fast embedding-cache forward pass for training and federated simulation.

        Args:
            news_indices:     (*) LongTensor of news corpus indices (0=PAD).
            embedding_cache:  MultimodalEmbeddingCache instance.

        Returns:
            (*, fused_dim) news representations.
        """
        text_emb = embedding_cache.get_text_embedding(news_indices).to(
            next(self.parameters()).device
        )
        image_emb = embedding_cache.get_image_embedding(news_indices).to(
            next(self.parameters()).device
        )
        return self.forward(text_emb, image_emb)
