"""Precomputed Multimodal Embedding Cache.

Precomputes and caches 768-dim DistilBERT text embeddings and 512-dim
ResNet-18 visual embeddings for the news corpus. Accelerates centralized
training and federated edge simulations by bypassing redundant forward passes.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import pandas as pd

from .multimodal_extractors import ImageFeatureExtractor, TextFeatureExtractor

logger = logging.getLogger(__name__)


class MultimodalEmbeddingCache:
    """Manages precomputed text and visual representations for all news articles."""

    def __init__(
        self,
        text_embeddings: Optional[torch.Tensor] = None,
        image_embeddings: Optional[torch.Tensor] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.text_embeddings = text_embeddings
        self.image_embeddings = image_embeddings
        self.metadata = metadata or {}

    @classmethod
    def precompute(
        cls,
        news_df: pd.DataFrame,
        text_extractor: TextFeatureExtractor,
        image_extractor: ImageFeatureExtractor,
        images_dir: Optional[Union[str, Path]] = None,
        text_model: Optional[nn.Module] = None,
        batch_size: int = 32,
        device: Optional[torch.device] = None,
    ) -> "MultimodalEmbeddingCache":
        """Precompute text and visual embeddings for all articles in the corpus.

        Args:
            news_df: Parsed news DataFrame with 'news_idx', 'title', 'news_id'
            text_extractor: DistilBERT / offline tokenizer
            image_extractor: ResNet18 visual extractor
            images_dir: Directory containing news JPEG images
            text_model: Optional loaded transformer model (if None, uses pooled projection)
            batch_size: Processing batch size
            device: Compute device

        Returns:
            Instance of MultimodalEmbeddingCache with populated tensor tables.
        """
        device = device or torch.device("cpu")
        max_idx = int(news_df["news_idx"].max())
        num_items = max_idx + 1  # 0 is PAD

        text_dim = 768
        image_dim = 512

        # Initialize tables with zeros (index 0 is PAD)
        text_table = torch.zeros(num_items, text_dim, dtype=torch.float32)
        image_table = torch.zeros(num_items, image_dim, dtype=torch.float32)

        images_dir_path = Path(images_dir) if images_dir else None

        print(f"Precomputing multimodal embeddings for {len(news_df)} news articles...")

        # 1. Process Text Embeddings
        titles = news_df["title"].tolist()
        indices = news_df["news_idx"].tolist()
        news_ids = news_df["news_id"].tolist()

        for start in range(0, len(titles), batch_size):
            end = min(start + batch_size, len(titles))
            batch_titles = titles[start:end]
            batch_indices = indices[start:end]

            input_ids, attn_mask = text_extractor.tokenize_batch(batch_titles, max_len=text_extractor.max_title_len)
            input_ids = input_ids.to(device)
            attn_mask = attn_mask.to(device)

            if text_model is not None:
                with torch.no_grad():
                    outputs = text_model(input_ids=input_ids, attention_mask=attn_mask)
                    # Use [CLS] representation (first token)
                    if hasattr(outputs, "last_hidden_state"):
                        batch_text_emb = outputs.last_hidden_state[:, 0, :]
                    else:
                        batch_text_emb = outputs[0][:, 0, :]
            else:
                # Deterministic semantic projection from token representations
                with torch.no_grad():
                    # Mean over token IDs projected to 768-dim
                    token_floats = input_ids.float().unsqueeze(-1)
                    # Fixed deterministic sinusoidal position/token projection
                    freqs = torch.arange(text_dim, device=device).float() * 0.01
                    proj = torch.sin(token_floats * freqs.unsqueeze(0).unsqueeze(0))
                    # Masked average pooling
                    mask_expanded = attn_mask.unsqueeze(-1).float()
                    batch_text_emb = (proj * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)

            for b_idx, target_idx in enumerate(batch_indices):
                text_table[target_idx] = batch_text_emb[b_idx].cpu()

        # 2. Process Image Embeddings
        print("  Extracting visual features via ResNet-18...")
        for idx_pos in range(0, len(news_ids), batch_size):
            end_pos = min(idx_pos + batch_size, len(news_ids))
            batch_nids = news_ids[idx_pos:end_pos]
            batch_indices = indices[idx_pos:end_pos]

            img_tensors = []
            for nid in batch_nids:
                img_path = (images_dir_path / f"{nid}.jpg") if images_dir_path else Path(f"{nid}.jpg")
                img_t = image_extractor.preprocess_image(img_path)
                img_tensors.append(img_t)

            batch_img_tensor = torch.stack(img_tensors, dim=0)
            with torch.no_grad():
                batch_img_emb = image_extractor.extract_embedding(batch_img_tensor)

            for b_idx, target_idx in enumerate(batch_indices):
                image_table[target_idx] = batch_img_emb[b_idx].cpu()

        print(f"Precomputed {num_items - 1} article embeddings (Text: {text_dim}D, Image: {image_dim}D).")

        metadata = {
            "num_articles": len(news_df),
            "max_news_idx": max_idx,
            "text_dim": text_dim,
            "image_dim": image_dim,
        }

        return cls(text_embeddings=text_table, image_embeddings=image_table, metadata=metadata)

    def save(self, file_path: Union[str, Path]) -> None:
        """Save embedding tables and metadata to disk."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "text_embeddings": self.text_embeddings,
                "image_embeddings": self.image_embeddings,
                "metadata": self.metadata,
            },
            path,
        )
        print(f"Saved multimodal embedding cache to {path}")

    @classmethod
    def load(cls, file_path: Union[str, Path]) -> "MultimodalEmbeddingCache":
        """Load precomputed embedding cache from disk."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Embedding cache not found at {path}")

        data = torch.load(path, map_location="cpu", weights_only=False)
        logger.info(
            "Loaded embedding cache from %s (%d articles).",
            path,
            data["metadata"].get("num_articles", 0),
        )
        return cls(
            text_embeddings=data["text_embeddings"],
            image_embeddings=data["image_embeddings"],
            metadata=data["metadata"],
        )

    def get_text_embedding(self, news_idx: torch.Tensor) -> torch.Tensor:
        """Retrieve text embeddings for given news index tensor."""
        if self.text_embeddings is None:
            raise ValueError("Text embeddings are not loaded.")
        return self.text_embeddings[news_idx]

    def get_image_embedding(self, news_idx: torch.Tensor) -> torch.Tensor:
        """Retrieve visual embeddings for given news index tensor."""
        if self.image_embeddings is None:
            raise ValueError("Image embeddings are not loaded.")
        return self.image_embeddings[news_idx]

    def get_multimodal_pair(self, news_idx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Retrieve both (text_emb, image_emb) for given news index tensor."""
        return self.get_text_embedding(news_idx), self.get_image_embedding(news_idx)
