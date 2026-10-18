"""PyTorch Datasets and DataLoaders for Multimodal News Recommendation.

Provides optimized Dataset classes and batch collation functions for:
1. MINDNewsDataset: Text and visual features for all news articles
2. MINDRecommendationDataset: Pointwise/pairwise training samples
3. MINDTestImpressionDataset: Session-based impression evaluation
"""

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd

from .multimodal_extractors import ImageFeatureExtractor, TextFeatureExtractor

logger = logging.getLogger(__name__)


class MINDNewsDataset(Dataset):
    """Encapsulates multimodal features (text tokens & visual tensors) for all news articles."""

    def __init__(
        self,
        news_df: pd.DataFrame,
        text_extractor: TextFeatureExtractor,
        image_extractor: ImageFeatureExtractor,
        images_dir: Optional[Union[str, Path]] = None,
        max_title_len: int = 30,
        max_abstract_len: int = 50,
        pre_extract_images: bool = False,
    ):
        self.news_df = news_df.sort_values("news_idx").reset_index(drop=True)
        self.text_extractor = text_extractor
        self.image_extractor = image_extractor
        self.images_dir = Path(images_dir) if images_dir else None
        self.max_title_len = max_title_len
        self.max_abstract_len = max_abstract_len

        # Precompute tokenized titles and abstracts for blazing fast indexing
        logger.info("Pre-tokenizing %d news titles and abstracts...", len(self.news_df))
        titles = self.news_df["title"].tolist()
        abstracts = self.news_df["abstract"].tolist()

        self.title_ids, self.title_masks = self.text_extractor.tokenize_batch(titles, max_title_len)
        self.abstract_ids, self.abstract_masks = self.text_extractor.tokenize_batch(abstracts, max_abstract_len)

        self.category_indices = torch.tensor(self.news_df["category_idx"].values, dtype=torch.long)
        self.subcategory_indices = torch.tensor(self.news_df["subcategory_idx"].values, dtype=torch.long)
        self.news_indices = torch.tensor(self.news_df["news_idx"].values, dtype=torch.long)
        self.news_ids = self.news_df["news_id"].tolist()

        # Optional pre-extraction of image tensors
        self.image_tensors: Optional[torch.Tensor] = None
        if pre_extract_images:
            self._precompute_images()

    def _precompute_images(self) -> None:
        """Precompute and store all normalized image tensors in memory."""
        logger.info("Pre-loading %d news image tensors...", len(self.news_ids))
        img_list = []
        for nid in self.news_ids:
            img_path = (self.images_dir / f"{nid}.jpg") if self.images_dir else Path(f"{nid}.jpg")
            img_t = self.image_extractor.preprocess_image(img_path)
            img_list.append(img_t)
        self.image_tensors = torch.stack(img_list, dim=0)

    def __len__(self) -> int:
        return len(self.news_df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {
            "news_idx": self.news_indices[idx],
            "category_idx": self.category_indices[idx],
            "subcategory_idx": self.subcategory_indices[idx],
            "title_ids": self.title_ids[idx],
            "title_mask": self.title_masks[idx],
            "abstract_ids": self.abstract_ids[idx],
            "abstract_mask": self.abstract_masks[idx],
        }

        if self.image_tensors is not None:
            item["image_tensor"] = self.image_tensors[idx]
        else:
            nid = self.news_ids[idx]
            img_path = (self.images_dir / f"{nid}.jpg") if self.images_dir else Path(f"{nid}.jpg")
            item["image_tensor"] = self.image_extractor.preprocess_image(img_path)

        return item


class MINDRecommendationDataset(Dataset):
    """PyTorch Dataset for training recommendation models on user interaction instances."""

    def __init__(self, samples: List[Dict[str, Any]]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        return {
            "user_idx": torch.tensor(sample["user_idx"], dtype=torch.long),
            "history_indices": torch.tensor(sample["history_indices"], dtype=torch.long),
            "candidate_idx": torch.tensor(sample["candidate_idx"], dtype=torch.long),
            "label": torch.tensor(sample["label"], dtype=torch.float),
        }


def collate_recommendation_batch(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Collate individual sample dicts into stacked batch tensors."""
    return {
        "user_idx": torch.stack([b["user_idx"] for b in batch]),
        "history_indices": torch.stack([b["history_indices"] for b in batch]),
        "candidate_idx": torch.stack([b["candidate_idx"] for b in batch]),
        "label": torch.stack([b["label"] for b in batch]),
    }


class MINDTestImpressionDataset(Dataset):
    """PyTorch Dataset for session-based evaluation of ranking metrics (AUC, MRR, nDCG@K)."""

    def __init__(self, sessions: List[Dict[str, Any]]):
        self.sessions = sessions

    def __len__(self) -> int:
        return len(self.sessions)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        session = self.sessions[idx]
        return {
            "impression_id": session.get("impression_id", idx + 1),
            "user_idx": torch.tensor(session["user_idx"], dtype=torch.long),
            "history_indices": torch.tensor(session["history_indices"], dtype=torch.long),
            "candidate_indices": torch.tensor(session["candidate_indices"], dtype=torch.long),
            "labels": torch.tensor(session["labels"], dtype=torch.float),
        }


def create_recommendation_dataloaders(
    train_samples: List[Dict[str, Any]],
    dev_samples: Optional[List[Dict[str, Any]]] = None,
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 0,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """Build PyTorch DataLoaders with custom collation functions.

    Args:
        train_samples: Training interaction samples with negative sampling
        dev_samples: Optional dev samples (pointwise or session-based)
        batch_size: Batch size
        shuffle: Whether to shuffle training data
        num_workers: Multi-process data loading workers

    Returns:
        (train_loader, dev_loader)
    """
    train_dataset = MINDRecommendationDataset(train_samples)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_recommendation_batch,
        pin_memory=False,
    )

    dev_loader = None
    if dev_samples is not None:
        # Check if dev_samples is session-based (contains candidate_indices)
        if len(dev_samples) > 0 and "candidate_indices" in dev_samples[0]:
            dev_dataset = MINDTestImpressionDataset(dev_samples)
            # Evaluate session by session (batch_size=1 or pass-through)
            dev_loader = DataLoader(
                dev_dataset,
                batch_size=1,
                shuffle=False,
                num_workers=num_workers,
                collate_fn=lambda x: x[0],  # Return individual session dict
            )
        else:
            dev_dataset = MINDRecommendationDataset(dev_samples)
            dev_loader = DataLoader(
                dev_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                collate_fn=collate_recommendation_batch,
            )

    return train_loader, dev_loader
