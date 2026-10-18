"""Preprocessing package for MIND multimodal news recommendation.

Exports dataset downloaders, TSV parsers, multimodal feature extractors,
PyTorch datasets, and precomputed embedding caches.
"""

from .dataset_downloader import download_mind, generate_demo_dataset
from .multimodal_extractors import TextFeatureExtractor, ImageFeatureExtractor, FallbackTokenizer
from .mind_parser import MINDParser
from .dataset import (
    MINDNewsDataset,
    MINDRecommendationDataset,
    MINDTestImpressionDataset,
    collate_recommendation_batch,
    create_recommendation_dataloaders,
)
from .embedding_cache import MultimodalEmbeddingCache
from .pipeline import run_pipeline

__all__ = [
    "download_mind",
    "generate_demo_dataset",
    "TextFeatureExtractor",
    "ImageFeatureExtractor",
    "FallbackTokenizer",
    "MINDParser",
    "MINDNewsDataset",
    "MINDRecommendationDataset",
    "MINDTestImpressionDataset",
    "collate_recommendation_batch",
    "create_recommendation_dataloaders",
    "MultimodalEmbeddingCache",
    "run_pipeline",
]
