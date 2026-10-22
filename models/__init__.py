"""Models package for multimodal news representation and recommendation.

Exports the core encoder modules and the top-level recommender model.
"""

from .news_encoder import TextEncoder, VisualEncoder, MultimodalNewsEncoder, CrossModalAttentionFusion
from .user_encoder import AdditiveAttention, AttentiveUserEncoder, MeanUserEncoder
from .recommender import MultimodalNewsRecommender

__all__ = [
    "TextEncoder",
    "VisualEncoder",
    "CrossModalAttentionFusion",
    "MultimodalNewsEncoder",
    "AdditiveAttention",
    "AttentiveUserEncoder",
    "MeanUserEncoder",
    "MultimodalNewsRecommender",
]
