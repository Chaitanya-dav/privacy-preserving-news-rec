"""Backend model loader — loads checkpoints and embedding cache once at startup."""

from torch._dynamo.package import cache_dir
import json
import logging
from pathlib import Path
from typing import Optional

import torch

logger = logging.getLogger(__name__)

_model = None
_embedding_cache = None
_parser = None          # MINDParser with vocabulary
_news_df = None         # news_parsed.tsv as DataFrame
_training_history = {}  # centralized + federated metric logs

BASE_DIR = Path(__file__).resolve().parents[1]   # project root
CHECKPOINT_DIR = BASE_DIR / "results" / "checkpoints"
PROCESSED_DIR  = BASE_DIR / "data" / "processed"
EMBEDDINGS_DIR = BASE_DIR / "data" / "embeddings"


def _load_model(checkpoint_path: Path, device: torch.device):
    """Rebuild and load the recommender from a checkpoint .pt file."""
    import sys
    sys.path.insert(0, str(BASE_DIR))

    from models.recommender import MultimodalNewsRecommender

    model = MultimodalNewsRecommender(
        text_input_dim=768,
        image_input_dim=512,
        fused_dim=256,
        fusion_type="concat_projection",
        user_pooling="mean",
        scorer_type="dot",
    )
    state = torch.load(checkpoint_path, map_location=device)
    # Checkpoints may store raw state_dict or {"model_state_dict": ...}
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    logger.info("Loaded model from %s", checkpoint_path)
    return model


def _load_embedding_cache(device: torch.device):
    from preprocessing.embedding_cache import MultimodalEmbeddingCache
    cache_path = EMBEDDINGS_DIR / "multimodal_embeddings.pt"
    cache = MultimodalEmbeddingCache(cache_dir=str(cache_path), device=device)
    logger.info("Loaded embedding cache from %s", cache_path)
    return cache


def _load_parser():
    from preprocessing.mind_parser import MINDParser
    parser = MINDParser(max_history_len=50)
    parser.load_processed(PROCESSED_DIR)
    logger.info("Loaded vocabulary: %d news, %d users", len(parser.news2idx), len(parser.user2idx))
    return parser


def _load_news_df():
    import pandas as pd
    tsv = PROCESSED_DIR / "news_parsed.tsv"
    df = pd.read_csv(tsv, sep="\t", dtype=str).fillna("")
    df["news_idx"] = df["news_idx"].astype(int)
    return df


def _load_training_history():
    result = {}
    for name, fname in [
        ("centralized", "centralized_training_history.json"),
        ("federated",   "federated_training_history.json"),
    ]:
        path = CHECKPOINT_DIR / fname
        if path.exists():
            with open(path) as f:
                result[name] = json.load(f)
    return result


def get_model():
    global _model
    if _model is None:
        _initialise()
    return _model


def get_embedding_cache():
    global _embedding_cache
    if _embedding_cache is None:
        _initialise()
    return _embedding_cache


def get_parser():
    global _parser
    if _parser is None:
        _initialise()
    return _parser


def get_news_df():
    global _news_df
    if _news_df is None:
        _initialise()
    return _news_df


def get_training_history():
    global _training_history
    if not _training_history:
        _training_history = _load_training_history()
    return _training_history


def _initialise():
    global _model, _embedding_cache, _parser, _news_df, _training_history
    device = torch.device("cpu")

    # Prefer best_centralized > last_centralized checkpoint
    for ckpt in ["best_centralized_model.pt", "last_centralized_model.pt"]:
        cp = CHECKPOINT_DIR / ckpt
        if cp.exists():
            _model = _load_model(cp, device)
            break

    _embedding_cache = _load_embedding_cache(device)
    _parser           = _load_parser()
    _news_df          = _load_news_df()
    _training_history = _load_training_history()
