"""End-to-End Multimodal News Data Preprocessing Pipeline.

Orchestrates raw data ingestion/generation, parsing of MIND TSV files,
vocabulary indexing, multimodal feature extraction, and precomputed caching.
Can be executed directly via:
    python -m preprocessing.pipeline
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict
import yaml
import torch

from .dataset_downloader import download_mind, generate_demo_dataset
from .mind_parser import MINDParser
from .multimodal_extractors import ImageFeatureExtractor, TextFeatureExtractor
from .dataset import MINDNewsDataset, create_recommendation_dataloaders
from .embedding_cache import MultimodalEmbeddingCache

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("preprocessing.pipeline")


def load_yaml_config(config_path: str = "configs/base_config.yaml") -> Dict[str, Any]:
    """Load configuration dictionary from YAML file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_pipeline(
    config_path: str = "configs/base_config.yaml",
    force_demo: bool = False,
    extract_embeddings: bool = True,
) -> Dict[str, Any]:
    """Execute complete Phase 2 data preprocessing pipeline."""
    start_time = time.time()
    print("=" * 70)
    print(" PRIVACY-PRESERVING MULTIMODAL NEWS RECOMMENDATION")
    print(" Phase 2: Data Ingestion, Preprocessing & Multimodal Pipeline")
    print("=" * 70)

    cfg = load_yaml_config(config_path)
    paths = cfg.get("paths", {})
    dataset_cfg = cfg.get("dataset", {})

    raw_dir = Path(paths.get("raw_data_dir", "data/raw"))
    processed_dir = Path(paths.get("processed_data_dir", "data/processed"))
    embeddings_dir = Path(paths.get("embeddings_dir", "data/embeddings"))

    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    embeddings_dir.mkdir(parents=True, exist_ok=True)

    variant = "demo" if force_demo else dataset_cfg.get("mind_variant", "demo")
    max_title_len = dataset_cfg.get("max_title_len", 30)
    max_abstract_len = dataset_cfg.get("max_abstract_len", 50)
    max_history_len = dataset_cfg.get("max_history_len", 50)
    neg_ratio = dataset_cfg.get("negative_sample_ratio", 4)

    # --------------------------------------------------------------------------
    # Step 1: Raw Data Ingestion / Generation
    # --------------------------------------------------------------------------
    print(f"\n[Step 1/5] Ingesting dataset (Variant: {variant})...")
    train_news_tsv = raw_dir / "train" / "news.tsv"
    train_behaviors_tsv = raw_dir / "train" / "behaviors.tsv"
    dev_news_tsv = raw_dir / "dev" / "news.tsv"
    dev_behaviors_tsv = raw_dir / "dev" / "behaviors.tsv"
    images_dir = raw_dir / "images"

    if variant == "demo" or not (train_news_tsv.exists() and train_behaviors_tsv.exists()):
        print("  Generating self-contained multimodal MIND demo dataset...")
        generate_demo_dataset(
            output_dir=str(raw_dir),
            num_news=100,
            num_users=50,
            num_train_impressions=200,
            num_dev_impressions=50,
            seed=cfg.get("system", {}).get("seed", 42),
            generate_images=True,
        )
    else:
        print(f"  Found existing raw files at {raw_dir}")

    # --------------------------------------------------------------------------
    # Step 2: MIND TSV Parsing & Vocabularies
    # --------------------------------------------------------------------------
    print("\n[Step 2/5] Parsing news and behaviors with negative sampling...")
    parser = MINDParser(
        max_history_len=max_history_len,
        negative_sample_ratio=neg_ratio,
        seed=cfg.get("system", {}).get("seed", 42),
    )

    news_df = parser.parse_news(train_news_tsv)
    train_samples = parser.parse_behaviors(train_behaviors_tsv, is_train=True)
    dev_sessions = parser.parse_behaviors(dev_behaviors_tsv, is_train=False)

    parser.save_processed(processed_dir)

    # Save parsed sample JSONs for fast access
    import json

    with open(processed_dir / "train_samples.json", "w", encoding="utf-8") as f:
        json.dump(train_samples, f)
    with open(processed_dir / "dev_sessions.json", "w", encoding="utf-8") as f:
        json.dump(dev_sessions, f)

    # --------------------------------------------------------------------------
    # Step 3: Multimodal Extractors Initialization
    # --------------------------------------------------------------------------
    print("\n[Step 3/5] Initializing DistilBERT Text & ResNet-18 Vision Extractors...")
    text_extractor = TextFeatureExtractor(
        model_name=cfg.get("models", {}).get("text_encoder", {}).get("model_name", "distilbert-base-uncased"),
        max_title_len=max_title_len,
        max_abstract_len=max_abstract_len,
    )
    image_extractor = ImageFeatureExtractor(
        embedding_dim=cfg.get("models", {}).get("image_encoder", {}).get("embedding_dim", 512),
        image_size=cfg.get("models", {}).get("image_encoder", {}).get("image_size", 224),
        freeze_weights=True,
    )

    # --------------------------------------------------------------------------
    # Step 4: Multimodal Embedding Cache Generation
    # --------------------------------------------------------------------------
    cache_path = embeddings_dir / "multimodal_embeddings.pt"
    if extract_embeddings:
        print("\n[Step 4/5] Precomputing Corpus Multimodal Embedding Cache...")
        cache = MultimodalEmbeddingCache.precompute(
            news_df=news_df,
            text_extractor=text_extractor,
            image_extractor=image_extractor,
            images_dir=images_dir,
            batch_size=32,
        )
        cache.save(cache_path)
    else:
        print("\n[Step 4/5] Skipping embedding precomputation (--no-extract-embeddings).")
        cache = None

    # --------------------------------------------------------------------------
    # Step 5: PyTorch Dataset & DataLoader Validation Check
    # --------------------------------------------------------------------------
    print("\n[Step 5/5] Validating PyTorch Datasets & DataLoaders...")
    news_dataset = MINDNewsDataset(
        news_df=news_df,
        text_extractor=text_extractor,
        image_extractor=image_extractor,
        images_dir=images_dir,
        max_title_len=max_title_len,
        max_abstract_len=max_abstract_len,
    )

    sample_news_item = news_dataset[0]
    print(f"  News Item Batch Shape Check:")
    print(f"    - Title Tokens Shape    : {list(sample_news_item['title_ids'].shape)} (Target: [{max_title_len}])")
    print(f"    - Title Mask Shape      : {list(sample_news_item['title_mask'].shape)} (Target: [{max_title_len}])")
    print(f"    - Image Tensor Shape    : {list(sample_news_item['image_tensor'].shape)} (Target: [3, 224, 224])")

    train_loader, dev_loader = create_recommendation_dataloaders(
        train_samples=train_samples,
        dev_samples=dev_sessions,
        batch_size=32,
        shuffle=True,
    )

    first_batch = next(iter(train_loader))
    print(f"  Recommendation Training Batch Check (Batch Size: 32):")
    print(f"    - User Indices Shape    : {list(first_batch['user_idx'].shape)} (Target: [32])")
    print(f"    - History Indices Shape : {list(first_batch['history_indices'].shape)} (Target: [32, {max_history_len}])")
    print(f"    - Candidate Shape       : {list(first_batch['candidate_idx'].shape)} (Target: [32])")
    print(f"    - Labels Shape          : {list(first_batch['label'].shape)} (Target: [32])")

    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print(" PHASE 2 PIPELINE EXECUTION SUMMARY")
    print("=" * 70)
    print(f"  Total Multimodal News Articles : {len(news_df)}")
    print(f"  Categories / Subcategories     : {len(parser.category2idx) - 1} / {len(parser.subcategory2idx) - 1}")
    print(f"  Total Simulated Users          : {len(parser.user2idx) - 1}")
    print(f"  Pointwise Training Samples     : {len(train_samples)} (1:4 Negative Sampling)")
    print(f"  Evaluation Impression Sessions : {len(dev_sessions)}")
    print(f"  Embedding Cache Created        : {cache_path.name} ({cache_path.stat().st_size / (1024*1024):.2f} MB)")
    print(f"  Elapsed Pipeline Time          : {elapsed:.2f} seconds")
    print("=" * 70)
    print(" RESULT: PHASE 2 PREPROCESSING PIPELINE READY & VERIFIED")
    print("=" * 70)

    return {
        "num_news": len(news_df),
        "num_users": len(parser.user2idx) - 1,
        "num_train_samples": len(train_samples),
        "num_dev_sessions": len(dev_sessions),
        "cache_path": str(cache_path),
        "elapsed_seconds": elapsed,
    }


def main():
    parser = argparse.ArgumentParser(description="MIND Multimodal Preprocessing Pipeline")
    parser.add_argument("--config", type=str, default="configs/base_config.yaml", help="Config file path")
    parser.add_argument("--demo", action="store_true", help="Force synthetic demo generation")
    parser.add_argument("--no-embeddings", action="store_true", help="Skip embedding cache extraction")
    args = parser.parse_args()

    run_pipeline(
        config_path=args.config,
        force_demo=args.demo,
        extract_embeddings=not args.no_embeddings,
    )


if __name__ == "__main__":
    main()
