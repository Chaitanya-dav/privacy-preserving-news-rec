"""Centralized Training Entry-Point (CLI).

Usage:
    python -m centralized.train --config configs/base_config.yaml --epochs 5
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("centralized.train")


def build_model_from_config(cfg: dict):
    """Instantiate MultimodalNewsRecommender from config dict."""
    from models.recommender import MultimodalNewsRecommender

    models_cfg = cfg.get("models", {})
    fused_dim = models_cfg.get("multimodal_fusion", {}).get("fused_dim", 256)
    fusion_type = models_cfg.get("multimodal_fusion", {}).get("fusion_type", "concat_projection")
    dropout = models_cfg.get("multimodal_fusion", {}).get("dropout", 0.2)
    pooling = models_cfg.get("user_model", {}).get("pooling", "attention")
    attn_dim = models_cfg.get("user_model", {}).get("attention_dim", 128)

    return MultimodalNewsRecommender(
        text_input_dim=768,
        image_input_dim=512,
        fused_dim=fused_dim,
        fusion_type=fusion_type,
        user_pooling=pooling,
        attention_dim=attn_dim,
        news_dropout=dropout,
        user_dropout=0.1,
    )


def main():
    parser = argparse.ArgumentParser(description="Centralized Baseline Trainer")
    parser.add_argument("--config", default="configs/base_config.yaml")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override epochs from config")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override batch size from config")
    parser.add_argument("--lr", type=float, default=None,
                        help="Override learning rate from config")
    parser.add_argument("--no-eval", action="store_true",
                        help="Skip dev evaluation during training")
    args = parser.parse_args()

    # Load config
    cfg_path = Path(args.config)
    if not cfg_path.exists():
        logger.error("Config not found: %s", cfg_path)
        sys.exit(1)

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Device
    sys_cfg = cfg.get("system", {})
    device_setting = sys_cfg.get("device", "auto")
    if device_setting == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_setting)

    torch.manual_seed(sys_cfg.get("seed", 42))

    # Training hyperparams
    train_cfg = cfg.get("centralized_training", {})
    num_epochs = args.epochs or train_cfg.get("epochs", 5)
    batch_size = args.batch_size or train_cfg.get("batch_size", 64)
    lr = args.lr or train_cfg.get("learning_rate", 1e-3)
    weight_decay = train_cfg.get("weight_decay", 1e-5)

    paths_cfg = cfg.get("paths", {})
    processed_dir = Path(paths_cfg.get("processed_data_dir", "data/processed"))
    embeddings_dir = Path(paths_cfg.get("embeddings_dir", "data/embeddings"))
    checkpoints_dir = Path(paths_cfg.get("checkpoints_dir", "results/checkpoints"))

    # ── Step 1: Load Embedding Cache ──────────────────────────────────────────
    cache_file = embeddings_dir / "multimodal_embeddings.pt"
    if not cache_file.exists():
        logger.error(
            "Embedding cache not found at %s. Run Phase 2 pipeline first:\n"
            "  python -m preprocessing.pipeline",
            cache_file,
        )
        sys.exit(1)

    from preprocessing.embedding_cache import MultimodalEmbeddingCache
    print(f"Loading embedding cache from {cache_file}...")
    embedding_cache = MultimodalEmbeddingCache.load(cache_file)
    print(f"  Cache loaded: {embedding_cache.metadata}")

    # ── Step 2: Load Processed Samples ────────────────────────────────────────
    train_samples_file = processed_dir / "train_samples.json"
    dev_sessions_file = processed_dir / "dev_sessions.json"

    if not train_samples_file.exists() or not dev_sessions_file.exists():
        logger.error(
            "Processed samples not found in %s. Run Phase 2 pipeline first.", processed_dir
        )
        sys.exit(1)

    with open(train_samples_file, "r", encoding="utf-8") as f:
        train_samples = json.load(f)
    with open(dev_sessions_file, "r", encoding="utf-8") as f:
        dev_sessions = json.load(f)

    print(f"Train samples: {len(train_samples):,}  |  Dev sessions: {len(dev_sessions):,}")

    # ── Step 3: Build DataLoader ───────────────────────────────────────────────
    from preprocessing.dataset import create_recommendation_dataloaders
    train_loader, _ = create_recommendation_dataloaders(
        train_samples=train_samples,
        batch_size=batch_size,
        shuffle=True,
        num_workers=sys_cfg.get("num_workers", 0),
    )

    # ── Step 4: Build Model ────────────────────────────────────────────────────
    model = build_model_from_config(cfg)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {num_params:,} trainable parameters  |  Device: {device}")

    # ── Step 5: Train ──────────────────────────────────────────────────────────
    from torch.optim.lr_scheduler import CosineAnnealingLR
    from centralized.trainer import CentralizedTrainer

    trainer = CentralizedTrainer(
        model=model,
        embedding_cache=embedding_cache,
        device=device,
        learning_rate=lr,
        weight_decay=weight_decay,
        checkpoint_dir=str(checkpoints_dir),
    )

    scheduler = CosineAnnealingLR(trainer.optimizer, T_max=num_epochs, eta_min=lr * 0.1)

    history = trainer.fit(
        train_loader=train_loader,
        dev_sessions=dev_sessions if not args.no_eval else [],
        num_epochs=num_epochs,
        scheduler=scheduler,
        eval_every=1,
    )

    print("\n[Centralized Training Complete]")
    if history and "AUC" in history[-1]:
        last = history[-1]
        print(f"  Final Epoch — AUC: {last['AUC']:.4f}  MRR: {last['MRR']:.4f}"
              f"  nDCG@5: {last['nDCG@5']:.4f}  nDCG@10: {last['nDCG@10']:.4f}")


if __name__ == "__main__":
    main()
