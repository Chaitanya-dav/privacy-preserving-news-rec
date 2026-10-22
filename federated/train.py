"""Federated Training Entry-Point (CLI) - Phase 12.

Usage:
    python -m federated.train --config configs/federated_config.yaml --rounds 5
"""

import argparse
import json
import logging
from pathlib import Path

import torch
import yaml

from models.recommender import MultimodalNewsRecommender
from preprocessing.embedding_cache import MultimodalEmbeddingCache
from .client import FlowerNewsClient
from .partitioner import FederatedDataPartitioner
from .server import FederatedServer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("federated.train")


def main():
    parser = argparse.ArgumentParser(description="Federated Learning Recommender Trainer")
    parser.add_argument("--config", default="configs/federated_config.yaml")
    parser.add_argument("--base-config", default="configs/base_config.yaml")
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--clients", type=int, default=None)
    parser.add_argument("--local-epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    args = parser.parse_args()

    # Load configs
    with open(args.base_config, "r") as f:
        base_cfg = yaml.safe_load(f)
    with open(args.config, "r") as f:
        fed_cfg = yaml.safe_load(f)

    device_str = base_cfg.get("system", {}).get("device", "auto")
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    # Resolve training params
    fed_section = fed_cfg.get("federated", {})
    local_section = fed_cfg.get("local_training", {})
    part_section = fed_cfg.get("partitioning", {})

    num_rounds = args.rounds or fed_section.get("num_rounds", 5)
    num_clients = args.clients or fed_section.get("num_clients", 5)
    local_epochs = args.local_epochs or local_section.get("local_epochs", 2)
    lr = args.lr or local_section.get("learning_rate", 1e-3)
    batch_size = local_section.get("batch_size", 32)
    strategy = part_section.get("strategy", "user_dirichlet")
    alpha = part_section.get("alpha", 0.5)

    processed_dir = Path(base_cfg["paths"]["processed_data_dir"])
    embeddings_dir = Path(base_cfg["paths"]["embeddings_dir"])
    checkpoints_dir = Path(base_cfg["paths"]["checkpoints_dir"])

    cache_file = embeddings_dir / "multimodal_embeddings.pt"
    if not cache_file.exists():
        raise FileNotFoundError(f"Embedding cache not found at {cache_file}.")
    cache = MultimodalEmbeddingCache.load(cache_file)

    with open(processed_dir / "train_samples.json", "r") as f:
        train_samples = json.load(f)
    with open(processed_dir / "dev_sessions.json", "r") as f:
        dev_sessions = json.load(f)

    # 1. Edge Data Partitioning (Phase 10)
    print(f"Partitioning {len(train_samples)} interaction samples across {num_clients} edge clients ({strategy})...")
    partitioner = FederatedDataPartitioner(
        train_samples=train_samples,
        dev_sessions=dev_sessions,
        num_clients=num_clients,
        strategy=strategy,
        alpha=alpha,
    )
    stats = partitioner.get_partition_stats()
    print(f"Partition Stats: Samples/client={stats['sample_counts']} | Imbalance={stats['imbalance_ratio']}")

    # 2. Instantiate Global Model & Edge Clients (Phase 11)
    models_cfg = base_cfg.get("models", {})
    fused_dim = models_cfg.get("multimodal_fusion", {}).get("fused_dim", 256)
    fusion_type = models_cfg.get("multimodal_fusion", {}).get("fusion_type", "concat_projection")

    global_model = MultimodalNewsRecommender(
        text_input_dim=768,
        image_input_dim=512,
        fused_dim=fused_dim,
        fusion_type=fusion_type,
        user_pooling=models_cfg.get("user_model", {}).get("pooling", "attention"),
        attention_dim=models_cfg.get("user_model", {}).get("attention_dim", 128),
        news_dropout=0.2,
        user_dropout=0.1,
    ).to(device)

    clients: List[FlowerNewsClient] = []
    for c_id in range(num_clients):
        # Client creates its own local model instance
        client_model = MultimodalNewsRecommender(
            text_input_dim=768,
            image_input_dim=512,
            fused_dim=fused_dim,
            fusion_type=fusion_type,
            user_pooling=models_cfg.get("user_model", {}).get("pooling", "attention"),
            attention_dim=models_cfg.get("user_model", {}).get("attention_dim", 128),
            news_dropout=0.2,
            user_dropout=0.1,
        ).to(device)

        c_train = partitioner.get_client_train_samples(c_id)
        c_dev = partitioner.get_client_dev_sessions(c_id)

        client = FlowerNewsClient(
            client_id=c_id,
            model=client_model,
            train_samples=c_train,
            dev_sessions=c_dev,
            embedding_cache=cache,
            device=device,
            batch_size=batch_size,
            learning_rate=lr,
            local_epochs=local_epochs,
        )
        clients.append(client)

    # 3. Federated Server Orchestration & FedAvg Convergence (Phase 12)
    server = FederatedServer(
        global_model=global_model,
        clients=clients,
        dev_sessions=dev_sessions,
        embedding_cache=cache,
        device=device,
        checkpoint_dir=str(checkpoints_dir),
    )

    history = server.fit_rounds(
        num_rounds=num_rounds,
        local_epochs=local_epochs,
        lr=lr,
    )
    print(f"\n[Federated Learning] Training complete. {len(history)} rounds saved to {checkpoints_dir}.")


if __name__ == "__main__":
    main()
