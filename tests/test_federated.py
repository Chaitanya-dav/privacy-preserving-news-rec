"""Unit tests for Phase 10, Phase 11, and Phase 12: Federated Learning.

Covers:
  - Phase 10: FL Partitioner (Uniform IID & Dirichlet Non-IID user splitting)
  - Phase 11: FlowerNewsClient (parameter getter/setter, local fit, local evaluate)
  - Phase 12: FedAvg aggregation mathematics & multi-round convergence tracking
"""

import tempfile
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest
import torch

from models.recommender import MultimodalNewsRecommender
from preprocessing.embedding_cache import MultimodalEmbeddingCache
from federated.partitioner import FederatedDataPartitioner
from federated.client import (
    FlowerNewsClient,
    get_model_parameters,
    set_model_parameters,
)
from federated.server import (
    FederatedServer,
    compute_param_delta_norm,
    fedavg_aggregate,
)


@pytest.fixture
def dummy_cache():
    """Tiny cache for 15 articles."""
    n = 15
    text_emb = torch.randn(n + 1, 768)
    text_emb[0] = 0.0
    image_emb = torch.randn(n + 1, 512)
    image_emb[0] = 0.0
    return MultimodalEmbeddingCache(
        text_embeddings=text_emb,
        image_embeddings=image_emb,
        metadata={"num_articles": n},
    )


@pytest.fixture
def synthetic_data():
    """Simulated behavioral train samples and dev sessions across 6 users."""
    train_samples = [
        {"user_idx": 1, "history_indices": [1, 2, 0, 0, 0], "candidate_idx": 3, "label": 1.0},
        {"user_idx": 1, "history_indices": [1, 2, 0, 0, 0], "candidate_idx": 4, "label": 0.0},
        {"user_idx": 2, "history_indices": [5, 6, 0, 0, 0], "candidate_idx": 7, "label": 1.0},
        {"user_idx": 2, "history_indices": [5, 6, 0, 0, 0], "candidate_idx": 8, "label": 0.0},
        {"user_idx": 3, "history_indices": [9, 0, 0, 0, 0], "candidate_idx": 10, "label": 1.0},
        {"user_idx": 4, "history_indices": [1, 3, 0, 0, 0], "candidate_idx": 5, "label": 0.0},
        {"user_idx": 5, "history_indices": [2, 4, 0, 0, 0], "candidate_idx": 6, "label": 1.0},
        {"user_idx": 6, "history_indices": [7, 8, 0, 0, 0], "candidate_idx": 9, "label": 1.0},
    ]
    dev_sessions = [
        {"user_idx": 1, "history_indices": [1, 2, 0, 0, 0], "candidate_indices": [3, 4], "labels": [1.0, 0.0]},
        {"user_idx": 2, "history_indices": [5, 6, 0, 0, 0], "candidate_indices": [7, 8], "labels": [1.0, 0.0]},
    ]
    return train_samples, dev_sessions


def test_phase10_partitioner_uniform(synthetic_data):
    """Verify Phase 10 uniform (IID) partition allocates users evenly."""
    train_samples, dev_sessions = synthetic_data
    partitioner = FederatedDataPartitioner(
        train_samples=train_samples,
        dev_sessions=dev_sessions,
        num_clients=3,
        strategy="user_uniform",
        seed=42,
    )

    stats = partitioner.get_partition_stats()
    assert stats["num_clients"] == 3
    assert stats["strategy"] == "user_uniform"
    assert sum(stats["sample_counts"]) == len(train_samples)
    assert all(c > 0 for c in stats["sample_counts"])


def test_phase10_partitioner_dirichlet(synthetic_data):
    """Verify Phase 10 Dirichlet (Non-IID) partition creates distribution skew."""
    train_samples, dev_sessions = synthetic_data
    partitioner = FederatedDataPartitioner(
        train_samples=train_samples,
        dev_sessions=dev_sessions,
        num_clients=3,
        strategy="user_dirichlet",
        alpha=0.5,
        seed=123,
    )

    stats = partitioner.get_partition_stats()
    assert stats["strategy"] == "user_dirichlet"
    assert stats["alpha"] == 0.5
    assert sum(stats["sample_counts"]) == len(train_samples)
    # Check that client sample getters work
    c0_samples = partitioner.get_client_train_samples(0)
    assert len(c0_samples) == stats["sample_counts"][0]


def test_phase11_client_parameter_roundtrip():
    """Verify extracting and setting model parameters preserves shapes and values."""
    model = MultimodalNewsRecommender(
        text_input_dim=768, image_input_dim=512, fused_dim=32,
        fusion_type="concat_projection", user_pooling="mean",
    )
    params = get_model_parameters(model)
    assert isinstance(params, list)
    assert len(params) > 0
    assert isinstance(params[0], np.ndarray)

    # Perturb one parameter and set back
    perturbed = [p.copy() for p in params]
    perturbed[0] += 0.5
    set_model_parameters(model, perturbed)
    new_params = get_model_parameters(model)
    np.testing.assert_allclose(new_params[0], perturbed[0], rtol=1e-5)


def test_phase11_flower_client_fit_and_eval(synthetic_data, dummy_cache):
    """Verify FlowerNewsClient executes local epochs and returns valid metrics."""
    train_samples, dev_sessions = synthetic_data
    model = MultimodalNewsRecommender(
        text_input_dim=768, image_input_dim=512, fused_dim=32,
        fusion_type="concat_projection", user_pooling="mean",
    )

    client = FlowerNewsClient(
        client_id=0,
        model=model,
        train_samples=train_samples,
        dev_sessions=dev_sessions,
        embedding_cache=dummy_cache,
        batch_size=4,
        local_epochs=1,
    )

    initial_params = client.get_parameters()
    updated_params, num_samples, metrics = client.fit(
        initial_params, config={"local_epochs": 1, "lr": 1e-3}
    )

    assert num_samples == len(train_samples)
    assert "train_loss" in metrics
    assert isinstance(metrics["train_loss"], float)
    assert len(updated_params) == len(initial_params)

    # Evaluate
    loss, eval_samples, eval_metrics = client.evaluate(updated_params)
    assert eval_samples == len(dev_sessions)
    assert "AUC" in eval_metrics
    assert "MRR" in eval_metrics


def test_phase12_fedavg_aggregate_math():
    """Verify FedAvg calculates exact sample-weighted parameter averages."""
    # Client 1: 10 samples, weight = [[1.0, 2.0]]
    # Client 2: 30 samples, weight = [[5.0, 6.0]]
    # Total samples = 40.
    # Expected layer 0 = (10/40)*[1, 2] + (30/40)*[5, 6] = [0.25 + 3.75, 0.5 + 4.5] = [4.0, 5.0]
    c1_params = [np.array([[1.0, 2.0]], dtype=np.float32)]
    c2_params = [np.array([[5.0, 6.0]], dtype=np.float32)]

    results = [(c1_params, 10), (c2_params, 30)]
    aggregated = fedavg_aggregate(results)

    expected = np.array([[4.0, 5.0]], dtype=np.float32)
    np.testing.assert_allclose(aggregated[0], expected, rtol=1e-5)


def test_phase12_federated_server_multi_round(synthetic_data, dummy_cache):
    """Verify FederatedServer coordinates rounds, delta norm, and saves checkpoints."""
    train_samples, dev_sessions = synthetic_data

    partitioner = FederatedDataPartitioner(
        train_samples=train_samples,
        dev_sessions=dev_sessions,
        num_clients=2,
        strategy="user_uniform",
    )

    global_model = MultimodalNewsRecommender(
        text_input_dim=768, image_input_dim=512, fused_dim=16,
        fusion_type="concat_projection", user_pooling="mean",
    )

    clients = []
    for c_id in range(2):
        c_model = MultimodalNewsRecommender(
            text_input_dim=768, image_input_dim=512, fused_dim=16,
            fusion_type="concat_projection", user_pooling="mean",
        )
        c = FlowerNewsClient(
            client_id=c_id,
            model=c_model,
            train_samples=partitioner.get_client_train_samples(c_id),
            dev_sessions=partitioner.get_client_dev_sessions(c_id),
            embedding_cache=dummy_cache,
            batch_size=4,
            local_epochs=1,
        )
        clients.append(c)

    with tempfile.TemporaryDirectory() as tmpdir:
        server = FederatedServer(
            global_model=global_model,
            clients=clients,
            dev_sessions=dev_sessions,
            embedding_cache=dummy_cache,
            device=torch.device("cpu"),
            checkpoint_dir=tmpdir,
        )

        history = server.fit_rounds(num_rounds=2, local_epochs=1, lr=1e-3)

        assert len(history) == 2
        assert "param_delta_norm" in history[0]
        assert "AUC" in history[0]
        assert (Path(tmpdir) / "last_federated_model.pt").exists()
        assert (Path(tmpdir) / "federated_training_history.json").exists()
