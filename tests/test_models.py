"""Unit tests for Phase 3: Multimodal News Recommendation Models & Centralized Training.

Covers:
  - TextEncoder, VisualEncoder, MultimodalNewsEncoder (concat + attention fusion)
  - AdditiveAttention, AttentiveUserEncoder, MeanUserEncoder (PAD masking)
  - MultimodalNewsRecommender (pointwise + session forward)
  - Evaluation metrics (AUC, MRR, nDCG@5, nDCG@10)
  - CentralizedTrainer: one training step + evaluation
"""

import json
import tempfile
import shutil
from pathlib import Path
from typing import List

import pytest
import torch
import torch.nn as nn

# ─── Model imports ────────────────────────────────────────────────────────────
from models.news_encoder import TextEncoder, VisualEncoder, MultimodalNewsEncoder
from models.user_encoder import AdditiveAttention, AttentiveUserEncoder, MeanUserEncoder
from models.recommender import MultimodalNewsRecommender

# ─── Evaluation imports ───────────────────────────────────────────────────────
from evaluation.metrics import roc_auc_score_simple, mrr_score, ndcg_score, evaluate_impressions

# ─── Preprocessing imports ────────────────────────────────────────────────────
from preprocessing.embedding_cache import MultimodalEmbeddingCache


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def device():
    return torch.device("cpu")


@pytest.fixture(scope="module")
def dummy_cache():
    """A tiny MultimodalEmbeddingCache for 20 articles (index 0 = PAD)."""
    n = 20
    text_emb = torch.randn(n + 1, 768)   # index 0 = PAD
    text_emb[0] = 0.0
    image_emb = torch.randn(n + 1, 512)
    image_emb[0] = 0.0
    return MultimodalEmbeddingCache(
        text_embeddings=text_emb,
        image_embeddings=image_emb,
        metadata={"num_articles": n, "text_dim": 768, "image_dim": 512},
    )


@pytest.fixture(scope="module")
def model_concat(device):
    return MultimodalNewsRecommender(
        text_input_dim=768, image_input_dim=512, fused_dim=64,
        fusion_type="concat_projection", user_pooling="attention",
        attention_dim=32, news_dropout=0.0, user_dropout=0.0,
    ).to(device)


@pytest.fixture(scope="module")
def model_attn(device):
    return MultimodalNewsRecommender(
        text_input_dim=768, image_input_dim=512, fused_dim=64,
        fusion_type="attention_fusion", user_pooling="mean",
        attention_dim=32, news_dropout=0.0, user_dropout=0.0,
    ).to(device)


# ─── Text Encoder ─────────────────────────────────────────────────────────────

def test_text_encoder_output_shape(device):
    enc = TextEncoder(input_dim=768, output_dim=64).to(device)
    x = torch.randn(8, 768, device=device)
    out = enc(x)
    assert out.shape == (8, 64)


def test_text_encoder_batch_dims(device):
    """Multi-dimensional batch (B, H, D) should work via broadcasting."""
    enc = TextEncoder(input_dim=768, output_dim=64).to(device)
    x = torch.randn(4, 10, 768, device=device)  # (B, H, D)
    out = enc(x)
    assert out.shape == (4, 10, 64)


# ─── Visual Encoder ───────────────────────────────────────────────────────────

def test_visual_encoder_output_shape(device):
    enc = VisualEncoder(input_dim=512, output_dim=64).to(device)
    x = torch.randn(8, 512, device=device)
    out = enc(x)
    assert out.shape == (8, 64)


# ─── Multimodal News Encoder ──────────────────────────────────────────────────

@pytest.mark.parametrize("fusion_type", ["concat_projection", "attention_fusion"])
def test_multimodal_news_encoder_forward(device, fusion_type):
    enc = MultimodalNewsEncoder(
        text_input_dim=768, image_input_dim=512, fused_dim=64,
        fusion_type=fusion_type, dropout=0.0,
    ).to(device)
    text = torch.randn(8, 768, device=device)
    image = torch.randn(8, 512, device=device)
    out = enc(text, image)
    assert out.shape == (8, 64)


def test_multimodal_encoder_from_cache(device, dummy_cache):
    enc = MultimodalNewsEncoder(
        text_input_dim=768, image_input_dim=512, fused_dim=64, dropout=0.0,
    ).to(device)
    indices = torch.tensor([1, 2, 3, 4, 5], dtype=torch.long)
    out = enc.encode_from_cache(indices, dummy_cache)
    assert out.shape == (5, 64)


# ─── Additive Attention ───────────────────────────────────────────────────────

def test_additive_attention_no_mask(device):
    attn = AdditiveAttention(input_dim=64, attention_dim=32).to(device)
    h = torch.randn(4, 10, 64, device=device)
    out, weights = attn(h)
    assert out.shape == (4, 64)
    assert weights.shape == (4, 10)
    # Weights should sum to 1
    assert torch.allclose(weights.sum(dim=-1), torch.ones(4, device=device), atol=1e-5)


def test_additive_attention_pad_masking(device):
    """PAD positions should receive zero attention weight."""
    attn = AdditiveAttention(input_dim=64, attention_dim=32).to(device)
    h = torch.randn(2, 5, 64, device=device)
    mask = torch.tensor([[False, False, True, True, True],
                         [False, True,  True, True, True]], dtype=torch.bool, device=device)
    _, weights = attn(h, key_padding_mask=mask)
    # Masked positions should have exactly 0 weight
    assert (weights[0, 2:] == 0.0).all()
    assert (weights[1, 1:] == 0.0).all()


# ─── User Encoders ────────────────────────────────────────────────────────────

def test_attentive_user_encoder(device):
    enc = AttentiveUserEncoder(news_dim=64, attention_dim=32, dropout=0.0).to(device)
    history = torch.randn(4, 10, 64, device=device)
    # Simulate 3 PAD positions at end
    history[:, 7:] = 0.0
    user_vec = enc(history)
    assert user_vec.shape == (4, 64)


def test_mean_user_encoder(device):
    enc = MeanUserEncoder(news_dim=64, dropout=0.0).to(device)
    history = torch.randn(4, 10, 64, device=device)
    history[:, 8:] = 0.0
    user_vec = enc(history)
    assert user_vec.shape == (4, 64)


# ─── Recommender Forward ──────────────────────────────────────────────────────

def test_pointwise_forward(model_concat, device, dummy_cache):
    B, H = 8, 10
    candidate_idx = torch.randint(1, 20, (B,), device=device)
    history_idx = torch.randint(0, 20, (B, H), device=device)
    # Ensure some PADs
    history_idx[:, 8:] = 0

    with torch.no_grad():
        logits = model_concat.pointwise_forward(candidate_idx, history_idx, dummy_cache)
    assert logits.shape == (B,)
    assert logits.dtype == torch.float32


def test_session_forward(model_concat, device, dummy_cache):
    B, H, C = 4, 10, 6
    candidate_idx = torch.randint(1, 20, (B, C), device=device)
    history_idx = torch.randint(0, 20, (B, H), device=device)
    history_idx[:, 8:] = 0

    with torch.no_grad():
        scores = model_concat.session_forward(candidate_idx, history_idx, dummy_cache)
    assert scores.shape == (B, C)


def test_attention_fusion_forward(model_attn, device, dummy_cache):
    B, H = 6, 12
    candidate_idx = torch.randint(1, 20, (B,), device=device)
    history_idx = torch.randint(0, 20, (B, H), device=device)
    history_idx[:, 10:] = 0

    with torch.no_grad():
        logits = model_attn.pointwise_forward(candidate_idx, history_idx, dummy_cache)
    assert logits.shape == (B,)


def test_unified_forward_dispatch(model_concat, device, dummy_cache):
    """model.forward() should dispatch based on candidate dims."""
    B, H, C = 3, 8, 5
    history_idx = torch.randint(1, 20, (B, H), device=device)
    # Pointwise
    cand_1d = torch.randint(1, 20, (B,), device=device)
    with torch.no_grad():
        out = model_concat(cand_1d, history_idx, dummy_cache)
    assert out.shape == (B,)
    # Session
    cand_2d = torch.randint(1, 20, (B, C), device=device)
    with torch.no_grad():
        out = model_concat(cand_2d, history_idx, dummy_cache)
    assert out.shape == (B, C)


# ─── Evaluation Metrics ───────────────────────────────────────────────────────

def test_auc_perfect_ranking():
    scores = [0.9, 0.8, 0.1, 0.05]
    labels = [1.0, 1.0, 0.0, 0.0]
    assert roc_auc_score_simple(labels, scores) == 1.0


def test_auc_worst_ranking():
    scores = [0.1, 0.05, 0.9, 0.8]
    labels = [1.0, 1.0, 0.0, 0.0]
    assert roc_auc_score_simple(labels, scores) == 0.0


def test_mrr():
    # First relevant item at rank 2
    scores = [0.8, 0.9, 0.3, 0.1]
    labels = [0.0, 1.0, 0.0, 0.0]
    assert abs(mrr_score(labels, scores) - 1.0) < 1e-6

    # First relevant item at rank 2
    scores = [0.9, 0.8, 0.3, 0.1]
    labels = [0.0, 1.0, 0.0, 0.0]
    assert abs(mrr_score(labels, scores) - 0.5) < 1e-6


def test_ndcg_at_5():
    # Perfect: only positive ranked first
    scores = [0.9, 0.2, 0.1, 0.05, 0.04, 0.03]
    labels = [1.0, 0.0, 0.0, 0.0,  0.0,  0.0]
    assert abs(ndcg_score(labels, scores, k=5) - 1.0) < 1e-5


def test_evaluate_impressions():
    all_scores = [[0.9, 0.2, 0.1], [0.5, 0.8, 0.3]]
    all_labels = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    results = evaluate_impressions(all_scores, all_labels)
    assert "AUC" in results
    assert "MRR" in results
    assert "nDCG@5" in results
    assert "nDCG@10" in results
    assert 0.0 <= results["AUC"] <= 1.0
    assert 0.0 <= results["MRR"] <= 1.0


# ─── Centralized Trainer: one step ────────────────────────────────────────────

def test_centralized_trainer_single_epoch(device, dummy_cache):
    """Smoke test: one training epoch + evaluation should not crash."""
    from torch.utils.data import DataLoader
    from preprocessing.dataset import MINDRecommendationDataset, collate_recommendation_batch
    from centralized.trainer import CentralizedTrainer

    samples = [
        {"user_idx": 1, "history_indices": [1, 2, 3, 4, 0, 0, 0, 0, 0, 0], "candidate_idx": 5, "label": 1.0},
        {"user_idx": 1, "history_indices": [1, 2, 3, 4, 0, 0, 0, 0, 0, 0], "candidate_idx": 6, "label": 0.0},
        {"user_idx": 2, "history_indices": [7, 8, 0, 0, 0, 0, 0, 0, 0, 0], "candidate_idx": 9, "label": 1.0},
        {"user_idx": 2, "history_indices": [7, 8, 0, 0, 0, 0, 0, 0, 0, 0], "candidate_idx": 10, "label": 0.0},
    ]
    dev_sessions = [
        {"history_indices": [1, 2, 3, 0, 0, 0, 0, 0, 0, 0], "candidate_indices": [5, 11, 12], "labels": [1.0, 0.0, 0.0]},
    ]

    train_loader = DataLoader(
        MINDRecommendationDataset(samples),
        batch_size=2,
        collate_fn=collate_recommendation_batch,
    )

    model = MultimodalNewsRecommender(
        text_input_dim=768, image_input_dim=512, fused_dim=32,
        fusion_type="concat_projection", user_pooling="attention",
        attention_dim=16, news_dropout=0.0,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = CentralizedTrainer(
            model=model, embedding_cache=dummy_cache,
            device=device, learning_rate=1e-3, checkpoint_dir=tmpdir,
        )
        history = trainer.fit(train_loader, dev_sessions, num_epochs=2, eval_every=1)

    assert len(history) == 2
    assert "train_loss" in history[0]
    assert "AUC" in history[0]
    assert isinstance(history[0]["train_loss"], float)
