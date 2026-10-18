"""Unit tests for Phase 2: Data Ingestion, Preprocessing & Multimodal Feature Pipeline.

Covers:
- MIND TSV parsing and vocabulary management
- DistilBERT NLP text tokenization
- ResNet-18 computer vision feature extraction
- User history padding and 1:4 negative sampling
- PyTorch Dataset batch collation
- Precomputed multimodal embedding caching and reloading
"""

import shutil
import tempfile
from pathlib import Path
import pytest
import torch
import pandas as pd

from preprocessing.dataset_downloader import generate_demo_dataset, generate_synthetic_image
from preprocessing.multimodal_extractors import (
    FallbackTokenizer,
    ImageFeatureExtractor,
    TextFeatureExtractor,
)
from preprocessing.mind_parser import MINDParser
from preprocessing.dataset import (
    MINDNewsDataset,
    MINDRecommendationDataset,
    MINDTestImpressionDataset,
    collate_recommendation_batch,
    create_recommendation_dataloaders,
)
from preprocessing.embedding_cache import MultimodalEmbeddingCache


@pytest.fixture(scope="module")
def temp_data_env():
    """Create a temporary directory with a minimal synthetic MIND dataset."""
    temp_dir = Path(tempfile.mkdtemp(prefix="mind_test_"))
    demo_files = generate_demo_dataset(
        output_dir=str(temp_dir),
        num_news=20,
        num_users=10,
        num_train_impressions=25,
        num_dev_impressions=10,
        seed=123,
        generate_images=True,
    )
    yield temp_dir, demo_files
    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


def test_demo_dataset_generation(temp_data_env):
    """Verify demo generator outputs valid files matching MIND specifications."""
    temp_dir, files = temp_data_env

    assert files["train_news"].exists()
    assert files["train_behaviors"].exists()
    assert files["dev_news"].exists()
    assert files["dev_behaviors"].exists()
    assert files["images_dir"].exists()

    # Verify news.tsv has 8 tab-separated columns
    with open(files["train_news"], "r", encoding="utf-8") as f:
        first_line = f.readline().strip().split("\t")
        assert len(first_line) == 8
        assert first_line[0].startswith("N")

    # Verify behaviors.tsv has 5 tab-separated columns
    with open(files["train_behaviors"], "r", encoding="utf-8") as f:
        first_line = f.readline().strip().split("\t")
        assert len(first_line) == 5
        assert first_line[1].startswith("U")


def test_mind_parser_vocabularies_and_sampling(temp_data_env):
    """Verify parser extracts vocabularies, builds mappings, and samples negatives."""
    temp_dir, files = temp_data_env

    parser = MINDParser(max_history_len=30, negative_sample_ratio=4, seed=42)
    news_df = parser.parse_news(files["train_news"])

    # Check news vocabulary
    assert len(news_df) == 20
    assert parser.news2idx["<PAD>"] == 0
    assert len(parser.news2idx) == 21
    assert len(parser.category2idx) > 1

    # Check behaviors parsing with negative sampling
    train_samples = parser.parse_behaviors(files["train_behaviors"], is_train=True)
    assert len(train_samples) > 0

    first_sample = train_samples[0]
    assert "user_idx" in first_sample
    assert "history_indices" in first_sample
    assert len(first_sample["history_indices"]) == 30  # Padded to max_history_len
    assert "candidate_idx" in first_sample
    assert first_sample["label"] in [0.0, 1.0]

    # Verify negative ratio balance across samples
    labels = [s["label"] for s in train_samples]
    pos_count = sum(1 for l in labels if l == 1.0)
    neg_count = sum(1 for l in labels if l == 0.0)
    assert pos_count > 0
    assert neg_count >= pos_count * 2  # Balanced with negative sampling

    # Check dev session parsing
    dev_sessions = parser.parse_behaviors(files["dev_behaviors"], is_train=False)
    assert len(dev_sessions) > 0
    first_session = dev_sessions[0]
    assert "candidate_indices" in first_session
    assert "labels" in first_session
    assert len(first_session["candidate_indices"]) == len(first_session["labels"])


def test_fallback_tokenizer():
    """Verify offline fallback tokenizer produces deterministic token IDs and masks."""
    tok = FallbackTokenizer(vocab_size=30522)
    text = "Breakthrough AI in Healthcare"
    tokens, mask = tok.encode(text, max_len=10)

    assert len(tokens) == 10
    assert len(mask) == 10
    assert tokens[0] == 101  # [CLS]
    assert mask[0] == 1
    # Check padding at the end
    assert tokens[-1] == 0
    assert mask[-1] == 0


def test_text_feature_extractor():
    """Verify TextFeatureExtractor tokenizes single strings and batches into PyTorch tensors."""
    extractor = TextFeatureExtractor(max_title_len=16, max_abstract_len=24)

    # Single text
    t_ids, t_mask = extractor.tokenize_text("Quantum Computing Breakthrough", max_len=16)
    assert isinstance(t_ids, torch.Tensor)
    assert list(t_ids.shape) == [16]
    assert list(t_mask.shape) == [16]

    # Batch texts
    batch_texts = [
        "First article headline",
        "Second article with longer title and extra details",
    ]
    b_ids, b_mask = extractor.tokenize_batch(batch_texts, max_len=16)
    assert list(b_ids.shape) == [2, 16]
    assert list(b_mask.shape) == [2, 16]


def test_image_feature_extractor(temp_data_env):
    """Verify ResNet-18 loads and extracts 512-dim visual embeddings."""
    temp_dir, files = temp_data_env
    images_dir = files["images_dir"]

    extractor = ImageFeatureExtractor(embedding_dim=512, image_size=224, freeze_weights=True)

    # Test loading existing image
    sample_img_path = next(images_dir.glob("*.jpg"))
    img_tensor = extractor.preprocess_image(sample_img_path)
    assert isinstance(img_tensor, torch.Tensor)
    assert list(img_tensor.shape) == [3, 224, 224]

    # Test missing image fallback
    missing_tensor = extractor.preprocess_image("non_existent_image.jpg")
    assert list(missing_tensor.shape) == [3, 224, 224]

    # Test 512-dim embedding extraction
    emb_single = extractor.extract_embedding(img_tensor)
    assert list(emb_single.shape) == [512]

    # Test batch embedding extraction
    batch_imgs = torch.stack([img_tensor, missing_tensor], dim=0)
    emb_batch = extractor.extract_embedding(batch_imgs)
    assert list(emb_batch.shape) == [2, 512]


def test_pytorch_datasets_and_dataloaders(temp_data_env):
    """Verify MINDNewsDataset and MINDRecommendationDataset PyTorch pipelines."""
    temp_dir, files = temp_data_env

    parser = MINDParser(max_history_len=20, negative_sample_ratio=4)
    news_df = parser.parse_news(files["train_news"])
    train_samples = parser.parse_behaviors(files["train_behaviors"], is_train=True)

    text_extractor = TextFeatureExtractor(max_title_len=15, max_abstract_len=25)
    image_extractor = ImageFeatureExtractor(image_size=224)

    # 1. MINDNewsDataset test
    news_ds = MINDNewsDataset(
        news_df=news_df,
        text_extractor=text_extractor,
        image_extractor=image_extractor,
        images_dir=files["images_dir"],
        max_title_len=15,
        max_abstract_len=25,
    )
    assert len(news_ds) == 20
    first_news = news_ds[0]
    assert "news_idx" in first_news
    assert list(first_news["title_ids"].shape) == [15]
    assert list(first_news["image_tensor"].shape) == [3, 224, 224]

    # 2. MINDRecommendationDataset & DataLoader test
    train_loader, _ = create_recommendation_dataloaders(
        train_samples=train_samples,
        batch_size=8,
        shuffle=False,
    )

    batch = next(iter(train_loader))
    assert list(batch["user_idx"].shape) == [8]
    assert list(batch["history_indices"].shape) == [8, 20]
    assert list(batch["candidate_idx"].shape) == [8]
    assert list(batch["label"].shape) == [8]
    assert batch["label"].dtype == torch.float32


def test_embedding_cache_precompute_and_load(temp_data_env):
    """Verify precomputing, saving, and querying multimodal embedding cache."""
    temp_dir, files = temp_data_env

    parser = MINDParser(max_history_len=10)
    news_df = parser.parse_news(files["train_news"])
    text_extractor = TextFeatureExtractor(max_title_len=12)
    image_extractor = ImageFeatureExtractor(image_size=224)

    cache = MultimodalEmbeddingCache.precompute(
        news_df=news_df,
        text_extractor=text_extractor,
        image_extractor=image_extractor,
        images_dir=files["images_dir"],
        batch_size=10,
    )

    cache_file = temp_dir / "test_embeddings.pt"
    cache.save(cache_file)
    assert cache_file.exists()

    # Reload from disk
    loaded_cache = MultimodalEmbeddingCache.load(cache_file)
    assert loaded_cache.text_embeddings is not None
    assert loaded_cache.image_embeddings is not None

    query_indices = torch.tensor([1, 2, 3], dtype=torch.long)
    text_embs = loaded_cache.get_text_embedding(query_indices)
    img_embs = loaded_cache.get_image_embedding(query_indices)

    assert list(text_embs.shape) == [3, 768]
    assert list(img_embs.shape) == [3, 512]
