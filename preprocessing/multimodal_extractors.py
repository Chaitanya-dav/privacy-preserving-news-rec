"""Multimodal Feature Extractors: DistilBERT NLP Text & ResNet-18 Computer Vision.

Provides tokenization, tensor formatting, and dense feature extraction
for news articles' textual titles/abstracts and visual image thumbnails.
Includes robust offline fallbacks to ensure deterministic local execution.
"""

import os
import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
from PIL import Image

logger = logging.getLogger(__name__)


class FallbackTokenizer:
    """Deterministic offline word-hash tokenizer used when HuggingFace hub is unreachable."""

    def __init__(self, vocab_size: int = 30522, unk_token_id: int = 100, pad_token_id: int = 0):
        self.vocab_size = vocab_size
        self.unk_token_id = unk_token_id
        self.pad_token_id = pad_token_id

    def _hash_word(self, word: str) -> int:
        h = int(hashlib.md5(word.lower().encode("utf-8")).hexdigest(), 16)
        # Avoid reserving 0 (PAD) or 100 (UNK)
        return 101 + (h % (self.vocab_size - 101))

    def encode(
        self,
        text: str,
        max_len: int = 30,
    ) -> Tuple[List[int], List[int]]:
        words = text.strip().split()
        tokens = [101]  # [CLS] token
        for w in words[: max_len - 2]:
            tokens.append(self._hash_word(w))
        tokens.append(102)  # [SEP] token

        attn_mask = [1] * len(tokens)
        pad_len = max_len - len(tokens)
        if pad_len > 0:
            tokens.extend([self.pad_token_id] * pad_len)
            attn_mask.extend([0] * pad_len)
        else:
            tokens = tokens[:max_len]
            attn_mask = attn_mask[:max_len]

        return tokens, attn_mask


class TextFeatureExtractor:
    """Tokenizes and encodes news text using DistilBERT or offline tokenizer."""

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        max_title_len: int = 30,
        max_abstract_len: int = 50,
        device: Optional[torch.device] = None,
        use_offline_fallback: bool = False,
    ):
        self.model_name = model_name
        self.max_title_len = max_title_len
        self.max_abstract_len = max_abstract_len
        self.device = device or torch.device("cpu")
        self.tokenizer = None
        self.fallback = FallbackTokenizer()

        if not use_offline_fallback:
            try:
                from transformers import AutoTokenizer

                # Attempt local cache or fast offline load
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    local_files_only=False,
                )
                logger.info("Loaded HuggingFace AutoTokenizer for %s", model_name)
            except Exception as e:
                logger.warning(
                    "Could not load HuggingFace tokenizer (%s). Using deterministic offline tokenizer fallback.",
                    e,
                )
                self.tokenizer = None
        else:
            self.tokenizer = None

    def tokenize_text(
        self,
        text: str,
        max_len: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Tokenize a single text string into input_ids and attention_mask tensors."""
        if self.tokenizer is not None:
            encoding = self.tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=max_len,
                return_tensors="pt",
            )
            return encoding["input_ids"].squeeze(0), encoding["attention_mask"].squeeze(0)
        else:
            input_ids, mask = self.fallback.encode(text, max_len=max_len)
            return torch.tensor(input_ids, dtype=torch.long), torch.tensor(mask, dtype=torch.long)

    def tokenize_batch(
        self,
        texts: List[str],
        max_len: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Tokenize a list of text strings into batched tensors."""
        if self.tokenizer is not None:
            encoding = self.tokenizer(
                texts,
                padding="max_length",
                truncation=True,
                max_length=max_len,
                return_tensors="pt",
            )
            return encoding["input_ids"], encoding["attention_mask"]
        else:
            ids_list, mask_list = [], []
            for t in texts:
                i, m = self.fallback.encode(t, max_len=max_len)
                ids_list.append(i)
                mask_list.append(m)
            return torch.tensor(ids_list, dtype=torch.long), torch.tensor(mask_list, dtype=torch.long)


class ImageFeatureExtractor:
    """Preprocesses news images and extracts 512-dim visual embeddings using ResNet-18."""

    def __init__(
        self,
        embedding_dim: int = 512,
        image_size: int = 224,
        device: Optional[torch.device] = None,
        freeze_weights: bool = True,
    ):
        self.embedding_dim = embedding_dim
        self.image_size = image_size
        self.device = device or torch.device("cpu")

        # Standard torchvision image transformation pipeline
        import torchvision.transforms as transforms

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        # Load ResNet-18 model backbone
        self.model = self._init_resnet(freeze_weights)

    def _init_resnet(self, freeze_weights: bool) -> nn.Module:
        """Initialize ResNet18 backbone with classification head removed."""
        import torchvision.models as models

        try:
            # Try loading default weights or unweighted backbone
            resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        except Exception as e:
            logger.warning("Could not download ImageNet weights (%s). Initializing unweighted ResNet18.", e)
            resnet = models.resnet18(weights=None)

        # Replace classification layer with Identity to extract 512-dim pooled vector
        resnet.fc = nn.Identity()

        if freeze_weights:
            for param in resnet.parameters():
                param.requires_grad = False

        resnet.eval()
        resnet.to(self.device)
        return resnet

    def preprocess_image(self, image_path: Union[str, Path]) -> torch.Tensor:
        """Load and transform image file into normalized (3, 224, 224) tensor."""
        path = Path(image_path)
        if path.exists():
            try:
                img = Image.open(path).convert("RGB")
                return self.transform(img)
            except Exception as e:
                logger.warning("Error reading image %s: %s. Using synthetic fallback.", path, e)

        # Deterministic fallback tensor seeded by file path string
        return self._generate_fallback_tensor(path.stem)

    def _generate_fallback_tensor(self, seed_key: str) -> torch.Tensor:
        """Create a deterministic pseudo-image tensor for articles with missing images."""
        h = int(hashlib.sha256(seed_key.encode("utf-8")).hexdigest()[:8], 16)
        rng = torch.Generator().manual_seed(h)
        return torch.randn(3, self.image_size, self.image_size, generator=rng)

    @torch.no_grad()
    def extract_embedding(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """Extract 512-dim feature embedding from an image tensor.

        Args:
            image_tensor: (3, 224, 224) or (B, 3, 224, 224)

        Returns:
            (512,) or (B, 512) feature tensor.
        """
        is_single = (image_tensor.dim() == 3)
        if is_single:
            image_tensor = image_tensor.unsqueeze(0)

        image_tensor = image_tensor.to(self.device)
        embedding = self.model(image_tensor)

        if is_single:
            embedding = embedding.squeeze(0)

        return embedding.cpu()
