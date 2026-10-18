"""MIND News & Behavior Log Parser with Vocabularies & Negative Sampling.

Parses raw MIND TSV files (news.tsv and behaviors.tsv), constructs
bidirectional entity/category dictionaries, formats user click histories
with padding, and generates balanced negative samples for training.
"""

import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class MINDParser:
    """Parses and manages MIND dataset files, vocabularies, and user interactions."""

    NEWS_COLUMNS = [
        "news_id",
        "category",
        "subcategory",
        "title",
        "abstract",
        "url",
        "title_entities",
        "abstract_entities",
    ]

    BEHAVIOR_COLUMNS = [
        "impression_id",
        "user_id",
        "time",
        "history",
        "impressions",
    ]

    def __init__(
        self,
        max_history_len: int = 50,
        negative_sample_ratio: int = 4,
        seed: int = 42,
    ):
        self.max_history_len = max_history_len
        self.negative_sample_ratio = negative_sample_ratio
        self.seed = seed
        self.rng = random.Random(seed)

        # Vocabularies and mappings (0 reserved for PAD)
        self.news2idx: Dict[str, int] = {"<PAD>": 0}
        self.idx2news: Dict[int, str] = {0: "<PAD>"}
        self.user2idx: Dict[str, int] = {"<PAD>": 0}
        self.idx2user: Dict[int, str] = {0: "<PAD>"}
        self.category2idx: Dict[str, int] = {"<PAD>": 0}
        self.subcategory2idx: Dict[str, int] = {"<PAD>": 0}

        # News data cache
        self.news_df: Optional[pd.DataFrame] = None

    def parse_news(self, news_path: Union[str, Path]) -> pd.DataFrame:
        """Parse news.tsv and update news, category, and subcategory vocabularies.

        Args:
            news_path: Path to news.tsv

        Returns:
            Cleaned pandas DataFrame with added index columns.
        """
        path = Path(news_path)
        if not path.exists():
            raise FileNotFoundError(f"news.tsv not found at {path}")

        df = pd.read_csv(
            path,
            sep="\t",
            header=None,
            names=self.NEWS_COLUMNS,
            dtype=str,
            quoting=3,  # QUOTE_NONE
        ).fillna("")

        # Populate vocabularies
        for _, row in df.iterrows():
            nid = row["news_id"]
            if nid not in self.news2idx:
                idx = len(self.news2idx)
                self.news2idx[nid] = idx
                self.idx2news[idx] = nid

            cat = row["category"].strip().lower()
            if cat and cat not in self.category2idx:
                self.category2idx[cat] = len(self.category2idx)

            subcat = row["subcategory"].strip().lower()
            if subcat and subcat not in self.subcategory2idx:
                self.subcategory2idx[subcat] = len(self.subcategory2idx)

        # Add integer index columns
        df["news_idx"] = df["news_id"].map(self.news2idx)
        df["category_idx"] = df["category"].str.lower().map(self.category2idx).fillna(0).astype(int)
        df["subcategory_idx"] = df["subcategory"].str.lower().map(self.subcategory2idx).fillna(0).astype(int)

        self.news_df = df
        logger.info(
            "Parsed news.tsv: %d articles, %d categories, %d subcategories.",
            len(df),
            len(self.category2idx) - 1,
            len(self.subcategory2idx) - 1,
        )
        return df

    def parse_history(self, history_str: str) -> List[int]:
        """Convert space-separated history news IDs into padded integer index sequence."""
        if not history_str or not isinstance(history_str, str):
            return [0] * self.max_history_len

        tokens = history_str.strip().split()
        indices = [self.news2idx[t] for t in tokens if t in self.news2idx]

        # Truncate to most recent articles if exceeding max_history_len
        if len(indices) > self.max_history_len:
            indices = indices[-self.max_history_len :]
        # Left-pad with 0 (<PAD>) if shorter
        elif len(indices) < self.max_history_len:
            indices = [0] * (self.max_history_len - len(indices)) + indices

        return indices

    def parse_behaviors(
        self,
        behaviors_path: Union[str, Path],
        is_train: bool = True,
    ) -> List[Dict[str, Any]]:
        """Parse behaviors.tsv and build training or evaluation sample records.

        Args:
            behaviors_path: Path to behaviors.tsv
            is_train: If True, applies negative sampling (1:4 ratio);
                      if False, keeps all candidate impressions with ground truth for ranking metrics.

        Returns:
            List of parsed interaction dictionaries.
        """
        path = Path(behaviors_path)
        if not path.exists():
            raise FileNotFoundError(f"behaviors.tsv not found at {path}")

        df = pd.read_csv(
            path,
            sep="\t",
            header=None,
            names=self.BEHAVIOR_COLUMNS,
            dtype=str,
            quoting=3,
        ).fillna("")

        samples = []
        all_news_indices = [idx for idx in self.idx2news if idx != 0]

        for _, row in df.iterrows():
            uid = row["user_id"]
            if uid not in self.user2idx:
                self.user2idx[uid] = len(self.user2idx)
                self.idx2user[len(self.idx2user)] = uid
            user_idx = self.user2idx[uid]

            history_indices = self.parse_history(row["history"])
            imp_items = row["impressions"].strip().split()

            if is_train:
                # Separate clicked (1) and unclicked (0) items
                pos_candidates = []
                neg_candidates = []

                for item in imp_items:
                    if "-" not in item:
                        continue
                    nid, label = item.rsplit("-", 1)
                    if nid in self.news2idx:
                        c_idx = self.news2idx[nid]
                        if label == "1":
                            pos_candidates.append(c_idx)
                        elif label == "0":
                            neg_candidates.append(c_idx)

                # For each positive click, sample K negative articles
                for pos_idx in pos_candidates:
                    # 1. Pointwise positive sample
                    samples.append({
                        "user_idx": user_idx,
                        "history_indices": history_indices,
                        "candidate_idx": pos_idx,
                        "label": 1.0,
                    })

                    # 2. Sample negative articles
                    if len(neg_candidates) >= self.negative_sample_ratio:
                        sampled_negs = self.rng.sample(neg_candidates, k=self.negative_sample_ratio)
                    elif len(neg_candidates) > 0:
                        sampled_negs = self.rng.choices(neg_candidates, k=self.negative_sample_ratio)
                    else:
                        # Fallback random negatives if no negatives in session
                        sampled_negs = self.rng.sample(
                            [i for i in all_news_indices if i != pos_idx],
                            k=min(self.negative_sample_ratio, len(all_news_indices) - 1),
                        )

                    for neg_idx in sampled_negs:
                        samples.append({
                            "user_idx": user_idx,
                            "history_indices": history_indices,
                            "candidate_idx": neg_idx,
                            "label": 0.0,
                        })
            else:
                # Evaluation mode: retain full candidate list and binary labels for ranking evaluation
                candidate_indices = []
                labels = []

                for item in imp_items:
                    if "-" not in item:
                        continue
                    nid, label = item.rsplit("-", 1)
                    if nid in self.news2idx:
                        candidate_indices.append(self.news2idx[nid])
                        labels.append(float(label))

                if candidate_indices:
                    samples.append({
                        "impression_id": int(row["impression_id"]) if row["impression_id"].isdigit() else len(samples) + 1,
                        "user_idx": user_idx,
                        "history_indices": history_indices,
                        "candidate_indices": candidate_indices,
                        "labels": labels,
                    })

        logger.info(
            "Parsed behaviors.tsv: %d rows -> %d %s samples.",
            len(df),
            len(samples),
            "train (sampled)" if is_train else "dev (sessions)",
        )
        return samples

    def save_processed(self, output_dir: Union[str, Path]) -> None:
        """Save parsed vocabularies and mappings to processed data directory."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        vocab_data = {
            "news2idx": self.news2idx,
            "category2idx": self.category2idx,
            "subcategory2idx": self.subcategory2idx,
            "user2idx": self.user2idx,
            "max_history_len": self.max_history_len,
            "negative_sample_ratio": self.negative_sample_ratio,
        }

        with open(out / "vocabularies.json", "w", encoding="utf-8") as f:
            json.dump(vocab_data, f, indent=2)

        if self.news_df is not None:
            self.news_df.to_csv(out / "news_parsed.tsv", sep="\t", index=False)

        print(f"Saved processed vocabularies and parsed news to {out}")

    def load_processed(self, input_dir: Union[str, Path]) -> None:
        """Load pre-parsed vocabularies from processed directory."""
        inp = Path(input_dir)
        vocab_file = inp / "vocabularies.json"
        if not vocab_file.exists():
            raise FileNotFoundError(f"vocabularies.json not found in {inp}")

        with open(vocab_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.news2idx = data["news2idx"]
        self.idx2news = {v: k for k, v in self.news2idx.items()}
        self.category2idx = data["category2idx"]
        self.subcategory2idx = data["subcategory2idx"]
        self.user2idx = data["user2idx"]
        self.idx2user = {v: k for k, v in self.user2idx.items()}
        self.max_history_len = data.get("max_history_len", 50)
        self.negative_sample_ratio = data.get("negative_sample_ratio", 4)

        news_tsv = inp / "news_parsed.tsv"
        if news_tsv.exists():
            self.news_df = pd.read_csv(news_tsv, sep="\t", dtype=str).fillna("")
            self.news_df["news_idx"] = self.news_df["news_idx"].astype(int)
            self.news_df["category_idx"] = self.news_df["category_idx"].astype(int)
            self.news_df["subcategory_idx"] = self.news_df["subcategory_idx"].astype(int)

        logger.info("Loaded pre-parsed vocabularies: %d news, %d users.", len(self.news2idx), len(self.user2idx))
