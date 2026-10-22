"""Federated Edge Data Partitioner (Phase 10).

Partitions user interaction samples and dev sessions across simulated edge clients
using either uniform IID partitioning or Dirichlet-based Non-IID partitioning.
"""

import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


class FederatedDataPartitioner:
    """Partitions user behavioral data into edge client subsets for Federated Learning.

    Supported strategies:
      - 'user_uniform': IID partition where unique users are distributed uniformly across clients.
      - 'user_dirichlet': Non-IID partition using a Dirichlet distribution (Dir(alpha))
        to create realistic activity and preference skew across edge devices.
    """

    def __init__(
        self,
        train_samples: List[Dict[str, Any]],
        dev_sessions: Optional[List[Dict[str, Any]]] = None,
        num_clients: int = 5,
        strategy: str = "user_dirichlet",
        alpha: float = 0.5,
        seed: int = 42,
    ):
        self.train_samples = train_samples
        self.dev_sessions = dev_sessions or []
        self.num_clients = num_clients
        self.strategy = strategy
        self.alpha = alpha
        self.seed = seed
        self.rng = np.random.RandomState(seed)

        # Build user-indexed lookup tables
        self.user_to_train_indices: Dict[int, List[int]] = {}
        for idx, sample in enumerate(self.train_samples):
            u_id = sample.get("user_idx", 0)
            self.user_to_train_indices.setdefault(u_id, []).append(idx)

        self.unique_users = sorted(list(self.user_to_train_indices.keys()))
        self.num_users = len(self.unique_users)

        # Partition allocations: client_id -> list of sample indices
        self.client_train_indices: Dict[int, List[int]] = {c: [] for c in range(self.num_clients)}
        self.client_user_ids: Dict[int, List[int]] = {c: [] for c in range(self.num_clients)}
        self.client_dev_sessions: Dict[int, List[Dict[str, Any]]] = {c: [] for c in range(self.num_clients)}

        self._execute_partition()

    def _execute_partition(self) -> None:
        """Run the configured partitioning strategy."""
        if self.strategy == "user_uniform":
            self._partition_uniform()
        elif self.strategy == "user_dirichlet":
            self._partition_dirichlet()
        else:
            raise ValueError(f"Unknown partitioning strategy '{self.strategy}'. Use 'user_uniform' or 'user_dirichlet'.")

        self._partition_dev_sessions()

    def _partition_uniform(self) -> None:
        """Evenly divide shuffled unique users across edge clients (IID)."""
        shuffled_users = self.unique_users.copy()
        self.rng.shuffle(shuffled_users)

        for i, user in enumerate(shuffled_users):
            client_id = i % self.num_clients
            self.client_user_ids[client_id].append(user)
            self.client_train_indices[client_id].extend(self.user_to_train_indices[user])

    def _partition_dirichlet(self) -> None:
        """Allocate users to clients using a Dirichlet distribution (Non-IID)."""
        # Sort users by activity count to ensure heterogeneity across volume
        sorted_users = sorted(
            self.unique_users,
            key=lambda u: len(self.user_to_train_indices[u]),
            reverse=True,
        )

        # Draw multinomial probabilities from Dirichlet distribution
        proportions = self.rng.dirichlet(np.repeat(self.alpha, self.num_clients))
        # Ensure no client gets 0 probability
        proportions = np.clip(proportions, 0.05, None)
        proportions /= proportions.sum()

        client_buckets = {c: [] for c in range(self.num_clients)}
        assignments = self.rng.choice(
            self.num_clients,
            size=len(sorted_users),
            p=proportions,
        )

        # Ensure every client receives at least one user
        unassigned_clients = set(range(self.num_clients)) - set(assignments)
        if unassigned_clients:
            for c_empty in unassigned_clients:
                # Steal one user from the client with the most users
                fullest_client = max(range(self.num_clients), key=lambda c: np.sum(assignments == c))
                idxs = np.where(assignments == fullest_client)[0]
                assignments[idxs[0]] = c_empty

        for user, client_id in zip(sorted_users, assignments):
            self.client_user_ids[client_id].append(user)
            self.client_train_indices[client_id].extend(self.user_to_train_indices[user])

    def _partition_dev_sessions(self) -> None:
        """Partition evaluation sessions across clients."""
        if not self.dev_sessions:
            return

        # If sessions have user_idx, map to matching client; otherwise split round-robin
        for i, session in enumerate(self.dev_sessions):
            u_id = session.get("user_idx")
            assigned = False
            if u_id is not None:
                for c, u_list in self.client_user_ids.items():
                    if u_id in u_list:
                        self.client_dev_sessions[c].append(session)
                        assigned = True
                        break
            if not assigned:
                client_id = i % self.num_clients
                self.client_dev_sessions[client_id].append(session)

    def get_client_train_samples(self, client_id: int) -> List[Dict[str, Any]]:
        """Retrieve local training samples for a specific client."""
        indices = self.client_train_indices.get(client_id, [])
        return [self.train_samples[i] for i in indices]

    def get_client_dev_sessions(self, client_id: int) -> List[Dict[str, Any]]:
        """Retrieve local evaluation sessions for a specific client."""
        return self.client_dev_sessions.get(client_id, [])

    def get_partition_stats(self) -> Dict[str, Any]:
        """Compute statistical summary of the edge partition."""
        sample_counts = [len(self.client_train_indices[c]) for c in range(self.num_clients)]
        user_counts = [len(self.client_user_ids[c]) for c in range(self.num_clients)]
        dev_counts = [len(self.client_dev_sessions[c]) for c in range(self.num_clients)]

        max_samples = max(sample_counts) if sample_counts else 0
        min_samples = min(sample_counts) if sample_counts else 0
        imbalance_ratio = (max_samples / max(min_samples, 1))

        # Entropy of sample distribution (maximum entropy = uniform)
        probs = np.array(sample_counts, dtype=np.float64) / max(sum(sample_counts), 1)
        probs = probs[probs > 0]
        entropy = float(-np.sum(probs * np.log2(probs)))
        max_entropy = float(np.log2(self.num_clients))

        return {
            "num_clients": self.num_clients,
            "strategy": self.strategy,
            "alpha": self.alpha if self.strategy == "user_dirichlet" else None,
            "sample_counts": sample_counts,
            "user_counts": user_counts,
            "dev_session_counts": dev_counts,
            "total_train_samples": sum(sample_counts),
            "imbalance_ratio": round(imbalance_ratio, 3),
            "entropy": round(entropy, 3),
            "max_entropy": round(max_entropy, 3),
        }
