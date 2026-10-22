"""Differential Privacy for Edge Federated Learning (Phase 14).

Implements update clipping, calibrated Gaussian noise injection,
and privacy budget (epsilon, delta) tracking for edge updates.
"""

import math
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def clip_parameter_update(
    update: List[np.ndarray],
    clip_norm: float = 1.0,
) -> Tuple[List[np.ndarray], float]:
    """Clip a list of parameter arrays by global L2-norm C.

    Args:
        update: List of parameter tensors (delta or weight arrays).
        clip_norm: Threshold C for clipping.

    Returns:
        clipped_update: Clipped parameter arrays.
        l2_norm: Measured pre-clipping L2 norm.
    """
    total_norm_sq = sum(float(np.sum(p ** 2)) for p in update)
    l2_norm = float(math.sqrt(total_norm_sq))

    if l2_norm > clip_norm and l2_norm > 0:
        factor = clip_norm / l2_norm
        clipped = [p * factor for p in update]
    else:
        clipped = [p.copy() for p in update]

    return clipped, l2_norm


def add_gaussian_noise(
    update: List[np.ndarray],
    sigma: float,
    seed: Optional[int] = None,
) -> List[np.ndarray]:
    """Inject zero-mean Gaussian noise N(0, sigma^2 * I) into parameter arrays."""
    if sigma <= 0:
        return update

    rng = np.random.RandomState(seed)
    noisy = []
    for p in update:
        noise = rng.normal(loc=0.0, scale=sigma, size=p.shape).astype(p.dtype)
        noisy.append(p + noise)
    return noisy


class DifferentialPrivacyModule:
    """Manages update clipping and calibrated Gaussian mechanism for edge federated updates."""

    def __init__(
        self,
        clip_norm: float = 1.0,
        noise_multiplier: float = 0.05,
        target_delta: float = 1e-5,
    ):
        self.clip_norm = clip_norm
        self.noise_multiplier = noise_multiplier
        self.target_delta = target_delta
        self.sigma = clip_norm * noise_multiplier

    def apply_dp_to_client_updates(
        self,
        client_updates: List[Tuple[List[np.ndarray], int]],
        seed: Optional[int] = None,
    ) -> List[Tuple[List[np.ndarray], int]]:
        """Clip each client's parameter update and inject calibrated Gaussian noise."""
        processed: List[Tuple[List[np.ndarray], int]] = []

        for i, (params, n_samples) in enumerate(client_updates):
            client_seed = (seed + i * 101) if seed is not None else None
            # 1. Clip norm
            clipped, _ = clip_parameter_update(params, clip_norm=self.clip_norm)
            # 2. Add Gaussian noise
            noisy = add_gaussian_noise(clipped, sigma=self.sigma, seed=client_seed)
            processed.append((noisy, n_samples))

        return processed

    def estimate_cumulative_epsilon(
        self,
        rounds: int,
        subsampling_ratio: float = 1.0,
    ) -> float:
        """Estimate cumulative privacy budget epsilon (RDP approximation) over R rounds."""
        if self.sigma <= 0:
            return float("inf")

        q = min(1.0, max(0.01, subsampling_ratio))
        # Standard composition approximation:
        # eps ~ q * sqrt(2 * rounds * ln(1/delta)) / noise_multiplier
        log_term = math.sqrt(2.0 * math.log(1.0 / self.target_delta))
        eps = (q * math.sqrt(rounds) * log_term) / max(self.noise_multiplier, 1e-6)
        return float(round(eps, 3))
