"""Secure Aggregation via Pairwise Additive Masking & Shamir Secret Sharing (Phase 13).

Implements the Bonawitz et al. Secure Aggregation protocol simulation:
1. Pairwise pseudo-random masking between all edge client pairs:
     w_tilde_u = w_u + sum_{v > u} M_{u,v} - sum_{v < u} M_{v,u}
2. Server aggregates masked vectors where all pairwise masks algebraically cancel to zero:
     sum_{u} w_tilde_u = sum_{u} w_u
3. Shamir (k-of-n) threshold secret sharing for surviving client dropout recovery.
"""

import hashlib
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def _prg_mask(seed: int, shape: Tuple[int, ...]) -> np.ndarray:
    """Deterministic Pseudorandom Generator (PRG) creating a zero-mean float32 mask."""
    rng = np.random.RandomState(seed % (2**32 - 1))
    return rng.normal(loc=0.0, scale=1.0, size=shape).astype(np.float32)


def generate_pairwise_seed(client_a: int, client_b: int, round_num: int = 1) -> int:
    """Generate shared symmetric key seed for client pair (a, b)."""
    low, high = min(client_a, client_b), max(client_a, client_b)
    key_str = f"secagg_pair_{low}_{high}_r{round_num}"
    h = hashlib.sha256(key_str.encode("utf-8")).hexdigest()[:8]
    return int(h, 16)


class PairwiseMaskingSecAgg:
    """Simulates Secure Aggregation with zero-sum pairwise additive masking."""

    def __init__(self, round_num: int = 1):
        self.round_num = round_num

    def mask_client_updates(
        self,
        client_updates: List[Tuple[List[np.ndarray], int]],
        client_ids: Optional[List[int]] = None,
    ) -> List[Tuple[List[np.ndarray], int]]:
        """Apply zero-sum pairwise masks across participating edge clients.

        Args:
            client_updates: List of (params_list, num_samples) per client.
            client_ids: Optional explicit client IDs (defaults to 0..K-1).

        Returns:
            List of (masked_params_list, num_samples) per client.
        """
        num_clients = len(client_updates)
        if num_clients < 2:
            return client_updates

        ids = client_ids or list(range(num_clients))
        num_layers = len(client_updates[0][0])

        # Precompute pairwise mask tensors for every pair (u, v) where u < v
        pairwise_masks: Dict[Tuple[int, int], List[np.ndarray]] = {}
        for i in range(num_clients):
            for j in range(i + 1, num_clients):
                u_id, v_id = ids[i], ids[j]
                seed = generate_pairwise_seed(u_id, v_id, round_num=self.round_num)
                masks = [
                    _prg_mask(seed + layer, client_updates[0][0][layer].shape)
                    for layer in range(num_layers)
                ]
                pairwise_masks[(u_id, v_id)] = masks

        # Apply masks: client u adds M_{u,v} if u < v, subtracts M_{v,u} if u > v
        masked_updates: List[Tuple[List[np.ndarray], int]] = []
        for i in range(num_clients):
            u_id = ids[i]
            params, n_samples = client_updates[i]
            masked_params = [p.copy() for p in params]

            for j in range(num_clients):
                if i == j:
                    continue
                v_id = ids[j]
                if u_id < v_id:
                    # u < v: add M_{u,v}
                    masks = pairwise_masks[(u_id, v_id)]
                    for layer in range(num_layers):
                        masked_params[layer] += masks[layer]
                else:
                    # u > v: subtract M_{v,u}
                    masks = pairwise_masks[(v_id, u_id)]
                    for layer in range(num_layers):
                        masked_params[layer] -= masks[layer]

            masked_updates.append((masked_params, n_samples))

        return masked_updates


# ─── Shamir (k-of-n) Secret Sharing ──────────────────────────────────────────

class ShamirSecretSharing:
    """Shamir's k-of-n threshold secret sharing over a finite prime field."""

    def __init__(self, threshold_k: int = 3, total_n: int = 5, prime: int = 104729):
        if threshold_k > total_n:
            raise ValueError(f"Threshold k ({threshold_k}) cannot exceed total n ({total_n}).")
        self.k = threshold_k
        self.n = total_n
        self.p = prime

    def split_secret(self, secret: int, rng: Optional[np.random.RandomState] = None) -> List[Tuple[int, int]]:
        """Split an integer secret into n shares using a degree-(k-1) random polynomial."""
        secret_mod = secret % self.p
        rng = rng or np.random.RandomState(42)

        # Coefficients: a_0 = secret, a_1..a_{k-1} random in [1, p-1]
        coeffs = [secret_mod] + [int(rng.randint(1, self.p)) for _ in range(self.k - 1)]

        shares: List[Tuple[int, int]] = []
        for x in range(1, self.n + 1):
            # Evaluate polynomial at x: f(x) = sum_j coeffs[j] * x^j mod p
            y = sum(c * (x ** j) for j, c in enumerate(coeffs)) % self.p
            shares.append((x, int(y)))

        return shares

    def reconstruct_secret(self, shares: List[Tuple[int, int]]) -> int:
        """Reconstruct the secret using Lagrange interpolation from any k shares."""
        if len(shares) < self.k:
            raise ValueError(f"Need at least {self.k} shares to reconstruct, got {len(shares)}.")

        k_shares = shares[: self.k]
        secret = 0

        for i, (x_i, y_i) in enumerate(k_shares):
            num = 1
            den = 1
            for j, (x_j, _) in enumerate(k_shares):
                if i == j:
                    continue
                num = (num * (-x_j)) % self.p
                den = (den * (x_i - x_j)) % self.p

            # Modular inverse of denominator: den^{-1} mod p
            inv_den = pow(den % self.p, self.p - 2, self.p)
            lagrange_l = (num * inv_den) % self.p
            secret = (secret + y_i * lagrange_l) % self.p

        return secret % self.p
