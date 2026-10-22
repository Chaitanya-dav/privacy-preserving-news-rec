"""Privacy-Preserving module: Secure Aggregation and Differential Privacy."""

from .secure_aggregation import (
    PairwiseMaskingSecAgg,
    ShamirSecretSharing,
    generate_pairwise_seed,
)
from .differential_privacy import (
    DifferentialPrivacyModule,
    clip_parameter_update,
    add_gaussian_noise,
)

__all__ = [
    "PairwiseMaskingSecAgg",
    "ShamirSecretSharing",
    "generate_pairwise_seed",
    "DifferentialPrivacyModule",
    "clip_parameter_update",
    "add_gaussian_noise",
]
