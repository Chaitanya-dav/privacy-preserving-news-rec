"""Federated Learning package for privacy-preserving news recommendation."""

from .partitioner import FederatedDataPartitioner
from .client import FlowerNewsClient, get_model_parameters, set_model_parameters
from .server import FederatedServer, fedavg_aggregate, compute_param_delta_norm

__all__ = [
    "FederatedDataPartitioner",
    "FlowerNewsClient",
    "FederatedServer",
    "fedavg_aggregate",
    "compute_param_delta_norm",
    "get_model_parameters",
    "set_model_parameters",
]
