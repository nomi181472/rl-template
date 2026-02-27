# Config dataclasses — no torch dependency, always safe to import
from ._configs import MLPConfig, CNNConfig
from .factory import (
    make_policy_network,
    make_value_network,
    make_twin_value_network,
    register_network,
)

__all__ = [
    "MLPConfig", "CNNConfig",
    "make_policy_network", "make_value_network", "make_twin_value_network",
    "register_network",
]
