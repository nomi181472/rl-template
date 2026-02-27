"""algorithms/factory.py — algorithm registry with relative module resolution."""
from __future__ import annotations
import importlib
from .base import BaseAlgorithm

_REGISTRY: dict[str, str] = {
    "ppo":  ".ppo:PPOAlgorithm",
    "sac":  ".sac:SACAlgorithm",
    # "td3":  ".td3:TD3Algorithm",
    # "ddpg": ".ddpg:DDPGAlgorithm",
}


def register_algorithm(name: str, module_path: str):
    """
    Register a custom algorithm.
    External: register_algorithm("dreamer", "my_pkg.dreamer:DreamerAlgorithm")
    Built-in: register_algorithm("td3", ".td3:TD3Algorithm")
    """
    _REGISTRY[name.lower()] = module_path


def list_algorithms() -> list[str]:
    return sorted(_REGISTRY.keys())


def make_algorithm(algo_cfg, obs_space, act_space, network_cfg=None) -> BaseAlgorithm:
    name = algo_cfg.name.lower()
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown algorithm '{name}'. "
            f"Available: {list_algorithms()}. "
            f"Register custom ones with register_algorithm()."
        )
    path = _REGISTRY[name]
    mod_path, cls_name = path.rsplit(":", 1)

    if mod_path.startswith("."):
        pkg      = __name__.rsplit(".", 1)[0]   # "rl_v2.algorithms"
        mod_path = pkg + mod_path

    try:
        cls = getattr(importlib.import_module(mod_path), cls_name)
    except (ImportError, AttributeError) as e:
        raise ImportError(
            f"Cannot load algorithm '{name}' from '{mod_path}:{cls_name}'.\nError: {e}"
        ) from e
    return cls(algo_cfg, obs_space, act_space, network_cfg)
