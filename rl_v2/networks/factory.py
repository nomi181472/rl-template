"""
networks/factory.py
===================
Built-in entries use ".submodule:Class" (relative to this package).
External entries use "full.dotted.module:Class".
Both resolve correctly whether the package is pip-installed or a plain folder.
"""
from __future__ import annotations
import importlib
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from ..envs.spaces import ObservationSpace, ActionSpace

_POLICY_REGISTRY: dict[str, str] = {
    "mlp": ".mlp:MLPPolicy",
    "cnn": ".cnn:CNNPolicy",
}
_VALUE_REGISTRY: dict[str, str] = {
    "mlp": ".mlp:MLPValue",
    "cnn": ".cnn:CNNValue",
}
_TWIN_VALUE_REGISTRY: dict[str, str] = {
    "mlp": ".mlp:TwinMLPValue",
}


def register_network(name: str, policy_path: str,
                     value_path: str = None, twin_value_path: str = None):
    """
    Register a custom network architecture.
    Use full dotted path for external packages: "my_pkg.nets:MyPolicy"
    Use relative path for nets inside rl_v2/networks/:  ".my_net:MyPolicy"
    """
    _POLICY_REGISTRY[name.lower()] = policy_path
    if value_path:      _VALUE_REGISTRY[name.lower()]       = value_path
    if twin_value_path: _TWIN_VALUE_REGISTRY[name.lower()]  = twin_value_path


def _load_cls(registry: dict, name: str):
    if name not in registry:
        raise ValueError(
            f"Unknown network '{name}'. Available: {sorted(registry)}. "
            "Register custom ones with register_network()."
        )
    path = registry[name]
    mod_path, cls_name = path.rsplit(":", 1)

    # ".mlp" → "rl_v2.networks.mlp"  (works installed or as folder)
    if mod_path.startswith("."):
        pkg      = __name__.rsplit(".", 1)[0]   # "rl_v2.networks"
        mod_path = pkg + mod_path

    try:
        return getattr(importlib.import_module(mod_path), cls_name)
    except (ImportError, AttributeError) as e:
        raise ImportError(
            f"Cannot load network '{name}' from '{mod_path}:{cls_name}'. "
            f"Make sure the module is importable.\nOriginal error: {e}"
        ) from e


def make_policy_network(obs_space, act_space, network_cfg: Any,
                         deterministic: bool = False):
    """Build actor/policy network from space descriptors + config."""
    return _load_cls(_POLICY_REGISTRY, network_cfg.name).build(
        obs_space, act_space, network_cfg, deterministic=deterministic
    )


def make_value_network(obs_space, network_cfg: Any,
                        include_action: bool = False, action_dim: int = 0):
    """Build critic/value network. Set include_action=True for Q(s,a)."""
    return _load_cls(_VALUE_REGISTRY, network_cfg.name).build(
        obs_space, network_cfg,
        include_action=include_action, action_dim=action_dim
    )


def make_twin_value_network(obs_space, act_space, network_cfg: Any):
    """Build twin Q-networks for SAC / TD3."""
    return _load_cls(_TWIN_VALUE_REGISTRY, network_cfg.name).build(
        obs_space, act_space, network_cfg
    )
