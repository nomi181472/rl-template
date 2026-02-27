"""envs/factory.py — env backend registry with relative module resolution."""
from __future__ import annotations
import importlib
from .base import BaseEnvAdapter

# Built-in: ".module:Class"  External: "full.dotted.module:Class"
_REGISTRY: dict[str, str] = {
    "gymnasium":  ".gymnasium_adapter:GymnasiumAdapter",
    "isaac":      ".isaac_adapter:IsaacAdapter",
    "omniverse":  ".omniverse_adapter:OmniverseAdapter",
}


def register_adapter(name: str, module_path: str):
    """
    Register a custom env adapter.
    External packages: register_adapter("my_sim", "my_pkg.adapter:MyAdapter")
    Built-in inside rl_v2/envs/: register_adapter("my_sim", ".my_adapter:MyAdapter")
    """
    _REGISTRY[name.lower()] = module_path


def make_env_adapter(env_cfg) -> BaseEnvAdapter:
    name = env_cfg.env_backend
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown env backend '{name}'. "
            f"Available: {sorted(_REGISTRY)}. "
            f"Register custom ones with register_adapter()."
        )
    path = _REGISTRY[name]
    mod_path, cls_name = path.rsplit(":", 1)

    # Resolve relative path: ".gymnasium_adapter" → "rl_v2.envs.gymnasium_adapter"
    if mod_path.startswith("."):
        pkg      = __name__.rsplit(".", 1)[0]   # "rl_v2.envs"
        mod_path = pkg + mod_path

    try:
        cls = getattr(importlib.import_module(mod_path), cls_name)
    except (ImportError, AttributeError) as e:
        raise ImportError(
            f"Cannot load env adapter '{name}' from '{mod_path}:{cls_name}'.\n"
            f"Make sure the required simulator is installed.\nError: {e}"
        ) from e
    return cls(env_cfg)
