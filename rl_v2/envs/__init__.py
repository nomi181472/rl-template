from .base import BaseEnvAdapter
from .spaces import ObservationSpace, ActionSpace, SpaceType, ActionType
from .factory import make_env_adapter, register_adapter

# GymnasiumAdapter imported lazily to avoid requiring gymnasium at import time.
# Use make_env_adapter(cfg) instead of importing GymnasiumAdapter directly.
def _get_gymnasium_adapter():
    from .gymnasium_adapter import GymnasiumAdapter
    return GymnasiumAdapter

__all__ = [
    "BaseEnvAdapter",
    "ObservationSpace", "ActionSpace", "SpaceType", "ActionType",
    "make_env_adapter", "register_adapter",
]
