"""rl_v2 — algorithm, network, and env-backend agnostic RL framework."""

# These are all safe to import without torch/gymnasium installed:
from .configs    import Config, EnvConfig, LogConfig
from .envs.spaces import ObservationSpace, ActionSpace, SpaceType, ActionType
from .envs.factory import make_env_adapter, register_adapter
from .networks   import MLPConfig, CNNConfig, make_policy_network, make_value_network, make_twin_value_network, register_network
from .algorithms import make_algorithm, register_algorithm, list_algorithms, PPOConfig, SACConfig

# Trainer, Logger, Runner require torch — import explicitly when needed:
#   from rl_v2.loggers import RunLogger
#   from rl_v2.trainers import Trainer
#   from rl_v2.runners import InferenceRunner

__all__ = [
    # configs
    "Config", "EnvConfig", "LogConfig",
    # spaces
    "ObservationSpace", "ActionSpace", "SpaceType", "ActionType",
    # env
    "make_env_adapter", "register_adapter",
    # networks
    "MLPConfig", "CNNConfig",
    "make_policy_network", "make_value_network", "make_twin_value_network",
    "register_network",
    # algorithms
    "make_algorithm", "register_algorithm", "list_algorithms",
    "PPOConfig", "SACConfig",
]
