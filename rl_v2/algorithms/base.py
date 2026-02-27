"""
algorithms/base.py
==================
Abstract base for all RL algorithms.

Key change from v1
------------------
Algorithms no longer build networks internally.
They receive ObservationSpace + ActionSpace and use the network factory.
This means:
  - Swapping MLP ↔ CNN ↔ LSTM requires only a config change
  - Algorithm code never hard-codes layer sizes or activation functions
  - The same PPO can run on CartPole (MLP) and Atari (CNN) unchanged
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..envs.spaces import ObservationSpace, ActionSpace
    from ..envs.base import BaseEnvAdapter


MetricsDict = Dict[str, float]


class BaseAlgorithm(ABC):
    """
    Interface every RL algorithm implements.

    Constructor receives:
      algo_cfg    — algorithm hyperparams  (PPOConfig, SACConfig, etc.)
      obs_space   — from adapter.obs_space
      act_space   — from adapter.act_space
      network_cfg — architecture config    (MLPConfig, CNNConfig, etc.)

    The algorithm calls:
      make_policy_network(obs_space, act_space, network_cfg)
      make_value_network(obs_space, network_cfg, ...)
    to build its networks.

    During rollout the algorithm calls:
      adapter.get_observation(raw_obs)   → tensor  (preprocessing)
      adapter.get_action(raw_output)     → action  (postprocessing)
    """

    ALGORITHM_NAME: str  = "base"
    ON_POLICY:      bool = True

    def __init__(
        self,
        algo_cfg:    Any,
        obs_space:   "ObservationSpace",
        act_space:   "ActionSpace",
        network_cfg: Any,
    ):
        self.algo_cfg    = algo_cfg
        self.obs_space   = obs_space
        self.act_space   = act_space
        self.network_cfg = network_cfg
        self.device      = "cpu"
        self._step       = 0

    # ──────────────────────────────────────────────────────────────────
    # Required
    # ──────────────────────────────────────────────────────────────────

    @abstractmethod
    def setup(self, device: str = "cpu"):
        """Build networks, optimisers, buffers. Called once."""

    @abstractmethod
    def select_action(
        self,
        obs_tensor: "torch.Tensor",   # already preprocessed by adapter.get_observation()
        eval_mode:  bool = False,
    ) -> Tuple[Any, Optional[float]]:
        """
        Parameters
        ----------
        obs_tensor : network-ready tensor (1, input_dim) from adapter.get_observation()
        eval_mode  : True → deterministic/greedy

        Returns
        -------
        (raw_network_output, log_prob_or_None)
        The caller passes raw_network_output to adapter.get_action() to get env action.
        """

    @abstractmethod
    def update(self, batch: Any) -> MetricsDict:
        """Run one gradient update. Returns loggable metric dict."""

    @abstractmethod
    def collect_data(self, adapter: "BaseEnvAdapter", n_steps: int) -> Any:
        """Collect experience. On-policy: return rollout. Off-policy: fill buffer."""

    @abstractmethod
    def state_dict(self) -> Dict[str, Any]:
        """Serialisable state for checkpointing."""

    @abstractmethod
    def load_state_dict(self, state: Dict[str, Any]):
        """Restore from checkpoint."""

    # ──────────────────────────────────────────────────────────────────
    # Optional hooks
    # ──────────────────────────────────────────────────────────────────

    def pre_update(self): pass
    def post_update(self, metrics: MetricsDict): pass
    def on_episode_end(self, ep_reward: float, ep_steps: int): pass
    def train_mode(self): pass
    def eval_mode(self): pass
    def named_parameters(self): return []

    # ──────────────────────────────────────────────────────────────────
    # Persistence
    # ──────────────────────────────────────────────────────────────────

    def save(self, path: str):
        import torch, os
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({
            "algorithm":   self.ALGORITHM_NAME,
            "step":        self._step,
            "obs_space":   self.obs_space,
            "act_space":   self.act_space,
            "algo_cfg":    self.algo_cfg,
            "network_cfg": self.network_cfg,
            "state_dict":  self.state_dict(),
        }, path)

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "BaseAlgorithm":
        import torch
        data = torch.load(path, map_location=device)
        algo = cls(data["algo_cfg"], data["obs_space"],
                   data["act_space"], data["network_cfg"])
        algo.setup(device)
        algo.load_state_dict(data["state_dict"])
        algo._step = data.get("step", 0)
        return algo

    @property
    def total_steps(self) -> int:
        return self._step

    def _increment_steps(self, n: int = 1):
        self._step += n

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}("
                f"obs={self.obs_space.network_input_dim}, "
                f"act={self.act_space.network_output_dim}, "
                f"net={getattr(self.network_cfg,'name','?')}, "
                f"device={self.device})")
