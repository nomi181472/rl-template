"""
networks/mlp.py
===============
Multi-Layer Perceptron policy and value networks.
Built entirely from ObservationSpace + ActionSpace — no env/algo code.

Supports:
  - Discrete action spaces   → logit head
  - Continuous action spaces → mean + log_std heads (for stochastic)
                             → single head (for deterministic TD3/DDPG)
  - Shared or separate trunk for actor/critic
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Type

import torch
import torch.nn as nn

from .base import BaseNetwork, BaseValueNetwork
from ..envs.spaces import ObservationSpace, ActionSpace, ActionType


# ══════════════════════════════════════════════════════════════════════
#  Config
# ══════════════════════════════════════════════════════════════════════

from ._configs import MLPConfig  # noqa: F401


# ══════════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════════

_ACTIVATIONS = {
    "relu":  nn.ReLU,
    "tanh":  nn.Tanh,
    "elu":   nn.ELU,
    "silu":  nn.SiLU,
    "gelu":  nn.GELU,
    "leaky": nn.LeakyReLU,
}


def _build_trunk(
    in_dim:       int,
    hidden_sizes: List[int],
    activation:   str,
    layer_norm:   bool,
    dropout:      float,
) -> nn.Sequential:
    """Build the shared hidden layers."""
    act_cls = _ACTIVATIONS.get(activation, nn.ReLU)
    layers  = []
    prev    = in_dim

    for h in hidden_sizes:
        layers.append(nn.Linear(prev, h))
        if layer_norm:
            layers.append(nn.LayerNorm(h))
        layers.append(act_cls())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        prev = h

    return nn.Sequential(*layers), prev   # returns (trunk, out_dim)


def _init_output(layer: nn.Linear, std: float = 0.01):
    nn.init.orthogonal_(layer.weight, gain=std)
    nn.init.constant_(layer.bias, 0.0)


# ══════════════════════════════════════════════════════════════════════
#  MLP Policy  (Actor)
# ══════════════════════════════════════════════════════════════════════

class MLPPolicy(BaseNetwork):
    """
    MLP policy network.

    Discrete → outputs logits  (batch, n_actions)
    Continuous stochastic → outputs (mean, log_std)  for SAC/PPO
    Continuous deterministic → outputs action directly  for TD3/DDPG
    """

    def __init__(
        self,
        obs_space:     ObservationSpace,
        act_space:     ActionSpace,
        cfg:           MLPConfig,
        deterministic: bool = False,
    ):
        super().__init__(obs_space, act_space, cfg)
        self.deterministic = deterministic

        trunk, trunk_out = _build_trunk(
            self.input_dim, cfg.hidden_sizes, cfg.activation,
            cfg.layer_norm, cfg.dropout
        )
        self.trunk = trunk

        if act_space.action_type == ActionType.DISCRETE:
            self.head = nn.Linear(trunk_out, act_space.n)
            _init_output(self.head, cfg.init_std)
            self._forward = self._forward_discrete

        elif act_space.action_type == ActionType.CONTINUOUS:
            if deterministic:
                # TD3 / DDPG style: single head + tanh
                self.head = nn.Sequential(
                    nn.Linear(trunk_out, act_space.network_output_dim),
                    nn.Tanh(),
                )
                _init_output(self.head[0], cfg.init_std)
                self._forward = self._forward_deterministic
            else:
                # SAC / PPO style: mean + log_std heads
                dim = act_space.network_output_dim
                self.mean_head    = nn.Linear(trunk_out, dim)
                self.log_std_head = nn.Linear(trunk_out, dim)
                _init_output(self.mean_head,    cfg.init_std)
                _init_output(self.log_std_head, cfg.init_std)
                self._forward = self._forward_stochastic

        else:
            raise NotImplementedError(
                f"MLPPolicy does not support action_type={act_space.action_type}. "
                "Use a custom network for HYBRID / MULTI_DISCRETE."
            )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self._forward(obs)

    def _forward_discrete(self, obs: torch.Tensor) -> torch.Tensor:
        return self.head(self.trunk(obs))   # (batch, n_actions)

    def _forward_deterministic(self, obs: torch.Tensor) -> torch.Tensor:
        return self.head(self.trunk(obs))   # (batch, act_dim)  — tanh squashed

    def _forward_stochastic(self, obs: torch.Tensor) -> torch.Tensor:
        h       = self.trunk(obs)
        mean    = self.mean_head(h)
        log_std = self.log_std_head(h).clamp(-20, 2)
        return torch.cat([mean, log_std], dim=-1)   # (batch, 2*act_dim)

    @classmethod
    def build(
        cls,
        obs_space:     ObservationSpace,
        act_space:     ActionSpace,
        cfg:           MLPConfig,
        deterministic: bool = False,
    ) -> "MLPPolicy":
        return cls(obs_space, act_space, cfg, deterministic=deterministic)


# ══════════════════════════════════════════════════════════════════════
#  MLP Value  (Critic)
# ══════════════════════════════════════════════════════════════════════

class MLPValue(BaseValueNetwork):
    """
    MLP value / critic network.
    Outputs a scalar state-value V(s) or action-value Q(s, a).
    """

    def __init__(
        self,
        obs_space:   ObservationSpace,
        cfg:         MLPConfig,
        include_action: bool = False,   # True for Q-networks (SAC, TD3, DDPG)
        action_dim:  int     = 0,
    ):
        super().__init__(obs_space, cfg)
        self.include_action = include_action
        in_dim = self.input_dim + (action_dim if include_action else 0)

        trunk, trunk_out = _build_trunk(
            in_dim, cfg.hidden_sizes, cfg.activation,
            cfg.layer_norm, cfg.dropout
        )
        self.trunk = trunk
        self.head  = nn.Linear(trunk_out, 1)
        _init_output(self.head, 1.0)

    def forward(
        self,
        obs:    torch.Tensor,
        action: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        obs    : (batch, obs_dim)
        action : (batch, act_dim)  — required when include_action=True
        """
        if self.include_action:
            if action is None:
                raise ValueError("MLPValue with include_action=True requires action input.")
            x = torch.cat([obs, action], dim=-1)
        else:
            x = obs
        return self.head(self.trunk(x))   # (batch, 1)

    @classmethod
    def build(
        cls,
        obs_space:      ObservationSpace,
        cfg:            MLPConfig,
        include_action: bool = False,
        action_dim:     int  = 0,
    ) -> "MLPValue":
        return cls(obs_space, cfg, include_action=include_action, action_dim=action_dim)


# ══════════════════════════════════════════════════════════════════════
#  Twin MLP Value  (for SAC / TD3)
# ══════════════════════════════════════════════════════════════════════

class TwinMLPValue(nn.Module):
    """Two independent Q-networks. Reduces overestimation bias."""

    def __init__(self, obs_space: ObservationSpace, act_space: ActionSpace, cfg: MLPConfig):
        super().__init__()
        act_dim  = act_space.network_output_dim
        self.q1  = MLPValue.build(obs_space, cfg, include_action=True, action_dim=act_dim)
        self.q2  = MLPValue.build(obs_space, cfg, include_action=True, action_dim=act_dim)

    def forward(self, obs: torch.Tensor, action: torch.Tensor):
        return self.q1(obs, action), self.q2(obs, action)

    def q1_only(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.q1(obs, action)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @classmethod
    def build(cls, obs_space, act_space, cfg) -> "TwinMLPValue":
        return cls(obs_space, act_space, cfg)
