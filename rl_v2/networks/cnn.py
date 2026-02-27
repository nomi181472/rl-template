"""
networks/cnn.py
===============
Convolutional policy and value networks for image-based observations.

Same BaseNetwork interface as MLP — algorithms use them identically.
The only change is in the network config and build() call.

Supports:
  - Nature DQN CNN  (Atari-style: 3 conv layers + MLP head)
  - Custom conv configs

Usage
-----
# In your algorithm / PPO config:
# from rl_v2.networks.cnn import CNNPolicy, CNNConfig

network_cfg = CNNConfig(
    channels=[32, 64, 64],
    kernel_sizes=[8, 4, 3],
    strides=[4, 2, 1],
    mlp_hidden=[512],
)
actor = CNNPolicy.build(obs_space, act_space, network_cfg)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import numpy as np

from .base import BaseNetwork, BaseValueNetwork
from .mlp  import _build_trunk, _init_output, _ACTIVATIONS
from ..envs.spaces import ObservationSpace, ActionSpace, ActionType, SpaceType


from ._configs import CNNConfig  # noqa: F401


def _build_cnn(
    in_channels: int,
    image_shape: Tuple[int, int],
    cfg: CNNConfig,
) -> Tuple[nn.Sequential, int]:
    """Build conv layers and compute flat output dim."""
    layers  = []
    act_cls = _ACTIVATIONS.get(cfg.activation, nn.ReLU)
    prev_c  = in_channels
    H, W    = image_shape

    for out_c, k, s in zip(cfg.channels, cfg.kernel_sizes, cfg.strides):
        layers.append(nn.Conv2d(prev_c, out_c, kernel_size=k, stride=s))
        layers.append(act_cls())
        H = (H - k) // s + 1
        W = (W - k) // s + 1
        prev_c = out_c

    flat_dim = prev_c * H * W
    return nn.Sequential(*layers), flat_dim


class CNNPolicy(BaseNetwork):
    """
    CNN feature extractor + MLP policy head.
    Input:  (batch, C, H, W) image tensor  — from env.get_observation()
    Output: logits (discrete) or mean+log_std (continuous)
    """

    def __init__(
        self,
        obs_space:     ObservationSpace,
        act_space:     ActionSpace,
        cfg:           CNNConfig,
        deterministic: bool = False,
    ):
        super().__init__(obs_space, act_space, cfg)
        self.normalise = cfg.normalise_pixels

        # Infer input channels and spatial dims from obs_space
        shape = obs_space.shape   # (C, H, W) or (H, W, C)
        if len(shape) == 3:
            C, H, W = shape if shape[0] <= 4 else (shape[2], shape[0], shape[1])
        else:
            raise ValueError(f"CNNPolicy expects 3D image obs, got shape {shape}")

        self.conv, flat_dim = _build_cnn(C, (H, W), cfg)

        # MLP trunk after flatten
        trunk, trunk_out = _build_trunk(
            flat_dim, cfg.mlp_hidden, cfg.activation, cfg.layer_norm, cfg.dropout
        )
        self.trunk = trunk

        # Policy head(s)
        if act_space.action_type == ActionType.DISCRETE:
            self.head = nn.Linear(trunk_out, act_space.n)
            _init_output(self.head, cfg.init_std)
        elif act_space.action_type == ActionType.CONTINUOUS:
            dim = act_space.network_output_dim
            self.mean_head    = nn.Linear(trunk_out, dim)
            self.log_std_head = nn.Linear(trunk_out, dim)
            _init_output(self.mean_head, cfg.init_std)
            _init_output(self.log_std_head, cfg.init_std)
        else:
            raise NotImplementedError(f"CNNPolicy: unsupported action_type {act_space.action_type}")

        self.act_space = act_space

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if self.normalise:
            obs = obs.float() / 255.0
        features = self.conv(obs).flatten(start_dim=1)
        h        = self.trunk(features)

        if self.act_space.action_type == ActionType.DISCRETE:
            return self.head(h)
        else:
            mean    = self.mean_head(h)
            log_std = self.log_std_head(h).clamp(-20, 2)
            return torch.cat([mean, log_std], dim=-1)

    @classmethod
    def build(
        cls,
        obs_space:     ObservationSpace,
        act_space:     ActionSpace,
        cfg:           CNNConfig,
        deterministic: bool = False,
    ) -> "CNNPolicy":
        return cls(obs_space, act_space, cfg, deterministic=deterministic)


class CNNValue(BaseValueNetwork):
    """CNN + MLP value network."""

    def __init__(
        self,
        obs_space:      ObservationSpace,
        cfg:            CNNConfig,
        include_action: bool = False,
        action_dim:     int  = 0,
    ):
        super().__init__(obs_space, cfg)
        self.normalise      = cfg.normalise_pixels
        self.include_action = include_action

        shape = obs_space.shape
        C, H, W = shape if shape[0] <= 4 else (shape[2], shape[0], shape[1])

        self.conv, flat_dim = _build_cnn(C, (H, W), cfg)
        in_dim = flat_dim + (action_dim if include_action else 0)

        trunk, trunk_out = _build_trunk(in_dim, cfg.mlp_hidden, cfg.activation,
                                        cfg.layer_norm, cfg.dropout)
        self.trunk = trunk
        self.head  = nn.Linear(trunk_out, 1)
        _init_output(self.head, 1.0)

    def forward(self, obs: torch.Tensor, action: Optional[torch.Tensor] = None):
        if self.normalise:
            obs = obs.float() / 255.0
        x = self.conv(obs).flatten(start_dim=1)
        if self.include_action and action is not None:
            x = torch.cat([x, action], dim=-1)
        return self.head(self.trunk(x))

    @classmethod
    def build(cls, obs_space, cfg, include_action=False, action_dim=0):
        return cls(obs_space, cfg, include_action, action_dim)
