"""
networks/base.py
================
Abstract base for all neural network architectures.

Design contract
---------------
Networks are built ONLY from:
  - ObservationSpace  (tells them what input they receive)
  - ActionSpace       (tells them what output to produce)
  - NetworkConfig     (hyperparams: hidden sizes, activation, etc.)

Networks know NOTHING about:
  - Which algorithm uses them
  - Which environment they run in
  - How observations are collected
  - Training details

This makes networks freely swappable:
  MLP ↔ CNN ↔ LSTM ↔ Transformer — without touching algorithm or env code.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple, TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from ..envs.spaces import ObservationSpace, ActionSpace


class BaseNetwork(ABC, nn.Module):
    """
    Base class for all network architectures.

    Subclasses implement
    --------------------
    forward(obs_tensor) -> output_tensor
        obs_tensor : (batch, *obs_shape)  — already preprocessed by env
        output     : (batch, output_dim)  — raw logits or values

    build(obs_space, act_space, cfg) -> BaseNetwork
        Class method. Constructs the network from space descriptors.
    """

    def __init__(
        self,
        obs_space:  "ObservationSpace",
        act_space:  "ActionSpace",
        cfg:        Any,
    ):
        super().__init__()
        self.obs_space  = obs_space
        self.act_space  = act_space
        self.cfg        = cfg
        self.input_dim  = obs_space.network_input_dim
        self.output_dim = act_space.network_output_dim

    @abstractmethod
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        obs : torch.Tensor  shape (batch, input_dim)
              Already preprocessed by env.get_observation().

        Returns
        -------
        torch.Tensor  shape (batch, output_dim)
        """

    @classmethod
    @abstractmethod
    def build(
        cls,
        obs_space: "ObservationSpace",
        act_space: "ActionSpace",
        cfg:       Any,
    ) -> "BaseNetwork":
        """Construct the network from space descriptors and config."""

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}("
                f"in={self.input_dim}, out={self.output_dim}, "
                f"params={self.num_parameters():,})")


class BaseValueNetwork(ABC, nn.Module):
    """
    Base class for critic / value networks.
    Outputs a scalar value estimate given an observation.
    """

    def __init__(self, obs_space: "ObservationSpace", cfg: Any):
        super().__init__()
        self.obs_space = obs_space
        self.cfg       = cfg
        self.input_dim = obs_space.network_input_dim

    @abstractmethod
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        obs : (batch, input_dim)

        Returns
        -------
        (batch, 1)
        """

    @classmethod
    @abstractmethod
    def build(cls, obs_space: "ObservationSpace", cfg: Any) -> "BaseValueNetwork":
        """Construct from space descriptor and config."""

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
