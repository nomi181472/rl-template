"""
envs/spaces.py
==============
Typed descriptors for observation and action spaces.

The environment is responsible for describing:
  - WHAT it observes  (ObservationSpace)
  - WHAT actions it accepts (ActionSpace)
  - HOW to pre-process raw obs into a network-ready tensor (preprocess)
  - HOW to post-process raw network output into an env action (postprocess)

This is the contract between the env and the network layer.
No env-specific code lives in the network. No network-specific code in the env.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import torch   # type hints only


# ══════════════════════════════════════════════════════════════════════
#  Enums
# ══════════════════════════════════════════════════════════════════════

class SpaceType(Enum):
    FLAT        = auto()   # 1-D float vector  (most envs)
    IMAGE       = auto()   # (C, H, W) pixels  (Atari, cameras)
    DICT        = auto()   # dict of tensors    (multi-modal: proprio + pixels)
    GRAPH       = auto()   # graph/point-cloud  (Isaac articulation state)
    SEQUENCE    = auto()   # variable-length    (NLP-style)


class ActionType(Enum):
    DISCRETE    = auto()   # integer index into N choices
    CONTINUOUS  = auto()   # float vector in [low, high]^D
    MULTI_DISCRETE = auto()# multiple discrete heads (e.g. multi-agent)
    HYBRID      = auto()   # mixed discrete + continuous (e.g. humanoid)


# ══════════════════════════════════════════════════════════════════════
#  ObservationSpace
# ══════════════════════════════════════════════════════════════════════

@dataclass
class ObservationSpace:
    """
    Describes what the environment outputs as observation,
    and how to transform it into a flat or structured tensor
    that the neural network can consume.

    Fields
    ------
    shape        : raw observation shape from the env
    dtype        : numpy dtype of raw obs
    space_type   : FLAT | IMAGE | DICT | GRAPH | SEQUENCE
    flat_dim     : pre-computed flat dimension (network input_dim)
    low / high   : bounds for normalisation (optional)
    components   : for DICT spaces, maps key -> sub-ObservationSpace
    extra        : backend-specific metadata

    Key method
    ----------
    preprocess(raw_obs) -> torch.Tensor
        Converts raw env output into a network-ready tensor.
        Override per subclass for custom preprocessing.
    """

    shape:       Tuple[int, ...]
    dtype:       np.dtype            = field(default_factory=lambda: np.float32)
    space_type:  SpaceType           = SpaceType.FLAT
    flat_dim:    int                 = 0          # filled in __post_init__
    low:         Optional[np.ndarray]= None
    high:        Optional[np.ndarray]= None
    components:  Dict[str, "ObservationSpace"] = field(default_factory=dict)
    extra:       Dict[str, Any]      = field(default_factory=dict)

    def __post_init__(self):
        if self.flat_dim == 0:
            self.flat_dim = int(np.prod(self.shape))

    def preprocess(
        self,
        raw_obs: Union[np.ndarray, dict],
        device:  str = "cpu",
        normalise: bool = False,
    ) -> "torch.Tensor":
        import torch  # lazy import — only needed at runtime
        """
        Transform raw env observation → network-ready tensor.

        Override this in subclasses or env adapters for:
          - image normalisation  (÷255, mean-std)
          - proprioception concatenation
          - graph feature extraction
          - any domain-specific transforms

        Default: flatten to (1, flat_dim) float32 tensor.
        """
        if isinstance(raw_obs, dict):
            # DICT space: concatenate all components
            parts = [
                self.components[k].preprocess(raw_obs[k], device, normalise).squeeze(0)
                for k in sorted(self.components.keys())
            ]
            tensor = torch.cat(parts, dim=0).unsqueeze(0)
        else:
            tensor = torch.tensor(
                np.asarray(raw_obs, dtype=np.float32).flatten(),
                dtype=torch.float32,
                device=device,
            ).unsqueeze(0)   # (1, flat_dim)

        if normalise and self.low is not None and self.high is not None:
            low  = torch.tensor(self.low.flatten(),  dtype=torch.float32, device=device)
            high = torch.tensor(self.high.flatten(), dtype=torch.float32, device=device)
            rng  = (high - low).clamp(min=1e-8)
            tensor = (tensor - low) / rng * 2.0 - 1.0   # → [-1, 1]

        return tensor   # shape: (1, flat_dim)

    @property
    def network_input_dim(self) -> int:
        """Flat dimension the network's first layer should accept."""
        if self.space_type == SpaceType.DICT:
            return sum(s.flat_dim for s in self.components.values())
        if self.space_type == SpaceType.IMAGE:
            return int(np.prod(self.shape))   # CNN expects shape, but flat_dim for MLP
        return self.flat_dim

    def __repr__(self) -> str:
        return (f"ObservationSpace(shape={self.shape}, type={self.space_type.name}, "
                f"flat_dim={self.flat_dim})")


# ══════════════════════════════════════════════════════════════════════
#  ActionSpace
# ══════════════════════════════════════════════════════════════════════

@dataclass
class ActionSpace:
    """
    Describes the action space and defines how to convert raw
    network output into a valid env action.

    Fields
    ------
    action_type  : DISCRETE | CONTINUOUS | MULTI_DISCRETE | HYBRID
    n            : number of discrete choices  (DISCRETE only)
    shape        : action vector shape         (CONTINUOUS only)
    low / high   : action bounds               (CONTINUOUS only)
    components   : sub-spaces for HYBRID/MULTI_DISCRETE

    Key method
    ----------
    postprocess(raw_network_output) -> env_action
        Converts network output (logits or floats) into a valid env action.
        Override for custom squashing, clipping, or multi-head decoding.
    """

    action_type: ActionType
    # Discrete
    n:           int                  = 0
    # Continuous
    shape:       Tuple[int, ...]      = field(default_factory=tuple)
    low:         Optional[np.ndarray] = None
    high:        Optional[np.ndarray] = None
    # Multi / Hybrid
    components:  List["ActionSpace"]  = field(default_factory=list)
    extra:       Dict[str, Any]       = field(default_factory=dict)

    @property
    def is_discrete(self) -> bool:
        return self.action_type == ActionType.DISCRETE

    @property
    def is_continuous(self) -> bool:
        return self.action_type == ActionType.CONTINUOUS

    @property
    def network_output_dim(self) -> int:
        """
        Flat dimension the network's last layer should output.
        For continuous: action_dim
        For discrete:   n  (logits)
        For hybrid:     sum of sub-dims
        """
        if self.action_type == ActionType.DISCRETE:
            return self.n
        if self.action_type == ActionType.CONTINUOUS:
            return int(np.prod(self.shape))
        if self.action_type in (ActionType.MULTI_DISCRETE, ActionType.HYBRID):
            return sum(c.network_output_dim for c in self.components)
        return 0

    def postprocess(
        self,
        raw_output: Union[np.ndarray, int, "torch.Tensor"],
        clip: bool = True,
    ) -> Any:
        """
        Convert raw network output → valid env action.

        Default behaviour:
          DISCRETE:   return int (argmax or sampled index)
          CONTINUOUS: clip to [low, high] if clip=True
          HYBRID:     split + postprocess each component

        Override in subclasses or adapters for custom transforms
        (e.g. rescale from [-1,1] to [low, high]).
        """
        import torch; 
        if isinstance(raw_output, torch.Tensor):
            raw_output = raw_output.detach().cpu().numpy()

        if self.action_type == ActionType.DISCRETE:
            return int(raw_output)

        if self.action_type == ActionType.CONTINUOUS:
            action = raw_output.astype(np.float32)
            if clip and self.low is not None and self.high is not None:
                action = np.clip(action, self.low, self.high)
            return action

        if self.action_type == ActionType.MULTI_DISCRETE:
            # Split flat output into per-head chunks
            idx, result = 0, []
            for comp in self.components:
                dim = comp.network_output_dim
                result.append(comp.postprocess(raw_output[idx:idx+dim]))
                idx += dim
            return result

        return raw_output

    def sample(self) -> Any:
        """Random action for warmup / exploration baseline."""
        if self.action_type == ActionType.DISCRETE:
            return np.random.randint(0, self.n)
        if self.action_type == ActionType.CONTINUOUS:
            lo = self.low  if self.low  is not None else -1.0
            hi = self.high if self.high is not None else  1.0
            return np.random.uniform(lo, hi, self.shape).astype(np.float32)
        return None

    def __repr__(self) -> str:
        if self.action_type == ActionType.DISCRETE:
            return f"ActionSpace(DISCRETE, n={self.n})"
        return f"ActionSpace(CONTINUOUS, shape={self.shape}, low={self.low}, high={self.high})"
