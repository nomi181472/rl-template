"""
envs/base.py
============
Abstract base class for all environment adapters.

Key design change from v1
-------------------------
The env is now responsible for:
  1. Declaring its ObservationSpace  →  tells network what input shape to expect
  2. Declaring its ActionSpace       →  tells network what output shape to build
  3. get_observation(raw, device)    →  converts raw obs → network-ready tensor
  4. get_action(raw_output)          →  converts network output → valid env action

This means:
  - Networks are built purely from obs_space and action_space descriptors
  - No env-specific preprocessing scattered in algorithm code
  - Swapping Isaac ↔ Gymnasium ↔ OmniVerse doesn't touch algorithm or network code
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple, TYPE_CHECKING

import numpy as np
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import torch  # only needed for type hints; real import deferred to runtime

from .spaces import ObservationSpace, ActionSpace

if TYPE_CHECKING:
    pass


class BaseEnvAdapter(ABC):
    """
    Unified environment interface.

    Subclasses must implement
    -------------------------
    setup()             → build/connect env, return (ObservationSpace, ActionSpace)
    reset()             → return raw observation
    step(action)        → return (raw_obs, reward, done, info)
    close()             → shut down

    Plus the observation / action bridge:
    get_observation()   → raw obs → network tensor    (can override for custom preproc)
    get_action()        → network output → env action (can override for custom postproc)

    Optionally override
    --------------------
    start_recording(path) / stop_recording()
    render() / seed() / get_state()
    make_torchrl_env()   — required for on-policy TorchRL collector
    """

    def __init__(self, cfg):
        self.cfg          = cfg
        self._obs_space:  Optional[ObservationSpace] = None
        self._act_space:  Optional[ActionSpace]      = None

    # ──────────────────────────────────────────────────────────────────
    # Required: lifecycle
    # ──────────────────────────────────────────────────────────────────

    @abstractmethod
    def setup(self) -> Tuple[ObservationSpace, ActionSpace]:
        """
        Initialise the backend, build the env/scene.

        Returns
        -------
        (ObservationSpace, ActionSpace)
            The network builder reads these to construct the right architecture.

        Side effect
        -----------
        Must call self._set_spaces(obs_space, act_space) before returning.
        """

    @abstractmethod
    def reset(self) -> np.ndarray:
        """Reset to initial state. Returns raw observation (numpy)."""

    @abstractmethod
    def step(self, action: Any) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Apply action, advance one step.
        Returns (raw_obs, reward, done, info).
        Action is already a valid env action (postprocessed by get_action).
        """

    @abstractmethod
    def close(self):
        """Cleanly shut down the environment."""

    # ──────────────────────────────────────────────────────────────────
    # Observation bridge: raw env output → network input
    # ──────────────────────────────────────────────────────────────────

    def get_observation(
        self,
        raw_obs: Any,
        device:  str  = "cpu",
        normalise: bool = False,
    ) -> "torch.Tensor":
        """
        Convert raw observation from the environment into a
        network-ready tensor.

        Default: delegates to obs_space.preprocess().
        Override for custom preprocessing (e.g. frame stacking,
        point-cloud encoding, sensor fusion).

        Returns
        -------
        torch.Tensor of shape (1, obs_space.network_input_dim)
        """
        return self._obs_space.preprocess(raw_obs, device=device, normalise=normalise)

    # ──────────────────────────────────────────────────────────────────
    # Action bridge: network output → valid env action
    # ──────────────────────────────────────────────────────────────────

    def get_action(
        self,
        raw_network_output: Any,
        clip: bool = True,
    ) -> Any:
        """
        Convert raw network output into a valid action for this env.

        Default: delegates to act_space.postprocess().
        Override for custom transforms (e.g. scaled joint torques,
        gripper open/close logic, composite action decoding).

        Parameters
        ----------
        raw_network_output : numpy array, int, or torch.Tensor
        clip               : whether to clip continuous actions to bounds

        Returns
        -------
        Action in the format the env's step() expects.
        """
        return self._act_space.postprocess(raw_network_output, clip=clip)

    # ──────────────────────────────────────────────────────────────────
    # Space accessors
    # ──────────────────────────────────────────────────────────────────

    @property
    def obs_space(self) -> ObservationSpace:
        if self._obs_space is None:
            raise RuntimeError("Call setup() before accessing obs_space.")
        return self._obs_space

    @property
    def act_space(self) -> ActionSpace:
        if self._act_space is None:
            raise RuntimeError("Call setup() before accessing act_space.")
        return self._act_space

    def _set_spaces(self, obs_space: ObservationSpace, act_space: ActionSpace):
        self._obs_space = obs_space
        self._act_space = act_space

    # ──────────────────────────────────────────────────────────────────
    # Optional overrides
    # ──────────────────────────────────────────────────────────────────

    def start_recording(self, save_path: str):
        """Start capturing video/frames."""

    def stop_recording(self) -> Optional[str]:
        """Stop capture. Return path to saved file or None."""
        return None

    def seed(self, seed: int):
        """Set random seed."""

    def render(self) -> Optional[np.ndarray]:
        """Return RGB frame (H×W×3 uint8) or None."""
        return None

    def get_state(self) -> Dict[str, Any]:
        """Return simulator state for checkpointing/replay."""
        return {}

    def make_torchrl_env(self):
        """
        Return a TorchRL-wrapped version for SyncDataCollector.
        Required by on-policy algorithms (PPO).
        Override in each adapter.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement make_torchrl_env() "
            "to be used with on-policy algorithms."
        )

    # ──────────────────────────────────────────────────────────────────
    # Convenience
    # ──────────────────────────────────────────────────────────────────

    def random_action(self) -> Any:
        """Sample a random action (useful for warmup)."""
        return self._act_space.sample()

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}("
                f"env={getattr(self.cfg,'env_name','?')}, "
                f"obs={self._obs_space}, "
                f"act={self._act_space})")
