"""
envs/gymnasium_adapter.py
=========================
Gymnasium backend.
Builds ObservationSpace and ActionSpace from gym's native space objects.
Implements get_observation() with optional normalisation.
"""

from __future__ import annotations
import os
from typing import Any, Dict, Optional, Tuple

import numpy as np
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import torch  # type hints only
import gymnasium as gym
from gymnasium.wrappers import RecordVideo

from .base import BaseEnvAdapter
from .spaces import (
    ObservationSpace, ActionSpace,
    SpaceType, ActionType,
)


class GymnasiumAdapter(BaseEnvAdapter):
    """
    Wraps any Gymnasium-compatible environment.

    What it provides to the network layer
    --------------------------------------
    obs_space.network_input_dim   →  actor/critic input size
    act_space.network_output_dim  →  actor output size
    get_observation(raw, device)  →  flat float32 tensor (1, obs_dim)
    get_action(network_output)    →  int (discrete) or np.ndarray (continuous)
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        self._env:          Optional[gym.Env] = None
        self._recording:    bool              = False
        self._record_path:  Optional[str]     = None
        self._pre_record_env: Optional[gym.Env] = None

    # ──────────────────────────────────────────────────────────────────
    # BaseEnvAdapter: lifecycle
    # ──────────────────────────────────────────────────────────────────

    def setup(self) -> Tuple[ObservationSpace, ActionSpace]:
        render_mode = getattr(self.cfg, "render_mode", None)
        self._env   = gym.make(self.cfg.env_name, render_mode=render_mode)

        obs_space = self._build_obs_space(self._env.observation_space)
        act_space = self._build_act_space(self._env.action_space)

        self._set_spaces(obs_space, act_space)
        return obs_space, act_space

    def reset(self) -> np.ndarray:
        obs, _ = self._env.reset(seed=getattr(self.cfg, "seed", None))
        return obs.astype(np.float32)

    def step(self, action: Any) -> Tuple[np.ndarray, float, bool, Dict]:
        obs, reward, terminated, truncated, info = self._env.step(action)
        return obs.astype(np.float32), float(reward), terminated or truncated, info

    def close(self):
        if self._env:
            self._env.close()
            self._env = None

    # ──────────────────────────────────────────────────────────────────
    # Observation bridge (override for custom preprocessing)
    # ──────────────────────────────────────────────────────────────────

    def get_observation(
        self,
        raw_obs: np.ndarray,
        device:  str  = "cpu",
        normalise: bool = False,
    ) -> torch.Tensor:
        """
        Flat envs: flatten + optional normalise → (1, obs_dim) tensor.

        Override here if you need:
          - Frame stacking
          - Running mean-std normalisation
          - Feature extraction (e.g. encode image first)
        """
        return self._obs_space.preprocess(raw_obs, device=device, normalise=normalise)

    # ──────────────────────────────────────────────────────────────────
    # Action bridge (override for custom postprocessing)
    # ──────────────────────────────────────────────────────────────────

    def get_action(self, raw_network_output: Any, clip: bool = True) -> Any:
        """
        Convert network output to a valid Gymnasium action.

        Override here if you need:
          - Rescaling from tanh range to env-specific bounds
          - Multi-head action decoding
          - Action smoothing / filtering
        """
        return self._act_space.postprocess(raw_network_output, clip=clip)

    # ──────────────────────────────────────────────────────────────────
    # Recording
    # ──────────────────────────────────────────────────────────────────

    def start_recording(self, save_path: str):
        if self._env is None:
            raise RuntimeError("Call setup() before start_recording().")
        os.makedirs(save_path, exist_ok=True)
        base_env = self._unwrap_recorder()
        # If the base env does not support image rendering, recreate it
        # with an appropriate `render_mode` so RecordVideo can capture frames.
        env_to_wrap = base_env
        if getattr(base_env, "render_mode", None) is None:
            # preserve reference to original env to restore later
            self._pre_record_env = self._env
            try:
                env_to_wrap = gym.make(self.cfg.env_name, render_mode="rgb_array")
            except Exception:
                # Fallback: try common alias
                env_to_wrap = gym.make(self.cfg.env_name, render_mode="rgb_array_list")

        self._env = RecordVideo(env_to_wrap, video_folder=save_path,
                                episode_trigger=lambda ep: True,
                                name_prefix="episode")
        self._recording = True
        self._record_path = save_path

    def stop_recording(self) -> Optional[str]:
        if not self._recording:
            return None
        path          = self._record_path
        # Close the recorder wrapper and restore the previous env if we
        # replaced it when starting recording.
        try:
            # if RecordVideo created recorded_frames etc, close it cleanly
            self._env.close()
        except Exception:
            pass
        if self._pre_record_env is not None:
            # Restore the original env that was active before recording.
            self._env = self._pre_record_env
            self._pre_record_env = None
        else:start_recording
            # If we wrapped the original env in-place, unwrap back to it
            self._env = self._unwrap_recorder()
        self._recording = False
        return path

    def render(self) -> Optional[np.ndarray]:
        if self._env and hasattr(self._env, "render"):
            return self._env.render()
        return None

    def seed(self, seed: int):
        self.cfg.seed = seed

    # ──────────────────────────────────────────────────────────────────
    # TorchRL integration (on-policy PPO)
    # ──────────────────────────────────────────────────────────────────

    def make_torchrl_env(self):
        from torchrl.envs import GymEnv, TransformedEnv, Compose, StepCounter, RewardSum
        env = GymEnv(self.cfg.env_name, device=getattr(self.cfg, "device", "cpu"))
        return TransformedEnv(env, Compose(StepCounter(), RewardSum()))

    # ──────────────────────────────────────────────────────────────────
    # Space builders
    # ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _build_obs_space(gym_space) -> ObservationSpace:
        """Translate a gymnasium observation space → ObservationSpace."""
        import gymnasium.spaces as S

        if isinstance(gym_space, S.Box):
            shape      = gym_space.shape
            is_image   = len(shape) == 3   # (C, H, W) or (H, W, C)
            space_type = SpaceType.IMAGE if is_image else SpaceType.FLAT
            return ObservationSpace(
                shape=shape,
                dtype=gym_space.dtype,
                space_type=space_type,
                flat_dim=int(np.prod(shape)),
                low=gym_space.low.flatten()  if gym_space.is_bounded("below") else None,
                high=gym_space.high.flatten() if gym_space.is_bounded("above") else None,
            )

        if isinstance(gym_space, S.Dict):
            components = {
                k: GymnasiumAdapter._build_obs_space(v)
                for k, v in gym_space.spaces.items()
            }
            total_dim = sum(s.flat_dim for s in components.values())
            return ObservationSpace(
                shape=(total_dim,),
                space_type=SpaceType.DICT,
                flat_dim=total_dim,
                components=components,
            )

        # Fallback: treat everything else as flat
        flat_dim = int(np.prod(gym_space.shape)) if hasattr(gym_space, "shape") else 1
        return ObservationSpace(shape=(flat_dim,), flat_dim=flat_dim)

    @staticmethod
    def _build_act_space(gym_space) -> ActionSpace:
        """Translate a gymnasium action space → ActionSpace."""
        import gymnasium.spaces as S

        if isinstance(gym_space, S.Discrete):
            return ActionSpace(action_type=ActionType.DISCRETE, n=int(gym_space.n))

        if isinstance(gym_space, S.Box):
            return ActionSpace(
                action_type=ActionType.CONTINUOUS,
                shape=gym_space.shape,
                low=gym_space.low,
                high=gym_space.high,
            )

        if isinstance(gym_space, S.MultiDiscrete):
            components = [
                ActionSpace(action_type=ActionType.DISCRETE, n=int(n))
                for n in gym_space.nvec
            ]
            return ActionSpace(
                action_type=ActionType.MULTI_DISCRETE,
                components=components,
            )

        raise NotImplementedError(f"Unsupported action space: {type(gym_space)}")

    # ──────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────

    def _unwrap_recorder(self) -> gym.Env:
        env = self._env
        while isinstance(env, RecordVideo):
            env = env.env
        return env
