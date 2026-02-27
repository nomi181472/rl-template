"""
envs/isaac_adapter.py
=====================
Isaac Lab / Isaac Sim adapter stub.

Shows how get_observation() and get_action() are used for a real robotics env
where observations are multi-modal (joint pos + vel + task state)
and actions are joint torques that need rescaling.
"""

from __future__ import annotations
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from .base import BaseEnvAdapter
from .spaces import ObservationSpace, ActionSpace, SpaceType, ActionType


class IsaacAdapter(BaseEnvAdapter):
    """
    Wraps Isaac Lab ManagerBasedRLEnv.

    Observation design for robotics
    --------------------------------
    Isaac envs typically provide a dict of tensors:
      "policy"  : flat proprioceptive state (joint_pos, joint_vel, ee_pose, ...)
      "critic"  : extended state for the critic (includes privileged info)

    get_observation() handles:
      1. Extracting the "policy" key
      2. Moving to correct device
      3. Reshaping to (1, obs_dim)

    Action design
    -------------
    Network outputs floats in roughly [-1, 1] (tanh range).
    get_action() rescales to physical joint torque limits.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        self._env = None

    # ──────────────────────────────────────────────────────────────────
    # BaseEnvAdapter: lifecycle
    # ──────────────────────────────────────────────────────────────────

    def setup(self) -> Tuple[ObservationSpace, ActionSpace]:
        # ── Real implementation ──────────────────────────────────────
        # from isaaclab.envs import ManagerBasedRLEnv
        # from isaaclab_tasks.utils import parse_env_cfg
        #
        # env_cfg    = parse_env_cfg(self.cfg.env_name,
        #                            use_gpu=(self.cfg.device != "cpu"),
        #                            num_envs=self.cfg.num_envs)
        # self._env  = ManagerBasedRLEnv(cfg=env_cfg)
        #
        # obs_dim    = self._env.observation_manager.group_obs_dim["policy"]
        # act_dim    = self._env.action_manager.total_action_dim
        # act_limits = self._env.action_manager.action_limits   # (act_dim, 2)
        #
        # obs_space  = ObservationSpace(
        #     shape=(obs_dim,),
        #     space_type=SpaceType.FLAT,
        #     flat_dim=obs_dim,
        #     extra={"includes_privileged": True}
        # )
        # act_space  = ActionSpace(
        #     action_type=ActionType.CONTINUOUS,
        #     shape=(act_dim,),
        #     low=act_limits[:, 0].cpu().numpy(),
        #     high=act_limits[:, 1].cpu().numpy(),
        # )
        # self._set_spaces(obs_space, act_space)
        # return obs_space, act_space
        # ─────────────────────────────────────────────────────────────

        raise NotImplementedError(
            "IsaacAdapter.setup(): install Isaac Lab and uncomment above."
        )

    def reset(self) -> np.ndarray:
        # obs_dict, _ = self._env.reset()
        # return obs_dict["policy"].cpu().numpy()[0]
        raise NotImplementedError

    def step(self, action: Any) -> Tuple[np.ndarray, float, bool, Dict]:
        # import torch
        # act = torch.tensor(action, device=self.cfg.device).unsqueeze(0)
        # obs_dict, rew, term, trunc, info = self._env.step(act)
        # obs  = obs_dict["policy"].cpu().numpy()[0]
        # return obs, float(rew[0]), bool(term[0]) or bool(trunc[0]), info
        raise NotImplementedError

    def close(self):
        if self._env:
            self._env.close()
            self._env = None

    # ──────────────────────────────────────────────────────────────────
    # Observation bridge — CUSTOM for Isaac (multi-modal tensor)
    # ──────────────────────────────────────────────────────────────────

    def get_observation(
        self,
        raw_obs: np.ndarray,
        device:  str  = "cpu",
        normalise: bool = False,
    ) -> torch.Tensor:
        """
        Isaac obs are already well-conditioned float32 vectors.
        We just convert and move to device.

        Override here to add:
          - Running normalisation (obs_rms)
          - History stacking (last N obs concatenated)
          - Privileged info masking during inference
        """
        obs_t = torch.tensor(raw_obs, dtype=torch.float32, device=device).unsqueeze(0)

        # Example running normalisation (uncomment when self._obs_rms is set up):
        # if normalise and hasattr(self, "_obs_rms"):
        #     obs_t = (obs_t - self._obs_rms.mean) / (self._obs_rms.var.sqrt() + 1e-8)
        #     obs_t = obs_t.clamp(-5.0, 5.0)

        return obs_t  # (1, obs_dim)

    # ──────────────────────────────────────────────────────────────────
    # Action bridge — CUSTOM for Isaac (rescale to joint torque limits)
    # ──────────────────────────────────────────────────────────────────

    def get_action(self, raw_network_output: Any, clip: bool = True) -> Any:
        """
        Network outputs values in [-1, 1] (tanh squashing).
        Rescale to physical joint torque limits.

        Override here for:
          - PD controller target positions (instead of torques)
          - Impedance control parameters
          - Whole-body control with contact forces
        """
        if isinstance(raw_network_output, torch.Tensor):
            raw_network_output = raw_network_output.detach().cpu().numpy()

        action = raw_network_output.astype(np.float32)

        # Rescale from [-1,1] → [low, high]
        if self._act_space.low is not None and self._act_space.high is not None:
            lo     = self._act_space.low
            hi     = self._act_space.high
            action = lo + (action + 1.0) * 0.5 * (hi - lo)

        if clip:
            action = np.clip(action, self._act_space.low, self._act_space.high)

        return action

    # ──────────────────────────────────────────────────────────────────
    # TorchRL integration
    # ──────────────────────────────────────────────────────────────────

    def make_torchrl_env(self):
        # from torchrl.envs.libs.isaaclab import IsaacLabWrapper
        # return IsaacLabWrapper(self._env)
        raise NotImplementedError("Uncomment IsaacLabWrapper when Isaac Lab is installed.")

    def get_state(self) -> Dict[str, Any]:
        # Full sim state for replay
        # return {k: v.cpu() for k, v in self._env.scene.get_state().items()}
        return {}
