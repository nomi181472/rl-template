"""
envs/omniverse_adapter.py
=========================
OmniVerse / OmniIsaacGymEnvs adapter stub.

Shows how get_observation() handles multi-modal observations:
  - RGB camera images  (processed by CNN feature extractor)
  - Proprioception     (joint state vector)
  Both concatenated before being passed to the actor network.
"""

from __future__ import annotations
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from .base import BaseEnvAdapter
from .spaces import ObservationSpace, ActionSpace, SpaceType, ActionType


class OmniverseAdapter(BaseEnvAdapter):

    def __init__(self, cfg):
        super().__init__(cfg)
        self._env     = None
        self._sim_app = None

    def setup(self) -> Tuple[ObservationSpace, ActionSpace]:
        # ── Real implementation ──────────────────────────────────────
        # from omni.isaac.kit import SimulationApp
        # self._sim_app = SimulationApp({"headless": self.cfg.headless})
        # ...
        # ─────────────────────────────────────────────────────────────
        raise NotImplementedError(
            "OmniverseAdapter.setup(): install OmniIsaacGymEnvs and uncomment above."
        )

    def reset(self) -> np.ndarray:
        raise NotImplementedError

    def step(self, action: Any) -> Tuple[np.ndarray, float, bool, Dict]:
        raise NotImplementedError

    def close(self):
        if self._env:     self._env.close()
        if self._sim_app: self._sim_app.close()

    # ──────────────────────────────────────────────────────────────────
    # Custom observation preprocessing for visual + proprio fusion
    # ──────────────────────────────────────────────────────────────────

    def get_observation(
        self,
        raw_obs: Dict[str, np.ndarray],
        device:  str  = "cpu",
        normalise: bool = False,
    ) -> torch.Tensor:
        """
        Example for an OmniVerse env that returns:
          raw_obs["image"]  : (3, H, W) RGB frame
          raw_obs["state"]  : (proprio_dim,) joint state

        The network receives the concatenation of:
          1. CNN-encoded image features  (image_feature_dim,)
          2. Flat proprio vector         (proprio_dim,)

        Override or extend for your specific sensor suite.
        """
        # ── Proprioception ─────────────────────────────────────────
        state_t = torch.tensor(raw_obs["state"], dtype=torch.float32, device=device)

        # ── Vision (example using a CNN feature extractor) ─────────
        # img_t = torch.tensor(raw_obs["image"], dtype=torch.float32, device=device)
        # img_t = img_t.unsqueeze(0) / 255.0          # normalise pixels
        # with torch.no_grad():
        #     img_features = self._cnn_encoder(img_t)  # (1, feature_dim)
        # fused = torch.cat([img_features.squeeze(0), state_t], dim=0).unsqueeze(0)

        # Fallback when no CNN: just use state
        fused = state_t.unsqueeze(0)
        return fused  # (1, obs_dim)

    # ──────────────────────────────────────────────────────────────────
    # Custom action postprocessing
    # ──────────────────────────────────────────────────────────────────

    def get_action(self, raw_network_output: Any, clip: bool = True) -> Any:
        """
        OmniVerse actions are typically normalised joint positions.
        Rescale + clip to physical limits.
        """
        return self._act_space.postprocess(raw_network_output, clip=clip)

    def make_torchrl_env(self):
        raise NotImplementedError
