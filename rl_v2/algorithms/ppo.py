"""
algorithms/ppo.py
=================
PPO — on-policy, actor-critic.

Network: built via make_policy_network() / make_value_network().
Any architecture (MLP, CNN, LSTM) works without changing this file.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor

from torchrl.collectors import SyncDataCollector
from torchrl.data import ReplayBuffer, LazyTensorStorage, SamplerWithoutReplacement
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE

from .base import BaseAlgorithm, MetricsDict
from ..envs.spaces import ObservationSpace, ActionSpace, ActionType
from ..networks.factory import make_policy_network, make_value_network
from ..networks.mlp import MLPConfig


from .configs import PPOConfig  # noqa: F401 (re-export)


class PPOAlgorithm(BaseAlgorithm):

    ALGORITHM_NAME = "ppo"
    ON_POLICY      = True

    def __init__(
        self,
        algo_cfg:    PPOConfig,
        obs_space:   ObservationSpace,
        act_space:   ActionSpace,
        network_cfg: Any = None,
    ):
        super().__init__(algo_cfg, obs_space, act_space,
                         network_cfg or MLPConfig())
        # TorchRL components (built in setup)
        self._actor:      Optional[ProbabilisticActor] = None
        self._critic:     Optional[ValueOperator]      = None
        self._loss_fn:    Optional[ClipPPOLoss]        = None
        self._optimizer:  Optional[Adam]               = None
        self._advantage:  Optional[GAE]                = None
        self._collector:  Optional[SyncDataCollector]  = None
        self._replay_buf: Optional[ReplayBuffer]       = None

    # ──────────────────────────────────────────────────────────────────
    # Setup
    # ──────────────────────────────────────────────────────────────────

    def setup(self, device: str = "cpu"):
        self.device = device
        cfg          = self.algo_cfg
        obs_space    = self.obs_space
        act_space    = self.act_space
        net_cfg      = self.network_cfg

        # ── Build policy net (from network factory) ────────────────
        policy_net = make_policy_network(
            obs_space, act_space, net_cfg, deterministic=False
        )
        policy_net.to(device)

        # ── Wrap into TorchRL ProbabilisticActor ───────────────────
        is_discrete = act_space.action_type == ActionType.DISCRETE

        if is_discrete:
            td_module = TensorDictModule(
                policy_net, in_keys=["observation"], out_keys=["logits"]
            )
            self._actor = ProbabilisticActor(
                module=td_module, in_keys=["logits"], out_keys=["action"],
                distribution_class=torch.distributions.Categorical,
                return_log_prob=True,
            )
        else:
            # policy_net outputs cat([mean, log_std], dim=-1) for stochastic
            act_dim     = act_space.network_output_dim
            splitter    = _MeanLogStdSplitter(policy_net, act_dim)
            td_module   = TensorDictModule(
                splitter, in_keys=["observation"], out_keys=["loc", "scale"]
            )
            self._actor = ProbabilisticActor(
                module=td_module, in_keys=["loc", "scale"], out_keys=["action"],
                distribution_class=TanhNormal, return_log_prob=True,
            )
        self._actor.to(device)

        # ── Build value net ────────────────────────────────────────
        value_net = make_value_network(obs_space, net_cfg)
        value_net.to(device)
        self._critic = ValueOperator(
            module=_ValueWrapper(value_net), in_keys=["observation"]
        )

        # ── Loss / advantage / optimizer / buffer ─────────────────
        self._advantage  = GAE(gamma=cfg.gamma, lmbda=cfg.lam,
                               value_network=self._critic, average_gae=True)
        self._loss_fn    = ClipPPOLoss(
            actor_network=self._actor, critic_network=self._critic,
            clip_epsilon=cfg.clip_eps, entropy_bonus=True,
            entropy_coeff=cfg.entropy_coef, critic_coeff=cfg.critic_coef,
            loss_critic_type="smooth_l1",
        )
        self._optimizer  = Adam(self._loss_fn.parameters(), lr=cfg.lr)
        self._replay_buf = ReplayBuffer(
            storage=LazyTensorStorage(cfg.frames_per_batch, device=device),
            sampler=SamplerWithoutReplacement(),
        )

    # ──────────────────────────────────────────────────────────────────
    # Action selection
    # ──────────────────────────────────────────────────────────────────

    def select_action(
        self,
        obs_tensor: torch.Tensor,
        eval_mode:  bool = False,
    ) -> Tuple[Any, Optional[float]]:
        """
        obs_tensor: already preprocessed by adapter.get_observation()
        Returns (raw_output, log_prob) — caller passes raw_output to adapter.get_action()
        """
        mode = ExplorationType.DETERMINISTIC if eval_mode else ExplorationType.RANDOM
        with torch.no_grad(), set_exploration_type(mode):
            td       = TensorDict({"observation": obs_tensor}, batch_size=[1], device=self.device)
            td       = self._actor(td)
            raw_out  = td["action"].squeeze(0).cpu().numpy()
            log_prob = td.get("sample_log_prob", torch.tensor([float("nan")])).item()
        return raw_out, log_prob

    # ──────────────────────────────────────────────────────────────────
    # Data collection (manual, non-TorchRL path)
    # ──────────────────────────────────────────────────────────────────

    def collect_data(self, adapter, n_steps: int) -> Any:
        obs = adapter.reset()
        for _ in range(n_steps):
            obs_t                 = adapter.get_observation(obs, device=self.device)
            raw_out, _            = self.select_action(obs_t, eval_mode=False)
            action                = adapter.get_action(raw_out)
            next_obs, rew, done, _= adapter.step(action)
            obs = next_obs if not done else adapter.reset()

    # ──────────────────────────────────────────────────────────────────
    # Update
    # ──────────────────────────────────────────────────────────────────

    def update(self, batch: Any) -> MetricsDict:
        cfg    = self.algo_cfg
        losses = {"total": [], "obj": [], "crit": [], "ent": [], "kl": []}

        with torch.no_grad():
            batch = self._advantage(batch)
        self._replay_buf.extend(batch.reshape(-1))

        # Iterate over epochs and sample minibatches from the ReplayBuffer.
        # torchrl.ReplayBuffer does not provide `sample_tensordicts`; use
        # `sample(batch_size)` instead and repeat for the number of
        # minibatches that fit in a full batch.
        batch_size = cfg.frames_per_batch
        mini_b = cfg.mini_batch_size
        n_minibatches = max(1, batch_size // mini_b)
        for _ in range(cfg.num_epochs):
            for _b in range(n_minibatches):
                mb = self._replay_buf.sample(mini_b)
                self._optimizer.zero_grad()
                loss_td = self._loss_fn(mb)
                loss    = (loss_td["loss_objective"] +
                           loss_td["loss_critic"] +
                           loss_td["loss_entropy"])
                loss.backward()
                nn.utils.clip_grad_norm_(self._loss_fn.parameters(), cfg.max_grad_norm)
                self._optimizer.step()

                losses["total"].append(loss.item())
                losses["obj"].append(loss_td["loss_objective"].item())
                losses["crit"].append(loss_td["loss_critic"].item())
                losses["ent"].append(loss_td["loss_entropy"].item())
                with torch.no_grad():
                    try:
                        lr = mb["sample_log_prob"] - mb["action_log_prob"]
                        kl = ((torch.exp(lr) - 1) - lr).mean().item()
                    except Exception:
                        kl = 0.0
                    losses["kl"].append(kl)

        def _m(lst): return sum(lst) / len(lst) if lst else 0.0
        return {
            "loss/total":     _m(losses["total"]),
            "loss/objective": _m(losses["obj"]),
            "loss/critic":    _m(losses["crit"]),
            "loss/entropy":   _m(losses["ent"]),
            "ppo/approx_kl":  _m(losses["kl"]),
        }

    # ──────────────────────────────────────────────────────────────────
    # TorchRL collector
    # ──────────────────────────────────────────────────────────────────

    def build_collector(self, make_env_fn, device: str) -> SyncDataCollector:
        return SyncDataCollector(
            create_env_fn=make_env_fn, policy=self._actor,
            frames_per_batch=self.algo_cfg.frames_per_batch,
            total_frames=self.algo_cfg.total_frames,
            device=device, storing_device=device,
        )

    # ──────────────────────────────────────────────────────────────────
    # Modes / persistence
    # ──────────────────────────────────────────────────────────────────

    def train_mode(self):
        self._actor.train(); self._critic.train()

    def eval_mode(self):
        self._actor.eval(); self._critic.eval()

    def state_dict(self) -> Dict[str, Any]:
        return {
            "actor":     self._actor.state_dict(),
            "critic":    self._critic.state_dict(),
            "optimizer": self._optimizer.state_dict(),
        }

    def load_state_dict(self, state: Dict[str, Any]):
        self._actor.load_state_dict(state["actor"])
        self._critic.load_state_dict(state["critic"])
        if self._optimizer and "optimizer" in state:
            self._optimizer.load_state_dict(state["optimizer"])

    def named_parameters(self):
        return list(self._loss_fn.named_parameters()) if self._loss_fn else []


# ── Internal helpers ──────────────────────────────────────────────────

class _MeanLogStdSplitter(nn.Module):
    """Splits cat([mean, log_std]) output into separate tensors."""
    def __init__(self, net: nn.Module, act_dim: int):
        super().__init__()
        self.net     = net
        self.act_dim = act_dim

    def forward(self, obs):
        out = self.net(obs)
        mean, log_std = out.chunk(2, dim=-1)
        std = log_std.exp().clamp(1e-6, None)
        return mean, std


class _ValueWrapper(nn.Module):
    """Wraps BaseValueNetwork to match TorchRL's expected (obs,) -> (value,) signature."""
    def __init__(self, value_net):
        super().__init__()
        self.net = value_net

    def forward(self, obs):
        return self.net(obs)
