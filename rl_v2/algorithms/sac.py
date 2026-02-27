"""
algorithms/sac.py
=================
Soft Actor-Critic — off-policy, continuous actions.
Uses network factory. Calls adapter.get_observation() / adapter.get_action().
"""

from __future__ import annotations
from collections import deque
from dataclasses import dataclass
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from .base import BaseAlgorithm, MetricsDict
from ..envs.spaces import ObservationSpace, ActionSpace, ActionType
from ..networks.factory import make_policy_network, make_twin_value_network
from ..networks.mlp import MLPConfig


class _ReplayBuffer:
    def __init__(self, capacity: int):
        self._buf = deque(maxlen=capacity)

    def push(self, obs_t, action, reward, next_obs_t, done):
        # Store tensors (already preprocessed)
        self._buf.append((
            obs_t.cpu().squeeze(0),
            torch.tensor(action, dtype=torch.float32).flatten(),
            torch.tensor([reward], dtype=torch.float32),
            next_obs_t.cpu().squeeze(0),
            torch.tensor([float(done)], dtype=torch.float32),
        ))

    def sample(self, n: int):
        batch   = random.sample(self._buf, n)
        obs, act, rew, nobs, done = zip(*batch)
        return (torch.stack(obs),  torch.stack(act),  torch.stack(rew),
                torch.stack(nobs), torch.stack(done))

    def __len__(self): return len(self._buf)


from .configs import SACConfig  # noqa: F401


class SACAlgorithm(BaseAlgorithm):

    ALGORITHM_NAME = "sac"
    ON_POLICY      = False

    def __init__(self, algo_cfg, obs_space, act_space, network_cfg=None):
        if act_space.action_type == ActionType.DISCRETE:
            raise ValueError("SAC requires continuous actions.")
        super().__init__(algo_cfg, obs_space, act_space, network_cfg or MLPConfig())
        self._policy     = None
        self._twin_q     = None
        self._twin_q_tgt = None
        self._buf        = None

    def setup(self, device: str = "cpu"):
        self.device = device
        cfg = self.algo_cfg

        # Policy: stochastic continuous (outputs mean + log_std)
        self._policy = make_policy_network(
            self.obs_space, self.act_space, self.network_cfg, deterministic=False
        ).to(device)

        # Twin Q-networks
        self._twin_q     = make_twin_value_network(
            self.obs_space, self.act_space, self.network_cfg
        ).to(device)
        self._twin_q_tgt = deepcopy(self._twin_q)
        for p in self._twin_q_tgt.parameters():
            p.requires_grad = False

        self._opt_actor  = Adam(self._policy.parameters(), lr=cfg.lr_actor)
        self._opt_critic = Adam(self._twin_q.parameters(), lr=cfg.lr_critic)

        # Auto entropy
        if cfg.auto_alpha:
            tgt = cfg.target_entropy if cfg.target_entropy else -self.act_space.network_output_dim
            self._target_entropy = float(tgt)
            self.log_alpha       = torch.zeros(1, requires_grad=True, device=device)
            self._opt_alpha      = Adam([self.log_alpha], lr=cfg.lr_alpha)
        else:
            self.log_alpha = torch.zeros(1, device=device)

        self._buf = _ReplayBuffer(cfg.buffer_size)

    def _sample_action(self, obs_t: torch.Tensor, deterministic: bool = False):
        """
        Forward pass through the stochastic policy.
        Returns (action_tensor, log_prob) using reparameterisation.
        """
        act_dim = self.act_space.network_output_dim
        out     = self._policy(obs_t)                     # (batch, 2*act_dim)
        mean, log_std = out[..., :act_dim], out[..., act_dim:]
        log_std = log_std.clamp(-20, 2)
        std     = log_std.exp()

        dist    = torch.distributions.Normal(mean, std)
        raw     = dist.rsample()
        action  = torch.tanh(raw)
        log_pi  = dist.log_prob(raw) - torch.log(1 - action.pow(2) + 1e-6)
        log_pi  = log_pi.sum(-1, keepdim=True)

        if deterministic:
            return torch.tanh(mean), log_pi
        return action, log_pi

    def select_action(self, obs_tensor: torch.Tensor, eval_mode: bool = False):
        if len(self._buf) < self.algo_cfg.warmup_steps and not eval_mode:
            raw = self.act_space.sample()
            return raw, None

        with torch.no_grad():
            action_t, lp = self._sample_action(obs_tensor.to(self.device), deterministic=eval_mode)
        return action_t.squeeze(0).cpu().numpy(), lp.item()

    def collect_data(self, adapter, n_steps: int) -> Any:
        obs = adapter.reset()
        for _ in range(n_steps):
            obs_t              = adapter.get_observation(obs, device=self.device)
            raw_out, _         = self.select_action(obs_t, eval_mode=False)
            action             = adapter.get_action(raw_out)
            nobs, rew, done, _ = adapter.step(action)
            nobs_t             = adapter.get_observation(nobs, device=self.device)

            # Store preprocessed tensors (consistent representation)
            self._buf.push(obs_t, raw_out, rew, nobs_t, done)
            self._increment_steps()
            obs = nobs if not done else adapter.reset()
        return None

    def update(self, batch: Any = None) -> MetricsDict:
        if len(self._buf) < self.algo_cfg.batch_size:
            return {}

        cfg    = self.algo_cfg
        device = self.device
        obs, act, rew, nobs, done = [x.to(device) for x in self._buf.sample(cfg.batch_size)]
        alpha = self.log_alpha.exp().detach()

        # Critic
        with torch.no_grad():
            na, nlp    = self._sample_action(nobs)
            q1n, q2n   = self._twin_q_tgt(nobs, na)
            q_target   = rew + cfg.gamma * (1 - done) * (torch.min(q1n, q2n) - alpha * nlp)

        q1, q2      = self._twin_q(obs, act)
        critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)
        self._opt_critic.zero_grad(); critic_loss.backward(); self._opt_critic.step()

        # Actor
        a_new, log_pi = self._sample_action(obs)
        q1p, q2p      = self._twin_q(obs, a_new)
        actor_loss    = (alpha * log_pi - torch.min(q1p, q2p)).mean()
        self._opt_actor.zero_grad(); actor_loss.backward(); self._opt_actor.step()

        # Alpha
        alpha_loss = torch.tensor(0.0)
        if cfg.auto_alpha:
            alpha_loss = -(self.log_alpha * (log_pi + self._target_entropy).detach()).mean()
            self._opt_alpha.zero_grad(); alpha_loss.backward(); self._opt_alpha.step()

        # Soft target
        for sp, tp in zip(self._twin_q.parameters(), self._twin_q_tgt.parameters()):
            tp.data.copy_(cfg.tau * sp.data + (1 - cfg.tau) * tp.data)

        return {
            "loss/critic":     critic_loss.item(),
            "loss/actor":      actor_loss.item(),
            "loss/alpha":      alpha_loss.item(),
            "sac/alpha":       alpha.item(),
            "sac/log_pi_mean": log_pi.mean().item(),
        }

    def train_mode(self): self._policy.train(); self._twin_q.train()
    def eval_mode(self):  self._policy.eval();  self._twin_q.eval()

    def state_dict(self):
        return {
            "policy":    self._policy.state_dict(),
            "twin_q":    self._twin_q.state_dict(),
            "twin_q_tgt":self._twin_q_tgt.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "opt_actor": self._opt_actor.state_dict(),
            "opt_critic":self._opt_critic.state_dict(),
        }

    def load_state_dict(self, state):
        self._policy.load_state_dict(state["policy"])
        self._twin_q.load_state_dict(state["twin_q"])
        self._twin_q_tgt.load_state_dict(state["twin_q_tgt"])
        self.log_alpha.data = state["log_alpha"].to(self.device)
        self._opt_actor.load_state_dict(state["opt_actor"])
        self._opt_critic.load_state_dict(state["opt_critic"])

    def named_parameters(self):
        return (list(self._policy.named_parameters()) +
                list(self._twin_q.named_parameters()))
