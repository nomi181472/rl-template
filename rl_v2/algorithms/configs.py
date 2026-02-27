"""PPO and SAC config dataclasses. No torch dependency."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional


@dataclass
class PPOConfig:
    name:             str   = "ppo"
    frames_per_batch: int   = 1_000
    total_frames:     int   = 100_000
    num_epochs:       int   = 10
    mini_batch_size:  int   = 256
    lr:               float = 3e-4
    gamma:            float = 0.99
    lam:              float = 0.95
    clip_eps:         float = 0.2
    entropy_coef:     float = 0.01
    critic_coef:      float = 0.5
    max_grad_norm:    float = 0.5


@dataclass
class SACConfig:
    name:             str   = "sac"
    buffer_size:      int   = 1_000_000
    batch_size:       int   = 256
    warmup_steps:     int   = 1_000
    lr_actor:         float = 3e-4
    lr_critic:        float = 3e-4
    lr_alpha:         float = 3e-4
    gamma:            float = 0.99
    tau:              float = 0.005
    auto_alpha:       bool  = True
    target_entropy:   Optional[float] = None
    total_frames:     int   = 1_000_000
    frames_per_batch: int   = 1_000
