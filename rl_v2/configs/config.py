"""
configs/config.py
=================
Master configuration.

Algorithm-specific hyperparams live in each algorithm's own config class
(PPOConfig, SACConfig, TD3Config, etc.) — not here.
This keeps the master config small and algorithm-agnostic.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class EnvConfig:
    env_name:    str            = "CartPole-v1"
    env_backend: str            = "gymnasium"     # "gymnasium" | "isaac" | "omniverse"
    render_mode: Optional[str]  = None
    seed:        int            = 42
    device:      str            = "cpu"
    num_envs:    int            = 1
    headless:    bool           = True


@dataclass
class IsaacEnvConfig(EnvConfig):
    env_backend:  str   = "isaac"
    device:       str   = "cuda:0"
    physics_dt:   float = 1 / 60


@dataclass
class OmniverseEnvConfig(EnvConfig):
    env_backend: str  = "omniverse"
    device:      str  = "cuda:0"
    task_cfg:    dict = field(default_factory=dict)


@dataclass
class LogConfig:
    run_dir:          Optional[str] = None   # auto-generated if None
    log_tensorboard:  bool  = True
    log_csv:          bool  = True
    log_json:         bool  = True
    checkpoint_every: int   = 20
    record_sessions:  List  = field(default_factory=lambda: [0, 5, -1])
    histogram_every:  int   = 10


@dataclass
class Config:
    env: EnvConfig = field(default_factory=EnvConfig)
    log: LogConfig = field(default_factory=LogConfig)
    # Note: algorithm config is NOT here — it lives on the algorithm object.
    # This keeps Config universal across all algorithms.
