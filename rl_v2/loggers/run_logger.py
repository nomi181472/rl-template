"""
loggers/run_logger.py
=====================
Algorithm-agnostic logger.

Logs any flat dict of scalar metrics. No algorithm-specific field names.
All metrics from algo.update() are forwarded directly to TensorBoard/CSV/JSONL.
"""

from __future__ import annotations
import os
import csv
import json
import time
import datetime
from typing import Dict, Any, Optional

from torch.utils.tensorboard import SummaryWriter

from ..configs.config import Config

# Fixed fields that always appear in CSV; algorithm-specific ones appended dynamically
_BASE_FIELDS = [
    "iteration", "global_step", "elapsed_sec",
    "mean_reward", "max_reward", "min_reward", "std_reward", "fps",
]


class RunLogger:
    """
    Unified logger: TensorBoard + CSV + JSONL + console.
    Works with any algorithm — logs whatever metrics dict update() returns.
    """

    def __init__(self, cfg: Config, algorithm_name: str = "unknown"):
        self.cfg            = cfg
        self.algorithm_name = algorithm_name
        self.run_dir        = self._resolve_run_dir()
        self.log_dir        = os.path.join(self.run_dir, "logs")
        self.ckpt_dir       = os.path.join(self.run_dir, "checkpoints")
        self.vid_dir        = os.path.join(self.run_dir, "videos")
        self.tb_dir         = os.path.join(self.run_dir, "tensorboard")

        self._writer:     Optional[SummaryWriter] = None
        self._csv_file                            = None
        self._csv_writer                          = None
        self._extra_fields: list                  = []
        self._t_start                             = None

    # ------------------------------------------------------------------
    # Setup / teardown
    # ------------------------------------------------------------------

    def setup(self, extra_metric_keys: list = None):
        """
        Parameters
        ----------
        extra_metric_keys : list of metric key strings the algorithm will emit,
                            e.g. ["loss/total", "ppo/approx_kl", "sac/alpha"]
                            Used to pre-declare CSV columns.
        """
        for d in [self.log_dir, self.ckpt_dir, self.vid_dir, self.tb_dir]:
            os.makedirs(d, exist_ok=True)

        if self.cfg.log.log_tensorboard:
            self._writer = SummaryWriter(log_dir=self.tb_dir)

        self._extra_fields = extra_metric_keys or []
        if self.cfg.log.log_csv:
            all_fields       = _BASE_FIELDS + self._extra_fields
            csv_path         = os.path.join(self.log_dir, "training_log.csv")
            self._csv_file   = open(csv_path, "w", newline="")
            self._csv_writer = csv.DictWriter(
                self._csv_file, fieldnames=all_fields, extrasaction="ignore"
            )
            self._csv_writer.writeheader()

        self._t_start = time.time()

        # Save config
        cfg_dict = self._config_to_dict()
        with open(os.path.join(self.log_dir, "config.json"), "w") as f:
            json.dump(cfg_dict, f, indent=2)

        if self._writer:
            self._writer.add_hparams(
                {k: str(v) for k, v in cfg_dict.items()},
                metric_dict={"hparam/placeholder": 0},
            )

        print(f"\nAlgorithm  : {self.algorithm_name.upper()}")
        print(f"Run dir    : {os.path.abspath(self.run_dir)}")
        if self._writer:
            print(f"TensorBoard: tensorboard --logdir {os.path.abspath(self.tb_dir)}")
        print()

    def close(self):
        if self._csv_file:
            self._csv_file.close()
        if self._writer:
            self._writer.close()

        summary = {
            "algorithm":       self.algorithm_name,
            "run_dir":         os.path.abspath(self.run_dir),
            "elapsed_sec":     round(time.time() - self._t_start, 1),
            "tensorboard_cmd": f"tensorboard --logdir {os.path.abspath(self.tb_dir)}",
        }
        with open(os.path.join(self.log_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def log(self, iteration: int, global_step: int, metrics: Dict[str, Any]):
        """
        Write all metrics everywhere.
        metrics can contain any keys — they're all forwarded to TensorBoard
        and written to CSV/JSONL.
        """
        elapsed = round(time.time() - self._t_start, 1)
        row     = {"iteration": iteration, "global_step": global_step,
                   "elapsed_sec": elapsed, **metrics}

        # ── TensorBoard: all scalars ──────────────────────────────
        if self._writer:
            # Fixed reward scalars
            for key in ["mean_reward", "max_reward", "min_reward", "std_reward", "fps"]:
                if key in metrics:
                    tb_key = f"reward/{key}" if "reward" in key else f"perf/{key}"
                    self._writer.add_scalar(tb_key, metrics[key], global_step)
            self._writer.add_scalar("perf/elapsed_sec", elapsed, global_step)

            # All algorithm-specific metrics (loss/*, ppo/*, sac/*, td3/*, etc.)
            for k, v in metrics.items():
                if k not in ("mean_reward","max_reward","min_reward","std_reward","fps"):
                    if isinstance(v, (int, float)):
                        self._writer.add_scalar(k, v, global_step)

        # ── CSV ───────────────────────────────────────────────────
        if self._csv_writer:
            self._csv_writer.writerow(row)
            self._csv_file.flush()

        # ── JSONL ─────────────────────────────────────────────────
        if self.cfg.log.log_json:
            with open(os.path.join(self.log_dir, "training_log.jsonl"), "a") as jf:
                jf.write(json.dumps(row) + "\n")

    def log_video_reward(self, global_step: int, reward: float):
        if self._writer:
            self._writer.add_scalar("video/episode_reward", reward, global_step)

    def log_histogram(self, global_step: int, named_params):
        if not self._writer:
            return
        for name, param in named_params:
            self._writer.add_histogram(f"weights/{name}", param.data, global_step)
            if param.grad is not None:
                self._writer.add_histogram(f"grads/{name}", param.grad, global_step)

    def log_graph(self, model, dummy_input):
        if self._writer:
            try:
                self._writer.add_graph(model, dummy_input)
            except Exception:
                pass

    def writer_add_scalar(self, tag: str, value: float, step: int):
        """Direct TensorBoard scalar write for custom tags."""
        if self._writer:
            self._writer.add_scalar(tag, value, step)

    def elapsed(self) -> float:
        return time.time() - self._t_start

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_run_dir(self) -> str:
        if self.cfg.log.run_dir:
            return self.cfg.log.run_dir
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        return os.path.join(
            "runs",
            f"{self.cfg.env.env_name}_{self.cfg.env.env_backend}_{self.algorithm_name}_{ts}"
        )

    def _config_to_dict(self) -> dict:
        out = {}
        for section_name in ("env", "log"):
            section = getattr(self.cfg, section_name)
            for k, v in vars(section).items():
                out[f"{section_name}.{k}"] = v
        out["algorithm"] = self.algorithm_name
        return out
