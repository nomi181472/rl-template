"""
trainers/trainer.py
===================
Universal trainer — works with any algorithm + any env backend.

Key change from v1
------------------
All observation preprocessing goes through adapter.get_observation().
All action postprocessing goes through adapter.get_action().
The trainer never touches raw numpy arrays directly.
"""

from __future__ import annotations
import time
import torch
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..algorithms.base import BaseAlgorithm
    from ..envs.base import BaseEnvAdapter
    from ..loggers.run_logger import RunLogger
    from ..configs.config import Config


class Trainer:
    """Universal RL trainer."""

    def __init__(self, cfg, algorithm, adapter, logger):
        self.cfg       = cfg
        self.algorithm = algorithm
        self.adapter   = adapter
        self.logger    = logger
        self._collector = None

    def setup(self):
        algo = self.algorithm
        if algo.ON_POLICY and hasattr(algo, "build_collector"):
            self._collector = algo.build_collector(
                self.adapter.make_torchrl_env, self.cfg.env.device
            )

    def train(self):
        algo     = self.algorithm
        lcfg     = self.cfg.log
        algo_cfg = algo.algo_cfg

        total_frames     = getattr(algo_cfg, "total_frames", 100_000)
        frames_per_batch = getattr(algo_cfg, "frames_per_batch", 1_000)
        num_iters        = total_frames // frames_per_batch
        last_iter        = num_iters - 1
        record_set       = {r if r >= 0 else last_iter for r in lcfg.record_sessions}
        global_step      = 0

        print(f"Algorithm : {algo.ALGORITHM_NAME.upper()} "
              f"({'on-policy' if algo.ON_POLICY else 'off-policy'})")
        print(f"Network   : {getattr(algo.network_cfg, 'name', '?').upper()}")
        print(f"Iterations: {num_iters} | Frames: {total_frames:,}\n")

        algo.train_mode()

        if algo.ON_POLICY:
            self._train_on_policy(num_iters, last_iter, record_set,
                                  frames_per_batch, global_step)
        else:
            self._train_off_policy(num_iters, last_iter, record_set,
                                   frames_per_batch, global_step)

        self._save_checkpoint(last_iter, "final")

    def _train_on_policy(self, num_iters, last_iter, record_set, fpb, global_step):
        for i, batch in enumerate(self._collector):
            t0      = time.time()
            metrics = self.algorithm.update(batch)

            rewards     = batch.get(("next", "reward")).flatten()
            global_step += fpb
            fps          = fpb / max(time.time() - t0, 1e-6)
            self._log_and_print(i, last_iter, global_step, rewards.numpy(), metrics, fps)

            if i in record_set:
                self._record_episode(i, global_step)
            if (i + 1) % self.cfg.log.checkpoint_every == 0:
                self._save_checkpoint(i, "periodic")
            if i % self.cfg.log.histogram_every == 0:
                self.logger.log_histogram(global_step, self.algorithm.named_parameters())

            self._collector.update_policy_weights_()

    def _train_off_policy(self, num_iters, last_iter, record_set, fpb, global_step):
        algo        = self.algorithm
        ep_rewards  = []
        obs         = self.adapter.reset()
        ep_reward   = 0.0
        ep_steps    = 0

        for i in range(num_iters):
            t0 = time.time()

            # Collect via adapter bridge
            for _ in range(fpb):
                # env → network input
                obs_t = self.adapter.get_observation(obs, device=algo.device)
                # network output
                raw_out, _ = algo.select_action(obs_t, eval_mode=False)
                # network output → env action
                action = self.adapter.get_action(raw_out)

                nobs, reward, done, _ = self.adapter.step(action)
                ep_reward += reward
                ep_steps  += 1

                nobs_t = self.adapter.get_observation(nobs, device=algo.device)
                algo._buf.push(obs_t, raw_out, reward, nobs_t, done)
                algo._increment_steps()
                obs = nobs if not done else self.adapter.reset()

                if done:
                    ep_rewards.append(ep_reward)
                    algo.on_episode_end(ep_reward, ep_steps)
                    ep_reward = 0.0
                    ep_steps  = 0

            metrics     = algo.update()
            global_step += fpb
            fps          = fpb / max(time.time() - t0, 1e-6)
            recent       = np.array(ep_rewards[-10:] or [0.0])

            self._log_and_print(i, last_iter, global_step, recent, metrics, fps)

            if i in record_set:
                self._record_episode(i, global_step)
            if (i + 1) % self.cfg.log.checkpoint_every == 0:
                self._save_checkpoint(i, "periodic")
            if i % self.cfg.log.histogram_every == 0:
                self.logger.log_histogram(global_step, algo.named_parameters())

    def _log_and_print(self, i, last_iter, global_step, rewards, metrics, fps):
        log_metrics = {
            "mean_reward": round(float(np.mean(rewards)), 4),
            "max_reward":  round(float(np.max(rewards)), 4),
            "min_reward":  round(float(np.min(rewards)), 4),
            "std_reward":  round(float(np.std(rewards)), 4),
            "fps":         round(fps, 1),
            **{k: round(v, 6) for k, v in (metrics or {}).items()},
        }
        self.logger.log(i, global_step, log_metrics)

        gnorm = sum(
            p.grad.norm().item() ** 2
            for _, p in self.algorithm.named_parameters()
            if p.grad is not None
        ) ** 0.5
        self.logger.writer_add_scalar("grad/total_norm", gnorm, global_step)

        loss_str = " | ".join(
            f"{k.split('/')[-1]}={v:.4f}"
            for k, v in (metrics or {}).items() if v
        )
        print(
            f"[{self.algorithm.ALGORITHM_NAME.upper()}/{getattr(self.algorithm.network_cfg,'name','?').upper()}] "
            f"Iter {i:4d}/{last_iter} | step {global_step:7,} | "
            f"rew {log_metrics['mean_reward']:7.3f} | {loss_str} | fps {fps:5.0f}"
        )

    def _record_episode(self, iteration: int, global_step: int):
        """
        Uses adapter.get_observation() and adapter.get_action() throughout.
        No raw obs handling in the trainer.
        """
        vid_folder = f"{self.logger.vid_dir}/iter_{iteration:04d}"
        self.adapter.start_recording(vid_folder)
        obs          = self.adapter.reset()
        done         = False
        total_reward = 0.0

        while not done:
            obs_t           = self.adapter.get_observation(obs, device=self.algorithm.device)
            raw_out, _      = self.algorithm.select_action(obs_t, eval_mode=True)
            action          = self.adapter.get_action(raw_out)
            obs, rew, done, _ = self.adapter.step(action)
            total_reward    += rew

        self.adapter.stop_recording()
        self.logger.log_video_reward(global_step, total_reward)
        print(f"  [Video] iter={iteration} | reward={total_reward:.1f} | → {vid_folder}")

    def _save_checkpoint(self, iteration: int, tag: str = ""):
        name = f"ckpt_{self.algorithm.ALGORITHM_NAME}_iter{iteration:04d}"
        if tag: name += f"_{tag}"
        path = f"{self.logger.ckpt_dir}/{name}.pt"
        torch.save({
            "iteration":   iteration,
            "global_step": self.algorithm.total_steps,
            "algorithm":   self.algorithm.ALGORITHM_NAME,
            "network":     getattr(self.algorithm.network_cfg, "name", "?"),
            "state_dict":  self.algorithm.state_dict(),
            "obs_space":   self.algorithm.obs_space,
            "act_space":   self.algorithm.act_space,
            "algo_cfg":    self.algorithm.algo_cfg,
            "network_cfg": self.algorithm.network_cfg,
        }, path)
        print(f"  [Checkpoint] → {path}")

    def teardown(self):
        if self._collector:
            self._collector.shutdown()
