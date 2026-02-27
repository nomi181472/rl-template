"""
runners/inference_runner.py
===========================
Algorithm + env agnostic inference runner.
Uses adapter.get_observation() and adapter.get_action() for the full bridge.
"""

from __future__ import annotations
import json, time, datetime
import numpy as np
from typing import List, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..algorithms.base import BaseAlgorithm
    from ..envs.base import BaseEnvAdapter


class InferenceRunner:

    def __init__(self, algorithm: "BaseAlgorithm", adapter: "BaseEnvAdapter",
                 max_steps: int = 10_000):
        self.algorithm = algorithm
        self.adapter   = adapter
        self.max_steps = max_steps

    def run_episodes(self, n=5, deterministic=True, record=False,
                     record_dir=None, verbose=True) -> List[Dict]:
        self.algorithm.eval_mode()
        results = []
        for ep in range(n):
            if record:
                self.adapter.start_recording(record_dir or f"inference_videos/ep_{ep:03d}")
            result = self._run_one(ep, deterministic)
            if record:
                result["video_path"] = self.adapter.stop_recording()
            if verbose:
                print(f"  Ep {ep:3d} | reward={result['total_reward']:8.2f} | "
                      f"steps={result['steps']:5d} | {result['elapsed_sec']:.2f}s")
            results.append(result)
        return results

    def benchmark(self, n=100, verbose=True) -> Dict:
        if verbose:
            print(f"[Benchmark] {n} episodes | algo={self.algorithm.ALGORITHM_NAME} | "
                  f"net={getattr(self.algorithm.network_cfg,'name','?')}\n")
        results = self.run_episodes(n, deterministic=True, verbose=verbose)
        stats   = self.compute_stats(results)
        self.print_summary(results)
        return stats

    def _run_one(self, ep_idx: int, deterministic: bool) -> Dict:
        obs          = self.adapter.reset()
        total_reward = 0.0
        steps        = 0
        t0           = time.time()

        for _ in range(self.max_steps):
            # env → network input
            obs_t = self.adapter.get_observation(obs, device=self.algorithm.device)
            # algorithm → raw network output
            raw_out, _ = self.algorithm.select_action(obs_t, eval_mode=deterministic)
            # network output → env action
            action = self.adapter.get_action(raw_out)

            obs, reward, done, _ = self.adapter.step(action)
            total_reward += reward
            steps        += 1
            if done: break

        return {
            "episode":       ep_idx,
            "total_reward":  total_reward,
            "steps":         steps,
            "elapsed_sec":   round(time.time() - t0, 3),
            "algorithm":     self.algorithm.ALGORITHM_NAME,
            "network":       getattr(self.algorithm.network_cfg, "name", "?"),
            "deterministic": deterministic,
        }

    @staticmethod
    def compute_stats(results: List[Dict]) -> Dict:
        rewards = [r["total_reward"] for r in results]
        steps   = [r["steps"]        for r in results]
        return {
            "n_episodes":  len(results),
            "mean_reward": round(float(np.mean(rewards)), 4),
            "std_reward":  round(float(np.std(rewards)),  4),
            "min_reward":  round(float(np.min(rewards)),  4),
            "max_reward":  round(float(np.max(rewards)),  4),
            "mean_steps":  round(float(np.mean(steps)),   1),
        }

    @staticmethod
    def print_summary(results: List[Dict]):
        stats = InferenceRunner.compute_stats(results)
        algo  = results[0].get("algorithm", "?") if results else "?"
        net   = results[0].get("network",   "?") if results else "?"
        print(f"""
+------------------------------------------------------+
|  Inference Summary  [{algo.upper()} / {net.upper()}]
|  Episodes    : {stats['n_episodes']}
|  Reward mean : {stats['mean_reward']:10.2f}
|         std  : {stats['std_reward']:10.2f}
|         min  : {stats['min_reward']:10.2f}
|         max  : {stats['max_reward']:10.2f}
+------------------------------------------------------+""")

    @staticmethod
    def save_results(results: List[Dict], path: str):
        with open(path, "w") as f:
            json.dump({
                "timestamp": datetime.datetime.now().isoformat(),
                "summary":   InferenceRunner.compute_stats(results),
                "episodes":  results,
            }, f, indent=2)
        print(f"Results saved → {path}")
