"""
infer.py
========
Run inference on a saved checkpoint.

    pip install -e .
    python infer.py --mode benchmark --episodes 100
    python infer.py --checkpoint runs/.../ckpt_ppo_iter0099_final.pt --mode watch
    python infer.py --checkpoint runs/.../ckpt_sac_iter0099_final.pt --mode record --episodes 5
"""

import argparse
import glob
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import torch

from rl_v2.configs    import EnvConfig
from rl_v2.envs       import make_env_adapter
from rl_v2.networks   import MLPConfig, CNNConfig
from rl_v2.algorithms import make_algorithm, PPOConfig, SACConfig
from rl_v2.runners    import InferenceRunner


def find_latest_checkpoint():
    files = sorted(
        glob.glob(os.path.join("runs", "**", "checkpoints", "*.pt"), recursive=True)
    )
    if not files:
        print("No checkpoints found under runs/. Pass --checkpoint explicitly.")
        sys.exit(1)
    print(f"Auto-detected checkpoint: {files[-1]}")
    return files[-1]


def main():
    p = argparse.ArgumentParser(description="RL v2 — Inference")
    p.add_argument("--checkpoint",  type=str, default=None,
                   help="Path to .pt checkpoint. Auto-detects latest if omitted.")
    p.add_argument("--mode",        type=str, default="benchmark",
                   choices=["watch", "record", "benchmark"])
    p.add_argument("--episodes",    type=int, default=10)
    p.add_argument("--max_steps",   type=int, default=10_000)
    p.add_argument("--stochastic",  action="store_true",
                   help="Use stochastic policy (default: deterministic)")
    p.add_argument("--video_dir",   type=str, default="inference_videos")
    p.add_argument("--results_out", type=str, default=None)
    p.add_argument("--device",      type=str, default="cpu")
    p.add_argument("--env",         type=str, default=None,
                   help="Override env name from checkpoint")
    p.add_argument("--backend",     type=str, default=None,
                   help="Override backend from checkpoint")
    args = p.parse_args()

    # ── Load checkpoint ───────────────────────────────────────────────
    ckpt_path = args.checkpoint or find_latest_checkpoint()
    print(f"\nLoading: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=args.device)

    algo_name  = ckpt.get("algorithm", "ppo")
    saved_cfg  = ckpt.get("cfg", None)
    obs_space  = ckpt["obs_space"]
    act_space  = ckpt["act_space"]
    algo_cfg   = ckpt.get("algo_cfg", None)
    network_cfg = ckpt.get("network_cfg", None)

    # ── Rebuild algorithm ─────────────────────────────────────────────
    _cfg_map = {"ppo": PPOConfig, "sac": SACConfig}
    if algo_cfg is None:
        algo_cfg = _cfg_map.get(algo_name, PPOConfig)()
    if network_cfg is None:
        network_cfg = MLPConfig()

    algorithm = make_algorithm(algo_cfg, obs_space, act_space, network_cfg)
    algorithm.setup(device=args.device)
    algorithm.load_state_dict(ckpt["state_dict"])
    algorithm.eval_mode()
    print(f"Algorithm : {algorithm}")

    # ── Rebuild env ───────────────────────────────────────────────────
    backend  = args.backend or (saved_cfg.env.env_backend if saved_cfg else "gymnasium")
    env_name = args.env     or (saved_cfg.env.env_name    if saved_cfg else "CartPole-v1")
    render_mode = {"watch": "human", "record": "rgb_array", "benchmark": None}[args.mode]

    env_cfg = EnvConfig(
        env_name=env_name,
        env_backend=backend,
        render_mode=render_mode,
        device=args.device,
    )
    adapter = make_env_adapter(env_cfg)
    adapter.setup()
    print(f"Env       : {env_name}  ({backend})\n")

    # ── Run ───────────────────────────────────────────────────────────
    runner        = InferenceRunner(algorithm, adapter, max_steps=args.max_steps)
    deterministic = not args.stochastic

    if args.mode == "watch":
        results = runner.run_episodes(args.episodes, deterministic=deterministic)
    elif args.mode == "record":
        results = runner.run_episodes(
            args.episodes, deterministic=deterministic,
            record=True, record_dir=args.video_dir,
        )
    else:
        results = runner.run_episodes(
            args.episodes, deterministic=deterministic, verbose=True
        )

    runner.print_summary(results)

    out = args.results_out or f"inference_{algo_name}_{env_name.replace('/', '_')}.json"
    runner.save_results(results, out)
    adapter.close()


if __name__ == "__main__":
    main()
