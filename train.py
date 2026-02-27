"""
train.py
========
Run from the project root after installing the package:

    pip install -e .
    python train.py --algo ppo --net mlp --env CartPole-v1

Or without installing (adds project root to sys.path automatically):

    python train.py --algo ppo --net mlp --env CartPole-v1

Three independent axes — all switchable via CLI:
  --algo     ppo | sac              algorithm
  --net      mlp | cnn              network architecture
  --backend  gymnasium | isaac | omniverse   env backend

Examples
--------
python train.py --algo ppo --net mlp --env CartPole-v1
python train.py --algo sac --net mlp --env Pendulum-v1
python train.py --algo ppo --net cnn --env ALE/Pong-v5    # needs: pip install ale-py
python train.py --algo ppo --backend isaac --env Isaac-Cartpole-v0 --device cuda:0
"""

import argparse
import sys
import os

# --- Make the package importable when run directly (no install needed) ---
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
# -------------------------------------------------------------------------

from rl_v2.configs    import Config, EnvConfig, LogConfig
from rl_v2.envs       import make_env_adapter
from rl_v2.networks   import MLPConfig, CNNConfig
from rl_v2.algorithms import make_algorithm, list_algorithms, PPOConfig, SACConfig
from rl_v2.loggers    import RunLogger
from rl_v2.trainers   import Trainer


def build_arg_parser():
    p = argparse.ArgumentParser(description="RL v2 — Universal Trainer")
    p.add_argument("--algo",    type=str, default="ppo",
                   help=f"Algorithm. Available: {list_algorithms()}")
    p.add_argument("--net",     type=str, default="mlp", choices=["mlp", "cnn"],
                   help="Network architecture")
    p.add_argument("--env",     type=str, default="CartPole-v1",
                   help="Gymnasium env ID or simulator task name")
    p.add_argument("--backend", type=str, default="gymnasium",
                   choices=["gymnasium", "isaac", "omniverse"],
                   help="Environment backend")
    p.add_argument("--device",  type=str, default=None,
                   help="Torch device: cpu | cuda:0 | mps. If omitted the code will auto‑select cuda if available.")
    p.add_argument("--total_frames", type=int, default=100_000)
    p.add_argument("--lr",      type=float, default=3e-4)
    p.add_argument("--hidden",  type=int,   default=256,
                   help="Hidden layer size (MLP only)")
    p.add_argument("--layers",  type=int,   default=2,
                   help="Number of hidden layers (MLP only)")
    p.add_argument("--record",  type=str,   default="0,5,-1",
                   help="Comma-separated iteration indices to record video")
    p.add_argument("--checkpoint_every", type=int, default=20)
    p.add_argument("--run_dir", type=str,   default=None,
                   help="Override auto-generated run directory")
    p.add_argument("--list_algos", action="store_true",
                   help="Print available algorithms and exit")
    return p


def main():
    args = build_arg_parser().parse_args()

    # auto‑detect device if user didn't specify one
    if args.device is None:
        try:
            import torch
            args.device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            args.device = "cpu"

    if args.list_algos:
        print(f"Available algorithms: {list_algorithms()}")
        sys.exit(0)

    # ── Network config ────────────────────────────────────────────────
    if args.net == "mlp":
        network_cfg = MLPConfig(hidden_sizes=[args.hidden] * args.layers)
    elif args.net == "cnn":
        network_cfg = CNNConfig()
    else:
        print(f"Unknown network '{args.net}'")
        sys.exit(1)

    # ── Algorithm config ──────────────────────────────────────────────
    _algo_cfg_map = {"ppo": PPOConfig, "sac": SACConfig}
    AlgoCfgCls = _algo_cfg_map.get(args.algo.lower())
    if AlgoCfgCls is None:
        print(f"Unknown algorithm '{args.algo}'. Available: {list_algorithms()}")
        sys.exit(1)

    algo_cfg = AlgoCfgCls()
    algo_cfg.total_frames = args.total_frames
    for attr in ("lr", "lr_actor", "lr_critic"):
        if hasattr(algo_cfg, attr):
            setattr(algo_cfg, attr, args.lr)

    # ── Master config ─────────────────────────────────────────────────
    cfg = Config(
        env=EnvConfig(
            env_name=args.env,
            env_backend=args.backend,
            device=args.device,
        ),
        log=LogConfig(
            run_dir=args.run_dir,
            record_sessions=[int(x) for x in args.record.split(",")],
            checkpoint_every=args.checkpoint_every,
        ),
    )

    # ── Build + connect ───────────────────────────────────────────────
    print(f"\nAlgorithm : {args.algo.upper()}")
    print(f"Network   : {args.net.upper()}  hidden={[args.hidden]*args.layers}")
    print(f"Env       : {args.env}  ({args.backend})")
    print(f"Device    : {args.device}")

    adapter = make_env_adapter(cfg.env)
    obs_space, act_space = adapter.setup()

    print(f"\nObs space : {obs_space}")
    print(f"Act space : {act_space}")

    algorithm = make_algorithm(algo_cfg, obs_space, act_space, network_cfg)
    algorithm.setup(device=args.device)
    print(f"\n{algorithm}\n")

    logger  = RunLogger(cfg, algorithm_name=args.algo)
    logger.setup()

    trainer = Trainer(cfg, algorithm, adapter, logger)
    trainer.setup()

    # ── Train ─────────────────────────────────────────────────────────
    try:
        trainer.train()
    finally:
        trainer.teardown()
        adapter.close()
        logger.close()

    print(f"\nDone.")
    print(f"Run dir    : {logger.run_dir}")
    print(f"TensorBoard: tensorboard --logdir {logger.tb_dir}")


if __name__ == "__main__":
    main()
