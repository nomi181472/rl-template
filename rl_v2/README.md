# rl_v2 — Three Independent Plugin Systems

## Quick start

Run the trainer or inference script from the project root after installing the package:

```bash
# install editable copy and required packages
pip install -e .
pip install torchrl tensordict gymnasium[classic-control] tensorboard moviepy

# example training runs
python train.py --algo ppo --net mlp --env CartPole-v1
python train.py --algo sac --net mlp --env Pendulum-v1
python train.py --algo ppo --net cnn --env ALE/Pong-v5    # requires ale-py

# list available algorithms
python train.py --list_algos
```

You can also import `rl_v2` and use `infer.py` for running pre‑trained models.

## Architecture

```
rl_v2/
├── envs/
│   ├── spaces.py              ← ObservationSpace + ActionSpace  ← NEW
│   ├── base.py                ← BaseEnvAdapter
│   │     get_observation()    ← raw obs  → network tensor       ← NEW
│   │     get_action()         ← net out  → env action           ← NEW
│   ├── gymnasium_adapter.py   ✅
│   ├── isaac_adapter.py       🔧 stub
│   └── omniverse_adapter.py   🔧 stub
│
├── networks/                                                     ← NEW
│   ├── base.py                ← BaseNetwork + BaseValueNetwork
│   ├── mlp.py                 ✅ MLPPolicy, MLPValue, TwinMLPValue
│   ├── cnn.py                 ✅ CNNPolicy, CNNValue
│   └── factory.py             ← make_policy_network(), make_value_network()
│
├── algorithms/
│   ├── base.py                ← BaseAlgorithm (takes obs/act space + net cfg)
│   ├── ppo.py                 ✅ uses network factory
│   ├── sac.py                 ✅ uses network factory
│   └── factory.py
│
├── configs/ loggers/ trainers/ runners/
└── train.py  infer.py
```

---

## Data flow

```
                    ┌─────────────┐
                    │    ENV      │
                    └──────┬──────┘
                           │ raw_obs (numpy)
                           ▼
          adapter.get_observation(raw_obs, device)
                           │
                           │ obs_tensor (B, input_dim)  ← network input
                           ▼
                    ┌─────────────┐
                    │   NETWORK   │
                    └──────┬──────┘
                           │ raw_output (B, output_dim)  ← network output
                           ▼
             adapter.get_action(raw_output)
                           │
                           │ action  ← valid env action
                           ▼
                    ┌─────────────┐
                    │    ENV      │
                    └─────────────┘
```

The algorithm NEVER sees raw numpy arrays. The env NEVER sees tensors.

---

## Three independent axes of extension

### 1. New environment backend
Implement `setup() → (ObservationSpace, ActionSpace)`, `reset()`, `step()`, `close()`.
Override `get_observation()` and `get_action()` for custom preprocessing.

```python
class MySimAdapter(BaseEnvAdapter):
    def setup(self):
        obs_space = ObservationSpace(shape=(48,), flat_dim=48)
        act_space = ActionSpace(ActionType.CONTINUOUS, shape=(6,))
        self._set_spaces(obs_space, act_space)
        return obs_space, act_space

    def get_observation(self, raw_obs, device="cpu", normalise=False):
        # Custom: concatenate sensor modalities, run normalisation, etc.
        return torch.tensor(my_preprocess(raw_obs), device=device).unsqueeze(0)

    def get_action(self, raw_network_output, clip=True):
        # Custom: rescale from [-1,1] to joint torque limits
        return rescale_to_torque_limits(raw_network_output)
```

### 2. New network architecture

```python
# networks/lstm.py
from rl_v2.networks.base import BaseNetwork

class LSTMPolicy(BaseNetwork):
    def __init__(self, obs_space, act_space, cfg):
        super().__init__(obs_space, act_space, cfg)
        self.lstm = nn.LSTM(self.input_dim, cfg.hidden_size, cfg.num_layers)
        self.head  = nn.Linear(cfg.hidden_size, self.output_dim)

    def forward(self, obs):                  # obs: (batch, seq, input_dim)
        out, _ = self.lstm(obs)
        return self.head(out[:, -1])          # last hidden → action

    @classmethod
    def build(cls, obs_space, act_space, cfg, **kw):
        return cls(obs_space, act_space, cfg)

# Register it
from rl_v2.networks import register_network
register_network("lstm", "my_pkg.networks.lstm:LSTMPolicy")

# Use it
python train.py --net lstm --algo ppo --env CartPole-v1
```

### 3. New algorithm

```python
# algorithms/dreamer.py
from rl_v2.algorithms.base import BaseAlgorithm

class DreamerAlgorithm(BaseAlgorithm):
    ALGORITHM_NAME = "dreamer"
    ON_POLICY      = False

    def setup(self, device):
        # Build world model, actor, critic using network factory
        self._world_model = make_policy_network(self.obs_space, ...)
        ...

    def select_action(self, obs_tensor, eval_mode=False):
        # obs_tensor already preprocessed by adapter.get_observation()
        return self._actor(obs_tensor).numpy(), None

    def collect_data(self, adapter, n_steps):
        obs = adapter.reset()
        for _ in range(n_steps):
            obs_t  = adapter.get_observation(obs)      # ← bridge
            raw, _ = self.select_action(obs_t)
            action = adapter.get_action(raw)            # ← bridge
            nobs, r, done, _ = adapter.step(action)
            ...

    def update(self, batch) -> dict:
        # train world model + actor + critic
        return {"loss/world_model": ..., "loss/actor": ...}

# Register
from rl_v2.algorithms import register_algorithm
register_algorithm("dreamer", "my_pkg.dreamer:DreamerAlgorithm")
```

---

## Quick start

```bash
pip install torchrl tensordict gymnasium[classic-control] tensorboard moviepy pygame

# MLP + PPO
python train.py --algo ppo --net mlp --env CartPole-v1

# MLP + SAC (continuous)
python train.py --algo sac --net mlp --env Pendulum-v1

# CNN + PPO (Atari — needs ale-py)
pip install ale-py
python train.py --algo ppo --net cnn --env ALE/Pong-v5

# Isaac Lab (once installed)
python train.py --algo ppo --net mlp --backend isaac --env Isaac-Cartpole-v0 --device cuda:0
```

---

## What ObservationSpace + ActionSpace give you

| Field | Used by |
|-------|---------|
| `obs_space.network_input_dim` | Network first-layer size |
| `obs_space.preprocess(raw)` | Default `get_observation()` |
| `obs_space.space_type` | Router between MLP / CNN |
| `act_space.network_output_dim` | Network last-layer size |
| `act_space.postprocess(raw)` | Default `get_action()` |
| `act_space.is_discrete` | Actor head type selection |
| `act_space.sample()` | Warmup / random baseline |

The network is built purely from these descriptors.
No env-specific code ever enters the network or algorithm.
