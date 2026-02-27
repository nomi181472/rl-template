from setuptools import setup, find_packages

setup(
    name="rl_v2",
    version="0.1.0",
    description="Algorithm, network, and env-backend agnostic RL framework",
    python_requires=">=3.9",
    packages=find_packages(),   # finds rl_v2 and all its sub-packages
    install_requires=[
        "torch>=2.0",
        "torchrl>=0.3",
        "tensordict>=0.3",
        "numpy>=1.24",
        "gymnasium>=0.29",
        "tensorboard>=2.13",
    ],
    extras_require={
        "classic": ["gymnasium[classic-control]", "pygame"],
        "atari":   ["gymnasium[atari]", "ale-py"],
        "mujoco":  ["gymnasium[mujoco]"],
        "video":   ["moviepy"],
    },
)
