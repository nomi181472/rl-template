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
        # Box2D environments require the extra gymnasium[box2d] and a
        # system SWIG installation.  Users can install via:
        #     sudo apt-get install swig   # or equivalent for your OS
        # and then use pip extras as shown below.
        "box2d":  ["gymnasium[box2d]"],
        "atari":   ["gymnasium[atari]", "ale-py"],
        "mujoco":  ["gymnasium[mujoco]"],
        "video":   ["moviepy"],
    },
)
