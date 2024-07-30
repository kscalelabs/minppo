<p align="center">
  <picture>
    <img alt="K-Scale Open Source Robotics" src="https://media.kscale.dev/kscale-open-source-header.png" style="max-width: 100%;">
  </picture>
</p>

<div align="center">

[![License](https://img.shields.io/badge/license-MIT-green)](https://github.com/kscalelabs/ksim/blob/main/LICENSE)
[![Discord](https://img.shields.io/discord/1224056091017478166)](https://discord.gg/k5mSvCkYQh)
[![Wiki](https://img.shields.io/badge/wiki-humanoids-black)](https://humanoids.wiki)
<br />
[![python](https://img.shields.io/badge/-Python_3.11-blue?logo=python&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![black](https://img.shields.io/badge/Code%20Style-Black-black.svg?labelColor=gray)](https://black.readthedocs.io/en/stable/)
[![ruff](https://img.shields.io/badge/Linter-Ruff-red.svg?labelColor=gray)](https://github.com/charliermarsh/ruff)
<br />
[![Python Checks](https://github.com/kscalelabs/humanoid-standup/actions/workflows/test.yml/badge.svg)](https://github.com/kscalelabs/humanoid-standup/actions/workflows/test.yml)

</div>

# Humanoid Standup

Minimal training and inference code for making a humanoid robot stand up.

## Getting started
- `export DISPLAY=:0` if in a headless environmnet
- Run `train.py`!

## Learned (incomplete)
- `vmap` can specify axes to vectorize over. For dictionaries, have to create our own broadcasting
- Minibatches for vmapping
- Have to specify different random keys by device axis in RNG

## Room for improvement
- Training step 
- Initialize weights different
- Loss fucntion?
- Make sure each input batch has enough data?

## TODO

- [ ] Implement simple MJX environment using [Unitree G1 simulation artifacts](https://humanoids.wiki/w/Robot_Descriptions_List), similar to [this](https://gymnasium.farama.org/environments/mujoco/humanoid_standup)
- [ ] Implement simple PPO policy to try to make the robot stand up

## Goals

- The goal for this repository is to provide a super minimal implementation of a PPO policy for making a humanoid robot stand up, with only three files:
  - `environment.py` defines a class for interacting with the environment
  - `train.py` defines the core training loop
  - `infer.py` generates a video clip of a trained model controlling a robot
