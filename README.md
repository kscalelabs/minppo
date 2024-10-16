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

# MinPPO

This repository implements a minimal version of PPO using Jax.

## Usage

To visualize the environment, run:

```bash
python environment.py configs/stompy_pro.yaml
```

To train the model, run:

```bash
python train.py configs/stompy_pro.yaml
```

To run inference on the trained model, run:

```bash
python infer.py configs/stompy_pro.yaml 'inference.model_path=path/to/trained/model.pkl'
```
