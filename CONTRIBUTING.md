# Contributing

We love your input! We want to make contributing to this project as easy and transparent as possible, whether it's:

- Reporting a bug
- Discussing the current state of the code
- Submitting a fix
- Proposing new features
- Becoming a maintainer

## Getting Started

- Fork the repo and clone it on your machine.
- Create a branch (`git checkout -b feature/myNewFeature`).

## Your First Contribution

Unsure where to begin contributing to our code? You can start by looking through the "TODO" list in the README.md or looking at the issues 


## Setting Up Development Environment

The repository is currently (and purposefully) very lightweight. `requirements.txt` is all you need! If you come acros any errors regarding MuJoCo's rendering, you may have to `export DISPLAY=:0` on a headless environment utilizing `xvfb`. If you are still getting errors, try `export MUJOCO_GL=egl`


## Pull Request Process

1. Ensure any install or build dependencies are removed before the end of the layer when doing a build through `pip install -r requirements.txt`.
2. Update the README.md with details of changes, this includes new features, further documentation, or possible TODOs.
3. You may merge the Pull Request in once you have the sign-off of two other developers, or if you do not have permission to do that, you may request the second reviewer to merge it for you.
4. Follow the linting process shown below.

To run linting, use:

```bash
black *.py
ruff check --fix *.py
```
