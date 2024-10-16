"""Runs inference for the trained model."""

import logging
import os
import pickle
import sys
from typing import Sequence

os.environ["MUJOCO_GL"] = "egl"
os.environ["DISPLAY"] = ":0"

# Add logger configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_model(filename: str) -> dict:
    with open(filename, "rb") as f:
        return pickle.load(f)


def main(args: Sequence[str] | None = None) -> None:
    """Runs inference with pretrained models."""
    if args is None:
        args = sys.argv[1:]

    raise NotImplementedError("Not implemented yet")


if __name__ == "__main__":
    main()
