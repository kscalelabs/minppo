"""Defines a command line interface for the package."""

import argparse

from minppo.env import main as env_main
from minppo.infer import main as infer_main
from minppo.train import main as train_main


def main() -> None:
    parser = argparse.ArgumentParser(description="MinPPO CLI")
    parser.add_argument("command", choices=["train", "env", "infer"], help="Command to run")
    args, other_args = parser.parse_known_args()

    if args.command == "train":
        train_main(other_args)
    elif args.command == "env":
        env_main(other_args)
    elif args.command == "infer":
        infer_main(other_args)
    else:
        raise ValueError(f"Invalid command: {args.command}")


if __name__ == "__main__":
    # python -m minppo.cli
    main()
