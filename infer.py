"""Runs inference for the trained model."""

import argparse
import pickle
from typing import Any, Tuple

import jax
import jax.numpy as jnp
import mediapy as media
from jax import Array

from environment import HumanoidEnv
from train import Actor, choose_action


def load_models(actor_path: str, critic_path: str) -> Tuple[Array, Array]:
    with open(actor_path, "rb") as f:
        actor_params = pickle.load(f)
    with open(critic_path, "rb") as f:
        critic_params = pickle.load(f)
    return actor_params, critic_params


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--actor_path", type=str, default="actor_params.pkl", help="path to actor model")
    parser.add_argument("--critic_path", type=str, default="critic_params.pkl", help="path to critic model")
    parser.add_argument("--num_episodes", type=int, default=1, help="number of episodes to run")
    parser.add_argument("--max_steps", type=int, default=1000, help="maximum steps per episode")
    parser.add_argument("--video_path", type=str, default="inference_video.mp4", help="path to save video")
    parser.add_argument("--render_every", type=int, default=2, help="how many frames to skip between renders")
    parser.add_argument("--video_length", type=float, default=10.0, help="desired length of video in seconds")
    parser.add_argument("--width", type=int, default=640, help="width of the video frame")
    parser.add_argument("--height", type=int, default=480, help="height of the video frame")
    args = parser.parse_args()

    env = HumanoidEnv()
    actor_params, _ = load_models(args.actor_path, args.critic_path)
    actor = Actor(action_size=env.action_size)

    reset_fn = jax.jit(env.reset)
    step_fn = jax.jit(env.step)

    fps = int(1 / env.dt)
    max_frames = int(args.video_length * fps)
    rollout: list[Any] = []

    for episode in range(args.num_episodes):
        rng = jax.random.PRNGKey(episode)
        state = reset_fn(rng)
        obs = state.obs

        total_reward = 0

        for step in range(args.max_steps):
            if len(rollout) < max_frames:
                rollout.append(state.pipeline_state)

            action = choose_action(actor_params, obs, actor)
            state = step_fn(state, action)
            obs = state.obs
            total_reward += state.reward

            if state.done:
                break

        print(f"Episode {episode + 1} total reward: {total_reward}")

        if len(rollout) >= max_frames:
            break

    # Trim rollout to desired length
    rollout = rollout[:max_frames]

    images = jnp.array(env.render(rollout[:: args.render_every], camera="side", width=args.width, height=args.height))
    print(f"Rendering video with {len(images)} frames at {fps} fps")
    media.write_video(args.video_path, images, fps=fps)
    print(f"Video saved to {args.video_path}")


if __name__ == "__main__":
    main()
