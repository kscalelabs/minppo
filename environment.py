"""Definition of base humanoids environment with reward system and termination conditions."""

import argparse
import logging
import os
from typing import Any

import jax
import jax.numpy as jp
import mujoco
from brax import base
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf
from brax.mjx.base import State as MjxState

logger = logging.getLogger(__name__)


class HumanoidEnv(PipelineEnv):
    initial_qpos: jp.ndarray
    _action_size: int
    reset_noise_scale: float = 2e-4

    def __init__(self) -> None:
        """Initializes system with initial joint positions, action size, the model, and update rate."""
        path: str = os.path.join(os.path.dirname(__file__), "environments", "stompy", "legs.xml")
        mj_model: mujoco.MjModel = mujoco.MjModel.from_xml_path(path)
        # mj_data: mujoco.MjData = mujoco.MjData(mj_model)
        # renderer: mujoco.Renderer = mujoco.Renderer(mj_model)
        self.initial_qpos = jp.array(mj_model.keyframe("default").qpos)
        self._action_size = mj_model.nu
        sys: base.System = mjcf.load_model(mj_model)

        physics_steps_per_control_step: int = 4
        super().__init__(sys, n_frames=physics_steps_per_control_step, backend="mjx")

    def reset(self, rng: jp.ndarray) -> State:
        """Resets the environment to an initial state."""
        rng, rng1, rng2 = jax.random.split(rng, 3)

        low, hi = -self.reset_noise_scale, self.reset_noise_scale

        # qpos = self.sys.qpos0 + jax.random.uniform(rng1, (self.sys.nq,), minval=low, maxval=hi)
        qpos = self.initial_qpos + jax.random.uniform(rng1, (self.sys.nq,), minval=low, maxval=hi)
        qvel = jax.random.uniform(rng2, (self.sys.nv,), minval=low, maxval=hi)

        # initialize mjx state
        state = self.pipeline_init(qpos, qvel)
        obs = self.get_obs(state, jp.zeros(self._action_size))
        reward, done, zero = jp.zeros(3)

        metrics: dict[str, Any] = {}

        return State(state, obs, reward, done, metrics)

    def step(self, env_state: State, action: jp.ndarray) -> State:
        """Run one timestep of the environment's dynamics and returns observations with rewards."""
        state = env_state.pipeline_state
        next_state = self.pipeline_step(state, action)
        obs = self.get_obs(state, action)

        reward = self.compute_reward(state, next_state, action)

        # Termination condition
        done = self.is_done(next_state)

        return env_state.replace(pipeline_state=next_state, obs=obs, reward=reward, done=done)

    def compute_reward(self, state: MjxState, next_state: MjxState, action: jp.ndarray) -> jp.ndarray:
        """Compute the reward for standing and height."""
        min_z, max_z = -1, 2.0
        is_healthy = jp.where(state.q[2] < min_z, 0.0, 1.0)
        is_healthy = jp.where(state.q[2] > max_z, 0.0, is_healthy)

        ctrl_cost = -jp.sum(jp.square(action))

        xpos = state.subtree_com[1][0]
        next_xpos = next_state.subtree_com[1][0]
        velocity = (next_xpos - xpos) / self.dt

        jax.debug.print(
            "velocity {}, xpos {}, next_xpos {}",
            velocity,
            xpos,
            next_xpos,
            ordered=True,
        )
        jax.debug.print("is_healthy {}, height {}", is_healthy, state.q[2], ordered=True)

        total_reward = 5.0 * is_healthy + 1.25 * velocity + 0.1 * ctrl_cost

        return total_reward

    def is_done(self, state: MjxState) -> jp.ndarray:
        """Check if the episode should terminate."""
        # Get the height of the robot's center of mass
        com_height = state.q[2]

        # Set a termination threshold
        termination_height = -1

        # Episode is done if the robot falls below the termination height
        done = jp.where(com_height < termination_height, 1.0, 0.0)

        return done

    def get_obs(self, data: MjxState, action: jp.ndarray) -> jp.ndarray:
        """Returns the observation of the environment to pass to actor/critic model."""
        position = data.qpos
        position = position[2:]  # excludes "current positions"

        # external_contact_forces are excluded
        return jp.concatenate(
            [
                position,
                data.qvel,
                data.cinert[1:].ravel(),
                data.cvel[1:].ravel(),
                data.qfrc_actuator,
            ]
        )


def run_environment_adhoc() -> None:
    """Runs the environment for a few steps with random actions, for debugging."""
    import mediapy as media
    from tqdm import tqdm

    parser = argparse.ArgumentParser()
    parser.add_argument("--actor_path", type=str, default="actor_params.pkl", help="path to actor model")
    parser.add_argument("--critic_path", type=str, default="critic_params.pkl", help="path to critic model")
    parser.add_argument("--num_episodes", type=int, default=20, help="number of episodes to run")
    parser.add_argument("--max_steps", type=int, default=200, help="maximum steps per episode")
    parser.add_argument("--video_path", type=str, default="inference_video.mp4", help="path to save video")
    parser.add_argument("--render_every", type=int, default=2, help="how many frames to skip between renders")
    parser.add_argument("--video_length", type=float, default=10.0, help="desired length of video in seconds")
    parser.add_argument("--width", type=int, default=640, help="width of the video frame")
    parser.add_argument("--height", type=int, default=480, help="height of the video frame")
    args = parser.parse_args()

    env = HumanoidEnv()
    action_size = env.action_size

    rng = jax.random.PRNGKey(0)
    reset_fn = jax.jit(env.reset)
    step_fn = jax.jit(env.step)

    fps = int(1 / env.dt)
    max_frames = int(args.video_length * fps)
    rollout: list[Any] = []

    for episode in range(args.num_episodes):
        rng, _ = jax.random.split(rng)
        state = reset_fn(rng)

        total_reward = 0

        for step in tqdm(range(args.max_steps), desc=f"Episode {episode + 1} Steps", leave=False):
            if len(rollout) < args.video_length * fps:
                rollout.append(state.pipeline_state)

            action = jax.random.uniform(rng, (action_size,), minval=-1.0, maxval=1.0)  # placeholder for an action
            state = step_fn(state, action)
            total_reward += state.reward

            if state.done:
                break

        logging.info(f"Episode {episode + 1} total reward: {total_reward}")

        if len(rollout) >= max_frames:
            break

    # Trim rollout to desired length
    rollout = rollout[:max_frames]

    images = jp.array(env.render(rollout[:: args.render_every], camera="side", width=args.width, height=args.height))
    logging.info(f"Rendering video with {len(images)} frames at {fps} fps")
    media.write_video(args.video_path, images, fps=fps)
    logging.info(f"Video saved to {args.video_path}")


if __name__ == "__main__":
    run_environment_adhoc()
