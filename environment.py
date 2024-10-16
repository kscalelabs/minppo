"""Definition of base humanoids environment with reward system and termination conditions."""

import argparse
import asyncio
import logging
import shutil
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import mujoco
from brax import base
from brax.envs.base import PipelineEnv
from brax.io import mjcf
from brax.mjx.base import State as MjxState
from kscale import KScale

logger = logging.getLogger(__name__)

DEFAULT_KSCALE_ID = "5eb3cb7f23232298"
INCLUDE_C_VALS = True
PHYSICS_FRAMES = 1


@dataclass
class RewardConfig:
    termination_height: float = field(default=-0.2)
    height_min_z: float = field(default=-0.2)
    height_max_z: float = field(default=2.0)
    is_healthy_reward: float = field(default=5)
    original_pos_reward_exp_coefficient: float = field(default=2)
    original_pos_reward_subtraction_factor: float = field(default=0.2)
    original_pos_reward_max_diff_norm: float = field(default=0.5)
    ctrl_cost_coefficient: float = field(default=0.1)
    weights_ctrl_cost: float = field(default=0.1)
    weights_original_pos_reward: float = field(default=4)
    weights_is_healthy: float = field(default=1)
    weights_velocity: float = field(default=1.25)


def load_mjcf_model(kscale_id: str) -> mujoco.MjModel:
    api = KScale()
    mjcf_path = asyncio.run(api.mjcf_path(kscale_id))

    # We need to fix up the MJCF model to allow it to work with Brax.
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_mjcf_dir = Path(temp_dir) / mjcf_path.parent.name
        shutil.copytree(mjcf_path.parent, temp_mjcf_dir)
        temp_mjcf_path = temp_mjcf_dir / mjcf_path.name
        tree = ET.parse(temp_mjcf_path)
        root = tree.getroot()

        # Updates the <mujoco><default><joint> element to remove frictionloss attrib.
        for joint in root.findall(".//default/joint"):
            if "frictionloss" in joint.attrib:
                del joint.attrib["frictionloss"]

        # Write the modified XML back to the file
        tree.write(temp_mjcf_path)

        model: mujoco.MjModel = mujoco.MjModel.from_xml_path(str(temp_mjcf_path))
        return model


@dataclass
class EnvMetrics:
    episode_returns: jnp.ndarray
    episode_lengths: jnp.ndarray
    returned_episode_returns: jnp.ndarray
    returned_episode_lengths: jnp.ndarray
    timestep: jnp.ndarray
    returned_episode: jnp.ndarray

    def tree_flatten(self) -> tuple[list[jnp.ndarray], None]:
        return (
            self.episode_returns,
            self.episode_lengths,
            self.returned_episode_returns,
            self.returned_episode_lengths,
            self.timestep,
            self.returned_episode,
        ), None

    @classmethod
    def tree_unflatten(cls, aux_data: Any, children: list[jnp.ndarray]) -> "EnvMetrics":  # noqa: ANN401
        return cls(*children)


jax.tree_util.register_pytree_node_class(EnvMetrics)


@dataclass
class EnvState:
    pipeline_state: Any  # Use Any for MjxState as it's not a standard JAX type
    obs: jnp.ndarray
    reward: jnp.ndarray
    done: jnp.ndarray
    metrics: EnvMetrics

    def tree_flatten(self) -> tuple[list[Any], None]:
        return (self.pipeline_state, self.obs, self.reward, self.done, self.metrics), None

    @classmethod
    def tree_unflatten(cls, aux_data: Any, children: list[Any]) -> "EnvState":  # noqa: ANN401
        return cls(*children)


jax.tree_util.register_pytree_node_class(EnvState)


class HumanoidEnv(PipelineEnv):
    """Defines the environment for controlling a humanoid robot.

    This environment uses Brax's `mjcf` module to load a MuJoCo model of a
    humanoid robot, which can then be controlled using the `PipelineEnv` API.

    Parameters:
        n_frames: The number of times to step the physics pipeline for each
            environment step. Setting this value to be greater than 1 means
            that the policy will run at a lower frequency than the physics
            simulation.
        backend: The backend to use for the physics simulation.
        kscale_id: The ID of the robot to load from K-Scale.
    """

    initial_qpos: jnp.ndarray
    _action_size: int
    reset_noise_scale: float = 0.0

    def __init__(
        self,
        n_frames: int = PHYSICS_FRAMES,
        backend: str = "mjx",
        kscale_id: str = DEFAULT_KSCALE_ID,
    ) -> None:
        # Loads the MJCF model.
        mj_model: mujoco.MjModel = load_mjcf_model(kscale_id)
        mj_model.opt.solver = mujoco.mjtSolver.mjSOL_CG
        mj_model.opt.iterations = 6
        mj_model.opt.ls_iterations = 6

        self._action_size = mj_model.nu
        sys: base.System = mjcf.load_model(mj_model)

        super().__init__(sys, n_frames=n_frames, backend=backend)

        self.initial_qpos = jnp.array(sys.qpos0)
        self.reward_config = RewardConfig()

        # Currently unused
        actuator_ctrlrange = []
        for i in range(mj_model.nu):
            ctrlrange = mj_model.actuator_ctrlrange[i]
            actuator_ctrlrange.append(ctrlrange)

        self.actuator_ctrlrange = jnp.array(actuator_ctrlrange)

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, rng: jnp.ndarray) -> EnvState:
        """Resets the environment to an initial state."""
        rng, rng1, rng2 = jax.random.split(rng, 3)

        low, hi = -self.reset_noise_scale, self.reset_noise_scale
        qpos = self.initial_qpos + jax.random.uniform(rng1, (self.sys.nq,), minval=low, maxval=hi)
        qvel = jax.random.uniform(rng2, (self.sys.nv,), minval=low, maxval=hi)

        # initialize mjx state
        state = self.pipeline_init(qpos, qvel)
        obs = self.get_obs(state, jnp.zeros(self._action_size))

        metrics = EnvMetrics(
            episode_returns=jnp.array(0.0),
            episode_lengths=jnp.array(0),
            returned_episode_returns=jnp.array(0.0),
            returned_episode_lengths=jnp.array(0),
            timestep=jnp.array(0),
            returned_episode=jnp.array(False),
        )

        return EnvState(state, obs, jnp.array(0.0), jnp.array(False), metrics)

    @partial(jax.jit, static_argnums=(0,))
    def step(self, env_state: EnvState, action: jnp.ndarray, rng: jnp.ndarray) -> EnvState:
        """Run one timestep of the environment's dynamics and returns observations with rewards."""
        state = env_state.pipeline_state
        metrics = env_state.metrics

        state_step = self.pipeline_step(state, action)
        obs_state = self.get_obs(state, action)

        # reset env if done
        rng, rng1, rng2 = jax.random.split(rng, 3)
        low, hi = -self.reset_noise_scale, self.reset_noise_scale

        qpos = self.initial_qpos + jax.random.uniform(rng1, (self.sys.nq,), minval=low, maxval=hi)
        qvel = jax.random.uniform(rng2, (self.sys.nv,), minval=low, maxval=hi)
        state_reset = self.pipeline_init(qpos, qvel)
        obs_reset = self.get_obs(state, jnp.zeros(self._action_size))

        # get obs/reward/done of action + states
        reward = self.compute_reward(state, state_step, action)
        done = self.is_done(state_step)

        # setting done = True if nans in next state
        is_nan = jax.tree_util.tree_map(lambda x: jnp.any(jnp.isnan(x)), state_step)
        any_nan = jax.tree_util.tree_reduce(jnp.logical_or, is_nan)
        done = jnp.logical_or(done, any_nan)

        # selectively replace state/obs with reset environment based on if done
        new_state = jax.tree.map(lambda x, y: jax.lax.select(done, x, y), state_reset, state_step)
        obs = jax.lax.select(done, obs_reset, obs_state)

        # Calculate new episode return and length
        new_episode_return = metrics.episode_returns + reward
        new_episode_length = metrics.episode_lengths + 1

        # Update metrics
        new_metrics = EnvMetrics(
            episode_returns=new_episode_return * (1 - done),
            episode_lengths=new_episode_length * (1 - done),
            returned_episode_returns=metrics.returned_episode_returns * (1 - done) + new_episode_return * done,
            returned_episode_lengths=metrics.returned_episode_lengths * (1 - done) + new_episode_length * done,
            timestep=metrics.timestep + 1,
            returned_episode=done,
        )

        return EnvState(new_state, obs, reward, done, new_metrics)

    @partial(jax.jit, static_argnums=(0,))
    def compute_reward(
        self,
        state: MjxState,
        next_state: MjxState,
        action: jnp.ndarray,
    ) -> jnp.ndarray:
        min_z = self.reward_config.height_min_z
        max_z = self.reward_config.height_max_z

        exp_coef = self.reward_config.original_pos_reward_exp_coefficient
        subtraction_factor = self.reward_config.original_pos_reward_subtraction_factor
        max_diff_norm = self.reward_config.original_pos_reward_max_diff_norm

        # MAINTAINING ORIGINAL POSITION REWARD
        qpos0_diff = self.initial_qpos - state.qpos
        original_pos_reward = jnp.exp(-exp_coef * jnp.linalg.norm(qpos0_diff)) - subtraction_factor * jnp.clip(
            jnp.linalg.norm(qpos0_diff), 0, max_diff_norm
        )

        # HEALTHY REWARD
        is_healthy = jnp.where(state.q[2] < min_z, 0.0, 1.0)
        is_healthy = jnp.where(state.q[2] > max_z, 0.0, is_healthy)

        ctrl_cost = -jnp.sum(jnp.square(action))

        xpos = state.subtree_com[1][0]
        next_xpos = next_state.subtree_com[1][0]
        velocity = (next_xpos - xpos) / self.dt

        # Calculate and print each weight * reward pairing
        ctrl_cost_weighted = self.reward_config.weights_ctrl_cost * ctrl_cost
        original_pos_reward_weighted = self.reward_config.weights_original_pos_reward * original_pos_reward
        velocity_weighted = self.reward_config.weights_velocity * velocity
        is_healthy_weighted = self.reward_config.weights_is_healthy * is_healthy

        total_reward = ctrl_cost_weighted + original_pos_reward_weighted + velocity_weighted + is_healthy_weighted

        return total_reward

    @partial(jax.jit, static_argnums=(0,))
    def is_done(self, state: MjxState) -> jnp.ndarray:
        """Check if the episode should terminate."""
        com_height = state.q[2]
        min_z, max_z = self.reward_config.height_min_z, self.reward_config.height_max_z
        height_condition = jnp.logical_not(jnp.logical_and(min_z < com_height, com_height < max_z))
        return height_condition

    @partial(jax.jit, static_argnums=(0,))
    def get_obs(self, data: MjxState, action: jnp.ndarray) -> jnp.ndarray:
        if INCLUDE_C_VALS:
            obs_components = [
                data.qpos,
                data.qvel,
                data.cinert[1:].ravel(),
                data.cvel[1:].ravel(),
                data.qfrc_actuator,
            ]
        else:
            obs_components = [
                data.qpos,
                data.qvel,
                data.qfrc_actuator,
            ]

        return jnp.concatenate(obs_components)


def run_environment_adhoc() -> None:
    """Runs the environment for a few steps with random actions, for debugging."""
    try:
        import mediapy as media
        from tqdm import tqdm
    except ImportError:
        raise ImportError("Please install `mediapy` and `tqdm` to run this script")

    if not shutil.which("ffmpeg"):
        raise ImportError("Please install `ffmpeg` to run this script")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=20,
        help="number of episodes to run",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=1024,
        help="maximum steps per episode",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="episode.mp4",
        help="path to save video",
    )
    parser.add_argument(
        "--render-every",
        type=int,
        default=2,
        help="how many frames to skip between renders",
    )
    parser.add_argument(
        "--video-length",
        type=float,
        default=5.0,
        help="desired length of video in seconds",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="width of the video frame",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="height of the video frame",
    )
    parser.add_argument(
        "--kscale-id",
        type=str,
        default=DEFAULT_KSCALE_ID,
        help="ID of the robot to load from K-Scale",
    )
    parser.add_argument(
        "--camera-name",
        type=str,
        default="track",
        help="name of the camera to use for rendering",
    )
    args = parser.parse_args()

    env = HumanoidEnv()
    action_size = env.action_size

    rng = jax.random.PRNGKey(0)
    reset_fn = jax.jit(env.reset)
    step_fn = jax.jit(env.step)

    fps = int(1 / env.dt)
    max_frames = int(args.video_length * fps)
    rollout: list[MjxState] = []

    for episode in range(args.num_episodes):
        rng, _ = jax.random.split(rng)
        env_state = reset_fn(rng)

        total_reward = 0

        for _ in tqdm(range(args.max_steps), desc=f"Episode {episode + 1} Steps", leave=False):
            if len(rollout) < args.video_length * fps:
                rollout.append(env_state.pipeline_state)

            rng, action_rng = jax.random.split(rng)
            action = jax.random.uniform(action_rng, (action_size,), minval=0, maxval=1.0)

            rng, step_rng = jax.random.split(rng)
            env_state = step_fn(env_state, action, step_rng)
            total_reward += env_state.reward

            if env_state.done:
                break

        logger.info("Episode %d total reward: %f", episode + 1, total_reward)

        if len(rollout) >= max_frames:
            break

    logger.info("Rendering video with %d frames at %d fps", len(rollout), fps)

    images = jnp.array(
        env.render(
            rollout[:: args.render_every],
            camera=args.camera_name,
            width=args.width,
            height=args.height,
        )
    )

    logger.info("Video rendered")
    media.write_video(args.output_path, images, fps=fps)
    logger.info("Video saved to %s", args.output_path)


if __name__ == "__main__":
    # python environment.py
    run_environment_adhoc()
