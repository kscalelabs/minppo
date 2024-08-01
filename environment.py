"""Definition of base humanoids environment with reward system and termination conditions."""

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

    def __init__(self) -> None:
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

        qpos = self.sys.qpos0 + jax.random.uniform(rng1, (self.sys.nq,), minval=low, maxval=hi)
        # qpos = self.initial_qpos + jax.random.uniform(rng1, (self.sys.nq,), minval=low, maxval=hi)
        qvel = jax.random.uniform(rng2, (self.sys.nv,), minval=low, maxval=hi)

        # initialize mjx state
        state = self.pipeline_init(qpos, qvel)
        obs = self.get_obs(state, jp.zeros(self._action_size))
        reward, done, zero = jp.zeros(3)

        metrics: dict[str, Any] = {}

        return State(state, obs, reward, done, metrics)

    def step(self, env_state: State, action: jp.ndarray) -> State:
        """Run one timestep of the environment's dynamics."""
        state = env_state.pipeline_state
        next_state = self.pipeline_step(state, action)
        obs = self.get_obs(state, action)

        # Reward function: encourage standing
        reward = self.compute_reward(state, next_state, action)

        # Termination condition
        done = self.is_done(next_state)

        return env_state.replace(pipeline_state=next_state, obs=obs, reward=reward, done=done)

    def compute_reward(self, state: MjxState, next_state: MjxState, action: jp.ndarray) -> jp.ndarray:
        """Compute the reward for standing and height."""
        min_z, max_z = -0.345, 0
        is_healthy = jp.where(state.qpos[2] < min_z, 0.0, 1.0)
        is_healthy = jp.where(state.qpos[2] > max_z, 0.0, is_healthy)

        ctrl_cost = -jp.sum(jp.square(action))

        xpos = state.subtree_com[1][0]
        next_xpos = next_state.subtree_com[1][0]
        velocity = (next_xpos - xpos) / self.dt

        # jax.debug.print(
        #     "velocity {}, xpos {}, next_xpos {}",
        #     velocity,
        #     xpos,
        #     next_xpos,
        #     ordered=True,
        # )
        # jax.debug.print("is_healthy {}, height {}", is_healthy, state.q[2], ordered=True)

        total_reward = 5.0 * is_healthy + 1.25 * velocity

        return total_reward

    def is_done(self, state: MjxState) -> jp.ndarray:
        """Check if the episode should terminate."""
        # Get the height of the robot's center of mass
        com_height = state.q[2]  # Assuming the 3rd element is the z-position

        # Set a termination threshold
        termination_height = -0.35  # For example, 50% of the initial height

        # Episode is done if the robot falls below the termination height
        done = jp.where(com_height < termination_height, 1.0, 0.0)

        return done

    # details you want to pass through to the actor/critic model
    def get_obs(self, data: MjxState, action: jp.ndarray) -> jp.ndarray:
        """Returns the observation of the environment."""
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

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_frames", type=int, default=1)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    # TODO: 1. Fill this out and make sure it works in isolation.

    raise NotImplementedError


if __name__ == "__main__":
    run_environment_adhoc()
