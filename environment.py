"""Definition of base humanoids environment with reward system and termination conditions."""

import os
from typing import Any

import jax.numpy as jp
import mujoco
from brax import base
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf
from brax.mjx.base import State as MjxState


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
        qpos = self.initial_qpos
        qvel = jp.zeros(len(qpos) - 1)

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
        reward = self.compute_reward(state, next_state)

        # Termination condition
        done = self.is_done(next_state)

        return env_state.replace(pipeline_state=next_state, obs=obs, reward=reward, done=done)

    def compute_reward(self, state: MjxState, next_state: MjxState) -> jp.ndarray:
        """Compute the reward for standing and height."""
        min_z, max_z = -0.345, 0
        is_healthy = jp.where(state.qpos[2] < min_z, 0.0, 1.0)
        is_healthy = jp.where(state.qpos[2] > max_z, 0.0, is_healthy)

        xpos = state.subtree_com[1][1]
        next_xpos = next_state.subtree_com[1][1]
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

    def is_done(self, mjx_state: MjxState) -> jp.ndarray:
        """Check if the episode should terminate."""
        # Get the height of the robot's center of mass
        com_height = mjx_state.qpos[2]  # Assuming the 3rd element is the z-position

        # Set a termination threshold
        termination_height = -0.35  # For example, 50% of the initial height

        # Episode is done if the robot falls below the termination height
        done = jp.where(com_height < termination_height, 1.0, 0.0)

        return done

    # details you want to pass through to the actor/critic model
    def get_obs(self, data: MjxState, action: jp.ndarray) -> jp.ndarray:
        """Returns the observation of the environment."""
        return data.qpos


##################


# class RobotMJXEnv(PipelineEnv):
#     """An environment for humanoid body position, velocities, and angles."""

#     def __init__(
#         self,
#         reward_params=None,  # TODO: change rewards
#         terminate_when_unhealthy: bool = True,
#         reset_noise_scale: float = 1e-2,
#         exclude_current_positions_from_observation: bool = True,
#         log_reward_breakdown: bool = True,
#         **kwargs,
#     ) -> None:
#         path = os.path.join(os.path.dirname(__file__), "stompy", "stompylegs.xml")
#         mj_model = mujoco.MjModel.from_xml_path(path)
#         mj_model.opt.solver = mujoco.mjtSolver.mjSOL_CG
#         mj_model.opt.iterations = 6
#         mj_model.opt.ls_iterations = 6

#         sys = mjcf.load_model(mj_model)

#         physics_steps_per_control_step = 4  # Should find way to perturb this value in the future
#         kwargs["n_frames"] = kwargs.get("n_frames", physics_steps_per_control_step)
#         kwargs["backend"] = "mjx"

#         super().__init__(sys, **kwargs)

#         self._reward_params = reward_params
#         self._terminate_when_unhealthy = terminate_when_unhealthy
#         self._reset_noise_scale = reset_noise_scale
#         self._exclude_current_positions_from_observation = exclude_current_positions_from_observation
#         self._log_reward_breakdown = log_reward_breakdown

#         # keyframe is not in the xml file
#         # print("initial_qpos", mj_model.keyframe("default").qpos)
#         # self.initial_qpos = mj_model.keyframe("default").qpos
#         self.reward_fn = get_reward_fn(self._reward_params, self.dt, include_reward_breakdown=True)

#     def reset(self, rng: jp.ndarray) -> State:
#         """Resets the environment to an initial state.

#         Args:
#             rng: Random number generator seed.

#         Returns:
#             The initial state of the environment.
#         """
#         rng, rng1, rng2 = jax.random.split(rng, 3)

#         low, hi = -self._reset_noise_scale, self._reset_noise_scale
#         breakpoint()
#         qpos = self.sys.qpos0 + jax.random.uniform(rng1, (self.sys.nq,), minval=low, maxval=hi)
#         qvel = jax.random.uniform(rng2, (self.sys.nv,), minval=low, maxval=hi)

#         mjx_state = self.pipeline_init(qpos, qvel)
#         assert isinstance(mjx_state, mjxState), f"mjx_state is of type {type(mjx_state)}"

#         obs = self._get_obs(mjx_state, jp.zeros(self.sys.nu))
#         reward, done, zero = jp.zeros(3)
#         metrics = {
#             "x_position": zero,
#             "y_position": zero,
#             "distance_from_origin": zero,
#             "x_velocity": zero,
#             "y_velocity": zero,
#         }
#         for key in self._reward_params.keys():
#             metrics[key] = zero

#         return State(mjx_state, obs, reward, done, metrics)

#     def step(self, state: State, action: jp.ndarray) -> State:
#         """Runs one timestep of the environment's dynamics.

#         Args:
#             state: The current state of the environment.
#             action: The action to take.

#         Returns:
#             A tuple of the next state, the reward, whether the episode has ended, and additional information.
#         """
#         mjx_state = state.pipeline_state
#         assert mjx_state, "state.pipeline_state was recorded as None"
#         # TODO: determine whether to raise an error or reset the environment

#         next_mjx_state = self.pipeline_step(mjx_state, action)

#         assert isinstance(next_mjx_state, mjxState), f"next_mjx_state is of type {type(next_mjx_state)}"
#         assert isinstance(mjx_state, mjxState), f"mjx_state is of type {type(mjx_state)}"
#         # mlutz: from what I've seen, .pipeline_state and .pipeline_step(...)
#         # actually return an brax.mjx.base.State object however, the type
#         # hinting suggests that it should return a brax.base.State object
#         # brax.mjx.base.State inherits from brax.base.State but also inherits
#         # from mjx.Data, which is needed for some rewards

#         obs = self._get_obs(mjx_state, action)
#         reward, is_healthy, reward_breakdown = self.reward_fn(mjx_state, action, next_mjx_state)

#         if self._terminate_when_unhealthy:
#             done = 1.0 - is_healthy
#         else:
#             done = jp.array(0)

#         state.metrics.update(
#             x_position=next_mjx_state.subtree_com[1][0],
#             y_position=next_mjx_state.subtree_com[1][1],
#             distance_from_origin=jp.linalg.norm(next_mjx_state.subtree_com[1]),
#             x_velocity=(next_mjx_state.subtree_com[1][0] - mjx_state.subtree_com[1][0]) / self.dt,
#             y_velocity=(next_mjx_state.subtree_com[1][1] - mjx_state.subtree_com[1][1]) / self.dt,
#         )

#         if self._log_reward_breakdown:
#             for key, val in reward_breakdown.items():
#                 state.metrics[key] = val

#         return state.replace(pipeline_state=next_mjx_state, obs=obs, reward=reward, done=done)

#     def _get_obs(self, data: mjxState, action: jp.ndarray) -> jp.ndarray:
#         """Observes humanoid body position, velocities, and angles.

#         Args:
#             data: The current state of the environment.
#             action: The current action.

#         Returns:
#             Observations of the environment.
#         """
#         position = data.qpos
#         if self._exclude_current_positions_from_observation:
#             position = position[2:]

#         # external_contact_forces are excluded
#         return jp.concatenate(
#             [
#                 position,
#                 data.qvel,
#                 data.cinert[1:].ravel(),
#                 data.cvel[1:].ravel(),
#                 data.qfrc_actuator,
#             ]
#         )


# ############ REWARDS ############


# class RewardDict(TypedDict):
#     weight: float
#     healthy_z_lower: float | None
#     healthy_z_upper: float | None


# RewardParams = dict[str, RewardDict]

# DEFAULT_REWARD_PARAMS: RewardParams = {
#     "rew_forward": {"weight": 1.25},
#     "rew_healthy": {"weight": 5.0, "healthy_z_lower": 1.0, "healthy_z_upper": 2.0},
#     "rew_ctrl_cost": {"weight": 0.1},
# }


# def get_reward_fn(
#     reward_params: RewardParams,
#     dt: jax.Array,
#     include_reward_breakdown: bool,
# ) -> Callable[[mjxState, jp.ndarray, mjxState], tuple[jp.ndarray, jp.ndarray, dict[str, jp.ndarray]]]:
#     """Get a combined reward function.

#     Args:
#         reward_params: Dictionary of reward parameters.
#         dt: Time step.
#         include_reward_breakdown: Whether to include a breakdown of the reward
#             into its components.

#     Returns:
#         A reward function that takes in a state, action, and next state and
#         returns a float wrapped in a jp.ndarray.
#     """

#     def reward_fn(
#         state: mjxState, action: jp.ndarray, next_state: mjxState
#     ) -> tuple[jp.ndarray, jp.ndarray, dict[str, jp.ndarray]]:
#         reward, is_healthy = jp.array(0.0), jp.array(1.0)
#         rewards = {}
#         for key, params in reward_params.items():
#             r, h = reward_functions[key](state, action, next_state, dt, params)
#             is_healthy *= h
#             reward += r
#             if include_reward_breakdown:  # For more detailed logging, can be disabled for performance
#                 rewards[key] = r
#         return reward, is_healthy, rewards

#     return reward_fn


# def forward_reward_fn(
#     state: mjxState,
#     action: jp.ndarray,
#     next_state: mjxState,
#     dt: jax.Array,
#     params: RewardDict,
# ) -> tuple[jp.ndarray, jp.ndarray]:
#     """Reward function for moving forward.

#     Args:
#         state: Current state.
#         action: Action taken.
#         next_state: Next state.
#         dt: Time step.
#         params: Reward parameters.

#     Returns:
#         A float wrapped in a jax array.
#     """
#     xpos = state.subtree_com[1][0]  # TODO: include stricter typing than mjxState to avoid this type error
#     next_xpos = next_state.subtree_com[1][0]
#     velocity = (next_xpos - xpos) / dt
#     forward_reward = params["weight"] * velocity

#     return forward_reward, jp.array(1.0)  # TODO: ensure everything is initialized in a size 2 array instead...


# def healthy_reward_fn(
#     state: mjxState,
#     action: jp.ndarray,
#     next_state: mjxState,
#     dt: jax.Array,
#     params: RewardDict,
# ) -> tuple[jp.ndarray, jp.ndarray]:
#     """Reward function for staying healthy.

#     Args:
#         state: Current state.
#         action: Action taken.
#         next_state: Next state.
#         dt: Time step.
#         params: Reward parameters.

#     Returns:
#         A float wrapped in a jax array.
#     """
#     min_z = params["healthy_z_lower"]
#     max_z = params["healthy_z_upper"]
#     is_healthy = jp.where(state.q[2] < min_z, 0.0, 1.0)
#     is_healthy = jp.where(state.q[2] > max_z, 0.0, is_healthy)
#     healthy_reward = jp.array(params["weight"]) * is_healthy

#     return healthy_reward, is_healthy


# def ctrl_cost_reward_fn(
#     state: mjxState,
#     action: jp.ndarray,
#     next_state: mjxState,
#     dt: jax.Array,
#     params: RewardDict,
# ) -> tuple[jp.ndarray, jp.ndarray]:
#     """Reward function for control cost.

#     Args:
#         state: Current state.
#         action: Action taken.
#         next_state: Next state.
#         dt: Time step.
#         params: Reward parameters.

#     Returns:
#         A float wrapped in a jax array.
#     """
#     ctrl_cost = -params["weight"] * jp.sum(jp.square(action))

#     return ctrl_cost, jp.array(1.0)


# RewardFunction = Callable[[mjxState, jp.ndarray, mjxState, jax.Array, RewardDict], tuple[jp.ndarray, jp.ndarray]]


# # NOTE: After defining the reward functions, they must be added here to be used in the combined reward function.
# reward_functions: dict[str, RewardFunction] = {
#     "rew_forward": forward_reward_fn,
#     "rew_healthy": healthy_reward_fn,
#     "rew_ctrl_cost": ctrl_cost_reward_fn,
# }
