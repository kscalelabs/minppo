"""Definition of base humanoids environment with reward system and termination conditions."""

import argparse
from functools import partial
import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import mujoco
from brax import base
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf
from brax.mjx.base import State as MjxState


logger = logging.getLogger(__name__)

REWARD_CONFIG = {
    "termination_height": -0.2,
    "height_limits": {"min_z": -0.2, "max_z": 2.0},
    "is_healthy_reward": 5,
    "original_pos_reward": {
        "exp_coefficient": 2,
        "subtraction_factor": 0.05,
        "max_diff_norm": 0.5,
    },
    "weights": {
        "ctrl_cost": 0.1,
        "original_pos_reward": 4,
        "is_healthy": 1,
        "velocity": 1.25,
    },
}


REPO_DIR = "humanoid_original"  # humanoid_original or stompy or dora
XML_NAME = "humanoid.xml"  # dora2

# keyframe for default positions (or None for self.sys.qpos0)
KEYFRAME_NAME = "default"
print(f"using {KEYFRAME_NAME} from {REPO_DIR}/{XML_NAME}")

# my testing :)
INCLUDE_C_VALS = True
PHYSICS_FRAMES = 1


def download_model_files(repo_url: str, repo_dir: str, local_path: str) -> None:
    """Downloads or updates model files (XML + meshes) from a GitHub repository.

    Args:
        repo_url: The URL of the GitHub repository.
        repo_dir: The directory within the repository containing the model files.
        local_path: The local path where files should be saved.

    Returns:
        None
    """
    target_path = Path(local_path) / repo_dir

    # Check if the target directory already exists
    if target_path.exists():
        logger.info(
            f"Model files are already present in {target_path}. Skipping download."
        )
        return

    # Create a temporary directory for cloning
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Clone the repository into the temporary directory
        subprocess.run(["git", "clone", "--depth", "1", repo_url, temp_dir], check=True)

        # Path to the repo_dir in the temporary directory
        temp_repo_dir_path = temp_path / repo_dir

        if temp_repo_dir_path.exists():
            # If the target directory does not exist, create it
            target_path.parent.mkdir(parents=True, exist_ok=True)
            # Move the repo_dir from the temporary directory to the target path
            if target_path.exists():
                # If the target path exists, remove it first (to avoid FileExistsError)
                shutil.rmtree(target_path)
            shutil.move(str(temp_repo_dir_path), str(target_path.parent))
            logger.info(f"Model files downloaded to {target_path}")
        else:
            logger.info(f"The directory {repo_dir} does not exist in the repository.")


class HumanoidEnv(PipelineEnv):
    """Defines the environment for controlling a humanoid robot.

    This environment uses Brax's `mjcf` module to load a MuJoCo model of a
    humanoid robot, which can then be controlled using the `PipelineEnv` API.

    Parameters:
        n_frames: The number of times to step the physics pipeline for each
            environment step. Setting this value to be greater than 1 means
            that the policy will run at a lower frequency than the physics
            simulation.
    """

    initial_qpos: jnp.ndarray
    _action_size: int
    reset_noise_scale: float = 0.0

    def __init__(self, n_frames: int = PHYSICS_FRAMES, backend: str = "mjx") -> None:
        """Initializes system with initial joint positions, action size, the model, and update rate."""
        # GitHub repository URL
        repo_url = "https://github.com/nathanjzhao/mujoco-models.git"

        # Local path where the files should be saved
        environments_path = os.path.join(os.path.dirname(__file__), "environments")

        # Download or update the model files
        download_model_files(repo_url, REPO_DIR, environments_path)

        # Now use the local path to load the model
        xml_path = os.path.join(environments_path, REPO_DIR, XML_NAME)
        mj_model: mujoco.MjModel = mujoco.MjModel.from_xml_path(xml_path)

        # can definitely look at this more https://mujoco.readthedocs.io/en/latest/APIreference/APItypes.html#mjtdisablebit
        mj_model.opt.solver = mujoco.mjtSolver.mjSOL_CG
        mj_model.opt.iterations = 6
        mj_model.opt.ls_iterations = 6

        self._action_size = mj_model.nu
        sys: base.System = mjcf.load_model(mj_model)

        super().__init__(sys, n_frames=n_frames, backend=backend)

        try:
            if KEYFRAME_NAME:
                self.initial_qpos = jnp.array(
                    mj_model.keyframe(self.keyframe_name).qpos
                )
        except:
            self.initial_qpos = jnp.array(sys.qpos0)
            print("No keyframe found, utilizing qpos0")

        actuator_ctrlrange = []
        for i in range(mj_model.nu):
            ctrlrange = mj_model.actuator_ctrlrange[i]
            actuator_ctrlrange.append(ctrlrange)

        self.actuator_ctrlrange = jnp.array(actuator_ctrlrange)

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, rng: jnp.ndarray) -> State:
        """Resets the environment to an initial state."""
        rng, rng1, rng2 = jax.random.split(rng, 3)

        low, hi = -self.reset_noise_scale, self.reset_noise_scale
        qpos = self.initial_qpos + jax.random.uniform(
            rng1, (self.sys.nq,), minval=low, maxval=hi
        )
        qvel = jax.random.uniform(rng2, (self.sys.nv,), minval=low, maxval=hi)

        # initialize mjx state
        state = self.pipeline_init(qpos, qvel)
        obs = self.get_obs(state, jnp.zeros(self._action_size))
        metrics = {
            "episode_returns": 0,
            "episode_lengths": 0,
            "returned_episode_returns": 0,
            "returned_episode_lengths": 0,
            "timestep": 0,
            "returned_episode": False,
        }

        return State(state, obs, jnp.array(0.0), False, metrics)

    @partial(jax.jit, static_argnums=(0,))
    def step(self, env_state: State, action: jnp.ndarray, rng: jnp.ndarray) -> State:
        """Run one timestep of the environment's dynamics and returns observations with rewards."""
        state = env_state.pipeline_state
        metrics = env_state.metrics

        state_step = self.pipeline_step(
            state, action
        )  # because scaled action so bad...
        obs_state = self.get_obs(state, action)

        # reset env if done
        rng, rng1, rng2 = jax.random.split(rng, 3)
        low, hi = -self.reset_noise_scale, self.reset_noise_scale

        qpos = self.initial_qpos + jax.random.uniform(
            rng1, (self.sys.nq,), minval=low, maxval=hi
        )
        qvel = jax.random.uniform(rng2, (self.sys.nv,), minval=low, maxval=hi)
        state_reset = self.pipeline_init(qpos, qvel)
        obs_reset = self.get_obs(state, jnp.zeros(self._action_size))

        # get obs/reward/done of action + states
        reward = self.compute_reward(
            state,
            state_step,
            action,
        )
        done = self.is_done(state_step)

        # setting done = True if nans in next state
        is_nan = jax.tree_util.tree_map(lambda x: jnp.any(jnp.isnan(x)), state_step)
        any_nan = jax.tree_util.tree_reduce(jnp.logical_or, is_nan, initializer=False)
        done = jnp.logical_or(done, any_nan)

        # selectively replace state/obs with reset environment based on if done
        new_state = jax.tree.map(
            lambda x, y: jax.lax.select(done, x, y), state_reset, state_step
        )
        obs = jax.lax.select(done, obs_reset, obs_state)

        ########### METRIC TRACKING ###########

        # Calculate new episode return and length
        new_episode_return = metrics["episode_returns"] + reward
        new_episode_length = metrics["episode_lengths"] + 1

        # Update metrics -- we only count episode
        metrics["episode_returns"] = new_episode_return * (1 - done)
        metrics["episode_lengths"] = new_episode_length * (1 - done)
        metrics["returned_episode_returns"] = (
            metrics["returned_episode_returns"] * (1 - done) + new_episode_return * done
        )
        metrics["returned_episode_lengths"] = (
            metrics["returned_episode_lengths"] * (1 - done) + new_episode_length * done
        )
        metrics["timestep"] = metrics["timestep"] + 1
        metrics["returned_episode"] = done

        return env_state.replace(
            pipeline_state=new_state, obs=obs, reward=reward, done=done, metrics=metrics
        )

    @partial(jax.jit, static_argnums=(0,))
    def compute_reward(
        self,
        state: MjxState,
        next_state: MjxState,
        action: jnp.ndarray,
    ) -> jnp.ndarray:
        """Compute the reward for standing and height."""
        min_z, max_z = (
            REWARD_CONFIG["height_limits"]["min_z"],
            REWARD_CONFIG["height_limits"]["max_z"],
        )

        exp_coef, subtraction_factor, max_diff_norm = (
            REWARD_CONFIG["original_pos_reward"]["exp_coefficient"],
            REWARD_CONFIG["original_pos_reward"]["subtraction_factor"],
            REWARD_CONFIG["original_pos_reward"]["max_diff_norm"],
        )

        # MAINTAINING ORIGINAL POSITION REWARD
        qpos0_diff = self.initial_qpos - state.qpos
        original_pos_reward = jnp.exp(
            -exp_coef * jnp.linalg.norm(qpos0_diff)
        ) - subtraction_factor * jnp.clip(jnp.linalg.norm(qpos0_diff), 0, max_diff_norm)

        # HEALTHY REWARD
        is_healthy = jnp.where(state.q[2] < min_z, 0.0, 1.0)
        is_healthy = jnp.where(state.q[2] > max_z, 0.0, is_healthy)

        ctrl_cost = -jnp.sum(jnp.square(action))

        xpos = state.subtree_com[1][0]
        next_xpos = next_state.subtree_com[1][0]
        velocity = (next_xpos - xpos) / self.dt


        # Calculate and print each weight * reward pairing
        ctrl_cost_weighted = REWARD_CONFIG["weights"]["ctrl_cost"] * ctrl_cost
        original_pos_reward_weighted = REWARD_CONFIG["weights"]["original_pos_reward"] * original_pos_reward
        velocity_weighted = REWARD_CONFIG["weights"]["velocity"] * velocity
        is_healthy_weighted = REWARD_CONFIG["weights"]["is_healthy"] * is_healthy

        # jax.debug.print("ctrl_cost_weighted: {}, original_pos_reward_weighted: {}, is_healthy_weighted {}, velocity_weighted: {}", ctrl_cost_weighted, original_pos_reward_weighted, is_healthy_weighted, velocity_weighted)

        total_reward = (
            ctrl_cost_weighted
            + original_pos_reward_weighted
            + velocity_weighted
            + is_healthy_weighted
        )

        return total_reward

    @partial(jax.jit, static_argnums=(0,))
    def is_done(self, state: MjxState) -> bool:
        """Check if the episode should terminate."""
        # Get the height of the robot's center of mass
        com_height = state.q[2]

        min_z, max_z = (
            REWARD_CONFIG["height_limits"]["min_z"],
            REWARD_CONFIG["height_limits"]["max_z"],
        )
        height_condition = jnp.logical_not(
            jnp.logical_and(min_z < com_height, com_height < max_z)
        )

        # Check if any element in qvel or qacc exceeds 1e5
        velocity_condition = jnp.any(jnp.abs(state.qvel) > 1e5)

        # Combine conditions
        condition = jnp.logical_or(height_condition, velocity_condition)

        return condition

    @partial(jax.jit, static_argnums=(0,))
    def get_obs(self, data: MjxState, action: jnp.ndarray) -> jnp.ndarray:
        if INCLUDE_C_VALS:
            # slicing cinert and cvel because always zeroes
            # no longer slicing qpos
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


################## TEST ENVIRONMENT RUN ##################


def run_environment_adhoc() -> None:
    """Runs the environment for a few steps with random actions, for debugging."""
    try:
        import mediapy as media
        from tqdm import tqdm
    except ImportError:
        raise ImportError("Please install `mediapy` and `tqdm` to run this script")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--actor_path", type=str, default="actor_params.pkl", help="path to actor model"
    )
    parser.add_argument(
        "--critic_path",
        type=str,
        default="critic_params.pkl",
        help="path to critic model",
    )
    parser.add_argument(
        "--num_episodes", type=int, default=20, help="number of episodes to run"
    )
    parser.add_argument(
        "--max_steps", type=int, default=1024, help="maximum steps per episode"
    )
    parser.add_argument(
        "--env_name",
        type=str,
        default="random",
        help="path to save video",
    )
    parser.add_argument(
        "--render_every",
        type=int,
        default=2,
        help="how many frames to skip between renders",
    )
    parser.add_argument(
        "--video_length",
        type=float,
        default=5.0,
        help="desired length of video in seconds",
    )
    parser.add_argument(
        "--width", type=int, default=640, help="width of the video frame"
    )
    parser.add_argument(
        "--height", type=int, default=480, help="height of the video frame"
    )
    args = parser.parse_args()

    env = HumanoidEnv()
    action_size = env.action_size

    rng = jax.random.PRNGKey(0)
    reset_fn = jax.jit(env.reset)
    step_fn = jax.jit(env.step)

    fps = int(1 / env.dt)
    max_frames = int(args.video_length * fps)
    rollout: list[Any] = []

    video_path = args.env_name + ".mp4"
    metrics = {"qpos_2": []}

    for episode in range(args.num_episodes):
        rng, _ = jax.random.split(rng)
        state = reset_fn(rng)

        total_reward = 0

        for step in tqdm(
            range(args.max_steps), desc=f"Episode {episode + 1} Steps", leave=False
        ):
            if len(rollout) < args.video_length * fps:
                rollout.append(state.pipeline_state)

            #### STORED METRICS ####
            metrics["qpos_2"].append(state.pipeline_state.qpos[2])

            rng, action_rng = jax.random.split(rng)
            action = jax.random.uniform(
                action_rng, (action_size,), minval=0, maxval=1.0
            )  # placeholder for an action

            rng, step_rng = jax.random.split(rng)
            state = step_fn(state, action, step_rng)
            total_reward += state.reward

            if state.done:
                break

        print(f"Episode {episode + 1} total reward: {total_reward}")

        if len(rollout) >= max_frames:
            break

    print(f"Rendering video with {len(rollout)} frames at {fps} fps")
    images = jnp.array(
        env.render(
            rollout[:: args.render_every],
            camera="side",
            width=args.width,
            height=args.height,
        )
    )

    print("Video rendered")
    media.write_video(video_path, images, fps=fps)
    print(f"Video saved to {video_path}")

    ###### ADDS DEBUG TEXT ON TOP OF VIDEO ######

    # video_path = video_path
    # cap = cv2.VideoCapture(video_path)

    # if not cap.isOpened():
    #     print(f"Error: Could not open video {video_path}")

    # # Get video properties
    # frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # fps = cap.get(cv2.CAP_PROP_FPS)

    # # Define the codec and create VideoWriter object
    # debug_video_path = args.env_name + "_debug.mp4"
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # out = cv2.VideoWriter(debug_video_path, fourcc, fps, (frame_width, frame_height))

    # # Loop through each frame
    # frame_index = 0

    # if not out.isOpened():
    #     print("Error: Could not open VideoWriter")

    # while cap.isOpened():
    #     ret, frame = cap.read()
    #     if not ret:
    #         break

    #     # Write each metric on the frame
    #     font = cv2.FONT_HERSHEY_SIMPLEX
    #     font_scale = 1
    #     color = (255, 0, 0)  # Blue color in BGR
    #     thickness = 2
    #     y_offset = 50  # Initial y position for the text

    #     for key, values in metrics.items():
    #         if frame_index < len(values):
    #             text = f'{key}: {values[frame_index]}'
    #             org = (50, y_offset)  # Position for the text
    #             frame = cv2.putText(frame, text, org, font, font_scale, color, thickness, cv2.LINE_AA)
    #             y_offset += 30  # Move to the next line for the next metric

    #     # Write the frame into the output video
    #     out.write(frame)
    #     frame_index += 1

    # # Release everything if job is finished
    # cap.release()
    # out.release()
    # cv2.destroyAllWindows()


if __name__ == "__main__":
    # python environment.py
    run_environment_adhoc()
