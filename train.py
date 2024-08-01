"""Trains a policy network to get a humanoid to stand up."""

import argparse
import logging
import os
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Callable, Dict, List, Mapping, Tuple

import jax
import jax.numpy as jnp
import mediapy as media
import numpy as np
import optax
from brax.envs import State
from brax.mjx.base import State as MjxState
from flax import linen as nn
from jax import Array
from tqdm import tqdm

from environment import HumanoidEnv

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


@dataclass
class Config:
    lr_actor: float = field(default=3e-4)
    lr_critic: float = field(default=3e-4)
    num_iterations: int = field(default=15000)
    num_envs: int = field(default=2048)
    max_steps: int = field(default=2048 * 1000)
    max_steps_per_epoch: int = field(default=16384)
    gamma: float = field(default=0.98)
    lambd: float = field(default=0.98)
    batch_size: int = field(default=64)
    epsilon: float = field(default=0.2)
    l2_rate: float = field(default=0.001)


class Ppo:
    def __init__(self, observation_size: int, action_size: int, config: Config, key: Array) -> None:
        """Initialize the PPO model's actor/critic structure and optimizers."""
        self.actor = Actor(action_size)
        self.critic = Critic()
        self.actor_params = self.actor.init(key, jnp.zeros((1, observation_size)))
        self.critic_params = self.critic.init(key, jnp.zeros((1, observation_size)))

        self.actor_optim = optax.adam(learning_rate=config.lr_actor)
        self.critic_optim = optax.adamw(learning_rate=config.lr_critic, weight_decay=config.l2_rate)

        self.actor_opt_state = self.actor_optim.init(self.actor_params)
        self.critic_opt_state = self.critic_optim.init(self.critic_params)

    def get_params(self) -> Dict[str, Any]:
        """Get the parameters of the PPO model."""
        return {
            "actor_params": self.actor_params,
            "critic_params": self.critic_params,
            "actor_opt_state": self.actor_opt_state,
            "critic_opt_state": self.critic_opt_state,
        }

    def update_params(self, new_params: Dict[str, Any]) -> None:
        """Update the parameters of the PPO model."""
        self.actor_params = new_params["actor_params"]
        self.critic_params = new_params["critic_params"]
        self.actor_opt_state = new_params["actor_opt_state"]
        self.critic_opt_state = new_params["critic_opt_state"]


def train_step(
    actor_apply: Callable[[Any, Array], Tuple[Array, Array]],
    critic_apply: Callable[[Any, Array], Array],
    actor_optim: optax.GradientTransformation,
    critic_optim: optax.GradientTransformation,
    params: Dict[str, Any],
    states: Array,
    actions: Array,
    rewards: Array,
    masks: Array,
    config: Config,
) -> Tuple[Dict[str, Any], Array, Array]:
    """Perform a single training step with PPO parameters."""
    actor_params, critic_params, actor_opt_state, critic_opt_state = params.values()

    values = critic_apply(critic_params, states).squeeze()
    returns, advants = get_gae(rewards, masks, values, config)

    old_mu, old_std = actor_apply(actor_params, states)
    old_log_prob = actor_log_prob(old_mu, old_std, actions)

    def actor_loss_fn(params: Array) -> Array:
        """Prioritizing advantagous actions over more training, clipping to prevent too much change."""
        mu, std = actor_apply(params, states)
        new_log_prob = actor_log_prob(mu, std, actions)

        # Loss is defined by amount of change to improved state
        # Multiplied with amount of improvement
        ratio = jnp.exp(new_log_prob - old_log_prob)
        surrogate_loss = ratio * advants

        clipped_loss = jnp.clip(ratio, 1.0 - config.epsilon, 1.0 + config.epsilon) * advants
        actor_loss = -jnp.mean(jnp.minimum(surrogate_loss, clipped_loss))
        return actor_loss

    def critic_loss_fn(params: Array) -> Array:
        """Prioritizing being able to predict the ground truth returns."""
        critic_returns = critic_apply(params, states).squeeze()
        critic_loss = jnp.mean((critic_returns - returns) ** 2)
        return critic_loss

    # Calculating actor loss and updating actor parameters
    actor_grad_fn = jax.value_and_grad(actor_loss_fn)
    actor_loss, actor_grads = actor_grad_fn(actor_params)
    actor_updates, new_actor_opt_state = actor_optim.update(actor_grads, actor_opt_state, actor_params)
    new_actor_params = optax.apply_updates(actor_params, actor_updates)

    # Calculating critic loss and updating actor parameters
    critic_grad_fn = jax.value_and_grad(critic_loss_fn)
    critic_loss, critic_grads = critic_grad_fn(critic_params)
    critic_updates, new_critic_opt_state = critic_optim.update(critic_grads, critic_opt_state, critic_params)
    new_critic_params = optax.apply_updates(critic_params, critic_updates)

    new_params = {
        "actor_params": new_actor_params,
        "critic_params": new_critic_params,
        "actor_opt_state": new_actor_opt_state,
        "critic_opt_state": new_critic_opt_state,
    }

    return new_params, actor_loss, critic_loss


def get_gae(rewards: Array, masks: Array, values: Array, config: Config) -> Tuple[Array, Array]:
    """Calculate the Generalized Advantage Estimation for rewards using jax.lax.scan."""

    def gae_step(carry: Tuple[Array, Array], inp: Tuple[Array, Array, Array]) -> Tuple[Tuple[Array, Array], Array]:
        """Single step of Generalized Advantage Estimation to be iterated over."""
        gae, next_value = carry
        reward, mask, value = inp
        delta = reward + config.gamma * next_value * mask - value
        gae = delta + config.gamma * config.lambd * mask * gae
        return (gae, value), gae

    # Calculating reward, with combination of immediate reward and diminishing value of future rewards
    _, advantages = jax.lax.scan(
        f=gae_step,
        init=(jnp.zeros_like(rewards[-1]), values[-1]),
        xs=(rewards[::-1], masks[::-1], values[::-1]),
        reverse=True,
    )
    returns = advantages + values
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    return returns, advantages


def train(ppo: Ppo, memory: List[Tuple[Array, Array, Array, Array]], config: Config) -> None:
    """Train the PPO model using the memory collected from the environment."""
    # NOTE: think this needs to be reimplemented for vectorization because currently,
    # doesn't account that memory order is maintained

    # Reorders memory according to states, actions, rewards, masks
    states = jnp.array([e[0] for e in memory])
    actions = jnp.array([e[1] for e in memory])
    rewards = jnp.array([e[2] for e in memory])
    masks = jnp.array([e[3] for e in memory])

    n = len(states)
    arr = jnp.arange(n)
    key = jax.random.PRNGKey(0)

    for epoch in range(1):
        key, subkey = jax.random.split(key)
        arr = jax.random.permutation(subkey, arr)
        for i in range(n // config.batch_size):

            # Batching the data
            batch_indices = arr[config.batch_size * i : config.batch_size * (i + 1)]
            b_states = states[batch_indices]
            b_actions = actions[batch_indices]
            b_rewards = rewards[batch_indices]
            b_masks = masks[batch_indices]

            params = ppo.get_params()
            new_params, actor_loss, critic_loss = train_step(
                ppo.actor.apply,  # type: ignore[arg-type]
                ppo.critic.apply,  # type: ignore[arg-type]
                ppo.actor_optim,
                ppo.critic_optim,
                params,
                b_states,
                b_actions,
                b_rewards,
                b_masks,
                config,
            )
            ppo.update_params(new_params)


def actor_log_prob(mu: Array, sigma: Array, actions: Array) -> Array:
    """Calculate the log probability of the actions given the actor network's output."""
    return jax.scipy.stats.norm.logpdf(actions, mu, sigma).sum(axis=-1)


def actor_distribution(mu: Array, sigma: Array) -> Array:
    """Get an action from the actor network from its probability distribution of actions."""
    return jax.random.normal(jax.random.PRNGKey(0), shape=mu.shape) * sigma + mu


class Actor(nn.Module):
    """Actor network for PPO."""

    action_size: int

    @nn.compact
    def __call__(self, x: Array) -> Tuple[Array, Array]:
        x = nn.tanh(nn.Dense(64)(x))
        x = nn.tanh(nn.Dense(64)(x))
        mu = nn.Dense(self.action_size, kernel_init=nn.initializers.constant(0.1))(x)
        log_sigma = nn.Dense(self.action_size)(x)
        return mu, jnp.exp(log_sigma)


class Critic(nn.Module):
    """Critic network for PPO."""

    @nn.compact
    def __call__(self, x: Array) -> Array:
        x = nn.tanh(nn.Dense(64)(x))
        x = nn.tanh(nn.Dense(64)(x))
        return nn.Dense(1, kernel_init=nn.initializers.constant(0.1))(x)


def unwrap_state_vectorization(state: State, config: Config) -> State:
    """Unwraps one environment the vectorized rollout so that the frames in videos are correctly ordered."""
    unwrapped_rollout = []
    # Get all attributes of the state
    attributes = dir(state)

    # NOTE: can change ordering of this to save runtiem if want to save more vectorized states.
    # NOTE: (but anyways, the video isn't correctly ordered then)
    # saves from only first vectorized state
    for i in range(1):
        # Create a new state with the first element of each attribute
        new_state = {}
        for attr in attributes:
            # Skip special methods and attributes
            if (
                not attr.startswith("_")
                and not callable(getattr(state, attr))
                and attr not in ["ne", "nl", "nf", "nefc"]
            ):
                value = getattr(state, attr)
                try:
                    new_state[attr] = value[i]
                except Exception:
                    logger.warning(f"Could not get first element of {attr}. Setting {attr} to {value}")
                    new_state[attr] = value
        unwrapped_rollout.append(type(state)(**new_state))

    return unwrapped_rollout


def screenshot(
    env: HumanoidEnv,
    rng: Array,
    width: int = 640,
    height: int = 480,
    filename: str = "screenshot.png",
) -> None:
    """Save a screenshot of the starting environment for debugging initial positions."""
    state = env.reset(rng)
    image_array = env.render(state.pipeline_state, camera="side", width=width, height=height)
    image_array = jnp.array(image_array).astype("uint8")
    os.makedirs("screenshots", exist_ok=True)
    media.write_image(os.path.join("screenshots", filename), image_array)

    logger.info(f"Screenshot saved as {filename}")


@partial(jax.jit, static_argnums=(2,))
def choose_action(actor_params: Mapping[str, Mapping[str, Any]], obs: Array, actor: Actor) -> Array:
    # Given a state, we do our forward pass and then sample from to maintain "random actions"
    mu, sigma = actor.apply(actor_params, obs)
    return actor_distribution(mu, sigma)  # type: ignore[arg-type]


@jax.jit
def update_memory(memory: Dict[str, Array], new_data: Dict[str, Array]) -> Dict[str, Array]:
    return jax.tree.map(lambda x, y: jnp.concatenate([x, y]), memory, new_data)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default="Humanoid-v2", help="name of environmnet to put into logs")
    parser.add_argument("--render", action="store_true", help="render the environment")
    parser.add_argument("--width", type=int, default=640, help="width of the video frame")
    parser.add_argument("--height", type=int, default=480, help="height of the video frame")
    parser.add_argument("--render_every", type=int, default=2, help="render the environment every N steps")
    parser.add_argument("--video_length", type=int, default=5, help="maxmimum length of video in seconds")
    parser.add_argument("--save_video_every", type=int, help="save video every N iterations")
    args = parser.parse_args()

    config = Config()

    env = HumanoidEnv()
    observation_size = env.observation_size
    action_size = env.action_size

    logger.info("action_size %s", action_size)
    logger.info("observation_size %s", observation_size)

    @jax.jit
    def reset_fn(rng: Array) -> State:
        rngs = jax.random.split(rng, config.num_envs)
        return jax.vmap(env.reset)(rngs)

    @jax.jit
    def step_fn(states: State, actions: jax.Array) -> State:
        return jax.vmap(env.step)(states, actions)

    rng = jax.random.PRNGKey(0)

    # screenshot(env, rng)
    # return

    reset_fn(rng)
    np.random.seed(500)

    ppo = Ppo(observation_size, action_size, config, rng)
    episodes: int = 0

    for i in range(1, config.num_iterations + 1):
        # Initialize memory as JAX arrays
        memory = {
            "states": jnp.empty((0, observation_size)),
            "actions": jnp.empty((0, action_size)),
            "rewards": jnp.empty((0,)),
            "masks": jnp.empty((0,)),
        }
        scores = []
        steps = 0
        rollout: List[MjxState] = []

        pbar = tqdm(total=config.max_steps_per_epoch, desc=f"Steps for iteration {i}")

        while steps < config.max_steps_per_epoch:
            episodes += config.num_envs

            rng, subrng = jax.random.split(rng)
            states = reset_fn(subrng)
            obs = jax.device_put(states.obs)
            score = jnp.zeros(config.num_envs)

            for _ in range(config.max_steps):
                actions = choose_action(ppo.actor_params, obs, ppo.actor)

                states = step_fn(states, actions)
                next_obs, rewards, dones = states.obs, states.reward, states.done
                masks = (1 - dones).astype(jnp.float32)

                # Update memory
                new_data = {"states": obs, "actions": actions, "rewards": rewards, "masks": masks}
                memory = update_memory(memory, new_data)

                score += rewards
                obs = next_obs
                steps += config.num_envs
                pbar.update(config.num_envs)

                # Capture first environment for video if enabled
                if (
                    args.save_video_every
                    and i % args.save_video_every == 0
                    and len(rollout) < args.video_length * int(1 / env.dt)
                ):
                    unwrapped_states = unwrap_state_vectorization(states.pipeline_state, config)
                    rollout.extend(unwrapped_states)

                if jnp.all(dones):
                    break

            with open("log_" + args.env_name + ".txt", "a") as outfile:
                outfile.write("\t" + str(episodes) + "\t" + str(jnp.mean(score)) + "\n")
            scores.append(jnp.mean(score))

        score_avg = float(jnp.mean(jnp.array(scores)))
        pbar.close()
        logger.info("Episode %s score is %.2f", episodes, score_avg)

        # Save video for this iteration
        if args.save_video_every and i % args.save_video_every == 0 and rollout:
            images = jnp.array(
                env.render(rollout[:: args.render_every], camera="side", width=args.width, height=args.height)
            )
            fps = int(1 / env.dt)
            logger.info("Saving video for iteration %d", i)
            media.write_video(f"videos/{args.env_name}_video{i}.mp4", images, fps=fps)

        # Convert memory to the format expected by ppo.train
        train_memory = [
            (s, a, r, m) for s, a, r, m in zip(memory["states"], memory["actions"], memory["rewards"], memory["masks"])
        ]
        train(ppo, train_memory, config)


if __name__ == "__main__":
    main()
