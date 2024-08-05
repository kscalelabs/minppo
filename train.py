"""Trains a policy network to get a humanoid to stand up."""

import argparse
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import mediapy as media
import numpy as np
import optax
from brax.envs import State
from brax.mjx.base import State as MjxState
from jax import Array
from tqdm import tqdm

from environment import HumanoidEnv

logger = logging.getLogger(__name__)


@dataclass
class Config:
    lr_actor: float = field(default=2.5e-4, metadata={"help": "Learning rate for the actor network."})
    lr_critic: float = field(default=2.5e-4, metadata={"help": "Learning rate for the critic network."})
    num_iterations: int = field(default=15000, metadata={"help": "Number of environment simulation iterations."})
    num_envs: int = field(default=16, metadata={"help": "Number of environments to run at once with vectorization."})
    max_steps_per_episode: int = field(
        default=128 * 16, metadata={"help": "Maximum number of steps per episode (across ALL environments)."}
    )
    max_steps_per_iteration: int = field(
        default=512 * 16,
        metadata={
            "help": "Maximum number of steps per iteration of simulating environments (across ALL environments)."
        },
    )
    gamma: float = field(default=0.99, metadata={"help": "Discount factor for future rewards."})
    lambd: float = field(default=0.95, metadata={"help": "Lambda parameter for GAE calculation."})
    batch_size: int = field(default=32, metadata={"help": "Batch size for training updates."})
    epsilon: float = field(default=0.2, metadata={"help": "Clipping parameter for PPO."})
    l2_rate: float = field(default=0.001, metadata={"help": "L2 regularization rate for the critic."})
    entropy_coeff: float = field(default=0.01, metadata={"help": "Coefficient for entropy loss."})


# NOTE: change how initialize weights?
class Actor(eqx.Module):
    """Actor network for PPO."""

    linear1: eqx.nn.Linear
    linear2: eqx.nn.Linear
    mu_layer: eqx.nn.Linear
    log_sigma_layer: eqx.nn.Linear

    def __init__(self, input_size: int, action_size: int, key: Array) -> None:
        keys = jax.random.split(key, 4)
        self.linear1 = eqx.nn.Linear(input_size, 256, key=keys[0])
        self.linear2 = eqx.nn.Linear(256, 256, key=keys[1])
        self.mu_layer = eqx.nn.Linear(256, action_size, key=keys[2])
        self.log_sigma_layer = eqx.nn.Linear(256, action_size, key=keys[3])

        # Parameter initialization according to Trick #2
        self.linear1 = self.initialize_layer(self.linear1, np.sqrt(2), keys[0])
        self.linear2 = self.initialize_layer(self.linear2, np.sqrt(2), keys[1])
        self.mu_layer = self.initialize_layer(self.mu_layer, 0.01, keys[2])
        self.log_sigma_layer = self.initialize_layer(self.log_sigma_layer, 0.01, keys[3])

    def __call__(self, x: Array) -> Tuple[Array, Array]:
        x = jax.nn.tanh(self.linear1(x))
        x = jax.nn.tanh(self.linear2(x))
        mu = self.mu_layer(x)
        log_sigma = self.log_sigma_layer(x)
        return mu, jnp.exp(log_sigma)

    def initialize_layer(self, layer: eqx.nn.Linear, scale: float, key: Array) -> eqx.nn.Linear:
        weight_shape = layer.weight.shape

        initializer = jax.nn.initializers.orthogonal()
        new_weight = initializer(key, weight_shape, jnp.float32) * scale
        new_bias = jnp.zeros(layer.bias.shape) if layer.bias is not None else None

        def where_weight(layer: eqx.nn.Linear) -> Array:
            return layer.weight

        def where_bias(layer: eqx.nn.Linear) -> Array | None:
            return layer.bias

        new_layer = eqx.tree_at(where_weight, layer, new_weight)
        new_layer = eqx.tree_at(where_bias, new_layer, new_bias)

        return new_layer


class Critic(eqx.Module):
    """Critic network for PPO."""

    linear1: eqx.nn.Linear
    linear2: eqx.nn.Linear
    value_layer: eqx.nn.Linear

    def __init__(self, input_size: int, key: Array) -> None:
        keys = jax.random.split(key, 3)
        self.linear1 = eqx.nn.Linear(input_size, 256, key=keys[0])
        self.linear2 = eqx.nn.Linear(256, 256, key=keys[1])
        self.value_layer = eqx.nn.Linear(256, 1, key=keys[2])

        # Parameter initialization according to Trick #2
        self.linear1 = self.initialize_layer(self.linear1, np.sqrt(2), keys[0])
        self.linear2 = self.initialize_layer(self.linear2, np.sqrt(2), keys[1])
        self.value_layer = self.initialize_layer(self.value_layer, 1.0, keys[2])

    def __call__(self, x: Array) -> Array:
        x = jax.nn.tanh(self.linear1(x))
        x = jax.nn.tanh(self.linear2(x))
        return self.value_layer(x)

    def initialize_layer(self, layer: eqx.nn.Linear, scale: float, key: Array) -> eqx.nn.Linear:
        weight_shape = layer.weight.shape

        initializer = jax.nn.initializers.orthogonal()
        new_weight = initializer(key, weight_shape, jnp.float32) * scale
        new_bias = jnp.zeros(layer.bias.shape) if layer.bias is not None else None

        def where_weight(layer: eqx.nn.Linear) -> Array:
            return layer.weight

        def where_bias(layer: eqx.nn.Linear) -> Array | None:
            return layer.bias

        new_layer = eqx.tree_at(where_weight, layer, new_weight)
        new_layer = eqx.tree_at(where_bias, new_layer, new_bias)

        return new_layer


class Ppo:
    def __init__(self, observation_size: int, action_size: int, config: Config, key: Array) -> None:
        """Initialize the PPO model's actor/critic structure and optimizers."""
        key, subkey1, subkey2 = jax.random.split(key, 3)
        self.actor = Actor(observation_size, action_size, subkey1)
        self.critic = Critic(observation_size, subkey2)

        total_timesteps = config.num_iterations

        # Learning rate annealing according to Trick #4
        self.actor_schedule = optax.linear_schedule(
            init_value=config.lr_actor, end_value=0, transition_steps=total_timesteps
        )
        self.critic_schedule = optax.linear_schedule(
            init_value=config.lr_critic, end_value=0, transition_steps=total_timesteps
        )

        # eps below according to Trick #3
        self.actor_optim = optax.chain(optax.adam(learning_rate=self.actor_schedule, eps=1e-5))
        self.critic_optim = optax.chain(
            optax.adamw(learning_rate=self.critic_schedule, weight_decay=config.l2_rate, eps=1e-5)
        )

        # Initialize optimizer states
        self.actor_opt_state = self.actor_optim.init(eqx.filter(self.actor, eqx.is_array))
        self.critic_opt_state = self.critic_optim.init(eqx.filter(self.critic, eqx.is_array))

    def get_params(self) -> Dict[str, Any]:
        """Get the parameters of the PPO model."""
        return {
            "actor": self.actor,
            "critic": self.critic,
            "actor_opt_state": self.actor_opt_state,
            "critic_opt_state": self.critic_opt_state,
        }

    def update_params(self, new_params: Dict[str, Any]) -> None:
        """Update the parameters of the PPO model."""
        self.actor = new_params["actor"]
        self.critic = new_params["critic"]
        self.actor_opt_state = new_params["actor_opt_state"]
        self.critic_opt_state = new_params["critic_opt_state"]


@jax.jit
def apply_critic(critic: Critic, state: Array) -> Array:
    return critic(state)


@jax.jit
def apply_actor(actor: Critic, state: Array) -> Array:
    return actor(state)


def train_step(
    actor_optim: optax.GradientTransformation,
    critic_optim: optax.GradientTransformation,
    params: Dict[str, Any],
    states_b: Array,
    actions_b: Array,
    returns_b: Array,
    advants_b: Array,
    old_log_prob_b: Array,
    config: Config,
) -> Tuple[Dict[str, Any], Array, Array]:
    """Perform a single training step with PPO parameters."""
    actor, critic, actor_opt_state, critic_opt_state = params.values()

    actor_vmap = jax.vmap(apply_actor, in_axes=(None, 0))
    critic_vmap = jax.vmap(apply_critic, in_axes=(None, 0))

    # Normalizing advantages *in minibatch* according to Trick #7
    advants_b = (advants_b - advants_b.mean()) / (advants_b.std() + 1e-8)

    @eqx.filter_value_and_grad
    def actor_loss_fn(actor: Actor) -> Array:
        """Prioritizing advantageous actions over more training."""
        mu_b, std_b = actor_vmap(actor, states_b)
        new_log_prob_b = actor_log_prob(mu_b, std_b, actions_b)

        # Calculating the ratio of new and old probabilities
        ratio_b = jnp.exp(new_log_prob_b - old_log_prob_b)
        surrogate_loss_b = ratio_b * advants_b

        # Clipping is done to prevent too much change if new advantages are very large
        clipped_loss_b = jnp.clip(ratio_b, 1.0 - config.epsilon, 1.0 + config.epsilon) * advants_b

        actor_loss = -jnp.mean(jnp.minimum(surrogate_loss_b, clipped_loss_b))
        entropy_loss = jnp.mean(0.5 * (jnp.log(2 * jnp.pi * std_b**2) + 1))
        
        total_loss = actor_loss - config.entropy_coeff * entropy_loss
        return total_loss

    @eqx.filter_value_and_grad
    def critic_loss_fn(critic: Critic) -> Array:
        """Prioritizing being able to predict the ground truth returns."""
        critic_returns_b = critic_vmap(critic, states_b).squeeze()
        critic_loss = jnp.mean((critic_returns_b - returns_b) ** 2)
        return critic_loss

    # Calculating actor loss and updating actor parameters
    actor_loss, actor_grads = actor_loss_fn(actor)
    actor_updates, new_actor_opt_state = actor_optim.update(actor_grads, actor_opt_state, params=actor)
    new_actor = eqx.apply_updates(actor, actor_updates)

    # Calculating critic loss and updating critic parameters
    critic_loss, critic_grads = critic_loss_fn(critic)
    critic_updates, new_critic_opt_state = critic_optim.update(critic_grads, critic_opt_state, params=critic)
    new_critic = eqx.apply_updates(critic, critic_updates)

    new_params = {
        "actor": new_actor,
        "critic": new_critic,
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
        xs=(rewards, masks, values),  # NOTE: correct direction?
        reverse=True,
    )
    returns = advantages + values
    return returns, advantages


def train(ppo: Ppo, memory: List[Tuple[Array, Array, Array, Array]], config: Config) -> None:
    """Train the PPO model using the memory collected from the environment."""

    # Reorders memory according to states, actions, rewards, masks
    states = jnp.array([e[0] for e in memory])
    actions = jnp.array([e[1] for e in memory])
    rewards = jnp.array([e[2] for e in memory])
    masks = jnp.array([e[3] for e in memory])

    # Calculate old log probabilities
    actor_vmap = jax.vmap(apply_actor, in_axes=(None, 0))
    old_mu, old_std = actor_vmap(ppo.actor, states)
    old_log_prob = actor_log_prob(old_mu, old_std, actions)

    # Calculate values for all states
    critic_vmap = jax.vmap(apply_critic, in_axes=(None, 0))
    values = critic_vmap(ppo.critic, states).squeeze()

    # NOTE: are the output shapes correct?

    # Calculate GAE and returns
    returns, advantages = get_gae(rewards, masks, values, config)

    n = len(states)
    arr = jnp.arange(n)
    key = jax.random.PRNGKey(0)

    for epoch in range(1):
        key, subkey = jax.random.split(key)
        arr = jax.random.permutation(subkey, arr)
        total_actor_loss = 0.0
        total_critic_loss = 0.0
        logger.info("Processing %d batches", n // config.batch_size)
        for i in range(n // config.batch_size):

            # Batching the data
            batch_indices = arr[config.batch_size * i : config.batch_size * (i + 1)]
            states_b = states[batch_indices]
            actions_b = actions[batch_indices]
            returns_b = returns[batch_indices]
            advantages_b = advantages[batch_indices]
            old_log_prob_b = old_log_prob[batch_indices]

            params = ppo.get_params()
            new_params, actor_loss, critic_loss = train_step(
                ppo.actor_optim,
                ppo.critic_optim,
                params,
                states_b,
                actions_b,
                returns_b,
                advantages_b,
                old_log_prob_b,
                config,
            )
            ppo.update_params(new_params)

            total_actor_loss += actor_loss.mean().item()
            total_critic_loss += critic_loss.mean().item()

        mean_actor_loss = total_actor_loss / (n // config.batch_size)
        mean_critic_loss = total_critic_loss / (n // config.batch_size)

        logger.info(f"Mean Actor Loss: {mean_actor_loss}, Mean Critic Loss: {mean_critic_loss}")


def actor_log_prob(mu: Array, sigma: Array, actions: Array) -> Array:
    """Calculate the log probability of the actions given the actor network's output."""
    return jax.scipy.stats.norm.logpdf(actions, mu, sigma).sum(axis=-1)


def actor_distribution(mu: Array, sigma: Array, rng: Array) -> Array:
    """Get an action from the actor network from its probability distribution of actions."""
    return jax.random.normal(rng, shape=mu.shape) * sigma + mu


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
            if not attr.startswith("_") and not callable(getattr(state, attr)):
                value = getattr(state, attr)
                try:
                    new_state[attr] = value[i]
                except Exception:
                    # logger.warning(f"Could not get first element of {attr}.")
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


@jax.jit
def choose_action(actor: Actor, obs: Array, rng: Array) -> Array:
    # Given a state, we do our forward pass and then sample from to maintain "random actions"
    mu, sigma = apply_actor(actor, obs)
    return actor_distribution(mu, sigma, rng)


@jax.jit
def update_memory(memory: Dict[str, Array], new_data: Dict[str, Array]) -> Dict[str, Array]:
    return jax.tree.map(lambda x, y: jnp.concatenate([x, y]), memory, new_data)


def reorder_memory(memory, num_envs):
    reordered_memory = {
        "states": jnp.concatenate([memory["states"][i::num_envs] for i in range(num_envs)], axis=0),
        "actions": jnp.concatenate([memory["actions"][i::num_envs] for i in range(num_envs)], axis=0),
        "rewards": jnp.concatenate([memory["rewards"][i::num_envs] for i in range(num_envs)], axis=0),
        "masks": jnp.concatenate([memory["masks"][i::num_envs] for i in range(num_envs)], axis=0),
    }
    return reordered_memory


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default="Humanoid-v2", help="name of environmnet to put into logs")
    parser.add_argument("--render", action="store_true", help="render the environment")
    parser.add_argument("--width", type=int, default=640, help="width of the video frame")
    parser.add_argument("--height", type=int, default=480, help="height of the video frame")
    parser.add_argument("--render_every", type=int, default=2, help="render the environment every N steps")
    parser.add_argument("--video_length", type=int, default=5, help="maxmimum length of video in seconds")
    parser.add_argument("--save_video_every", type=int, default=100, help="save video every N iterations")
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
        scores = []
        steps = 0
        rollout: List[MjxState] = []

        rng, reset_rng = jax.random.split(rng)
        states = reset_fn(reset_rng)
        pbar = tqdm(total=config.max_steps_per_iteration, desc=f"Steps for iteration {i}")

        while steps < config.max_steps_per_iteration:
            episodes += config.num_envs

            obs = jax.device_put(states.obs)
            score = jnp.zeros(config.num_envs)

            memory = {
                "states": jnp.empty((0, observation_size)),
                "actions": jnp.empty((0, action_size)),
                "rewards": jnp.empty((0,)),
                "masks": jnp.empty((0,)),
            }

            for _ in range(config.max_steps_per_episode):

                # Choosing actions
                choose_action_vmap = jax.vmap(choose_action, in_axes=(None, 0, 0))
                rng, *action_rng = jax.random.split(rng, num=config.num_envs + 1)
                actions = choose_action_vmap(ppo.actor, obs, jnp.array(action_rng))

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
                    rng, reset_rng = jax.random.split(rng)
                    states = reset_fn(reset_rng)
                    break

            # with open("log_" + args.env_name + ".txt", "a") as outfile:
            #     outfile.write("\t" + str(episodes) + "\t" + str(jnp.mean(score)) + "\n")
            scores.append(jnp.mean(score))

            # Convert memory to the format expected by ppo.train

            memory = reorder_memory(memory, config.num_envs)

            train_memory = [
                (s, a, r, m)
                for s, a, r, m in zip(memory["states"], memory["actions"], memory["rewards"], memory["masks"])
            ]
            train(ppo, train_memory, config)

        score_avg = float(jnp.mean(jnp.array(scores)))
        pbar.close()
        logger.info("Episode %s score is %.2f", episodes, score_avg)

        # Save video for this iteration
        if args.save_video_every and i % args.save_video_every == 0 and rollout:
            images = jnp.array(
                env.render(rollout[:: args.render_every], camera="side", width=args.width, height=args.height)
            )
            fps = int(1 / env.dt)

            script_dir = os.path.dirname(__file__)
            videos_dir = os.path.join(script_dir, "videos")

            if not os.path.exists(videos_dir):
                os.makedirs(videos_dir)

            video_path = os.path.join(videos_dir, f"{args.env_name}_video{i}.mp4")

            logger.info("Saving video to %s for iteration %d", video_path, i)
            media.write_video(video_path, images, fps=fps)


if __name__ == "__main__":
    main()
