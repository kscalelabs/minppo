"""Trains a policy network to get a humanoid to stand up."""

import argparse
import atexit
import contextlib
import datetime
import logging
import os
import pickle
from dataclasses import dataclass, field
from functools import partial
import shutil
import signal
import distrax
import sys
from typing import Any, Dict, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from brax.envs import State
from einops import rearrange
from jax import Array

from environment import HumanoidEnv

logger = logging.getLogger(__name__)


@dataclass
class Config:
    lr_actor: float = field(default=3e-4, metadata={"help": "Learning rate for the actor network."})
    lr_critic: float = field(default=3e-4, metadata={"help": "Learning rate for the critic network."})
    num_iterations: int = field(default=1500, metadata={"help": "Number of environment simulation iterations."})
    num_envs: int = field(default=32, metadata={"help": "Number of environments to run at once with vectorization."})
    max_steps_per_episode: int = field(
        default=512,
        metadata={"help": "Maximum number of steps per episode in a single environment."},
    )
    max_episodes_per_iteration: int = field(
        default=4,
        metadata={"help": "Maximum number of episodes per iteration before running collected data through train."},
    )
    minibatch_size: int = field(
        default=64, metadata={"help": "Minibatch size for batching training episodes from a data collection cycle."}
    )
    gamma: float = field(default=0.99, metadata={"help": "Discount factor for future rewards."})
    lambd: float = field(default=0.95, metadata={"help": "Lambda parameter for GAE calculation."})
    epsilon: float = field(default=0.2, metadata={"help": "Clipping parameter for PPO."})
    l2_rate: float = field(default=0.001, metadata={"help": "L2 regularization rate for the critic."})
    entropy_coeff: float = field(default=0.01, metadata={"help": "Coefficient for entropy loss."})  # 0.01
    minimal: bool = field(default=False, metadata={"help": "Make minimal PPO (no std) for breakpoint debugging"})
    anneal_on_train_step: bool = field(
        default=False, metadata={"help": "Whether to anneal learning rate on train() or train_step()"}
    )


# Separate actor/critic models
class Actor(eqx.Module):
    """Actor network for PPO."""

    # Defined parameters for Equinox to learn
    linear1: eqx.nn.Linear
    linear2: eqx.nn.Linear
    mu_layer: eqx.nn.Linear
    log_std: Array
    log_sigma_layer: eqx.nn.Linear

    def __init__(self, input_size: int, action_size: int, key: Array) -> None:
        keys = jax.random.split(key, 4)
        self.linear1 = eqx.nn.Linear(input_size, 256, key=keys[0])
        self.linear2 = eqx.nn.Linear(256, 256, key=keys[1])
        self.mu_layer = eqx.nn.Linear(256, action_size, key=keys[2])

        # learnable parameter for general std
        self.log_std = jnp.zeros(action_size)

        self.log_sigma_layer = eqx.nn.Linear(256, action_size, key=keys[3])

        # Orthogonal parameter initialization according to Trick #2
        self.linear1 = self.initialize_layer(self.linear1, np.sqrt(2), keys[0])
        self.linear2 = self.initialize_layer(self.linear2, np.sqrt(2), keys[1])
        self.mu_layer = self.initialize_layer(self.mu_layer, 0.01, keys[2])
        self.log_sigma_layer = self.initialize_layer(self.log_sigma_layer, 0.01, keys[3])

    def __call__(self, x: Array) -> Tuple[Array, Array]:
        x = jax.nn.tanh(self.linear1(x))
        x = jax.nn.tanh(self.linear2(x))
        mu = self.mu_layer(x)

        # State-independent standard deviation
        # log_std = jnp.broadcast_to(self.log_std, mu.shape)

        # State-dependent standard deviation
        log_std = self.log_sigma_layer(x)

        return mu, jnp.exp(log_std)

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

        # Orthogonal parameter initialization according to Trick #2
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

        if config.anneal_on_train_step:
            # Number times train() ran * number of times train_step() ran inside train()
            num_minibatches = (config.max_steps_per_episode) // config.minibatch_size
            total_timesteps = (config.max_episodes_per_iteration * config.num_iterations) * num_minibatches
        else:
            # Annealing at each call of train()
            total_timesteps = config.max_episodes_per_iteration * config.num_iterations

        # Learning rate annealing according to Trick #4
        self.actor_schedule = optax.linear_schedule(
            init_value=config.lr_actor, end_value=0, transition_steps=total_timesteps
        )
        
        # eps below according to Trick #3
        self.actor_optim = optax.chain(
            optax.clip_by_global_norm(0.5), optax.adam(learning_rate=self.actor_schedule, eps=1e-5)
        )
        self.critic_optim = optax.chain(
            optax.clip_by_global_norm(0.5),
            optax.adamw(learning_rate=config.lr_critic, weight_decay=config.l2_rate, eps=1e-5),
        )

        # Initialize optimizer states
        self.actor_opt_state = self.actor_optim.init(eqx.filter(self.actor, eqx.is_array))
        self.critic_opt_state = self.critic_optim.init(eqx.filter(self.critic, eqx.is_array))

    def get_params(self) -> Tuple[Any]:
        """Get the parameters of the PPO model."""
        return (
            self.actor,
            self.critic,
            self.actor_opt_state,
            self.critic_opt_state,
        )

    def update_params(self, new_params: Tuple[Any]) -> None:
        """Update the parameters of the PPO model."""
        self.actor = new_params[0]
        self.critic = new_params[1]
        self.actor_opt_state = new_params[2]
        self.critic_opt_state = new_params[3]


@jax.jit
def apply_critic(critic: Critic, state: Array) -> Array:
    return critic(state)


@jax.jit
def apply_actor(actor: Critic, state: Array) -> Array:
    return actor(state)


# NOTE: vmappable and pmappable given enough effort...
def train_step(
    actor_optim: optax.GradientTransformation,
    critic_optim: optax.GradientTransformation,
    params: Dict[str, Any],
    states_b: Array,
    actions_b: Array,
    returns_b: Array,
    advants_b: Array,
    old_log_prob_b: Array,
    epsilon: float = 0.2,
    entropy_coeff: float = 0.01,
    critic_coeff: float = 0.5,
    minimal: bool = False,
) -> Tuple[Tuple[Any], Tuple[Array, Array]]:
    """Perform a single training step with PPO parameters."""
    actor, critic, actor_opt_state, critic_opt_state = params

    actor_vmap = jax.vmap(apply_actor, in_axes=(None, 0))
    critic_vmap = jax.vmap(apply_critic, in_axes=(None, 0))

    # Normalizing advantages in batch to improve training
    advants_b = (advants_b - advants_b.mean()) / (advants_b.std() + 1e-4)

    @partial(eqx.filter_value_and_grad, has_aux=True)
    def actor_loss_fn(actor: Actor) -> Array:
        """Prioritizing advantage actions to be more probable in actor distribution with more training."""
        mu_b, std_b = actor_vmap(actor, states_b)
        new_log_prob_b = actor_log_prob(mu_b, std_b, actions_b)

        # Calculating the ratio of new and old probabilities
        ratio_b = jnp.exp(new_log_prob_b - old_log_prob_b)
        surrogate_loss_b = ratio_b * advants_b

        # Clipping is done to prevent too much change if new advantages are very large
        clipped_loss_b = jnp.clip(ratio_b, 1.0 - epsilon, 1.0 + epsilon) * advants_b

        # Choosing the smallest magnitude loss
        actor_loss = -jnp.mean(jnp.minimum(surrogate_loss_b, clipped_loss_b))

        # Entropy loss to encourage exploration
        entropy_loss = jnp.mean(0.5 * (jnp.log(2 * jnp.pi * (std_b + 1e-4) ** 2) + 1))

        total_loss = actor_loss - entropy_coeff * entropy_loss
        return total_loss, (new_log_prob_b, mu_b, std_b, actor_loss, entropy_loss, surrogate_loss_b, clipped_loss_b)

    @eqx.filter_value_and_grad
    def critic_loss_fn(critic: Critic) -> Array:
        """Prioritizing being able to predict the ground truth returns."""
        critic_returns_b = critic_vmap(critic, states_b).squeeze()
        critic_loss = jnp.mean((critic_returns_b - returns_b) ** 2)

        return critic_loss

    # Calculating actor loss and updating actor parameters --- outputting auxillary data for logging
    (actor_loss, values), actor_grads = actor_loss_fn(actor)
    actor_updates, new_actor_opt_state = actor_optim.update(actor_grads, actor_opt_state, params=actor)
    new_actor = eqx.apply_updates(actor, actor_updates)

    # if minimal:
    #     breakpoint()

    # if jnp.any(jnp.isnan(actor_loss)):
    #     breakpoint()

    # Calculating critic loss and updating critic parameters
    critic_loss, critic_grads = critic_loss_fn(critic)
    critic_updates, new_critic_opt_state = critic_optim.update(critic_grads, critic_opt_state, params=critic)
    new_critic = eqx.apply_updates(critic, critic_updates)

    new_params = (
        new_actor,
        new_critic,
        new_actor_opt_state,
        new_critic_opt_state,
    )

    return new_params, (actor_loss, critic_loss)


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


# NOTE: should do next value stuff?
# def get_gae(rewards: Array, masks: Array, values: Array, config: Config) -> Tuple[Array, Array]:
#     """Calculate the Generalized Advantage Estimation for rewards using jax.lax.scan."""

#     def gae_step(carry: Tuple[Array, Array], inp: Tuple[Array, Array, Array]) -> Tuple[Tuple[Array, Array], Array]:
#         """Single step of Generalized Advantage Estimation to be iterated over."""
#         advantages, last_advantage, last_value = carry

#         reward, mask, value, t = inp
#         delta = reward + config.gamma * last_value * mask - value
#         last_advantage = delta + config.gamma * config.lambd * mask * last_advantage

#         advantages = advantages.at[t].set(last_advantage)

#         return (advantages, last_advantage, value), None

#     # Calculating reward, with combination of immediate reward and diminishing value of future rewards
#     (advantages, _, _), _ = jax.lax.scan(
#         f=gae_step,
#         init=(jnp.zeros_like(rewards), jnp.array(0.0), jnp.array(0.0)),
#         xs=(rewards, masks, values, jnp.arange(len(rewards))),  # NOTE: correct direction?
#         reverse=True,
#     )

#     returns = advantages + values

#     return returns, advantages


def train(ppo: Ppo, states: Array, actions: Array, rewards: Array, masks: Array, config: Config) -> None:
    """Train the PPO model using the memory collected from the environment."""
    # Calculate old log probabilities
    actor_vmap = jax.vmap(apply_actor, in_axes=(None, 0))
    old_mu, old_std = actor_vmap(ppo.actor, states)
    old_log_prob = actor_log_prob(old_mu, old_std, actions)

    # Calculate values for all states
    critic_vmap = jax.vmap(apply_critic, in_axes=(None, 0))
    values = critic_vmap(ppo.critic, states).squeeze()

    # Calculate GAE and returns
    returns, advantages = get_gae(rewards, masks, values, config)

    assert (
        config.max_steps_per_episode % config.minibatch_size == 0
    ), "Batch size must divide max_steps_per_episode to properly calculate GAE"

    batch_size = len(states)
    arr = jnp.arange(batch_size)
    key = jax.random.PRNGKey(0)
    num_minibatches = batch_size // config.minibatch_size

    for epoch in range(1):
        key, subkey = jax.random.split(key)
        arr = jax.random.permutation(subkey, arr)

        # Calculate average advantages and returns
        total_actor_loss = 0.0
        total_critic_loss = 0.0

        actor_optim = ppo.actor_optim
        critic_optim = ppo.critic_optim

        def scan_body(carry: Tuple[Dict[str, Any], Tuple[float, float]], batch_indices: Array) -> Tuple[Tuple, None]:
            """Scan function to run through training loop quicker."""
            params, total_losses = carry

            # Batching the data
            states_b = states[batch_indices]
            actions_b = actions[batch_indices]
            returns_b = returns[batch_indices]
            advantages_b = advantages[batch_indices]
            old_log_prob_b = old_log_prob[batch_indices]

            new_params, (actor_loss, critic_loss) = train_step(
                actor_optim,
                critic_optim,
                params,
                states_b,
                actions_b,
                returns_b,
                advantages_b,
                old_log_prob_b,
                epsilon=config.epsilon,
                entropy_coeff=config.entropy_coeff,
                minimal=config.minimal,
            )

            # Distributed data parallel --- all gather of parameter updates

            total_actor_loss, total_critic_loss = total_losses
            total_actor_loss += actor_loss.mean()
            total_critic_loss += critic_loss.mean()

            return (new_params, (total_actor_loss, total_critic_loss)), None

        key, subkey = jax.random.split(key)
        arr = jax.random.permutation(subkey, arr)

        # Pre-batch the indices
        batched_indices = jnp.array(
            [arr[i * config.minibatch_size : (i + 1) * config.minibatch_size] for i in range(num_minibatches)]
        )

        # Calculate average advantages and returns
        initial_total_losses = (0.0, 0.0)  # (total_actor_loss, total_critic_loss)

        logger.info("Processing %d batches", num_minibatches)

        # TypeError: unhashable type: 'jaxlib.xla_extension.ArrayImpl'
        # Run the optimized loop using jax.lax.scan
        (new_params, (total_actor_loss, total_critic_loss)), _ = jax.lax.scan(
            scan_body, (ppo.get_params(), initial_total_losses), batched_indices
        )

        ppo.update_params(new_params)

        mean_actor_loss = total_actor_loss / num_minibatches
        mean_critic_loss = total_critic_loss / num_minibatches

        logger.info(f"Mean Actor Loss: {mean_actor_loss}, Mean Critic Loss: {mean_critic_loss}")


#################### ACTION INFERENCE ####################


def actor_log_prob(mu: Array, sigma: Array, actions: Array) -> Array:
    """Calculate the log probability of the actions given the actor network's output for actor loss."""
    # Summing across the number of actions after logpdf of each relative to mu/sigma, determined by state
    return jax.scipy.stats.norm.logpdf(actions, mu, sigma + 1e-4).sum(axis=1)


@jax.jit
def choose_action(actor: Actor, obs: Array, rng: Array) -> Array:
    """Jitted function taking the actor mu/sigma and sampling from its normal distribution to get action."""
    # Given a state, we do our forward pass and then sample from the mu/sigma output to maintain "random actions"
    mu, sigma = apply_actor(actor, obs)
    action = jax.random.normal(rng, shape=mu.shape) * (sigma + 1e-4) + mu
    return action


#################### MODEL SAVING + DEBUG ####################


def save_models(ppo: Ppo, iteration: int, save_dir: str, env_name: str) -> None:
    """Saves the actor and critic models."""
    os.makedirs(save_dir, exist_ok=True)

    actor_path = os.path.join(save_dir, f"{env_name}_actor_{iteration}.pkl")
    critic_path = os.path.join(save_dir, f"{env_name}_critic_{iteration}.pkl")

    with open(actor_path, "wb") as f:
        pickle.dump(ppo.actor, f)

    with open(critic_path, "wb") as f:
        pickle.dump(ppo.critic, f)

    logger.info(f"Saved models at iteration {iteration} to {actor_path} and {critic_path}")


class StdoutCapture:
    def __init__(self, dir):
        self.dir = dir
        # Ensure the directory exists
        if not os.path.exists(self.dir):
            os.makedirs(self.dir)
        self.captured_output = []

        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        backup_file_path = os.path.join(self.dir, f"{self.timestamp}.backup")
        shutil.copy(__file__, backup_file_path)
        logger.info(f"Backed up file to {backup_file_path}")

    def write(self, text):
        self.captured_output.append(text)
        sys.__stdout__.write(text)

    def flush(self):
        sys.__stdout__.flush()

    def save_output(self):
        output_file_path = os.path.join(self.dir, f"{self.timestamp}.out")
        logger.info(output_file_path)
        with open(output_file_path, "w") as f:
            f.writelines(self.captured_output)


def setup_interrupt_handling(stdout_capture):
    def signal_handler(signum, frame):
        logger.info("\nInterrupt received. Saving output and exiting...")
        stdout_capture.save_output()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    atexit.register(stdout_capture.save_output)


def print_config_and_args(config, args):
    logger.info("Configuration:")
    for key, value in vars(config).items():
        print(f"  {key}: {value}")
    logger.info("\nArguments:")
    for key, value in vars(args).items():
        print(f"  {key}: {value}")
    logger.info("\n")


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
    parser.add_argument("--video_length", type=int, default=10, help="maxmimum length of video in seconds")
    parser.add_argument("--save_video_every", type=int, default=100, help="save video every N iterations")
    parser.add_argument("--envs_to_sample", type=int, default=4, help="number of environments to sample for video")
    parser.add_argument("--save_every", type=int, default=10, help="save models every N iterations")
    parser.add_argument("--save_dir", type=str, default="models", help="directory to save models")

    # testing
    parser.add_argument("--debug", action="store_true", help="whether to log whole file and config/args")
    parser.add_argument("--anneal_on_train_step", action="store_true", help="whether to log whole file and config/args")
    args = parser.parse_args()

    config = Config()
    config.anneal_on_train_step = args.anneal_on_train_step

    env = HumanoidEnv()
    observation_size = env.observation_size
    action_size = env.action_size

    logger.info("Action size: %s", action_size)
    logger.info("Observation size: %s", observation_size)

    @jax.jit
    def reset_fn(rng: Array) -> State:
        rngs = jax.random.split(rng, config.num_envs)
        return jax.vmap(env.reset)(rngs)

    @jax.jit
    def step_fn(states: State, actions: jax.Array) -> State:
        return jax.vmap(env.step)(states, actions)

    rng = jax.random.PRNGKey(0)
    np.random.seed(500)

    ppo = Ppo(observation_size, action_size, config, rng)
    episodes: int = 0

    if args.debug:
        stdout_capture = StdoutCapture(f"logs/{args.env_name}")
        setup_interrupt_handling(stdout_capture)
        stdout_context = contextlib.redirect_stdout(stdout_capture)
    else:
        # If not using StdoutCapture, use a dummy context manager that does nothing
        @contextlib.contextmanager
        def dummy_context_manager():
            yield

        stdout_context = dummy_context_manager()

    with stdout_context:
        print_config_and_args(config, args)

        # NOTE: should start doing > output.txt
        for i in range(1, config.num_iterations + 1):
            # Initialize memory as JAX arrays
            scores = []
            rng, reset_rng = jax.random.split(rng)
            states = reset_fn(reset_rng)

            for _ in range(config.max_episodes_per_iteration):
                # NOTE: this actually repeats the same starting states? higher max episodes is better?
                # NOTE: this shouldnt be expected behavior which means there's a greater bug
                # NOTE: training loop should actually track prev obs maybe, so as if training on
                # small blocks from episode
                episodes += config.num_envs

                # Normalizing observations improves training
                norm_obs = (states.obs - jnp.mean(states.obs, axis=1, keepdims=True)) / (
                    jnp.std(states.obs, axis=1, keepdims=True) + 1e-4
                )

                # Initialize memory as full matrices of zeros
                memory = {
                    "states": jnp.zeros((config.max_steps_per_episode, config.num_envs, observation_size)),
                    "actions": jnp.zeros((config.max_steps_per_episode, config.num_envs, action_size)),
                    "rewards": jnp.zeros((config.max_steps_per_episode, config.num_envs)),
                    "masks": jnp.zeros((config.max_steps_per_episode, config.num_envs)),
                    # "step number": jnp.zeros((config.max_steps_per_episode, config.num_envs)),
                }

                def step_loop(carry, t: int) -> Tuple[Tuple, None]:
                    """Environment step/data collection function to be scanned quickly."""
                    rng, obs, states, memory, score = carry

                    # Calculating actions
                    rng, *action_rng = jax.random.split(rng, num=config.num_envs + 1)
                    actions = jax.vmap(choose_action, in_axes=(None, 0, 0))(ppo.actor, obs, jnp.array(action_rng))
                    next_states = step_fn(states, actions)

                    # Normalizing observations improves training
                    norm_obs = (next_states.obs - jnp.mean(next_states.obs, axis=1, keepdims=True)) / (
                        jnp.std(next_states.obs, axis=1, keepdims=True) + 1e-4
                    )

                    # Check for NaNs in obs or actions and set values to zero if NaNs are found
                    nan_mask = (
                        jnp.any(jnp.isnan(norm_obs))
                        | jnp.any(jnp.isnan(actions))
                        | jnp.any(jnp.isnan(next_states.reward))
                    )
                    rewards = jnp.where(nan_mask, jnp.zeros_like(next_states.reward), next_states.reward)
                    norm_obs = jnp.where(nan_mask, jnp.zeros_like(norm_obs), norm_obs)
                    actions = jnp.where(nan_mask, jnp.zeros_like(actions), actions)
                    dones = jnp.where(nan_mask, jnp.ones_like(next_states.done), next_states.done)

                    # nan_mask = (
                    #     jnp.isnan(norm_obs)
                    #     | jnp.isnan(actions)
                    #     | jnp.isnan(next_states.reward)
                    # )
                    # rewards = jnp.where(nan_mask, jnp.zeros_like(next_states.reward), next_states.reward)
                    # norm_obs = jnp.where(nan_mask, jnp.zeros_like(norm_obs), norm_obs)
                    # actions = jnp.where(nan_mask, jnp.zeros_like(actions), actions)
                    # dones = jnp.where(nan_mask, jnp.ones_like(next_states.done), next_states.done)

                    masks = (1 - dones).astype(jnp.float32)

                    # Update memory at the current timestep
                    memory = {
                        "states": memory["states"].at[t].set(obs),
                        "actions": memory["actions"].at[t].set(actions),
                        "rewards": memory["rewards"].at[t].set(rewards),
                        "masks": memory["masks"].at[t].set(masks),
                        # "step number": memory["step number"].at[t].set(jnp.array([t] * config.num_envs)),
                    }

                    new_score = score + rewards * masks

                    return (rng, norm_obs, next_states, memory, new_score), None

                (final_rng, final_obs, final_states, final_memory, final_score), _ = jax.lax.scan(
                    step_loop,
                    (rng, norm_obs, states, memory, jnp.zeros(config.num_envs)),
                    xs=jnp.arange(config.max_steps_per_episode),
                )

                rng = final_rng
                scores.append(jnp.mean(final_score))
                rng, reset_rng = jax.random.split(rng)
                states = reset_fn(reset_rng)

                if jnp.any(final_score > 1500) or jnp.any(final_score < -1500):
                    breakpoint()

                # We want all episodes that are of the same environment to be adjacent in memory (hence, n t)
                states_memory = rearrange(final_memory["states"], "t n obs_size -> (n t) obs_size")
                actions_memory = rearrange(final_memory["actions"], "t n action_size -> (n t) action_size")
                rewards_memory = rearrange(final_memory["rewards"], "t n -> (n t)")
                masks_memory = rearrange(final_memory["masks"], "t n -> (n t)")

                # Technically, all this "memory" is a batch
                train(ppo, states_memory, actions_memory, rewards_memory, masks_memory, config)

            score_avg = float(jnp.mean(jnp.array(scores)))
            logger.info("Episode %s score of iteration %d is %.2f", episodes, i, score_avg)

            if args.save_every and i % args.save_every == 0:
                save_models(ppo, i, args.save_dir, args.env_name)


if __name__ == "__main__":
    main()
