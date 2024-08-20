"""Defines the training script."""

import argparse
import importlib
import os
import pickle
from dataclasses import dataclass, field
import logging
from typing import Callable, NamedTuple

from omegaconf import OmegaConf
import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from brax.envs import State
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
from jax import Array
from omegaconf import OmegaConf

logger = logging.getLogger(__name__)


@dataclass
class Config:
    lr: float = field(default=3e-4, metadata={"help": "Learning rate"})
    num_envs: int = field(default=2048, metadata={"help": "Number of environments"})
    num_steps: int = field(default=10, metadata={"help": "Number of steps"})
    total_timesteps: int = field(default=1_000_000_000, metadata={"help": "Total timesteps"})
    update_epochs: int = field(default=4, metadata={"help": "Number of epochs for update"})
    num_minibatches: int = field(default=32, metadata={"help": "Number of minibatches"})
    gamma: float = field(default=0.99, metadata={"help": "Discount factor"})
    gae_lambda: float = field(default=0.95, metadata={"help": "GAE lambda"})
    clip_eps: float = field(default=0.2, metadata={"help": "Clipping epsilon"})
    ent_coef: float = field(default=0.0, metadata={"help": "Entropy coefficient"})
    vf_coef: float = field(default=0.5, metadata={"help": "Value function coefficient"})
    max_grad_norm: float = field(default=0.5, metadata={"help": "Maximum gradient norm"})
    activation: str = field(default="tanh", metadata={"help": "Activation function"})
    env_module: str = field(default="brax.envs", metadata={"help": "Environment module"})
    anneal_lr: bool = field(default=True, metadata={"help": "Anneal learning rate"})
    normalize_env: bool = field(default=True, metadata={"help": "Normalize environment"})
    debug: bool = field(default=True, metadata={"help": "Debug mode"})
    env_name: str = field(default="humanoid", metadata={"help": "Name of the environment"})


def get_activation(act: str) -> Callable[[Array], Array]:
    match act:
        case "relu":
            return nn.relu
        case "tanh":
            return nn.tanh
        case _:
            raise ValueError(f"Activation {act} not supported")


class ActorCritic(nn.Module):
    action_dim: int
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x: Array) -> tuple[distrax.Distribution, Array]:
        activation = getattr(nn, self.activation)
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh
        actor_mean = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(128, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(actor_mean)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(actor_mean)
        actor_logtstd = self.param("log_std", nn.initializers.zeros, (self.action_dim,))
        pi = distrax.MultivariateNormalDiag(actor_mean, jnp.exp(actor_logtstd))

        critic = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        critic = activation(critic)
        critic = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(critic)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(critic)

        return pi, jnp.squeeze(critic, axis=-1)


class Memory(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray


def save_model(params, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "wb") as f:
        pickle.dump(params, f)


def make_train(config):
    config["NUM_UPDATES"] = config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    config["MINIBATCH_SIZE"] = config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]

    env_module = importlib.import_module(config["ENV_MODULE"])

    env = env_module.HumanoidEnv()

    def linear_schedule(count):
        frac = 1.0 - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"])) / config["NUM_UPDATES"]
        return config["LR"] * frac

    def train(rng):
        network = ActorCritic(env.action_size, activation=config["ACTIVATION"])
        rng, _rng = jax.random.split(rng)
        init_x = jnp.zeros(env.observation_size)
        network_params = network.init(_rng, init_x)

        if config["ANNEAL_LR"]:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )

        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )

        # jit-ifying and vmap-ing functions
        @jax.jit
        def reset_fn(rng: jnp.ndarray) -> State:
            rngs = jax.random.split(rng, config["NUM_ENVS"])
            return jax.vmap(env.reset)(rngs)

        @jax.jit
        def step_fn(states: State, actions: jnp.ndarray, rng: jnp.ndarray) -> State:
            return jax.vmap(env.step)(states, actions, rng)

        # INIT ENV
        rng, reset_rng = jax.random.split(rng)
        env_state = reset_fn(jnp.array(reset_rng))

        obs = env_state.obs

        # TRAIN LOOP
        def _update_step(runner_state, unused):
            """Update steps of the model --- environment memory colelction then network update"""

            # COLLECT MEMORY
            def _env_step(runner_state, unused):
                """Runs NUM_STEPS across all environments and collects memory"""
                train_state, env_state, last_obs, rng = runner_state

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                pi, value = network.apply(train_state.params, last_obs)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)

                # STEP ENV
                # rngs in case environment "done" (terminates" and needs to be reset)
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])

                env_state = step_fn(env_state, action, rng_step)

                # Normalizing observations improves training
                obs = env_state.obs

                reward = env_state.reward
                done = env_state.done
                info = env_state.metrics

                # STORE MEMORY
                memory = Memory(done, action, value, reward, log_prob, last_obs, info)
                runner_state = (train_state, env_state, obs, rng)

                # jax.debug.print("info {}", info)
                return runner_state, memory

            runner_state, mem_batch = jax.lax.scan(_env_step, runner_state, None, config["NUM_STEPS"])

            # CALCULATE ADVANTAGE
            train_state, env_state, last_obs, rng = runner_state
            _, last_val = network.apply(train_state.params, last_obs)

            def _calculate_gae(mem_batch, last_val):
                def _get_advantages(gae_and_next_value, memory):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        memory.done,
                        memory.value,
                        memory.reward,
                    )
                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                    gae = delta + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    mem_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + mem_batch.value

            advantages, targets = _calculate_gae(mem_batch, last_val)

            def _update_epoch(update_state, unused):
                """Scanned function for updating networkfrom all state frames collected above."""

                def _update_minibatch(train_state, batch_info):
                    """Scanned function for updating from a single minibatch (single network update)."""
                    mem_batch, advantages, targets = batch_info

                    def _loss_fn(params, mem_batch, gae, targets):
                        # RERUN NETWORK
                        pi, value = network.apply(params, mem_batch.obs)
                        log_prob = pi.log_prob(mem_batch.action)

                        # CALCULATE VALUE LOSS
                        # want to critic model's ability to predict value
                        # clip prevents from too drastic changes
                        value_pred_clipped = mem_batch.value + (value - mem_batch.value).clip(
                            -config["CLIP_EPS"], config["CLIP_EPS"]
                        )
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()

                        # CALCULATE ACTOR LOSS
                        # want to maximize improvement (log prob diff)
                        # clip prevents from too drastic changes
                        ratio = jnp.exp(log_prob - mem_batch.log_prob)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                            jnp.clip(
                                ratio,
                                1.0 - config["CLIP_EPS"],
                                1.0 + config["CLIP_EPS"],
                            )
                            * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean()
                        entropy = pi.entropy().mean()

                        total_loss = loss_actor + config["VF_COEF"] * value_loss - config["ENT_COEF"] * entropy
                        return total_loss, (value_loss, loss_actor, entropy)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(train_state.params, mem_batch, advantages, targets)
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                train_state, mem_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)
                batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
                assert (
                    batch_size == config["NUM_STEPS"] * config["NUM_ENVS"]
                ), "batch size must be equal to number of steps * number of envs"
                permutation = jax.random.permutation(_rng, batch_size)
                batch = (mem_batch, advantages, targets)

                batch = jax.tree_util.tree_map(lambda x: x.reshape((batch_size,) + x.shape[2:]), batch)
                shuffled_batch = jax.tree_util.tree_map(lambda x: jnp.take(x, permutation, axis=0), batch)

                # organize into minibatches
                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])),
                    shuffled_batch,
                )

                train_state, total_loss = jax.lax.scan(_update_minibatch, train_state, minibatches)
                update_state = (train_state, mem_batch, advantages, targets, rng)
                return update_state, total_loss

            update_state = (train_state, mem_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(_update_epoch, update_state, None, config["UPDATE_EPOCHS"])
            train_state = update_state[0]
            metric = mem_batch.info
            rng = update_state[-1]

            # jax.debug.breakpoint()
            if config.get("DEBUG"):

                def callback(info):
                    return_values = info["returned_episode_returns"][info["returned_episode"]]
                    timesteps = info["timestep"][info["returned_episode"]] * config["NUM_ENVS"]
                    for t in range(len(timesteps)):
                        print(f"global step={timesteps[t]}, episodic return={return_values[t]}")

                jax.debug.callback(callback, metric)

            runner_state = (train_state, env_state, last_obs, rng)
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, env_state, obs, _rng)
        runner_state, metric = jax.lax.scan(_update_step, runner_state, None, config["NUM_UPDATES"])

        return {"runner_state": runner_state, "metrics": metric}

    return train


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Train a model with specified environment name.")
    parser.add_argument("config_path", type=str, nargs="?", help="Path to the config file")
    args = parser.parse_args()

    # Loads the experiment config.
    config = OmegaConf.structured(Config)
    if args.config_path:
        config = OmegaConf.merge(config, OmegaConf.load(args.config_path))

    # Runs the training loop.
    rng = jax.random.PRNGKey(30)
    train_jit = jax.jit(make_train(config))
    out = train_jit(rng)
    logger.info("Training finished.")

    save_model(out["runner_state"][0].params, f"models/{args.env_name}_model.pkl")
    logger.info("Model saved.")
