"""Train a model with a specified environment module."""

import argparse
import logging
import os
import pickle
from typing import Any, Callable, NamedTuple, Sequence

import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from brax.envs import State
from flax.core import FrozenDict
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState

from config import Config, load_config
from environment import HumanoidEnv

logger = logging.getLogger(__name__)


class MLP(nn.Module):
    features: Sequence[int]
    activation: str

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for feat in self.features[:-1]:
            x = nn.Dense(feat, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
            if self.activation == "relu":
                x = nn.relu(x)
            else:
                x = nn.tanh(x)
        return nn.Dense(self.features[-1], kernel_init=orthogonal(0.01), bias_init=constant(0.0))(x)


class ActorCritic(nn.Module):
    action_dim: int
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> tuple[distrax.Distribution, jnp.ndarray]:
        actor_mean = MLP([256, 128, self.action_dim], activation=self.activation)(x)
        actor_logtstd = self.param("log_std", nn.initializers.zeros, (self.action_dim,))
        pi = distrax.MultivariateNormalDiag(actor_mean, jnp.exp(actor_logtstd))

        critic = MLP([256, 256, 1], activation=self.activation)(x)

        return pi, jnp.squeeze(critic, axis=-1)


class Memory(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: dict[str, Any] | FrozenDict[str, Any] | Any


def save_model(params: FrozenDict, filename: str) -> None:
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "wb") as f:
        pickle.dump(params, f)


def make_train(config: Config) -> Callable[[jnp.ndarray], dict[str, Any]]:
    config.num_updates = config.total_timesteps // config.num_steps // config.num_envs
    config.minibatch_size = config.num_envs * config.num_steps // config.num_minibatches

    # Instantiates the environment.
    env = HumanoidEnv(
        n_frames=config.physics_n_frames,
        kscale_id=config.kscale_id,
    )

    def linear_schedule(count: int) -> float:
        frac = 1.0 - (count // (config.num_minibatches * config.update_epochs)) / config.num_updates
        return config.lr * frac

    def train(rng: jnp.ndarray) -> dict[str, Any]:
        # INIT NETWORK
        network = ActorCritic(env.action_size, activation=config.activation)
        rng, _rng = jax.random.split(rng)
        init_x = jnp.zeros(env.observation_size)
        network_params = network.init(_rng, init_x)

        if config.anneal_lr:
            tx = optax.chain(
                optax.clip_by_global_norm(config.max_grad_norm),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(config.max_grad_norm),
                optax.adam(config.lr, eps=1e-5),
            )

        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )

        # jit-ifying and vmap-ing functions
        @jax.jit
        def reset_fn(rng: jnp.ndarray) -> State:
            rngs = jax.random.split(rng, config.num_envs)
            return jax.vmap(env.reset)(rngs)

        @jax.jit
        def step_fn(states: State, actions: jnp.ndarray, rng: jnp.ndarray) -> State:
            return jax.vmap(env.step)(states, actions, rng)

        # INIT ENV
        rng, reset_rng = jax.random.split(rng)
        env_state = reset_fn(jnp.array(reset_rng))

        obs = env_state.obs

        def _update_step(
            runner_state: tuple[TrainState, State, jnp.ndarray, jnp.ndarray],
            unused: Memory,
        ) -> tuple[tuple[TrainState, State, jnp.ndarray, jnp.ndarray], Any]:
            """Update steps of the model --- environment memory colelction then network update."""

            def _env_step(
                runner_state: tuple[TrainState, State, jnp.ndarray, jnp.ndarray],
                unused: Memory,
            ) -> tuple[tuple[TrainState, State, jnp.ndarray, jnp.ndarray], Memory]:
                """Runs num_steps across all environments and collects memory."""
                train_state, env_state, last_obs, rng = runner_state

                # Runs the network to get the action distribution and value function.
                pi, value = network.apply(train_state.params, last_obs)
                rng, action_rng = jax.random.split(rng)
                action = pi.sample(seed=action_rng)
                log_prob = pi.log_prob(action)

                # Updates the environment state using the simulation.
                rng, *step_rngs = jax.random.split(rng, config.num_envs + 1)
                env_state = step_fn(env_state, action, step_rngs)

                # Normalizing observations improves training
                obs = env_state.obs

                reward = env_state.reward
                done = env_state.done
                info = env_state.metrics

                # Stores the memory for the current step.
                memory = Memory(done, action, value, reward, log_prob, last_obs, info)  # type: ignore[arg-type]
                runner_state = (train_state, env_state, obs, rng)

                return runner_state, memory

            runner_state, mem_batch = jax.lax.scan(_env_step, runner_state, None, config.num_steps)

            # CALCULATE ADVANTAGE
            train_state, env_state, last_obs, rng = runner_state
            _, last_val = network.apply(train_state.params, last_obs)
            last_val = jnp.array(last_val)

            def _calculate_gae(mem_batch: Memory, last_val: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
                def _get_advantages(
                    gae_and_next_value: tuple[jnp.ndarray, jnp.ndarray], memory: Memory
                ) -> tuple[tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        memory.done,
                        memory.value,
                        memory.reward,
                    )
                    delta = reward + config.gamma * next_value * (1 - done) - value
                    gae = delta + config.gamma * config.gae_lambda * (1 - done) * gae
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

            def _update_epoch(
                update_state: tuple[TrainState, Memory, jnp.ndarray, jnp.ndarray, jnp.ndarray],
                unused: tuple[jnp.ndarray, jnp.ndarray],
            ) -> tuple[tuple[TrainState, Memory, jnp.ndarray, jnp.ndarray, jnp.ndarray], Any]:
                """Scanned function for updating networkfrom all state frames collected above."""

                def _update_minibatch(
                    train_state: TrainState, batch_info: tuple[Memory, jnp.ndarray, jnp.ndarray]
                ) -> tuple[TrainState, Any]:
                    """Scanned function for updating from a single minibatch (single network update)."""
                    mem_batch, advantages, targets = batch_info

                    def _loss_fn(
                        params: FrozenDict, mem_batch: Memory, gae: jnp.ndarray, targets: jnp.ndarray
                    ) -> tuple[jnp.ndarray, tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
                        # Rerun network
                        pi, value = network.apply(params, mem_batch.obs)
                        log_prob = pi.log_prob(mem_batch.action)

                        # CALCULATE VALUE LOSS
                        # want to critic model's ability to predict value
                        # clip prevents from too drastic changes
                        value_pred_clipped = mem_batch.value + (value - mem_batch.value).clip(
                            -config.clip_eps, config.clip_eps
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
                                1.0 - config.clip_eps,
                                1.0 + config.clip_eps,
                            )
                            * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean()
                        entropy = pi.entropy().mean()

                        total_loss = loss_actor + config.vf_coef * value_loss - config.ent_coef * entropy
                        return total_loss, (value_loss, loss_actor, entropy)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(train_state.params, mem_batch, advantages, targets)
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                train_state, mem_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)
                batch_size = config.minibatch_size * config.num_minibatches
                assert (
                    batch_size == config.num_steps * config.num_envs
                ), "batch size must be equal to number of steps * number of envs"
                permutation = jax.random.permutation(_rng, batch_size)
                batch = (mem_batch, advantages, targets)

                batch = jax.tree_util.tree_map(lambda x: x.reshape((batch_size,) + x.shape[2:]), batch)
                shuffled_batch = jax.tree_util.tree_map(lambda x: jnp.take(x, permutation, axis=0), batch)

                # organize into minibatches
                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(x, [config.num_minibatches, -1] + list(x.shape[1:])),
                    shuffled_batch,
                )

                train_state, total_loss = jax.lax.scan(_update_minibatch, train_state, minibatches)
                update_state = (train_state, mem_batch, advantages, targets, rng)
                return update_state, total_loss

            update_state = (train_state, mem_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(_update_epoch, update_state, None, config.update_epochs)
            train_state = update_state[0]
            metric = mem_batch.info
            rng = update_state[-1]

            # jax.debug.breakpoint()
            if config.debug:

                def callback(info: dict[str, Any]) -> None:
                    return_values = info["returned_episode_returns"][info["returned_episode"]]
                    timesteps = info["timestep"][info["returned_episode"]] * config.num_envs
                    for t in range(len(timesteps)):
                        print("global step=%d, episodic return=%f" % (timesteps[t], return_values[t]))

                jax.debug.callback(callback, metric)

            runner_state = (train_state, env_state, last_obs, rng)
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, env_state, obs, _rng)
        runner_state, metric = jax.lax.scan(_update_step, runner_state, None, config.num_updates)

        return {"runner_state": runner_state, "metrics": metric}

    return train


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a model with specified environment name.")
    parser.add_argument("config", type=str, help="Path to the config file")
    args = parser.parse_args()
    config = load_config(args.config)

    # Trains the model.
    rng = jax.random.PRNGKey(30)
    train_jit = jax.jit(make_train(config))
    out = train_jit(rng)

    model_save_path = f"models/{config.env_name}_model.pkl"
    logger.info("Finished training. Saving model to %s...", model_save_path)
    save_model(out["runner_state"][0].params, model_save_path)


if __name__ == "__main__":
    # python train.py
    main()
