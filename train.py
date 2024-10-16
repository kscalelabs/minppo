"""Train a model with a specified environment module."""

import logging
import os
import pickle
from dataclasses import dataclass
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

from config import Config, load_config_from_cli
from environment import HumanoidEnv

logger = logging.getLogger(__name__)


@dataclass
class RunnerState:
    train_state: TrainState
    env_state: State
    last_obs: jnp.ndarray
    rng: jnp.ndarray


@dataclass
class UpdateState:
    train_state: TrainState
    mem_batch: "Memory"
    advantages: jnp.ndarray
    targets: jnp.ndarray
    rng: jnp.ndarray


@dataclass
class TrainOutput:
    runner_state: RunnerState
    metrics: Any


class MLP(nn.Module):
    features: Sequence[int]
    use_tanh: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for feat in self.features[:-1]:
            x = nn.Dense(feat, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
            if self.use_tanh:
                x = nn.tanh(x)
            else:
                x = nn.relu(x)
        return nn.Dense(self.features[-1], kernel_init=orthogonal(0.01), bias_init=constant(0.0))(x)


class ActorCritic(nn.Module):
    num_layers: int
    hidden_size: int
    action_dim: int
    use_tanh: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> tuple[distrax.Distribution, jnp.ndarray]:
        actor_mean = MLP([self.hidden_size] * self.num_layers + [self.action_dim], use_tanh=self.use_tanh)(x)
        actor_logtstd = self.param("log_std", nn.initializers.zeros, (self.action_dim,))
        pi = distrax.MultivariateNormalDiag(actor_mean, jnp.exp(actor_logtstd))
        critic = MLP([self.hidden_size] * self.num_layers + [1], use_tanh=False)(x)
        return pi, jnp.squeeze(critic, axis=-1)


class Memory(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: Any


def save_model(params: FrozenDict, filename: str) -> None:
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "wb") as f:
        pickle.dump(params, f)


def make_train(config: Config) -> Callable[[jnp.ndarray], TrainOutput]:
    # Computes the number of updates to run and the minibatch size from
    # parameters that are easier to reason about.
    num_updates = config.training.total_timesteps // config.training.num_steps // config.training.num_envs
    minibatch_size = config.training.num_envs * config.training.num_steps // config.training.num_minibatches

    env = HumanoidEnv(
        n_frames=config.environment.n_frames,
        kscale_id=config.kscale_id,
    )

    def linear_schedule(count: int) -> float:
        frac = 1.0 - (count // (minibatch_size * config.training.update_epochs)) / num_updates
        return config.training.lr * frac

    def train(rng: jnp.ndarray) -> TrainOutput:
        network = ActorCritic(
            num_layers=config.model.num_layers,
            hidden_size=config.model.hidden_size,
            action_dim=env.action_size,
            use_tanh=config.model.use_tanh,
        )
        rng, _rng = jax.random.split(rng)
        init_x = jnp.zeros(env.observation_size)
        network_params = network.init(_rng, init_x)

        if config.training.anneal_lr:
            tx = optax.chain(
                optax.clip_by_global_norm(config.opt.max_grad_norm),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )

        else:
            tx = optax.chain(
                optax.clip_by_global_norm(config.opt.max_grad_norm),
                optax.adam(config.opt.lr, eps=1e-5),
            )

        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )

        # jit-ifying and vmap-ing functions
        @jax.jit
        def reset_fn(rng: jnp.ndarray) -> State:
            rngs = jax.random.split(rng, config.training.num_envs)
            return jax.vmap(env.reset)(rngs)

        @jax.jit
        def step_fn(states: State, actions: jnp.ndarray, rng: jnp.ndarray) -> State:
            return jax.vmap(env.step)(states, actions, rng)

        # INIT ENV
        rng, reset_rng = jax.random.split(rng)
        env_state = reset_fn(jnp.array(reset_rng))

        obs = env_state.obs

        def _update_step(
            runner_state: RunnerState,
            unused: Memory,
        ) -> tuple[RunnerState, Any]:
            """Update steps of the model --- environment memory collection then network update."""

            def _env_step(
                runner_state: RunnerState,
                unused: Memory,
            ) -> tuple[RunnerState, Memory]:
                """Runs num_steps across all environments and collects memory."""
                train_state, env_state, last_obs, rng = runner_state

                # Runs the network to get the action distribution and value function.
                pi, value = network.apply(train_state.params, last_obs)
                rng, action_rng = jax.random.split(rng)
                action = pi.sample(seed=action_rng)
                log_prob = pi.log_prob(action)

                # Updates the environment state using the simulation.
                rng, *step_rngs = jax.random.split(rng, config.training.num_envs + 1)
                env_state: State = step_fn(env_state, action, step_rngs)

                # Normalizing observations improves training
                obs = env_state.obs

                reward = env_state.reward
                done = env_state.done
                info = env_state.metrics

                # Stores the memory for the current step.
                memory = Memory(done, action, value, reward, log_prob, last_obs, info)
                runner_state = RunnerState(train_state, env_state, obs, rng)

                return runner_state, memory

            # Runs the environment for some steps to collect memory.
            runner_state, mem_batch = jax.lax.scan(_env_step, runner_state, None, config.rl.num_env_steps)

            # CALCULATE ADVANTAGE
            _, last_val = network.apply(runner_state.train_state.params, runner_state.last_obs)
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
                    delta = reward + config.rl.gamma * next_value * (1 - done) - value
                    gae = delta + config.rl.gamma * config.rl.gae_lambda * (1 - done) * gae
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
                update_state: UpdateState,
                unused: tuple[jnp.ndarray, jnp.ndarray],
            ) -> tuple[UpdateState, Any]:

                def _update_minibatch(
                    train_state: TrainState, batch_info: tuple[Memory, jnp.ndarray, jnp.ndarray]
                ) -> tuple[TrainState, Any]:
                    mem_batch, advantages, targets = batch_info

                    def _loss_fn(
                        params: FrozenDict, mem_batch: Memory, gae: jnp.ndarray, targets: jnp.ndarray
                    ) -> tuple[jnp.ndarray, tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
                        # Rerun network
                        pi, value = network.apply(params, mem_batch.obs)
                        log_prob = pi.log_prob(mem_batch.action)

                        # Calculates the value loss - we want to minimize the
                        # difference between the predicted value and the
                        # target value, clipped to prevent too drastic changes.
                        value_pred_clipped = mem_batch.value + (value - mem_batch.value).clip(
                            -config.rl.clip_eps, config.rl.clip_eps
                        )
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()

                        # Calculates the actor loss - we want to maximize the
                        # improvement (log prob diff) while preventing too
                        # drastic changes.
                        ratio = jnp.exp(log_prob - mem_batch.log_prob)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                            jnp.clip(
                                ratio,
                                1.0 - config.rl.clip_eps,
                                1.0 + config.rl.clip_eps,
                            )
                            * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean()
                        entropy = pi.entropy().mean()

                        total_loss = loss_actor + config.rl.vf_coef * value_loss - config.rl.ent_coef * entropy
                        return total_loss, (value_loss, loss_actor, entropy)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(train_state.params, mem_batch, advantages, targets)
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                train_state, mem_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)
                batch_size = minibatch_size * config.training.num_minibatches
                if batch_size != config.training.num_steps * config.training.num_envs:
                    raise ValueError("`batch_size` must be equal to `num_steps * num_envs`")

                permutation = jax.random.permutation(_rng, batch_size)
                batch = (mem_batch, advantages, targets)

                batch = jax.tree_util.tree_map(lambda x: x.reshape((batch_size,) + x.shape[2:]), batch)
                shuffled_batch = jax.tree_util.tree_map(lambda x: jnp.take(x, permutation, axis=0), batch)

                # organize into minibatches
                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(x, [config.training.num_minibatches, -1] + list(x.shape[1:])),
                    shuffled_batch,
                )

                train_state, total_loss = jax.lax.scan(_update_minibatch, train_state, minibatches)
                update_state = UpdateState(train_state, mem_batch, advantages, targets, rng)
                return update_state, total_loss

            update_state = UpdateState(runner_state.train_state, mem_batch, advantages, targets, runner_state.rng)
            update_state, loss_info = jax.lax.scan(_update_epoch, update_state, None, config.training.update_epochs)

            runner_state = RunnerState(
                update_state.train_state, runner_state.env_state, runner_state.last_obs, update_state.rng
            )
            return runner_state, mem_batch.info

        rng, _rng = jax.random.split(rng)
        runner_state = RunnerState(train_state, env_state, obs, _rng)
        runner_state, metric = jax.lax.scan(_update_step, runner_state, None, num_updates)

        return TrainOutput(runner_state=runner_state, metrics=metric)

    return train


def main() -> None:
    config = load_config_from_cli()

    # Trains the model.
    rng = jax.random.PRNGKey(config.training.seed)
    train_jit = jax.jit(make_train(config))
    out = train_jit(rng)

    # Saves the trained model.
    logger.info("Finished training. Saving model to %s...", config.training.model_save_path)
    save_model(out.runner_state.train_state.params, config.training.model_save_path)


if __name__ == "__main__":
    # python train.py
    main()
