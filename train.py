"""Trains a policy network to get a humanoid to stand up."""

import argparse
import atexit
import contextlib
import datetime
import os
import pickle
from dataclasses import dataclass, field
from functools import partial
import shutil
import signal
import sys
from typing import Any, Dict, List, Mapping, Tuple

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

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"


################ CONFIG ################


# brax setup, with lr adjusted
@dataclass
class Config:
    lr_actor: float = field(default=3e-4)  # NOTE: to be experimented with
    lr_critic: float = field(default=3e-4)
    truncate_after: int = field(default=128)
    num_iterations: int = field(default=20)
    num_envs: int = field(default=1024)  # if too high, can just end up not finishing episodes (especially in rollout)
    max_steps: int = field(default=1600 * 1024)  # 400 steps each environment
    max_steps_per_iteration: int = field(
        default=15000000
        // 15
        # default=150000000 // 15
    )  # increase to get more rollout + training for ppo each iteration
    gamma: float = field(default=0.98)
    lambd: float = field(default=0.98)
    minibatch_size: int = field(default=32)
    batch_size: int = field(default=512)
    epsilon: float = field(default=0.2)
    l2_rate: float = field(default=0.001)
    beta: int = field(default=3)


# @dataclass
# class Config:
#     lr_actor: float = field(default=0.00003)
#     lr_critic: float = field(default=0.00003)
#     num_iterations: int = field(default=20)
#     num_envs: int = field(default=4096)  # if too high, can just end up not finishing episodes (especially in rollout)
#     max_steps: int = field(default=1000 * 4096)
#     max_steps_per_iteration: int = field(
#         default=150000000 // 15
#     )  # increase to get more rollout + training for ppo each iteration
#     gamma: float = field(default=0.98)
#     lambd: float = field(default=0.98)
#     minibatch_size: int = field(default=32)
#     batch_size: int = field(default=512)
#     epsilon: float = field(default=0.2)
#     l2_rate: float = field(default=0.001)
#     beta: int = field(default=3)

################ PPO MODELS ################


class Ppo:
    def __init__(self, observation_size: int, action_size: int, config: Config, key: Array) -> None:
        self.actor = Actor(action_size)  # take in minibatches of states
        self.critic = Critic()
        self.actor_params = self.actor.init(key, jnp.zeros((1, observation_size)))
        self.critic_params = self.critic.init(key, jnp.zeros((1, observation_size)))
        self.config = config

        # vectorized functions for applying actor and critic, (None, 0) for the non-batched dimensions
        self.actor_apply = jax.jit(jax.vmap(lambda params, x: self.actor.apply(params, x), in_axes=(None, 0)))
        self.critic_apply = jax.jit(jax.vmap(lambda params, x: self.critic.apply(params, x), in_axes=(None, 0)))

        self.actor_optim = optax.adam(learning_rate=config.lr_actor)
        self.critic_optim = optax.adamw(learning_rate=config.lr_critic, weight_decay=config.l2_rate)

        self.actor_opt_state = self.actor_optim.init(self.actor_params)
        self.critic_opt_state = self.critic_optim.init(self.critic_params)

    def get_params(self) -> Tuple[Array, Array, optax.OptState, optax.OptState]:
        return (
            self.actor_params,
            self.critic_params,
            self.actor_opt_state,
            self.critic_opt_state,
        )

    def update_params(self, new_params: Tuple[Array, Array, optax.OptState, optax.OptState]) -> None:
        self.actor_params, self.critic_params, self.actor_opt_state, self.critic_opt_state = new_params

    # puts train steps on different gpus
    # self is a start broadcasted type, but params is unhashable so can't be broadcastsed
    @partial(jax.pmap, axis_name="devices", static_broadcasted_argnums=(0,))
    def pmap_train_step(
        self,
        states: Array,
        actions: Array,
        rewards: Array,
        masks: Array,
        old_log_prob: Array,
    ) -> Tuple[Dict[str, Any], Array, Array]:

        # split across minibatches
        @partial(jax.vmap)
        def train_step(
            states: Array, actions: Array, rewards: Array, masks: Array, old_log_prob: Array
        ) -> Tuple[Dict[str, Any], Array, Array]:
            # vectorizations are using different parameters in vmap
            actor_params, critic_params, actor_opt_state, critic_opt_state = self.get_params()

            values = self.critic_apply(critic_params, states).squeeze()
            returns, advants = self.get_gae(rewards, masks, values)

            # NOTE: add entropy loss? https://github.com/google/brax/blob/c89a3ad7900648613ead04dedacc91e923e10a68/brax/training/agents/ppo/losses.py#L173C1-L174C1

            # maximizing advantage while minimizing change
            def actor_loss_fn(params: Array) -> Array:
                mu, std = self.actor_apply(params, states)
                new_log_prob = actor_log_prob(mu, std, actions)

                # generally, we want our model to explore --> need variance
                ratio = jnp.exp(new_log_prob - old_log_prob)
                surrogate_loss = ratio * advants

                clipped_loss = jnp.clip(ratio, 1.0 - self.config.epsilon, 1.0 + self.config.epsilon) * advants
                actor_loss = -jnp.mean(jnp.minimum(surrogate_loss, clipped_loss))
                return actor_loss

            # want the critic to best approximate rewards/returns
            def critic_loss_fn(params: Array) -> Array:
                critic_returns = self.critic_apply(params, states).squeeze()
                critic_loss = jnp.mean((critic_returns - returns) ** 2)
                return critic_loss

            # graidents + backpropogation
            actor_grad_fn = jax.value_and_grad(actor_loss_fn)

            actor_loss, actor_grads = actor_grad_fn(actor_params)
            # jax.debug.print("actor_loss, {}, actor_grads, {}", actor_loss, actor_grads)
            actor_updates, new_actor_opt_state = self.actor_optim.update(actor_grads, actor_opt_state, actor_params)
            new_actor_params = optax.apply_updates(actor_params, actor_updates)

            critic_grad_fn = jax.value_and_grad(critic_loss_fn)
            critic_loss, critic_grads = critic_grad_fn(critic_params)
            critic_updates, new_critic_opt_state = self.critic_optim.update(
                critic_grads, critic_opt_state, critic_params
            )
            new_critic_params = optax.apply_updates(critic_params, critic_updates)

            new_params = (new_actor_params, new_critic_params, new_actor_opt_state, new_critic_opt_state)

            return new_params, actor_loss, critic_loss

        return train_step(states, actions, rewards, masks, old_log_prob)

    # def get_gae(self, rewards: Array, masks: Array, values: Array) -> Tuple[Array, Array]:

    #     def gae_step(carry: Tuple[Array, Array], inp: Tuple[Array, Array, Array]) -> Tuple[Tuple[Array, Array], Array]:
    #         gae, next_value = carry
    #         reward, mask, value = inp
    #         delta = reward + self.config.gamma * next_value * mask - value
    #         gae = delta + self.config.gamma * self.config.lambd * mask * gae  # mask for not counting reward when done
    #         return (gae, value), gae

    #     # calculating advantage = combination of immediate reward and diminishing value of future rewards
    #     _, advantages = jax.lax.scan(
    #         f=gae_step,
    #         init=(jnp.zeros_like(rewards[-1]), values[-1]),
    #         xs=(rewards[::-1], masks[::-1], values[::-1]),
    #         reverse=True,
    #     )

    #     # return = weighting between critic valuation + advantage
    #     returns = advantages + values
    #     advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    #     return returns, advantages

    # Truncation: https://arxiv.org/pdf/2301.10920
    def get_gae(self, rewards: Array, masks: Array, values: Array) -> Tuple[Array, Array]:

        def gae_step(
            carry: Tuple[Array, Array, Array], inp: Tuple[Array, Array, Array]
        ) -> Tuple[Tuple[Array, Array, Array], Array]:
            gae, next_value, step_count = carry
            reward, mask, value = inp

            delta = reward + self.config.gamma * next_value * mask - value
            new_gae = jnp.where(
                self.config.truncate_after is None or step_count < self.config.truncate_after,
                delta + self.config.gamma * self.config.lambd * mask * gae,
                gae,
            )

            return (new_gae, value, step_count + 1), new_gae

        # calculating advantage with truncation. NOTE: correct direction?
        _, advantages = jax.lax.scan(
            f=gae_step,
            init=(jnp.zeros_like(rewards[0]), values[0], jnp.array(0)),
            xs=(rewards, masks, values),
            reverse=True,
        )

        # return = weighting between critic valuation + advantage
        returns = advantages + values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)  # normalize advantages
        return returns, advantages

    def train(self, states, actions, rewards, masks, config: Config) -> None:

        n = len(states)
        num_devices = jax.local_device_count()
        batch_size = config.batch_size
        minibatch_size = config.minibatch_size

        actor_params, _, _, _ = self.get_params()
        old_mu, old_std = self.actor_apply(actor_params, states)

        # interesting -- beginning indices always high. more exploration for first moves
        old_log_prob = actor_log_prob(old_mu, old_std, actions)

        assert (
            batch_size % (num_devices * minibatch_size) == 0
        ), "Batch size must be divisible by num_devices * minibatch_size"

        for epoch in range(1):

            perm = jax.random.permutation(jax.random.PRNGKey(epoch), n)
            states = states[perm]
            actions = actions[perm]
            rewards = rewards[perm]
            masks = masks[perm]
            actor_loss = 0
            critic_loss = 0

            for i in range(n // batch_size):
                batch_start = i * batch_size
                batch_end = (i + 1) * batch_size

                # shaping variables to be split across GPUs, then vectorized
                # according to remaining diensions (minibatch left to PPO policy)
                b_states = states[batch_start:batch_end].reshape(num_devices, -1, minibatch_size, states.shape[-1])
                b_actions = actions[batch_start:batch_end].reshape(num_devices, -1, minibatch_size, actions.shape[-1])
                b_rewards = rewards[batch_start:batch_end].reshape(num_devices, -1, minibatch_size)
                b_masks = masks[batch_start:batch_end].reshape(num_devices, -1, minibatch_size)

                b_old_log_prob = old_log_prob[batch_start:batch_end].reshape(num_devices, -1, minibatch_size)

                # outputs with (num_devices, vectors_per_device, ...)
                new_params, actor_loss, critic_loss = self.pmap_train_step(
                    b_states,
                    b_actions,
                    b_rewards,
                    b_masks,
                    b_old_log_prob,
                )

                # print("actor loss {}, critic loss {}", actor_loss, critic_loss)

                # averaging across all devices and vectorizations
                new_params = jax.tree.map(lambda x: jnp.mean(x, axis=(0, 1)), new_params)
                actor_loss = jnp.mean(actor_loss)
                critic_loss = jnp.mean(critic_loss)

                self.update_params(new_params)

            print(f"Epoch {epoch+1}, Actor Loss: {actor_loss:.4f}, Critic Loss: {critic_loss:.4f}")


# log probability of actions given a normal distribution
def actor_log_prob(mu: Array, sigma: Array, actions: Array) -> Array:
    return jax.scipy.stats.norm.logpdf(actions, mu, sigma).sum(axis=-1)


# Optimizer vs. necessary?
# class TanhBijector:
#   """Tanh Bijector."""

#   def forward(self, x):
#     return jnp.tanh(x)

#   def inverse(self, y):
#     return jnp.arctanh(y)

#   def forward_log_det_jacobian(self, x):
#     return 2. * (jnp.log(2.) - x - jax.nn.softplus(-2. * x))

# class NormalTanhDistribution(ParametricDistribution):
#   """Normal distribution followed by tanh."""

#   def __init__(self, event_size, min_std=0.001, var_scale=1):
#     super().__init__(
#         param_size=2 * event_size,
#         postprocessor=TanhBijector(),
#         event_ndims=1,
#         reparametrizable=True)
#     self._min_std = min_std
#     self._var_scale = var_scale

#   def create_dist(self, parameters):
#     loc, scale = jnp.split(parameters, 2, axis=-1)
#     scale = (jax.nn.softplus(scale) + self._min_std) * self._var_scale
#     return NormalDistribution(loc=loc, scale=scale)


class Actor(nn.Module):
    action_size: int

    @nn.compact
    def __call__(self, x: Array) -> Tuple[Array, Array]:

        x = nn.Dense(64, dtype=jnp.float32)(x)
        x = nn.swish(x)
        x = nn.Dense(64, dtype=jnp.float32)(x)
        x = nn.swish(x)
        x = nn.Dense(64, dtype=jnp.float32)(x)
        x = nn.swish(x)
        x = nn.Dense(64, dtype=jnp.float32)(x)
        x = nn.swish(x)
        x = nn.Dense(64, dtype=jnp.float32)(x)
        x = nn.swish(x)
        mu = nn.Dense(self.action_size, kernel_init=nn.initializers.constant(0.1), dtype=jnp.float32)(x)
        log_sigma = nn.Dense(self.action_size, dtype=jnp.float32)(x)

        return mu, jnp.exp(log_sigma)


class Critic(nn.Module):
    @nn.compact
    def __call__(self, x: Array) -> Array:

        x = nn.Dense(256, dtype=jnp.float32)(x)
        x = nn.swish(x)
        x = nn.Dense(256, dtype=jnp.float32)(x)
        x = nn.swish(x)
        x = nn.Dense(256, dtype=jnp.float32)(x)
        x = nn.swish(x)
        x = nn.Dense(256, dtype=jnp.float32)(x)
        x = nn.swish(x)
        x = nn.Dense(256, dtype=jnp.float32)(x)
        x = nn.swish(x)

        return nn.Dense(1, kernel_init=nn.initializers.constant(0.1), dtype=jnp.float32)(x)


# for normalizaing observation states
class Normalize:
    def __init__(self, observation_size: int) -> None:
        self.mean: Array = jnp.zeros((observation_size,))
        self.std: Array = jnp.zeros((observation_size,))
        self.stdd: Array = jnp.zeros((observation_size,))
        self.n: int = 0

    def __call__(self, x: Array) -> Array:
        x = jnp.asarray(x)
        self.n += 1
        if self.n == 1:
            self.mean = x
        else:
            old_mean = self.mean.copy()
            self.mean = old_mean + (x - old_mean) / self.n
            self.stdd = self.stdd + (x - old_mean) * (x - self.mean)
        if self.n > 1:
            self.std = jnp.sqrt(self.stdd / (self.n - 1))
        else:
            self.std = self.mean

        x = x - self.mean
        x = x / (self.std + 1e-8)
        x = jnp.clip(x, -5, +5)
        return x


################ UTILS ################


# show "starting image" of xml for testing
def screenshot(
    env: HumanoidEnv,
    rng: Array,
    width: int = 640,
    height: int = 480,
    filename: str = "screenshot.png",
) -> None:
    state = env.reset(rng)
    image_array = env.render(state.pipeline_state, camera="side", width=width, height=height)
    image_array = jnp.array(image_array).astype("uint8")
    os.makedirs("screenshots", exist_ok=True)
    media.write_image(os.path.join("screenshots", filename), image_array)

    print(f"Screenshot saved as {filename}")


def unwrap_state_vectorization_parallelization(state: State, envs_to_sample: int, num_devices: int) -> State:
    unwrapped_rollout = []
    # Get all attributes of the state
    attributes = dir(state)

    # NOTE: can change ordering of this to save runtime if want to save more vectorized states.
    # saves from only first vectorized state
    for env in range(envs_to_sample):
        i, j = divmod(env, num_devices)
        # Create a new state with the first element of each attribute
        new_state = {}
        for attr in attributes:
            # Skip special methods and attributes
            if not attr.startswith("_") and not callable(getattr(state, attr)):
                value = getattr(state, attr)
                try:
                    new_state[attr] = value[j][i]
                except Exception:
                    new_state[attr] = value
        unwrapped_rollout.append((type(state)(**new_state), i))

    return unwrapped_rollout


def print_config_and_args(config, args):
    print("Configuration:")
    for key, value in vars(config).items():
        print(f"  {key}: {value}")
    print("\nArguments:")
    for key, value in vars(args).items():
        print(f"  {key}: {value}")
    print("\n")


# for logging stdout for debugging/config/args iteration
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

    def write(self, text):
        self.captured_output.append(text)
        sys.__stdout__.write(text)

    def flush(self):
        sys.__stdout__.flush()

    def save_output(self):
        output_file_path = os.path.join(self.dir, f"{self.timestamp}.out")
        print(output_file_path)
        with open(output_file_path, "w") as f:
            f.writelines(self.captured_output)


def setup_interrupt_handling(stdout_capture):
    def signal_handler(signum, frame):
        print("\nInterrupt received. Saving output and exiting...")
        stdout_capture.save_output()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    atexit.register(stdout_capture.save_output)


################ MAIN ################


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--env_name", type=str, default="base", help="name of environmnet to put into logs")
    parser.add_argument("-ps", "--parallelize_sim", action="store_true", help="parallelize simulations across GPUs")
    parser.add_argument("-d", "--debug", action="store_true", help="whether to log outputs")
    parser.add_argument("-v", "--save_video", action="store_true", help="whether to save a video")
    parser.add_argument("-r", "--render", action="store_true", help="render the environment")
    parser.add_argument("-x", "--width", type=int, default=640, help="width of the video frame")
    parser.add_argument("-y", "--height", type=int, default=480, help="height of the video frame")
    parser.add_argument("-re", "--render_every", type=int, default=2, help="render the environment every N steps")
    parser.add_argument("-vl", "--video_length", type=int, default=5, help="maxmimum length of video in seconds")
    parser.add_argument("--envs_to_sample", type=int, default=4, help="number of environments to sample for video")
    parser.add_argument("--save_video_every", type=int, default=1, help="save video every N iterations")
    args = parser.parse_args()

    config = Config()
    args.envs_to_sample = min(args.envs_to_sample, config.num_envs)
    log_dir = f"logs/{args.env_name}/"

    # Setup stdout capture and current python file saving
    if args.debug:
        stdout_capture = StdoutCapture(log_dir)
        setup_interrupt_handling(stdout_capture)
        stdout_context = contextlib.redirect_stdout(stdout_capture)
    else:
        # If not using StdoutCapture, use a dummy context manager that does nothing
        @contextlib.contextmanager
        def dummy_context_manager():
            yield

        stdout_context = dummy_context_manager()

    # Redirect stdout to our capture object
    with stdout_context:

        print_config_and_args(config, args)

        env = HumanoidEnv()
        observation_size = env.observation_size
        action_size = env.action_size
        print("action_size", action_size)
        print("observation_size", observation_size)

        rng = jax.random.PRNGKey(0)

        ppo = Ppo(observation_size, action_size, config, rng)
        normalize = Normalize(observation_size)
        episodes: int = 0

        # num_devices = 1
        num_devices = jax.device_count()
        envs_per_device = config.num_envs // num_devices

        print("devices", num_devices)
        print("vectorization", config.batch_size // (num_devices * config.minibatch_size))
        print("minibatches", config.minibatch_size)
        print("batches", config.batch_size)

        assert config.num_envs % num_devices == 0, "Number of environments must be divisible by number of devices"

        @jax.jit
        def update_memory(memory, new_data):
            def concat_and_reshape(x, y):
                concat = jnp.concatenate([x, y])
                return concat

            return jax.tree.map(concat_and_reshape, memory, new_data)

        @partial(jax.pmap, axis_name="devices")
        def pmap_reset_fn(rng):
            rngs = jax.random.split(rng, envs_per_device)  # each device different random number

            return jax.vmap(env.reset)(rngs)

        @partial(jax.pmap, axis_name="devices")
        def pmap_step_fn(states, actions):
            return jax.vmap(env.step)(states, actions)

        @partial(jax.pmap, axis_name="devices")
        def pmap_normalize(obs):
            return jax.vmap(normalize)(obs)

        for i in range(1, config.num_iterations + 1):

            batch_states = []
            batch_actions = []
            batch_rewards = []
            batch_masks = []

            scores = []
            steps = 0
            rollout = []

            # must be in loop in order to get updated params
            # @partial(jax.jit, static_argnums=(1,))
            def choose_action(obs: Array, rng) -> Array:
                # given a state, we do our forward pass and then sample from to maintain "random actions"
                mu, sigma = ppo.actor.apply(ppo.actor_params, obs)
                return jax.random.normal(rng, shape=mu.shape) * sigma + mu

            @partial(jax.pmap, axis_name="devices")
            def pmap_choose_action(obs, rng):
                # want specific rng keys for each device
                # rng = jax.random.fold_in(rng, jax.lax.axis_index("devices"))
                rngs = jax.random.split(rng, envs_per_device)
                return jax.vmap(choose_action)(obs, rngs)

            pbar = tqdm(total=config.max_steps_per_iteration, desc=f"Steps for iteration {i}")

            while steps < config.max_steps_per_iteration:
                episodes += config.num_envs

                rng, *subrngs = jax.random.split(rng, num_devices + 1)  # each device/vector different random number
                subrngs = jnp.array(subrngs)
                states = pmap_reset_fn(subrngs)
                obs = states.obs
                # obs = pmap_normalize(states.obs)
                score = jnp.zeros((num_devices, envs_per_device))

                for _ in range(config.max_steps // config.num_envs):
                    # each device/vector different random number
                    rng, *action_rngs = jax.random.split(rng, num_devices + 1)
                    action_rngs = jnp.array(action_rngs)
                    actions = pmap_choose_action(obs, action_rngs)
                    # print("actions", actions)

                    states = pmap_step_fn(states, actions)

                    # these are (num_devices, envs_per_device, ...)
                    next_obs, rewards, dones = states.obs, states.reward, states.done
                    masks = (1 - dones).astype(jnp.float32)

                    # NOTE: need to arrange states to be adjacent
                    batch_states.append(obs.reshape(num_devices * envs_per_device, -1))
                    batch_actions.append(actions.reshape(num_devices * envs_per_device, -1))
                    batch_rewards.append(rewards.reshape(num_devices * envs_per_device))
                    batch_masks.append(masks.reshape(num_devices * envs_per_device))

                    # memory = update_memory(memory, new_data)

                    score += rewards
                    obs = next_obs
                    steps += config.num_envs
                    pbar.update(config.num_envs)

                    if (
                        args.save_video
                        and i % args.save_video_every == 0
                        and len(rollout) < args.video_length * int(1 / env.dt)
                    ):
                        # need to unwrap states to render
                        unwrapped_states = unwrap_state_vectorization_parallelization(
                            jax.device_get(states.pipeline_state), args.envs_to_sample, num_devices
                        )
                        rollout.extend(unwrapped_states)

                    if jnp.all(dones):
                        break

                # Process batch when it reaches a certain size
                if len(batch_states) * num_devices * envs_per_device >= config.batch_size:

                    ppo.train(
                        # reshaping so that all same environments next to each other
                        jnp.concatenate(batch_states).reshape(
                            num_devices * envs_per_device, -1, batch_states[0].shape[-1]
                        ),
                        jnp.concatenate(batch_actions).reshape(
                            num_devices * envs_per_device, -1, batch_actions[0].shape[-1]
                        ),
                        jnp.concatenate(batch_rewards).reshape(num_devices * envs_per_device, -1),
                        jnp.concatenate(batch_masks).reshape(num_devices * envs_per_device, -1),
                        config,
                    )
                    batch_states, batch_actions, batch_rewards, batch_masks = [], [], [], []

                # with open(os.path.join(log_dir, f"log_{args.env_name}.txt"), "a") as outfile:
                #     outfile.write("\t" + str(episodes) + "\t" + str(jnp.mean(score)) + "\n")
                scores.append(jnp.mean(score))

            # Process any remaining data
            if batch_states:
                ppo.train(
                    jnp.concatenate(batch_states),
                    jnp.concatenate(batch_actions),
                    jnp.concatenate(batch_rewards),
                    jnp.concatenate(batch_masks),
                    config,
                )

            score_avg = float(jnp.mean(jnp.array(scores)))
            pbar.close()
            print("{} episode score is {:.2f}".format(episodes, score_avg))

            # Save video for this iteration
            if args.save_video and i % args.save_video_every == 0 and rollout:
                print(f"Total frames: {len(rollout)}")

                # Reorder frames to group by environment
                reordered_rollout = [
                    frame for i in range(args.envs_to_sample) for frame in rollout[i :: args.envs_to_sample]
                ]

                images = jnp.array(
                    env.render(
                        [frame[0] for frame in reordered_rollout[:: args.render_every]],
                        camera="side",
                        width=args.width,
                        height=args.height,
                    )
                )
                fps = int(1 / env.dt)
                print(f"Find video at videos/{args.env_name}_video{i}.mp4 with fps={fps}")
                media.write_video(f"videos/{args.env_name}_video{i}.mp4", images, fps=fps)

        with open("actor_params.pkl", "wb") as f:
            pickle.dump(ppo.actor_params, f)

        with open("critic_params.pkl", "wb") as f:
            pickle.dump(ppo.critic_params, f)

        print("Training completed. Models saved.")


if __name__ == "__main__":
    main()
