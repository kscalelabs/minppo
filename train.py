"""Trains a policy network to get a humanoid to stand up."""

import argparse
from functools import partial
import os
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Tuple

import jax
import jax.numpy as jnp
import mediapy as media
import numpy as np
import torch
from brax.envs import State  # type: ignore[import-untyped]
from jax import random

# from torch import nn, optim
from tqdm import tqdm
import optax
from flax import linen as nn

from environment import HumanoidEnv


@dataclass
class Config:
    lr_actor: float = field(default=0.0003)
    lr_critic: float = field(default=0.0003)
    num_iterations: int = field(default=15000)
    num_envs: int = field(default=16)
    max_steps: int = field(default=10000)
    max_steps_per_epoch: int = field(default=16)
    gamma: float = field(default=0.98)
    lambd: float = field(default=0.98)
    batch_size: int = field(default=64)
    epsilon: float = field(default=0.2)
    l2_rate: float = field(default=0.001)
    beta: int = field(default=3)


class Ppo:
    def __init__(self, observation_size: int, action_size: int, config: Config, key: jnp.ndarray):
        self.actor = Actor(action_size)
        self.critic = Critic()
        self.actor_params = self.actor.init(key, jnp.zeros((1, observation_size)))
        self.critic_params = self.critic.init(key, jnp.zeros((1, observation_size)))

        self.actor_optim = optax.adam(learning_rate=config.lr_actor)
        self.critic_optim = optax.adamw(learning_rate=config.lr_critic, weight_decay=config.l2_rate)

        self.actor_opt_state = self.actor_optim.init(self.actor_params)
        self.critic_opt_state = self.critic_optim.init(self.critic_params)

    def get_params(self):
        return {
            "actor_params": self.actor_params,
            "critic_params": self.critic_params,
            "actor_opt_state": self.actor_opt_state,
            "critic_opt_state": self.critic_opt_state,
        }

    def update_params(self, new_params):
        self.actor_params = new_params["actor_params"]
        self.critic_params = new_params["critic_params"]
        self.actor_opt_state = new_params["actor_opt_state"]
        self.critic_opt_state = new_params["critic_opt_state"]


def train_step(actor_apply, critic_apply, actor_optim, critic_optim, params, states, actions, rewards, masks, config):
    actor_params, critic_params, actor_opt_state, critic_opt_state = params.values()

    values = critic_apply(critic_params, states).squeeze()
    returns, advants = get_gae(rewards, masks, values, config)

    old_mu, old_std = actor_apply(actor_params, states)
    old_log_prob = actor_log_prob(old_mu, old_std, actions)

    def actor_loss_fn(params):
        mu, std = actor_apply(params, states)
        new_log_prob = actor_log_prob(mu, std, actions)
        ratio = jnp.exp(new_log_prob - old_log_prob)
        surrogate_loss = ratio * advants
        clipped_loss = jnp.clip(ratio, 1.0 - config.epsilon, 1.0 + config.epsilon) * advants
        actor_loss = -jnp.mean(jnp.minimum(surrogate_loss, clipped_loss))
        return actor_loss

    def critic_loss_fn(params):
        critic_returns = critic_apply(params, states).squeeze()
        critic_loss = jnp.mean((critic_returns - returns) ** 2)
        return critic_loss

    actor_grad_fn = jax.value_and_grad(actor_loss_fn)
    actor_loss, actor_grads = actor_grad_fn(actor_params)
    actor_updates, new_actor_opt_state = actor_optim.update(actor_grads, actor_opt_state)
    new_actor_params = optax.apply_updates(actor_params, actor_updates)

    critic_grad_fn = jax.value_and_grad(critic_loss_fn)
    critic_loss, critic_grads = critic_grad_fn(critic_params)
    critic_updates, new_critic_opt_state = critic_optim.update(critic_grads, critic_opt_state)
    new_critic_params = optax.apply_updates(critic_params, critic_updates)

    new_params = {
        "actor_params": new_actor_params,
        "critic_params": new_critic_params,
        "actor_opt_state": new_actor_opt_state,
        "critic_opt_state": new_critic_opt_state,
    }

    return new_params, actor_loss, critic_loss


@jax.jit
def get_gae(rewards, masks, values, config):
    def gae_step(carry, inp):
        gae, next_value = carry
        reward, mask, value = inp
        delta = reward + config.gamma * next_value * mask - value
        gae = delta + config.gamma * config.lambd * mask * gae
        return (gae, value), gae

    _, advantages = jax.lax.scan(
        f=gae_step,
        init=(jnp.zeros_like(rewards[-1]), values[-1]),
        xs=(rewards[::-1], masks[::-1], values[::-1]),
        reverse=True,
    )
    returns = advantages + values
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    return returns, advantages


def train(ppo, memory, config):
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
            batch_indices = arr[config.batch_size * i : config.batch_size * (i + 1)]
            b_states = states[batch_indices]
            b_actions = actions[batch_indices]
            b_rewards = rewards[batch_indices]
            b_masks = masks[batch_indices]

            params = ppo.get_params()
            new_params, actor_loss, critic_loss = train_step(
                ppo.actor.apply,
                ppo.critic.apply,
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


# class Ppo:
#     def __init__(self, observation_size: int, action_size: int, config: Config) -> None:
#         self.actor_net = Actor(observation_size, action_size)
#         self.critic_net = Critic(observation_size)
#         self.actor_optim = optim.Adam(self.actor_net.parameters(), lr=config.lr_actor)
#         self.critic_optim = optim.Adam(self.critic_net.parameters(), lr=config.lr_critic, weight_decay=config.l2_rate)
#         self.critic_loss_func = torch.nn.MSELoss()

#     def train(self, memory: Deque[Tuple[np.ndarray, np.ndarray, float, float]], config: Config) -> None:
#         states = torch.tensor(np.vstack([e[0] for e in memory]), dtype=torch.float32)
#         actions = torch.tensor(np.array([e[1] for e in memory]), dtype=torch.float32)
#         rewards = torch.tensor(np.array([e[2] for e in memory]), dtype=torch.float32)
#         masks = torch.tensor(np.array([e[3] for e in memory]), dtype=torch.float32)

#         # parallelization for many devices -- requires models to be jax-compatible
#         # num_devices = jax.device_count()
#         # if num_devices > 1:
#         #     self.actor_net = jax.pmap(self.actor_net)
#         #     self.critic_net = jax.pmap(self.critic_net)

#         values = self.critic_net(states)

#         # generalized advantage estimation for advantage
#         # at a certain point based on immediate and future rewards
#         returns, advants = self.get_gae(rewards, masks, values, config)
#         old_mu, old_std = self.actor_net(states)
#         pi = self.actor_net.distribution(old_mu, old_std)

#         old_log_prob = pi.log_prob(actions).sum(1, keepdim=True)

#         n = len(states)
#         arr = np.arange(n)
#         for epoch in range(1):
#             np.random.shuffle(arr)
#             for i in range(n // config.batch_size):
#                 b_index = arr[config.batch_size * i : config.batch_size * (i + 1)]
#                 b_states = states[b_index]
#                 b_advants = advants[b_index].unsqueeze(1)
#                 b_actions = actions[b_index]
#                 b_returns = returns[b_index].unsqueeze(1)

#                 mu, std = self.actor_net(b_states)
#                 pi = self.actor_net.distribution(mu, std)

#                 # generally, we don't want our policy's output distribution
#                 # to change too much between iterations
#                 new_prob = pi.log_prob(b_actions).sum(1, keepdim=True)
#                 old_prob = old_log_prob[b_index].detach()
#                 ratio = torch.exp(new_prob - old_prob)

#                 surrogate_loss = ratio * b_advants
#                 critic_returns = self.critic_net(b_states)

#                 # want the critic to best approximate rewards/returns
#                 # so can evaluate how poor actor's actions are
#                 critic_loss = self.critic_loss_func(critic_returns, b_returns)

#                 self.critic_optim.zero_grad()
#                 critic_loss.backward()
#                 self.critic_optim.step()

#                 ratio = torch.clamp(ratio, 1.0 - config.epsilon, 1.0 + config.epsilon)

#                 clipped_loss = ratio * b_advants
#                 actor_loss = -torch.min(surrogate_loss, clipped_loss).mean()

#                 # KL_penalty = self.kl_divergence(old_mu[b_index],old_std[b_index],mu,std)
#                 # actor_loss = -(surrogate_loss-beta*KL_penalty).mean()

#                 self.actor_optim.zero_grad()
#                 actor_loss.backward()

#                 self.actor_optim.step()

#     def kl_divergence(
#         self, old_mu: torch.Tensor, old_sigma: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor
#     ) -> torch.Tensor:
#         old_mu = old_mu.detach()
#         old_sigma = old_sigma.detach()

#         kl = (
#             torch.log(old_sigma)
#             - torch.log(sigma)
#             + (old_sigma.pow(2) + (old_mu - mu).pow(2)) / (2.0 * sigma.pow(2))
#             - 0.5
#         )
#         return kl.sum(1, keepdim=True)

#     def get_gae(
#         self, rewards: torch.Tensor, masks: torch.Tensor, values: torch.Tensor, config: Config
#     ) -> Tuple[torch.Tensor, torch.Tensor]:
#         rewards = torch.Tensor(rewards)
#         masks = torch.Tensor(masks)
#         returns = torch.zeros_like(rewards)
#         advants = torch.zeros_like(rewards)

#         running_returns = torch.zeros(1, device=values.device, dtype=values.dtype)
#         previous_value = torch.zeros(1, device=values.device, dtype=values.dtype)
#         running_advants = torch.zeros(1, device=values.device, dtype=values.dtype)

#         # calculating reward, with combination of immediate reward and diminishing value of future rewards
#         for t in reversed(range(len(rewards))):
#             running_returns = rewards[t] + config.gamma * running_returns * masks[t]
#             running_tderror = rewards[t] + config.gamma * previous_value * masks[t] - values.data[t]
#             running_advants = running_tderror + config.gamma * config.lambd * running_advants * masks[t]

#             returns[t] = running_returns
#             previous_value = values.data[t]
#             advants[t] = running_advants

#         advants = (advants - advants.mean()) / advants.std()
#         return returns, advants


def actor_log_prob(mu, sigma, actions):
    return jax.scipy.stats.norm.logpdf(actions, mu, sigma).sum(axis=-1)


def actor_distribution(mu, sigma):
    return jax.random.normal(jax.random.PRNGKey(0), shape=mu.shape) * sigma + mu


class Actor(nn.Module):
    action_size: int

    @nn.compact
    def __call__(self, x):
        x = nn.tanh(nn.Dense(64)(x))
        x = nn.tanh(nn.Dense(64)(x))
        mu = nn.Dense(self.action_size, kernel_init=nn.initializers.constant(0.1))(x)
        log_sigma = nn.Dense(self.action_size)(x)
        return mu, jnp.exp(log_sigma)


class Critic(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.tanh(nn.Dense(64)(x))
        x = nn.tanh(nn.Dense(64)(x))
        return nn.Dense(1, kernel_init=nn.initializers.constant(0.1))(x)


# class Actor(nn.Module):
#     def __init__(self, observation_size: int, action_size: int) -> None:
#         super(Actor, self).__init__()
#         self.fc1 = nn.Linear(observation_size, 64)
#         self.fc2 = nn.Linear(64, 64)
#         self.sigma = nn.Linear(64, action_size)
#         self.mu = nn.Linear(64, action_size)
#         self.mu.weight.data.mul_(0.1)
#         self.mu.bias.data.mul_(0.0)
#         # self.set_init([self.fc1,self.fc2, self.mu, self.sigma])

#         # randomness to provide variation for next model inference so not limited to few actions
#         self.distribution = torch.distributions.Normal

#     def set_init(self, layers: List[nn.Module]) -> None:
#         for layer in layers:
#             nn.init.normal_(layer.weight, mean=0.0, std=0.1)
#             nn.init.constant_(layer.bias, 0.0)

#     def forward(self, s: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
#         x = torch.tanh(self.fc1(s))
#         x = torch.tanh(self.fc2(x))

#         mu = self.mu(x)
#         log_sigma = self.sigma(x)
#         # log_sigma = torch.zeros_like(mu)
#         sigma = torch.exp(log_sigma)
#         return mu, sigma

#     def choose_action(self, s: torch.Tensor) -> np.ndarray:
#         # given a state, we do our forward pass and then sample from to maintain "random actions"
#         mu, sigma = self.forward(s)
#         pi = self.distribution(mu, sigma)
#         return pi.sample().numpy()


# class Critic(nn.Module):
#     def __init__(self, observation_size: int) -> None:
#         super(Critic, self).__init__()
#         self.fc1 = nn.Linear(observation_size, 64)
#         self.fc2 = nn.Linear(64, 64)
#         self.fc3 = nn.Linear(64, 1)
#         self.fc3.weight.data.mul_(0.1)
#         self.fc3.bias.data.mul_(0.0)

#     def set_init(self, layers: List[nn.Module]) -> None:
#         for layer in layers:
#             nn.init.normal_(layer.weight, mean=0.0, std=0.1)
#             nn.init.constant_(layer.bias, 0.0)

#     def forward(self, s: torch.Tensor) -> torch.Tensor:
#         x = torch.tanh(self.fc1(s))
#         x = torch.tanh(self.fc2(x))
#         returns = self.fc3(x)
#         return returns


# for normalizaing observation states
class Normalize:
    def __init__(self, observation_size: int) -> None:
        self.mean: jnp.ndarray = jnp.zeros((observation_size,))
        self.std: jnp.ndarray = jnp.zeros((observation_size,))
        self.stdd: jnp.ndarray = jnp.zeros((observation_size,))
        self.n: int = 0

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
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


def unwrap_state_vectorization(state: State, config: Config) -> State:
    unwrapped_rollout = []
    # Get all attributes of the state
    attributes = dir(state)

    ne = getattr(state, "ne", 1)  # Default to 1 if not present
    nl = state.nl
    nefc = state.nefc
    nf = state.nf

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
                    if hasattr(value, "__getitem__"):
                        new_state[attr] = value[i]
                    else:
                        new_state[attr] = value
                except Exception:
                    print(f"Could not get first element of {attr}")
        unwrapped_rollout.append(type(state)(ne, nl, nefc, nf, **new_state))

    return unwrapped_rollout


# show "starting image" of xml for testing
def screenshot(
    env: HumanoidEnv, rng: jnp.ndarray, width: int = 640, height: int = 480, filename: str = "screenshot.png"
) -> None:
    state = env.reset(rng)
    image_array = env.render(state.pipeline_state, camera="side", width=width, height=height)
    image_array = jnp.array(image_array).astype("uint8")
    os.makedirs("screenshots", exist_ok=True)
    media.write_image(os.path.join("screenshots", filename), image_array)

    print(f"Screenshot saved as {filename}")


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
    print("action_size", action_size)
    print("observation_size", observation_size)

    @jax.jit
    def reset_fn(rng: jnp.ndarray) -> State:
        rngs = jax.random.split(rng, config.num_envs)
        return jax.vmap(env.reset)(rngs)

    @jax.jit
    def step_fn(states: State, actions: jax.Array) -> State:
        return jax.vmap(env.step)(states, actions)

    rng = jax.random.PRNGKey(0)

    # screenshot(env, rng)
    # return

    reset_fn(rng)
    torch.manual_seed(500)
    np.random.seed(500)

    ppo = Ppo(observation_size, action_size, config, rng)
    normalize = Normalize(observation_size)
    episodes: int = 0

    @partial(jax.jit, static_argnums=(2,))
    def choose_action(actor_params, obs, actor):
        mu, sigma = actor.apply(actor_params, obs)
        return actor_distribution(mu, sigma)

    @jax.jit
    def update_memory(memory, new_data):
        return jax.tree.map(lambda x, y: jnp.concatenate([x, y]), memory, new_data)

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
        rollout = []

        pbar = tqdm(total=config.max_steps_per_epoch, desc=f"Steps for iteration {i}")

        while steps < config.max_steps_per_epoch:
            episodes += config.num_envs

            rng, subrng = jax.random.split(rng)
            states = reset_fn(subrng)
            obs = jax.device_put(normalize(states.obs))
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

        score_avg = jnp.mean(jnp.array(scores))
        pbar.close()
        print("{} episode score is {:.2f}".format(episodes, score_avg))

        # Save video for this iteration
        if args.save_video_every and i % args.save_video_every == 0 and rollout:
            print(len(rollout))
            images = jnp.array(
                env.render(rollout[:: args.render_every], camera="side", width=args.width, height=args.height)
            )
            fps = int(1 / env.dt)
            print(f"Find video at video.mp4 with fps={fps}")
            media.write_video(f"videos/{args.env_name}_video{i}.mp4", images, fps=fps)

        # Convert memory to the format expected by ppo.train
        train_memory = [
            (s, a, r, m) for s, a, r, m in zip(memory["states"], memory["actions"], memory["rewards"], memory["masks"])
        ]
        train(ppo, train_memory, config)


if __name__ == "__main__":
    main()
