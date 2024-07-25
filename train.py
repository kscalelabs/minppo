"""Trains a policy network to get a humanoid to stand up."""

import argparse
from collections import deque
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
from flax import linen
import mediapy as media
import numpy as np
import torch
from brax.envs import State
from torch import nn, optim
from tqdm import tqdm

from environment import HumanoidEnv


@dataclass
class Config:
    lr_actor: float = field(default=0.0003)
    lr_critic: float = field(default=0.0003)
    num_iterations: int = field(default=15000)
    num_envs: int = field(default=16)
    max_steps: int = field(default=10000)
    max_steps_per_epoch: int = field(default=2048)
    gamma: float = field(default=0.98)
    lambd: float = field(default=0.98)
    batch_size: int = field(default=64)
    epsilon: float = field(default=0.2)
    l2_rate: float = field(default=0.001)
    beta: int = field(default=3)


class Ppo:
    def __init__(self, observation_size, action_size, config: Config) -> None:
        self.actor_net = Actor(observation_size, action_size)
        self.critic_net = Critic(observation_size)
        self.actor_optim = optim.Adam(self.actor_net.parameters(), lr=config.lr_actor)
        self.critic_optim = optim.Adam(self.critic_net.parameters(), lr=config.lr_critic, weight_decay=config.l2_rate)
        self.critic_loss_func = torch.nn.MSELoss()

    def train(self, memory, config):
        states = torch.tensor(np.vstack([e[0] for e in memory]), dtype=torch.float32)
        actions = torch.tensor(np.array([e[1] for e in memory]), dtype=torch.float32)
        rewards = torch.tensor(np.array([e[2] for e in memory]), dtype=torch.float32)
        masks = torch.tensor(np.array([e[3] for e in memory]), dtype=torch.float32)

        # parallelization for many devices -- requires models to be jax-compatible
        # num_devices = jax.device_count()
        # if num_devices > 1:
        #     self.actor_net = jax.pmap(self.actor_net)
        #     self.critic_net = jax.pmap(self.critic_net)

        values = self.critic_net(states)

        # generalized advantage estimation for advantage
        # at a certain point based on immediate and future rewards
        returns, advants = self.get_gae(rewards, masks, values, config)
        old_mu, old_std = self.actor_net(states)
        pi = self.actor_net.distribution(old_mu, old_std)

        old_log_prob = pi.log_prob(actions).sum(1, keepdim=True)

        n = len(states)
        arr = np.arange(n)
        for epoch in range(1):
            np.random.shuffle(arr)
            for i in range(n // config.batch_size):
                b_index = arr[config.batch_size * i : config.batch_size * (i + 1)]
                b_states = states[b_index]
                b_advants = advants[b_index].unsqueeze(1)
                b_actions = actions[b_index]
                b_returns = returns[b_index].unsqueeze(1)

                mu, std = self.actor_net(b_states)
                pi = self.actor_net.distribution(mu, std)

                # generally, we don't want our policy's output distribution
                # to change too much between iterations
                new_prob = pi.log_prob(b_actions).sum(1, keepdim=True)
                old_prob = old_log_prob[b_index].detach()
                ratio = torch.exp(new_prob - old_prob)

                surrogate_loss = ratio * b_advants
                critic_returns = self.critic_net(b_states)

                # want the critic to best approximate rewards/returns
                # so can evaluate how poor actor's actions are
                critic_loss = self.critic_loss_func(critic_returns, b_returns)

                self.critic_optim.zero_grad()
                critic_loss.backward()
                self.critic_optim.step()

                ratio = torch.clamp(ratio, 1.0 - config.epsilon, 1.0 + config.epsilon)

                clipped_loss = ratio * b_advants
                actor_loss = -torch.min(surrogate_loss, clipped_loss).mean()

                # KL_penalty = self.kl_divergence(old_mu[b_index],old_std[b_index],mu,std)
                # actor_loss = -(surrogate_loss-beta*KL_penalty).mean()

                self.actor_optim.zero_grad()
                actor_loss.backward()

                self.actor_optim.step()

    def kl_divergence(self, old_mu, old_sigma, mu, sigma):
        old_mu = old_mu.detach()
        old_sigma = old_sigma.detach()

        kl = (
            torch.log(old_sigma)
            - torch.log(sigma)
            + (old_sigma.pow(2) + (old_mu - mu).pow(2)) / (2.0 * sigma.pow(2))
            - 0.5
        )
        return kl.sum(1, keepdim=True)

    def get_gae(self, rewards, masks, values, config: Config):
        rewards = torch.Tensor(rewards)
        masks = torch.Tensor(masks)
        returns = torch.zeros_like(rewards)
        advants = torch.zeros_like(rewards)
        running_returns = 0
        previous_value = 0
        running_advants = 0

        # calculating reward, with sum of current reward and diminishing value of future rewards
        for t in reversed(range(0, len(rewards))):
            running_returns = rewards[t] + config.gamma * running_returns * masks[t]
            running_tderror = rewards[t] + config.gamma * previous_value * masks[t] - values.data[t]
            running_advants = running_tderror + config.gamma * config.lambd * running_advants * masks[t]

            returns[t] = running_returns
            previous_value = values.data[t]
            advants[t] = running_advants

        advants = (advants - advants.mean()) / advants.std()
        return returns, advants


class Actor(nn.Module):
    def __init__(self, observation_size, action_size):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(observation_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.sigma = nn.Linear(64, action_size)
        self.mu = nn.Linear(64, action_size)
        self.mu.weight.data.mul_(0.1)
        self.mu.bias.data.mul_(0.0)
        # self.set_init([self.fc1,self.fc2, self.mu, self.sigma])

        # randomness to provide variation for next model inference so not limited to few actions
        self.distribution = torch.distributions.Normal

    def set_init(self, layers):
        for layer in layers:
            nn.init.normal_(layer.weight, mean=0.0, std=0.1)
            nn.init.constant_(layer.bias, 0.0)

    def forward(self, s):
        x = torch.tanh(self.fc1(s))
        x = torch.tanh(self.fc2(x))

        mu = self.mu(x)
        log_sigma = self.sigma(x)
        # log_sigma = torch.zeros_like(mu)
        sigma = torch.exp(log_sigma)
        return mu, sigma

    def choose_action(self, s):
        # given a state, we do our forward pass and then sample from to maintain "random actions"
        mu, sigma = self.forward(s)
        Pi = self.distribution(mu, sigma)
        return Pi.sample().numpy()


class Critic(nn.Module):
    def __init__(self, observation_size):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(observation_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)
        self.fc3.weight.data.mul_(0.1)
        self.fc3.bias.data.mul_(0.0)
        # self.set_init([self.fc1, self.fc2, self.fc2])

    def set_init(self, layers):
        for layer in layers:
            nn.init.normal_(layer.weight, mean=0.0, std=0.1)
            nn.init.constant_(layer.bias, 0.0)

    def forward(self, s):
        x = torch.tanh(self.fc1(s))
        x = torch.tanh(self.fc2(x))
        returns = self.fc3(x)
        return returns


# for normalizaing observation states
class Normalize:
    def __init__(self, observation_size):
        self.mean = jnp.zeros((observation_size,))
        self.std = jnp.zeros((observation_size,))
        self.stdd = jnp.zeros((observation_size,))
        self.n = 0

    def __call__(self, x):
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


def unwrap_state_vectorization(state, config):
    if config.num_envs == 1:
        return state
    # Get all attributes of the state
    attributes = dir(state)

    ne = getattr(state, "ne", 1)  # Default to 1 if not present
    nl = state.nl
    nefc = state.nefc
    nf = state.nf

    # Create a new state with the first element of each attribute
    new_state = {}
    for attr in attributes:
        # Skip special methods and attributes
        if not attr.startswith("_") and not callable(getattr(state, attr)) and attr not in ["ne", "nl", "nf", "nefc"]:
            value = getattr(state, attr)
            try:
                if hasattr(value, "__getitem__"):
                    new_state[attr] = value[0]
                else:
                    new_state[attr] = value
            except:
                print(f"Could not get first element of {attr}")
    return type(state)(ne, nl, nefc, nf, **new_state)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default="Humanoid-v2", help="name of environmnet to put into logs")
    parser.add_argument("--render", action="store_true", help="render the environment")
    parser.add_argument("--save_video", action="store_true", help="save video of the environment")
    parser.add_argument("--width", type=int, default=640, help="width of the video frame")
    parser.add_argument("--height", type=int, default=480, help="height of the video frame")
    parser.add_argument("--render_every", type=int, default=2, help="render the environment every N steps")
    args = parser.parse_args()

    config = Config()

    env = HumanoidEnv()
    observation_size = env.observation_size
    action_size = env.action_size
    print("action_size", action_size)
    print("observation_size", observation_size)

    @jax.jit
    def reset_fn(rng):
        rngs = jax.random.split(rng, config.num_envs)
        return jax.vmap(env.reset)(rngs)

    @jax.jit
    def step_fn(states, actions):
        return jax.vmap(env.step)(states, actions)

    # reset_fn = jax.jit(env.reset)
    # step_fn = jax.jit(env.step)
    rng = jax.random.PRNGKey(0)

    reset_fn(rng)
    torch.manual_seed(500)
    np.random.seed(500)

    ppo = Ppo(observation_size, action_size, config)
    normalize = Normalize(observation_size)
    episodes = 0

    for i in range(config.num_iterations):
        memory = deque()
        scores = []
        steps = 0
        rollout = []

        pbar = tqdm(total=config.max_steps_per_epoch, desc=f"Steps for epoch {i}")

        while steps < config.max_steps_per_epoch:
            episodes += config.num_envs
            # print(episodes)

            states = reset_fn(rng)
            obs = jax.device_put(normalize(states.obs))
            score = 0

            # do as many episodes from random starting point
            for _ in range(config.max_steps):
                actions = ppo.actor_net.choose_action(torch.from_numpy(np.array(obs).astype(np.float32)).unsqueeze(0))[
                    0
                ]

                # reward defined in environment
                states = step_fn(states, actions)
                next_obs, rewards, dones = states.obs, states.reward, states.done

                # Capture first environment for video if enabled
                if args.save_video:
                    single_state = unwrap_state_vectorization(states.pipeline_state, config)
                    rollout.append(single_state)

                masks = (1 - dones) * 1

                # splitting for batches
                for s, a, r, m, s_ in zip(obs, actions, rewards, masks, next_obs):
                    # keeping track of all episodes to train with actor/critic
                    memory.append([s, a, r, m])

                score += r
                obs = next_obs
                steps += config.num_envs
                pbar.update(config.num_envs)
                if dones.all():
                    break
            with open("log_" + args.env_name + ".txt", "a") as outfile:
                outfile.write("\t" + str(episodes) + "\t" + str(score) + "\n")
            scores.append(score)

        score_avg = np.mean(scores)
        pbar.close()
        print("{} episode score is {:.2f}".format(episodes, score_avg))

        # Save video for this iteration
        if args.save_video and rollout:
            images = jnp.array(
                env.render(rollout[:: args.render_every], camera="side", width=args.width, height=args.height)
            )
            fps = int(1 / env.dt)
            print(f"Find video at video.mp4 with fps={fps}")
            media.write_video(f"videos/{args.env_name}_video{i}.mp4", images, fps=fps)

        ppo.train(memory, config)


if __name__ == "__main__":
    main()
