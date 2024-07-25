"""Trains a policy network to get a humanoid to stand up."""

import argparse
from collections import deque
from dataclasses import dataclass, field

import jax
import mediapy as media
import numpy as np
import torch
from torch import nn, optim
from tqdm import tqdm

from environment import HumanoidEnv


@dataclass
class Config:
    lr_actor: float = field(default=0.0003)
    lr_critic: float = field(default=0.0003)
    num_iterations: int = field(default=15000)
    max_steps: int = field(default=10000)
    max_steps_per_epoch: int = field(default=2048)
    gamma: float = field(default=0.98)
    lambd: float = field(default=0.98)
    batch_size: int = field(default=64)
    epsilon: float = field(default=0.2)
    l2_rate: float = field(default=0.001)
    beta: int = field(default=3)


class Ppo:
    def __init__(self, N_S, N_A, config: Config) -> None:
        self.actor_net = Actor(N_S, N_A)
        self.critic_net = Critic(N_S)
        self.actor_optim = optim.Adam(self.actor_net.parameters(), lr=config.lr_actor)
        self.critic_optim = optim.Adam(self.critic_net.parameters(), lr=config.lr_critic, weight_decay=config.l2_rate)
        self.critic_loss_func = torch.nn.MSELoss()

    def train(self, memory):
        states = torch.tensor(np.vstack([e[0] for e in memory]), dtype=torch.float32)

        actions = torch.tensor(np.array([e[1] for e in memory]), dtype=torch.float32)
        rewards = torch.tensor(np.array([e[2] for e in memory]), dtype=torch.float32)
        masks = torch.tensor(np.array([e[3] for e in memory]), dtype=torch.float32)

        values = self.critic_net(states)

        # generalized advantage estimation for advantage
        # at a certain point based on immediate and future rewards
        returns, advants = self.get_gae(rewards, masks, values)
        old_mu, old_std = self.actor_net(states)
        pi = self.actor_net.distribution(old_mu, old_std)

        old_log_prob = pi.log_prob(actions).sum(1, keepdim=True)

        n = len(states)
        arr = np.arange(n)
        for epoch in range(1):
            np.random.shuffle(arr)
            for i in range(n // batch_size):
                b_index = arr[batch_size * i : batch_size * (i + 1)]
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

                ratio = torch.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon)

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

    def get_gae(self, rewards, masks, values):
        rewards = torch.Tensor(rewards)
        masks = torch.Tensor(masks)
        returns = torch.zeros_like(rewards)
        advants = torch.zeros_like(rewards)
        running_returns = 0
        previous_value = 0
        running_advants = 0

        # calculating reward, with sum of current reward and diminishing value of future rewards
        for t in reversed(range(0, len(rewards))):
            running_returns = rewards[t] + gamma * running_returns * masks[t]
            running_tderror = rewards[t] + gamma * previous_value * masks[t] - values.data[t]
            running_advants = running_tderror + gamma * lambd * running_advants * masks[t]

            returns[t] = running_returns
            previous_value = values.data[t]
            advants[t] = running_advants

        advants = (advants - advants.mean()) / advants.std()
        return returns, advants


class Actor(nn.Module):
    def __init__(self, N_S, N_A):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(N_S, 64)
        self.fc2 = nn.Linear(64, 64)
        self.sigma = nn.Linear(64, N_A)
        self.mu = nn.Linear(64, N_A)
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
    def __init__(self, N_S):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(N_S, 64)
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


def state_to_flattened_array(state):
    # Get all attributes of the state object
    attributes = vars(state)
    # Initialize an empty list to hold flattened arrays
    flattened_arrays = []
    for attr in attributes.values():
        # Ensure the attribute is a NumPy array
        np_attr = np.asarray(attr)
        # Flatten the attribute and add it to the list
        flattened_arrays.append(np_attr.flatten())
    # Concatenate all flattened arrays into a single array
    flattened_state = np.concatenate(flattened_arrays)
    return flattened_state


# for normalizaing observation states
class Normalize:
    def __init__(self, N_S):
        self.mean = np.zeros((N_S,))
        self.std = np.zeros((N_S,))
        self.stdd = np.zeros((N_S,))
        self.n = 0

    def __call__(self, x):
        x = np.asarray(x)
        self.n += 1
        if self.n == 1:
            self.mean = x
        else:
            old_mean = self.mean.copy()
            self.mean = old_mean + (x - old_mean) / self.n
            self.stdd = self.stdd + (x - old_mean) * (x - self.mean)
        if self.n > 1:
            self.std = np.sqrt(self.stdd / (self.n - 1))
        else:
            self.std = self.mean

        x = x - self.mean

        x = x / (self.std + 1e-8)

        x = np.clip(x, -5, +5)

        return x


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default="Humanoid-v2", help="name of Mujoco environement")
    parser.add_argument("--render", action="store_true", help="render the environment")
    parser.add_argument("--save_video", action="store_true", help="save video of the environment")
    parser.add_argument("--width", type=int, default=640, help="width of the video frame")
    parser.add_argument("--height", type=int, default=480, help="height of the video frame")
    parser.add_argument("--render_every", type=int, default=2, help="render the environment every N steps")
    args = parser.parse_args()

    env = HumanoidEnv()
    N_S = env.observation_size
    N_A = env.action_size
    print("N_A", N_A)
    print("N_S", N_S)

    reset_fn = jax.jit(env.reset)
    step_fn = jax.jit(env.step)
    # reset_fn = env.reset
    # step_fn = env.step
    rng = jax.random.PRNGKey(0)

    reset_fn(rng)
    torch.manual_seed(500)
    np.random.seed(500)

    ppo = Ppo(N_S, N_A)
    normalize = Normalize(N_S)
    episodes = 0
    eva_episodes = 0

    for i in range(Iter):
        memory = deque()
        scores = []
        steps = 0
        rollout = []

        pbar = tqdm(total=MAX_STEPS_PER_EPOCH, desc=f"Steps for epoch {i}")

        while steps < MAX_STEPS_PER_EPOCH:
            episodes += 1
            # print(episodes)

            state = reset_fn(rng)
            s = jax.device_put(normalize(state.obs))
            score = 0

            # do as many episodes from random starting point
            for _ in range(MAX_STEP):
                steps += 1

                a = ppo.actor_net.choose_action(torch.from_numpy(np.array(s).astype(np.float32)).unsqueeze(0))[0]

                # reward defined in environment
                state = step_fn(state, a)
                s_, r, done = state.obs, state.reward, state.done
                s_ = jax.device_put(normalize(s_))

                # Capture frame for video if enabled
                if args.save_video:
                    rollout.append(state.pipeline_state)

                mask = (1 - done) * 1
                # keeping track of all episodes to train with actor/critic
                memory.append([s, a, r, mask])

                score += r
                s = s_
                pbar.update(1)
                if done.all():
                    break
            with open("log_" + args.env_name + ".txt", "a") as outfile:
                outfile.write("\t" + str(episodes) + "\t" + str(score) + "\n")
            scores.append(score)

        score_avg = np.mean(scores)
        pbar.close()
        print("{} episode score is {:.2f}".format(episodes, score_avg))

        # Save video for this iteration
        if args.save_video and rollout:
            images = np.array(
                env.render(rollout[:: args.render_every], camera="side", width=args.width, height=args.height)
            )
            fps = int(1 / env.dt)
            print(f"Find video at video.mp4 with fps={fps}")
            media.write_video(f"videos/video{i}.mp4", images, fps=fps)

        ppo.train(memory)


if __name__ == "__main__":
    main()
