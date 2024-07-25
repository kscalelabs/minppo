import torch
import torch.nn as nn
from parameters import *
import torch.optim as optim
import numpy as np
import jax


class Ppo:
    def __init__(self, N_S, N_A):
        self.actor_net = Actor(N_S, N_A)
        self.critic_net = Critic(N_S)
        self.actor_optim = optim.Adam(self.actor_net.parameters(), lr=lr_actor)
        self.critic_optim = optim.Adam(self.critic_net.parameters(), lr=lr_critic, weight_decay=l2_rate)
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
