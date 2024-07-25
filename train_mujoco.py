import os
from typing import Any
import gym
import imageio
import jax
import jax.numpy as jp
import torch
import numpy as np
import argparse
from PPO import Ppo
from collections import deque
from parameters import *
import mujoco_py
import mediapy as media
from brax import base, envs

from environments.robot_mjx import HumanoidEnv, RobotMJXEnv
import environments


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

    env = gym.make(args.env_name)
    N_S = env.observation_space.shape[0]
    N_A = env.action_space.shape[0]

    rng = jax.random.PRNGKey(0)

    env.reset(seed=500)
    torch.manual_seed(500)
    np.random.seed(500)

    ppo = Ppo(N_S, N_A)
    normalize = Normalize(N_S)
    episodes = 0
    eva_episodes = 0

    if args.render or args.save_video:
        viewer = mujoco_py.MjViewer(env.sim)

    for i in range(Iter):
        memory = deque()
        scores = []
        steps = 0
        frames = []  # To store frames for video

        while steps < 2048:
            episodes += 1

            s = normalize(env.reset())
            score = 0
            rollout = []

            # do as many episodes from random starting point
            for _ in range(MAX_STEP):
                steps += 1

                # reward defined in environment
                a=ppo.actor_net.choose_action(torch.from_numpy(np.array(s).astype(np.float32)).unsqueeze(0))[0]
                s_ , r ,done, _,info = env.step(a)
                s_ = normalize(s_)

                if args.render:
                    viewer.render()

                # Capture frame for video if enabled
                if args.save_video:
                    rollout.append(s_)

                mask = (1 - done) * 1
                # keeping track of all episodes to train with actor/critic
                memory.append([s, a, r, mask])

                score += r
                s = s_
                if done:
                    break
            with open("log_" + args.env_name + ".txt", "a") as outfile:
                outfile.write("\t" + str(episodes) + "\t" + str(score) + "\n")
            scores.append(score)
        score_avg = np.mean(scores)
        print("{} episode score is {:.2f}".format(episodes, score_avg))

        # Save video for this iteration
        if args.save_video and frames:
            images = np.array(env.render(rollout[::2], camera="side", width=640, height=480))
            fps = int(1 / env.dt)
            print(f"Find video at video.mp4 with fps={fps}")
            media.write_video("video.mp4", images, fps=fps)
        ppo.train(memory)

    if args.render or args.save_video:
        viewer.close()


if __name__ == "__main__":
    main()
