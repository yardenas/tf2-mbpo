import os
import sys

import gym
import numpy as np


def generate_sequence(env, horizon):
    obs_sequence = np.empty((horizon,) + env.observation_space.shape)
    act_sequence = np.empty((horizon - 1,) + env.action_space.shape)

    def policy():
        return np.random.uniform(env.action_space.low, env.action_space.high)

    obs = env.reset()
    for t in range(horizon - 1):
        obs_sequence[t] = obs
        action = policy()
        act_sequence[t] = action
        obs, _, done, _ = env.step(action)
        assert not done, "Pendulum ends only on time limit."
    obs_sequence[-1] = obs
    return obs_sequence, act_sequence


if __name__ == "__main__":
    T = 50
    N = 100
    M = 200
    target_dir = os.path.join('data', 'inverted_pendulum')
    os.makedirs(target_dir, exist_ok=True)
    env = gym.make('Pendulum-v0')
    for j in range(M):
        print('.', end='')
        sys.stdout.flush()
        obs_data = np.empty((N, T) + env.observation_space.shape)
        action_data = np.empty((N, T - 1) + env.action_space.shape)
        for i in range(N):
            obs_data[i], action_data[i] = generate_sequence(env, T)
        np.savez_compressed(os.path.join(target_dir, 'train_%03d' % j),
                            observation=obs_data, action=action_data)
    N = 100
    M = 10
    for j in range(M):
        obs_data = np.empty((N, T) + env.observation_space.shape)
        action_data = np.empty((N, T - 1) + env.action_space.shape)
        for i in range(N):
            obs_data[i], action_data[i] = generate_sequence(env, T)
        np.savez_compressed(os.path.join(target_dir, 'test_%03d' % j),
                            observation=obs_data, action=action_data)
