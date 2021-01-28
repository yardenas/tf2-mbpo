import os
import sys

import gym
import numpy as np


class RandomObs(gym.ObservationWrapper):
    def __init__(self, env, stddev=0.1):
        super(RandomObs,
              self).__init__(env)
        self._stddev = stddev

    def observation(self, observation):
        return observation + np.random.normal(np.zeros_like(observation),
                                              np.ones_like(observation)) * self._stddev


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


def main():
    T = 50
    N = 100
    M = 200
    target_dir = os.path.join('data', 'inverted_pendulum')
    os.makedirs(target_dir, exist_ok=True)
    env = RandomObs(gym.make('Pendulum-v0'))
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


if __name__ == "__main__":
    main()