from collections import defaultdict

import gym
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from gym.wrappers import RescaleAction
from tensorboardX import SummaryWriter
from tqdm import tqdm

from mbpo.env_wrappers import ActionRepeat


# Following https://github.com/tensorflow/probability/issues/840 and
# https://github.com/tensorflow/probability/issues/840.
class StableTanhBijector(tfp.bijectors.Tanh):
    def __init__(self, validate_args=False, name='tanh_stable_bijector'):
        super(StableTanhBijector, self).__init__(validate_args=validate_args, name=name)

    def _inverse(self, y):
        y = tf.where(
            tf.less_equal(tf.abs(y), 1.),
            tf.clip_by_value(y, -1.0 + 1e-6, 1.0 - 1e-6),
            y)
        return tf.atanh(y)


class SampleDist(object):
    def __init__(self, dist, seed, samples=500):
        self._dist = dist
        self._samples = samples
        # Use a stateless seed to get the same samples everytime -
        # this simulates the fact that the mean, entropy and mode are deterministic.
        self._seed = (0, seed)

    @property
    def name(self):
        return 'SampleDist'

    def __getattr__(self, name):
        return getattr(self._dist, name)

    def mean(self):
        samples = self._dist.sample(self._samples, seed=self._seed)
        return tf.reduce_mean(samples, 0)

    def mode(self):
        sample = self._dist.sample(self._samples, seed=self._seed)
        logprob = self._dist.log_prob(sample)
        return tf.gather(sample, tf.argmax(logprob))[0]

    def entropy(self):
        sample = self._dist.sample(self._samples, seed=self._seed)
        logprob = self.log_prob(sample)
        return -tf.reduce_mean(logprob, 0)


class TrainingLogger(object):
    def __init__(self, config):
        self._writer = SummaryWriter(config.log_dir)
        self._metrics = defaultdict(tf.metrics.Mean)
        dump_string(pretty_print(config), config.log_dir + '/params.txt')

    def __getitem__(self, item):
        return self._metrics[item]

    def __setitem__(self, key, value):
        self._metrics[key] = value

    def log_evaluation_summary(self, summary, step):
        for k, v in summary.items():
            self._writer.add_scalar(k, float(v), step)
        self._writer.flush()

    def log_metrics(self, step):
        print("\n----Training step {} summary----".format(step))
        for k, v in self._metrics.items():
            print("{:<40} {:<.2f}".format(k, float(v.result())))
            self._writer.add_scalar(k, float(v.result()), step)
            v.reset_states()
        self._writer.flush()

    # (N, T, C, H, W)
    def log_video(self, images, step=None, name='Evaluation policy', fps=15):
        self._writer.add_video(name, images, step, fps=fps)
        self._writer.flush()


def do_episode(agent, training, environment, config, pbar, render, reset_function=None):
    observation = environment.reset() if reset_function is None else reset_function()
    episode_summary = defaultdict(list)
    steps = 0
    done = False
    while not done:
        action = agent(observation, training).squeeze()
        next_observation, reward, done, info = environment.step(action)
        terminal = done and not info.get('TimeLimit.truncated')
        if training:
            agent.observe(dict(observation=observation.astype(np.float32),
                               next_observation=next_observation.astype(np.float32),
                               action=action.astype(np.float32),
                               reward=np.array(reward, dtype=np.float32),
                               terminal=np.array(terminal, dtype=np.bool),
                               info=info,
                               steps=info.get('steps', config.action_repeat)))
        observation = next_observation
        if render:
            episode_summary['image'].append(environment.render(mode='rgb_array'))
        pbar.update(info.get('steps', config.action_repeat))
        steps += info.get('steps', config.action_repeat)
        episode_summary['observation'].append(observation)
        episode_summary['next_observation'].append(next_observation)
        episode_summary['action'].append(action)
        episode_summary['reward'].append(reward)
        episode_summary['terminal'].append(terminal)
        episode_summary['info'].append(info)
    episode_summary['steps'] = [steps]
    return steps, episode_summary


def interact(agent, environment, steps, config, training=True):
    pbar = tqdm(total=steps)
    steps_count = 0
    episodes = []
    while steps_count < steps:
        episode_steps, episode_summary = \
            do_episode(agent, training,
                       environment, config,
                       pbar, len(episodes) < config.render_episodes and not training)
        steps_count += episode_steps
        episodes.append(episode_summary)
        print("\nFinished episode with return: {}".format(sum(episode_summary['reward'])))
    pbar.close()
    return steps, episodes


def make_env(name, episode_length, action_repeat, seed):
    env = gym.make(name)
    if not isinstance(env, gym.wrappers.TimeLimit):
        env = gym.wrappers.TimeLimit(env, max_episode_steps=episode_length)
    else:
        # https://github.com/openai/gym/issues/499
        env._max_episode_steps = episode_length
    env = ActionRepeat(env, action_repeat)
    env = RescaleAction(env, -1.0, 1.0)
    env.seed(seed)
    # train_env = ObservationNormalize(env)
    # test_env = TestObservationNormalize(env, train_env.normalize)
    return env, env


# Reading the errors produced by this function should assume all obsersvations are normalized to
# [-1, 1]
def evaluate_model(episodes_summaries, agent):
    observations_mse = 0.0
    rewards_mse = 0.0
    terminal_accuracy = 0.0
    n_episodes = min(len(episodes_summaries), 30)
    prediction_horizon = 25
    observations = np.empty([n_episodes, episodes_summaries[0]['observation'][0].shape[0]])
    actions = np.empty([n_episodes, prediction_horizon] +
                       [max(1, sum(episodes_summaries[0]['action'][0].shape))])
    for i in range(n_episodes):
        observations[i, :] = np.array(episodes_summaries[i]['observation'][0])
        episode_action = np.array(episodes_summaries[i]['action'])[-prediction_horizon:, ...]
        episode_action = episode_action[:, None] if len(
            episode_action.shape) == 1 else episode_action
        actions[i, :episode_action.shape[0]] = episode_action
    predicted_rollouts = agent.model(tf.constant(observations, tf.float32),
                                     tf.constant(actions, tf.float32))
    # observations_mse += (np.asarray(
    #     predicted_rollouts['next_observation'].numpy() -
    #     episodes_summaries[i]['next_observation'][-prediction_horizon:]) ** 2).mean() \
    #                     / n_episodes
    # rewards_mse += (np.asarray(
    #     predicted_rollouts['reward'].numpy() -
    #     episodes_summaries[i]['reward'][-prediction_horizon:]) ** 2).mean() / n_episodes
    # terminal_accuracy += (1.0 - (np.abs(predicted_rollouts['terminal'] -
    #                                     episodes_summaries[i]['terminal'][-prediction_horizon:])
    #                              < 1e-5)).mean() / n_episodes
    return dict(observations_mse=observations_mse,
                rewards_mse=rewards_mse,
                terminal_accuracy=terminal_accuracy)


def pretty_print(config, indent=0):
    summary = str()
    align = 30 - indent * 2
    for key, value in vars(config).items():
        summary += '  ' * indent + '{:{align}}'.format(str(key), align=align)
        summary += '{}\n'.format(str(value))
    return summary


def dump_string(string, filename):
    with open(filename, 'w+') as file:
        file.write(string)


def preprocess(image):
    return tf.cast(image, tf.float32) / 255.0 - 0.5


def clone_model(a, b):
    for var_a, var_b in zip(a.variables, b.variables):
        var_b.assign(var_a)


def split_batch(batch, split_size):
    div = tf.shape(batch)[0] // split_size
    return tf.split(batch, [div] * (split_size - 1) +
                    [div + tf.shape(batch)[0] % split_size])


def standardize_video(sequence, modality, transpose=True):
    shape = tf.shape(sequence)
    if modality == 'binary_image' and shape[-1] != 1:
        standardized = tf.transpose(tf.reshape(tf.transpose(sequence, [0, 2, 3, 1, 4]),
                                  [shape[0], shape[2], shape[3], -1, 1]), [0, 3, 1, 2, 4])
    elif modality == 'rgb_image' and shape[-1] != 3:
        standardized = tf.transpose(tf.reshape(tf.transpose(sequence, [0, 2, 3, 1, 4]),
                                               [shape[0], shape[2], shape[3], -1, 3]),
                                    [0, 3, 1, 2, 4])
    else:
        standardized = sequence
    if transpose:
        return tf.transpose(standardized, [0, 1, 4, 2, 3]).numpy()
    else:
        return standardized.numpy()
