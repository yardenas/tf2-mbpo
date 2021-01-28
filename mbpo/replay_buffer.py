import collections
import functools
import os

import numpy as np
import tensorflow as tf
from tensorflow_probability import stats as tfps

import mbpo.utils as utils


class ReplayBuffer(object):
    def __init__(self, data_path, observation_type, observation_shape, action_shape,
                 sequence_length, batch_size, refresh_period=100):
        self._data_path = data_path
        self._sequence_length = sequence_length
        self._observation_type = observation_type
        self._dataset = iter(self._make_dataset(observation_shape, action_shape, observation_type,
                                                batch_size))
        self._refresh_period = refresh_period
        self._buffer = {'observation': [],
                        'action': [],
                        'reward': [],
                        'terminal': [],
                        'info': []}
        self._obs_mean = tf.Variable(tf.zeros(observation_shape),
                                     dtype=np.float32, trainable=False)
        self._obs_variance = tf.Variable(tf.zeros(observation_shape),
                                         dtype=np.float32, trainable=False)
        self._obs_moving_stats_count = tf.Variable(0)

    # https://github.com/danijar/dreamer/blob/02
    # f0210f5991c7710826ca7881f19c64a012290c/tools.py  # L157
    def _load_data(self):
        data = {}
        while True:
            for filename in os.listdir(self._data_path):
                file_path = os.path.join(self._data_path, filename)
                data[filename] = np.load(file_path)
            ordered_data = collections.OrderedDict(sorted(data.items()))
            keys = ordered_data.keys()
            for episode_id in np.random.choice(len(keys), self._refresh_period):
                episode_data = ordered_data[keys[episode_id]]
                episode_length = episode_data['action'].shape[1]
                spare = episode_length - self._sequence_length
                if spare < 1:
                    print('Episode too short, skipping it.')
                    continue
                index = int(np.random.randint(0, spare))
                yield {k: v[index:index + self._sequence_length] for k, v in episode_data.items()}

    def _make_dataset(self, observation_shape, action_shape, observation_type,
                      batch_size):
        dataset = tf.data.Dataset.from_generator(
            self._load_data,
            output_types={'observation': np.float32,
                          'action': np.float32,
                          'reward': np.float32,
                          'terminal': np.float32},
            output_shapes={
                'observation': (self._sequence_length,) + observation_shape,
                'action': (self._sequence_length - 1,) + action_shape,
                'reward': (self._sequence_length - 1,),
                'terminal': (self._sequence_length - 1,)})
        dataset = dataset.map(functools.partial(
            self._preprocess, observation_type=observation_type))
        dataset = dataset.prefetch(128)
        dataset = dataset.batch(batch_size, drop_remainder=True)
        return dataset

    def _finalize_episode(self):
        # ANd postprocessing: change to uint, and resize.
        episode_data = {k: np.array(v) for k, v in self._buffer.items()}
        if self._observation_type == 'dense':
            self._update_statistics(episode_data['observation'])
        elif self._observation_type == 'rgb_image' or self._observation_type == 'binary_image':
            episode_data['observation'] *= 255
            episode_data['observation'].astype(np.uint8)
        np.savez_compressed(self._data_path, **episode_data)

    def _update_statistics(self, observation):
        tfps.moving_stats.assign_moving_mean_variance(
            observation,
            self._obs_mean,
            self._obs_variance,
            zero_debias_count=self._obs_moving_stats_count)

    def _preprocess(self, data, observation_type):
        if observation_type == 'dense':
            data['observation'] = utils.normalize_clip(
                data['observation'], tf.convert_to_tensor(self._obs_mean),
                tf.sqrt(tf.convert_to_tensor(self._obs_variance)), 10.0)
        elif observation_type == 'rgb_image':
            data['observation'] = utils.preprocess(data['observation'])
        elif observation_type == 'binary_image':
            data['observation'] = tf.where(
                utils.preprocess(data['observation']) > 0.0, 1.0, 0.0)
        else:
            raise RuntimeError("Invalid observation type")

    def store(self, transition):
        if not self._buffer['observation']:
            for k, v in self._buffer.items():
                v.append(transition[k])
            self._buffer['observation'].append(transition['next_observation'])
        else:
            for k, v in self._buffer.items():
                if k == 'observation':
                    v.append(transition['next_observation'])
                else:
                    v.append(transition[k])
        if transition['terminal'] or transition['info'].get('TimeLimit.truncated'):
            self._finalize_episode()
            for v in self._buffer.values():
                v.clear()

    def sample(self):
        return next(self._dataset)
