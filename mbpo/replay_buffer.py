import numpy as np


class SequenceBuffer(object):
    def __init__(self, sequence_length, observation_dim, action_dim):
        self._buffer = {
            'observation': np.empty((sequence_length, observation_dim),
                                    dtype=np.float32),
            'next_observation': np.empty((sequence_length, observation_dim),
                                         dtype=np.float32),
            'prev_action': np.empty((sequence_length, action_dim), dtype=np.float32),
            'reward': np.empty(sequence_length, dtype=np.float32),
            'terminal': np.empty(sequence_length, dtype=np.bool),
            'info': np.empty(sequence_length, dtype=dict)
        }
        self._sequence_length = sequence_length
        self._ptr = 0

    def store(self, transition):
        added_items = 0
        # In case the episode ended and this was the last transition, we cannot complete the
        # sequence -> discard it.
        if transition['info'].get('TimeLimit.truncated') and self._ptr < self._sequence_length - 1:
            self._force_reset()
        else:
            for k, v in transition.items():
                self._buffer[k][self._ptr:self._ptr + 1, ...] = transition[k]
            added_items += 1
        # If this transition led to a terminal state, pad the sequence with the last items.
        if transition['terminal'] and self._ptr + 1 < self._sequence_length:
            for k, v in self._buffer.items():
                self._buffer[k][self._ptr + 1:] = self._buffer[k][self._ptr + 1]
            added_items += self._sequence_length - self._ptr - 2
        assert self._ptr + added_items <= self._sequence_length, (self._ptr + added_items)
        self._ptr = (self._ptr + added_items) % self._sequence_length

    def _force_reset(self):
        self._ptr = 0

    @property
    def full(self):
        # When full, ptr overflows to zero.
        return self._ptr == 0

    def flush(self):
        return self._buffer


class ReplayBuffer(object):
    def __init__(self, observation_dim, action_dim, sequence_length, memory_capacity=1000000):
        self._memory_capacity = memory_capacity
        self._data = {
            'observation': np.empty((memory_capacity, sequence_length, observation_dim),
                                    dtype=np.float32),
            'next_observation': np.empty((memory_capacity, sequence_length, observation_dim),
                                         dtype=np.float32),
            'prev_action': np.empty((memory_capacity, sequence_length, action_dim), dtype=np.float32),
            'reward': np.empty((memory_capacity, sequence_length), dtype=np.float32),
            'terminal': np.empty((memory_capacity, sequence_length), dtype=np.bool),
            'info': np.empty((memory_capacity, sequence_length), dtype=dict)
        }
        self._sequence_buffer = SequenceBuffer(sequence_length, observation_dim, action_dim)
        self._size = 0
        self._ptr = 0
        self._sequence_length = sequence_length
        self.obs_mean = np.zeros((observation_dim,), dtype=np.float32)
        self.obs_stddev = np.ones_like(self.obs_mean)

    def update_statistics(self):
        cat = np.concatenate([
            self._data['observation'][:self._size].ravel(),
            self._data['next_observation'][:self._size].ravel()], axis=0)
        self.obs_mean = np.mean(cat, axis=0)
        self.obs_stddev = np.std(cat, axis=0)

    def store(self, transition):
        self._sequence_buffer.store(transition)
        if self._sequence_buffer.full:
            for k, v in self._sequence_buffer.flush():
                self._data[k][self._ptr] = v.copy()
            self._size = min(self._size + 1, self._memory_capacity)
        self._ptr = (self._ptr + 1) % self._memory_capacity

    def sample(self, batch_size):
        indices = np.random.randint(0, self._size, batch_size)
        out = {k: v[indices, ...] for k, v in self._data.items()}
        out['observation'] = np.clip((out['observation'] - self.obs_mean) / self.obs_stddev,
                                     -10.0, 10.0)
        out['next_observation'] = np.clip(
            (out['next_observation'] - self.obs_mean) / self.obs_stddev,
            -10.0, 10.0)
        return {k: v for k, v in out.items() if k != 'info'}
