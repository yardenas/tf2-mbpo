import tensorflow as tf
from tensorflow_probability import distributions as tfd

from mbpo.swag import SWAG
from mbpo.world_models import BayesianWorldModel


class SwagMultistepModel(BayesianWorldModel):
    def __init__(self, config, logger, observation_shape):
        super(SwagMultistepModel, self).__init__(config, logger)
        self._optimizer = SWAG(
            tf.optimizers.Adam(
                config.model_learning_rate,
                clipnorm=config.grad_clip_norm),
            2000, 5)
        self._posterior_samples = config.posterior_samples
        self._head = tf.keras.Sequential(
            [tf.keras.layers.Dense(config.units, tf.nn.relu) for _ in range(2)] +
            [tf.keras.layers.Dense(2 * observation_shape[-1])])

    def _update_beliefs(self, prev_action, current_observation):
        pass

    def _generate_sequences_posterior(self, initial_belief, horizon, actor,
                                      actions, log_sequences):
        posterior_samples = []
        for _ in range(self._posterior_samples):
            self._optimizer.sample_and_assign(1.0, self.trainable_variables)
            sample = self._unroll_sequence(initial_belief, horizon, actions=actions).sample()
            posterior_samples.append(sample)
        stacked_all = tf.stack(posterior_samples, 0)
        return {'stochastic': stacked_all, 'deterministic': stacked_all}, stacked_all

    def _reconstruct_sequences_posterior(self, batch):
        next_observations = batch['observation'][:, 1:]
        initial_observation = batch['observation'][:, 0]
        samples = []
        actions = batch['action']
        for i in range(self._posterior_samples):
            self._optimizer.sample_and_assign(1.0, self.trainable_variables)
            sample = self._unroll_sequence(initial_observation,
                                           tf.shape(next_observations)[1],
                                           actions,
                                           next_observations).sample()
            samples.append(sample)
        stacked_all = tf.stack(samples, 0)
        return stacked_all, {'stochastic': stacked_all, 'deterministic': stacked_all}

    def _training_step(self, batch, log_sequences):
        observations = batch['observation'][:, :-1]
        next_observations = batch['observation'][:, 1:]
        action = batch['action']
        with tf.GradientTape() as tape:
            sequence = self._unroll_sequence(
                observations[:, 0], tf.shape(next_observations)[1],
                actions=action)
            loss = -tf.reduce_mean(sequence.log_prob(next_observations))
        grads = tape.gradient(loss, self.trainable_variables)
        self._optimizer.apply_gradients(zip(grads, self.trainable_variables))
        sequences = tf.reshape(sequence.mean(),
                               tf.shape(batch['observation'][:, 1:])) if log_sequences else None
        return None, sequences

    def _unroll_sequence(self, initial_observation,
                         horizon, actions,
                         next_observations=None):
        raise NotImplementedError


class SwagRnn(SwagMultistepModel):
    def __init__(self, config, logger, observation_shape):
        super(SwagRnn, self).__init__(config, logger, observation_shape)
        self._cell = tf.keras.layers.GRUCell(config.deterministic_size)

    def _unroll_sequence(self, initial_observation,
                         horizon, actions,
                         next_observations=None):
        sequence = tf.TensorArray(tf.float32, horizon)
        observation = initial_observation
        state = self._cell.get_initial_state(None, tf.shape(actions)[0])
        for t in range(horizon):
            action = actions[:, t]
            cat = tf.concat([observation, action], -1)
            state, _ = self._cell(cat, state)
            mu, stddev = tf.split(self._head(state), 2, -1)
            mu += observation
            stddev = tf.math.softplus(stddev)
            prediction = tf.concat([mu, stddev], -1)
            observation = tf.random.normal(mu, stddev).sample() if \
                next_observations is None else next_observations[:, t]
            sequence = sequence.write(t, prediction)
        transposed = tf.transpose(sequence.stack(), [1, 0, 2])
        mu, stddev = tf.split(transposed, 2, -1)
        return tfd.Independent(tfd.Normal(mu, stddev), 1)


class SwagFeedForward(SwagMultistepModel):
    def __init__(self, config, logger, observation_shape):
        super(SwagFeedForward, self).__init__(config, logger, observation_shape)
        self._deter = tf.keras.layers.Dense(config.deterministic_size, tf.nn.relu)
        self._n_step_loss = config.n_step_loss

    def _step(self, observation, action):
        cat = tf.concat([observation, action], -1)
        state = self._deter(cat)
        mu, stddev = tf.split(self._head(state), 2, -1)
        stddev = tf.math.softplus(stddev)
        mu += observation
        return mu, stddev

    def _unroll_sequence(self, initial_observation,
                         horizon, actions,
                         next_observations=None):
        sequence = tf.TensorArray(tf.float32, horizon)
        observation = initial_observation
        for t in range(horizon):
            action = actions[:, t]
            mu, stddev = self._step(observation, action)
            prediction = tf.concat([mu, stddev], -1)
            observation = tf.random.normal(mu, stddev).sample() if \
                next_observations is None else next_observations[:, t]
            sequence = sequence.write(t, prediction)
        transposed = tf.transpose(sequence.stack(), [1, 0, 2])
        mu, stddev = tf.split(transposed, 2, -1)
        return tfd.Independent(tfd.Normal(mu, stddev), 1)

    def _training_step(self, batch, log_sequences):
        observations, next_observations, action = self._make_training_step_data(batch)
        with tf.GradientTape() as tape:
            if self._n_step_loss:
                predicted_next_observation = self._unroll_sequence(
                    observations[:, 0], tf.shape(next_observations)[1],
                    actions=action)
            else:
                mu, stddev = self._step(observations, action)
                predicted_next_observation = tfd.Normal(mu, stddev)
            loss = -tf.reduce_mean(predicted_next_observation.log_prob(next_observations))
        grads = tape.gradient(loss, self.trainable_variables)
        self._optimizer.apply_gradients(zip(grads, self.trainable_variables))
        sequence = tf.reshape(predicted_next_observation.mean(),
                              tf.shape(batch['observation'][:, 1:])) if log_sequences else None
        return None, sequence

    def _make_training_step_data(self, batch):
        if self._n_step_loss:
            observations = batch['observation'][:, :-1]
            next_observations = batch['observation'][:, 1:]
            actions = batch['action']
        else:
            # We should actually shuffle the data to ensure that it is i.i.d but...(?)
            observations = tf.reshape(batch['observation'][:, :-1], (-1,) + self._shape)
            next_observations = tf.reshape(batch['observation'][:, 1:], (-1,) + self._shape)
            actions = tf.reshape(batch['action'], [-1, tf.shape(batch['action'])[-1]])
        return observations, next_observations, actions
