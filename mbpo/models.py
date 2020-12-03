import tensorflow as tf
from tensorflow_probability import distributions as tfd

import mbpo.utils as utils


class WorldModel(tf.Module):
    def __init__(self, dynamics_size, dynamics_layers, units, reward_layers=1, terminal_layers=1,
                 min_stddev=1e-4,
                 activation=tf.nn.relu):
        super().__init__()
        self._dynamics = tf.keras.Sequential(
            [tf.keras.layers.Dense(units=units, activation=activation) for _ in
             range(dynamics_layers)])
        self._next_observation_residual_mu = tf.keras.layers.Dense(dynamics_size)
        self._next_observation_stddev = tf.keras.layers.Dense(
            dynamics_size,
            activation=lambda t: tf.math.softplus(t) + min_stddev)
        # Assuming reward with a unit standard deviation.
        # TODO (yarden): not sure if this is too simplifying.
        self._reward_mu = tf.keras.Sequential(
            [tf.keras.layers.Dense(units=units, activation=activation) for _ in
             range(reward_layers)] + [
                tf.keras.layers.Dense(1)])
        self._terminal_logit = tf.keras.Sequential(
            [tf.keras.layers.Dense(units=units, activation=activation) for _ in
             range(terminal_layers)] + [
                tf.keras.layers.Dense(1)])

    def __call__(self, observation, action):
        cat = tf.concat([observation, action], axis=1)
        x = self._dynamics(cat)
        next_observations = self._next_observation_residual_mu(x) + tf.stop_gradient(observation)
        # The world model predicts the difference between next_observation and observation.
        return dict(next_observation=tfd.MultivariateNormalDiag(
            loc=next_observations,
            scale_diag=self._next_observation_stddev(x)),
            reward=tfd.Normal(loc=tf.squeeze(self._reward_mu(
                tf.concat([cat, next_observations], axis=1)), axis=1), scale=1.0),
            terminal=tfd.Bernoulli(logits=tf.squeeze(self._terminal_logit(cat), axis=1),
                                   dtype=tf.float32))


# class WorldModelEnsemble(tf.Module):
#     def __init__(self, ensemble_size, dynamics_size, dynamics_layers, units, reward_layers=1,
#                  terminal_layers=1,
#                  min_stddev=1e-4,
#                  activation=tf.nn.relu):
#         super().__init__()
#         self._ensemble = [WorldModel(
#             dynamics_size,
#             dynamics_layers,
#             units, reward_layers, terminal_layers, min_stddev, activation)
#             for _ in range(ensemble_size)]
#
#     def __call__(self, observation, action):
#         states = []
#         rewards = []
#         terminals = []
#         for model in self._ensemble:


class Actor(tf.Module):
    def __init__(self, size, layers, units, seed, min_stddev=1e-4, activation=tf.nn.relu):
        super().__init__()
        self._policy = tf.keras.Sequential(
            [tf.keras.layers.Dense(units=units, activation=activation) for _ in range(layers)]
        )
        self._mu = tf.keras.layers.Dense(size)
        self._stddev = tf.keras.layers.Dense(
            size,
            activation=lambda t: tf.math.softplus(t + 5) + min_stddev)
        self._seed = seed

    def __call__(self, observation):
        x = self._policy(observation)
        multivariate_normal_diag = tfd.MultivariateNormalDiag(
            loc=5.0 * tf.tanh(self._mu(x) / 5.0),
            scale_diag=self._stddev(x))
        # Squash actions to [-1, 1]
        squashed = tfd.TransformedDistribution(multivariate_normal_diag, utils.StableTanhBijector())
        return utils.SampleDist(squashed, seed=self._seed)


class Critic(tf.Module):
    def __init__(self, layers, units, activation=tf.nn.relu, output_regularization=1e-3):
        super().__init__()
        self._action_value = tf.keras.Sequential(
            [tf.keras.layers.Dense(units=units, activation=activation) for _ in range(layers)] +
            [tf.keras.layers.Dense(units=1, activity_regularizer=tf.keras.regularizers.l2(
                output_regularization))])

    def __call__(self, observation):
        mu = tf.squeeze(self._action_value(observation), axis=2)
        return tfd.Independent(tfd.Normal(loc=mu, scale=1.0), 0)
