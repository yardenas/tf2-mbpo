import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd

import mbpo.utils as utils


class WorldModel(tf.Module):
    def __init__(self, dynamics_size, cell_size, dynamics_layers, units, seed,
                 reward_layers=1, terminal_layers=1, min_stddev=1e-4,
                 activation=tf.nn.relu):
        super().__init__()
        self._min_stddev = min_stddev
        self._cell = tf.keras.layers.GRUCell(cell_size)
        self._hidden_encoder = tf.keras.layers.Dense(units, activation=activation)
        self._hidden_decoder = tf.keras.layers.Dense(units, activation=activation)
        self._observation_decoder = tf.keras.Sequential(
            [tf.keras.layers.Dense(units=units, activation=activation) for _ in
             range(dynamics_layers)] + [tf.keras.layers.Dense(2 * dynamics_size)])
        # Assuming reward with a unit standard deviation.
        self._reward_decoder = tf.keras.Sequential(
            [tf.keras.layers.Dense(units=units, activation=activation) for _ in
             range(reward_layers)] + [tf.keras.layers.Dense(1)])
        self._terminal_logit_decoder = tf.keras.Sequential(
            [tf.keras.layers.Dense(units=units, activation=activation) for _ in
             range(terminal_layers)] + [tf.keras.layers.Dense(1)])
        self._rng = tf.random.Generator.from_seed(seed)

    def __call__(self, observation, horizon, actions=None, actor=None, next_observations=None):
        return self._imagine_rollouts(observation, horizon, actions, actor, next_observations)

    def _step(self, action, observation, deterministic):
        cat = tf.concat([observation, action], -1)
        cat = self._hidden_encoder(cat)
        x, deterministic = self._cell(cat, deterministic)
        hidden_states = self._hidden_decoder(x)
        next_observation_mu, next_observation_stddev = tf.split(
            self._observation_decoder(hidden_states), 2, -1)
        # TODO (yarde): maybe actor acts on latent space?
        next_observation_mu += tf.stop_gradient(observation)
        next_observation_stddev = tf.math.softplus(next_observation_stddev) + self._min_stddev
        return next_observation_mu, next_observation_stddev, deterministic

    def _imagine_rollouts(self, observations, horizon,
                          actions, actor, next_observations):
        state = {'stochastic': tf.TensorArray(tf.float32, horizon),
                 'deterministic': tf.TensorArray(tf.float32, horizon),
                 'next_obs_mu': tf.TensorArray(tf.float32, horizon),
                 'next_obs_stddev': tf.TensorArray(tf.float32, horizon)}
        seeds = tf.cast(self._rng.make_seeds(horizon), tf.int32)
        deterministic = self._cell.get_initial_state(batch_size=tf.shape(observations)[0],
                                                     dtype=tf.float32)
        sequence_mode = next_observations is not None
        observation = observations[:, 0, ...] if sequence_mode else observations
        for k in tf.range(horizon):
            action_seed, obs_seed = tfp.random.split_seed(seeds[:, k], 2, "imagine_rollouts")
            action = actor(tf.stop_gradient(observation)).sample(
                seed=action_seed) if actions is None else actions[:, k, ...]
            next_observation_mu, next_observation_stddev, deterministic = self._step(
                observation, action, deterministic)
            if sequence_mode:
                observation = next_observations[:, k, ...]
            else:
                observation = tfd.MultivariateNormalDiag(
                    next_observation_mu, next_observation_stddev).sample(seed=obs_seed)
            state['stochastic'] = state['stochastic'].write(k, observation)
            state['deterministic'] = state['deterministic'].write(k, deterministic)
            state['next_obs_mu'] = state['next_obs_mu'].write(k, next_observation_mu)
            state['next_obs_stddev'] = state['next_obs_stddev'].write(k, next_observation_stddev)
        state = {k: tf.transpose(v.stack(), [1, 0, 2]) for k, v in state.items()}
        next_obs_dist = tfd.Independent(tfd.MultivariateNormalDiag(
            state['next_obs_mu'], state['next_obs_stddev']), 0)
        features = tf.concat([state['stochastic'], state['deterministic']], -1)
        reward_dist = tfd.Independent(tfd.Normal(
            tf.squeeze(self._reward_decoder(features), -1), 1.0), 0)
        terminal_dist = tfd.Independent(tfd.Bernoulli(
            logits=tf.squeeze(self._terminal_logit_decoder(features), -1), dtype=tf.float32), 0)
        return {'next_observation': next_obs_dist,
                'reward': reward_dist,
                'terminal': terminal_dist}


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
