import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd

import mbpo.utils as utils


class WorldModel(tf.Module):
    def __init__(self, observation_shape, stochastic_size, deterministic_size, units, seed,
                 free_nats=3, kl_scale=0.8, observation_layers=3, reward_layers=1,
                 terminal_layers=1, min_stddev=1e-4, activation=tf.nn.relu):
        super().__init__()
        self._min_stddev = min_stddev
        self._cell = tf.keras.layers.GRUCell(deterministic_size)
        self._predict_encoder = tf.keras.layers.Dense(units, activation=activation)
        self._correct_encoder = tf.keras.layers.Dense(units, activation)
        self._observation_encoder = tf.keras.Sequential(
            [tf.keras.layers.Dense(units, activation) for _ in range(3)])
        self._observation_decoder = DenseDecoder(observation_shape, observation_layers, activation)
        self._reward_decoder = DenseDecoder((), reward_layers, units, activation)
        self._terminal_decoder = DenseDecoder((), terminal_layers, units, activation, 'bernoulli')
        self._posterior_decoder = tf.keras.Sequential(
            [tf.keras.layers.Dense(units, activation) for _ in range(2)] +
            [tf.keras.layers.Dense(2 * stochastic_size)])
        self._prior_decoder = tf.keras.Sequential(
            [tf.keras.layers.Dense(units, activation) for _ in range(2)] +
            [tf.keras.layers.Dense(2 * stochastic_size)])
        self._current_belief = None
        self._rng = tf.random.Generator.from_seed(seed)
        self._stochastic_size = stochastic_size
        self._free_nats = free_nats
        self._kl_scale = kl_scale

    def __call__(self, belief, horizon, actions=None, actor=None):
        seeds = tf.cast(self._rng.make_seeds(horizon), tf.int32)
        features = tf.concat([belief['stochastic'], belief['deterministic']], -1)
        all_features = [features]
        ##### TODO (yardeb))!!!!! think if belief and actions are of the same time step@@@
        for t in range(horizon):
            action_seed, obs_seed = tfp.random.split_seed(seeds[:, t], 2, "imagine_rollouts")
            action = actor(tf.stop_gradient(features)).sample(
                seed=action_seed) if actions is None else actions[:, t, ...]
            prior, belief = self.predict(action, belief)
            features.append(tf.concat(belief['stochastic'], belief['deterministic'], -1))
        return tf.stack(all_features, 1)

    def predict(self, prev_action, prev_belief):
        cat_stochastic = tf.concat([prev_belief['deterministic'], prev_action], -1)
        x = self._predict_encoder(cat_stochastic)
        prior_mu, prior_stddev = tf.split(self._prior_decoder(x), 2, -1)
        prior_stddev = tf.math.softplus(prior_stddev) + 0.1
        prior = tfd.MultivariateNormalDiag(prior_mu, prior_stddev)
        prior_latent_stochastic = prior.sample()
        cat_deterministic = tf.concat([prior_latent_stochastic, prev_action], -1)
        prior_latent_deterministic, _ = self._cell(cat_deterministic, prev_belief['deterministic'])
        return prior, {'stochastic': prior_latent_stochastic,
                       'deterministic': prior_latent_deterministic}

    def _correct(self, embeddings, prev_action, prev_belief):
        x = tf.concat([embeddings, prev_belief['deterministic'], prev_action])
        x = self._correct_encoder(x)
        posterior_mu, posterior_stddev = tf.split(self._posterior_decoder(x), 2, -1)
        posterior_stddev = tf.math.softplus(posterior_stddev) + 0.1
        posterior = tfd.MultivariateNormalDiag(posterior_mu, posterior_stddev)
        posterior_latent_stochastic = posterior.sample()
        posterior_latent_deterministic, _ = self._cell(
            tf.concat([posterior_latent_stochastic, prev_action], -1), prev_belief['deterministic'])
        return posterior, {'stochastic': posterior_latent_stochastic,
                           'deterministic': posterior_latent_deterministic}

    def update_belief(self, observation, action):
        embeddings = self._observation_encoder(observation)
        posterior, self._current_belief = self._correct(embeddings, action, self._current_belief)

    def reset(self, batch_size, training=False):
        initial = {'stochastic': tf.zeros([batch_size, self._stochastic_size], tf.float32),
                   'deterministic': self._cell.get_initial_state(None, batch_size, tf.float32)}
        if not training:
            self._current_belief = initial
        return initial

    def compute_loss(self, batch):
        next_observations = batch['next_observation']
        actions = batch['action']
        horizon = tf.shape(next_observations)[1]
        embeddings = self._observation_encoder(next_observations)
        belief = self.reset(tf.shape(actions)[0])
        loss = 0.0
        stochastics = []
        deterministics = []
        for t in range(horizon):
            prior, _ = self.predict(actions[:, t], belief)
            posterior, belief = self._correct(embeddings[:, t], actions[:, t], belief)
            stochastics.append(belief['stochastic'])
            deterministics.append(belief['deterministic'])
            kl = tf.reduce_mean(tfd.kl_divergence(posterior, prior))
            kl = self._kl_scale * tf.maximum(kl, self._free_nats)
            loss += kl
        total_kl = loss
        features = tf.concat([tf.stack(stochastics, 1), tf.stack(deterministics, 1)], -1)
        rewards = batch['reward']
        log_p_rewards = tf.reduce_mean(self._reward_decoder(features).log_prob(rewards))
        terminals = batch['terminal']
        log_p_terminals = tf.reduce_mean(self._terminal_decoder(features).log_prob(terminals))
        log_p_observations = tf.reduce_mean(
            self._observation_decoder(features).log_prob(next_observations))
        loss -= (log_p_rewards + log_p_observations + log_p_terminals)
        return loss, log_p_rewards, log_p_terminals, log_p_observations, total_kl


class DenseEncoder(tf.Module):
    def __init__(self, layers, units, activation=tf.nn.relu):
        super().__init__()
        self._layers = tf.keras.Sequential(
            [tf.keras.layers.Dense(units, activation) for _ in range(layers)])

    def __call__(self, inputs):
        return self._layers(inputs)


class DenseDecoder(tf.Module):
    def __init__(self, shape, layers, units, activation=tf.nn.relu, dist='normal'):
        super(DenseDecoder, self).__init__()
        self._layers = tf.keras.Sequential(
            [tf.keras.layers.Dense(units, activation) for _ in range(layers)] +
            [tf.keras.layers.Dense(len(shape))])
        self._shape = shape
        self._dist = dist

    def __call__(self, inputs):
        x = self._layers(inputs)
        x = tf.reshape(x, tf.shape(inputs)[:-1] + self._shape)
        if self._dist == 'normal':
            return tfd.Independent(tfd.MultivariateNormalDiag(x, 1.0), len(self._shape))
        elif self._dist == 'bernoulli':
            return tfd.Independent(tfd.Bernoulli(x, dtype=tf.float32), len(self._shape))


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
