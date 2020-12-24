import tensorflow as tf
from tensorflow_probability import distributions as tfd

import mbpo.building_blocks as blocks
import mbpo.utils as utils


class WorldModel(tf.Module):
    # Following formulation of https://arxiv.org/pdf/1605.07571.pdf
    def __init__(self, observation_type, observation_shape, stochastic_size, deterministic_size,
                 units, seed, free_nats=3, kl_scale=1.0, observation_layers=3, reward_layers=1,
                 terminal_layers=1, activation=tf.nn.relu):
        super().__init__()
        self._cell = tf.keras.layers.GRUCell(deterministic_size)
        self._g = tf.keras.layers.GRU(deterministic_size, return_sequences=True, go_backwards=True)
        self._observation_encoder = blocks.encoder(observation_type, observation_shape,
                                                   observation_layers, units)
        self._observation_decoder = blocks.decoder(observation_type, observation_shape, 3, units)
        self._reward_decoder = blocks.DenseDecoder((), reward_layers, units, activation)
        self._terminal_decoder = blocks.DenseDecoder(
            (), terminal_layers, units, activation, 'bernoulli')
        self._posterior_decoder = tf.keras.Sequential(
            [tf.keras.layers.Dense(units, activation) for _ in range(1)] +
            [tf.keras.layers.Dense(2 * stochastic_size)])
        self._prior_decoder = tf.keras.Sequential(
            [tf.keras.layers.Dense(units, activation) for _ in range(1)] +
            [tf.keras.layers.Dense(2 * stochastic_size)])
        self._current_belief = None
        self._rng = tf.random.Generator.from_seed(seed)
        self._stochastic_size = stochastic_size
        self._free_nats = float(free_nats)
        self._kl_scale = kl_scale

    def __call__(self, belief, horizon, actions=None, actor=None):
        pass
        # TODO (yarden): make sure that action is one step *behind* observation
        # prior, prediction = self.predict(action, self._current_belief)
        # embeddings = self._observation_encoder(observation)
        # _, self._current_belief = self._correct(
        # embeddings, self._current_belief, prediction, prior)

    def predict(self, prev_action, prev_belief, prev_embeddings, seed=None):
        u_t = tf.concat([prev_embeddings, prev_action], -1)
        d_t, _ = self._cell(u_t, prev_belief['deterministic'])
        d_t_z_t_1 = tf.concat([prev_belief['stochastic'], d_t], -1)
        prior_mu, prior_stddev = tf.split(self._prior_decoder(d_t_z_t_1), 2, -1)
        prior_stddev = tf.math.softplus(prior_stddev) + 0.1
        prior = tfd.MultivariateNormalDiag(prior_mu, prior_stddev)
        z_t = prior.sample(seed=seed)
        return prior, {'stochastic': z_t, 'deterministic': d_t}

    def _correct(self, current_smoothed, prev_stochastic, prior_mu, seed=None):
        # Name alias to keep naming convention of https://arxiv.org/pdf/1605.07571.pdf
        z_t_1 = prev_stochastic
        a_t = current_smoothed
        # The posterior decoder predicts the residual mean from the prior, as suggested in
        # https://arxiv.org/pdf/1605.07571.pdf (eq. 12)
        posterior_mu_residual, posterior_stddev = tf.split(
            self._posterior_decoder(tf.concat([z_t_1, a_t], -1)), 2, -1)
        posterior_mu = posterior_mu_residual + tf.stop_gradient(prior_mu)
        posterior_stddev = tf.math.softplus(posterior_stddev) + 0.1
        posterior = tfd.MultivariateNormalDiag(posterior_mu, posterior_stddev)
        z_t = posterior.sample(seed=seed)
        return posterior, z_t

    def _smooth(self, deterministic_states, embeddings):
        cat = tf.concat([deterministic_states, embeddings], -1)
        return tf.reverse(self._g(cat), [1])

    def reset(self, batch_size, training=False):
        initial = {'stochastic': tf.zeros([batch_size, self._stochastic_size], tf.float32),
                   'deterministic': self._cell.get_initial_state(None, batch_size, tf.float32)}
        if not training:
            self._current_belief = initial
        return initial

    def compute_loss(self, batch):
        # TODO (yarden): why next observation? because to infer o_t we need a_t-1
        next_observations = batch['next_observation']
        actions = batch['action']
        horizon = tf.shape(next_observations)[1]
        embeddings = self._observation_encoder(next_observations)
        belief = self.reset(tf.shape(actions)[0], True)
        total_kl = 0.0
        stochastics = []
        deterministics = []
        # TODO (yarden): maybe should stop the gradient here for prev belief?
        # In trainig should we feed in the actual images or z_t_1 as imput????????
        # In prediction for sure z_t_1 because we don't want to generate image at every timestep
        for t in range(horizon):
            prior, belief_prediction = self.predict(actions[:, t], belief)
            posterior, belief = self._correct(embeddings[:, t], belief, belief_prediction, prior)
            stochastics.append(belief['stochastic'])
            deterministics.append(belief['deterministic'])
            kl = tf.reduce_mean(tfd.kl_divergence(posterior, prior))
            kl = self._kl_scale * tf.maximum(kl, self._free_nats)
            total_kl += kl
        features = tf.concat([tf.stack(stochastics, 1), tf.stack(deterministics, 1)], -1)
        rewards = batch['reward']
        log_p_rewards = tf.reduce_mean(self._reward_decoder(features).log_prob(rewards))
        terminals = batch['terminal']
        log_p_terminals = tf.reduce_mean(self._terminal_decoder(features).log_prob(terminals))
        log_p_observations = tf.reduce_mean(
            self._observation_decoder(features).log_prob(next_observations))
        loss = self._kl_scale * tf.maximum(
            total_kl, self._free_nats) - (log_p_rewards + log_p_observations + log_p_terminals)
        return loss, log_p_rewards, log_p_terminals, log_p_observations, total_kl


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
