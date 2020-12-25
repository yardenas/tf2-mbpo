import tensorflow as tf
import tensorflow_probability as tfp
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
        self._g = tf.keras.layers.GRU(deterministic_size, return_sequences=True)
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

    def __call__(self, prev_embeddings, prev_action, current_observation):
        seeds = tf.cast(self._rng.make_seeds(), tf.int32)
        predict_seed, correct_seed = tfp.random.split_seed(seeds[:, 0], 2, "update_belief")
        prior, belief = self.predict(prev_action, prev_embeddings,
                                     self._current_belief, predict_seed)
        current_embeddings = self._observation_encoder(current_observation[:, None, ...])
        smoothed = self._smooth(belief['deterministic'][:, None, ...],
                                current_embeddings[:, None, ...])
        _, z_t = self._correct(smoothed, self._current_belief['stochastic'],
                               prior.mean(), correct_seed)
        self._current_belief = {'stochastic': z_t, 'deterministic': belief['deterministic']}

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
        cat = tf.reverse(cat, [1])
        return tf.reverse(self._g(cat), [1])

    def reset(self, batch_size, training=False):
        initial = {'stochastic': tf.zeros([batch_size, self._stochastic_size], tf.float32),
                   'deterministic': self._cell.get_initial_state(None, batch_size, tf.float32)}
        if not training:
            self._current_belief = initial
        return initial

    def observe_sequence(self, batch):
        embeddings = self._observation_encoder(batch['observation'])
        prev_embeddings = embeddings[:, :-1]
        belief = self.reset(tf.shape(prev_embeddings)[0], True)
        horizon = tf.shape(prev_embeddings)[1]
        predictions = {'deterministics': tf.TensorArray(tf.float32, horizon),
                       'prior_mus': tf.TensorArray(tf.float32, horizon),
                       'prior_stddevs': tf.TensorArray(tf.float32, horizon)}
        seeds = tf.cast(self._rng.make_seeds(horizon), tf.int32)
        actions = batch['action']
        for t in tf.range(horizon):
            prior, belief = self.predict(actions[:, t], belief, prev_embeddings[:, t],
                                         seed=seeds[:, t])
            predictions['deterministics'] = predictions['deterministics'].write(
                t, belief['deterministic'])
            predictions['prior_mus'] = predictions['prior_mus'].write(t, prior.mean())
            predictions['prior_stddevs'] = predictions['prior_stddevs'].write(
                t, prior.stddev())
        stacked_predictions = {k: tf.transpose(v.stack(), [1, 0, 2]) for k, v in
                               predictions.items()}
        current_embeddings = embeddings[:, 1:]
        smoothed = self._smooth(stacked_predictions['deterministics'], current_embeddings)
        inferred = {'stochastics': tf.TensorArray(tf.float32, horizon),
                    'posterior_mus': tf.TensorArray(tf.float32, horizon),
                    'posterior_stddevs': tf.TensorArray(tf.float32, horizon)}
        seeds = tf.cast(self._rng.make_seeds(horizon), tf.int32)
        z_t = self.reset(tf.shape(actions)[0], True)['stochastic']
        for t in tf.range(horizon):
            posterior, z_t = self._correct(
                smoothed[:, t], z_t, stacked_predictions['prior_mus'][:, t], seeds[:, t])
            inferred['stochastics'] = inferred['stochastics'].write(t, z_t)
            inferred['posterior_mus'] = inferred['posterior_mus'].write(
                t, posterior.mean())
            inferred['posterior_stddevs'] = inferred['posterior_stddevs'].write(
                t, posterior.stddev())
        stacked_inferred = {k: tf.transpose(v.stack(), [1, 0, 2]) for k, v in inferred.items()}
        beliefs = {'stochastic': stacked_inferred['stochastics'],
                   'deterministic': stacked_predictions['deterministics']}
        prior = tfd.MultivariateNormalDiag(stacked_predictions['prior_mus'],
                                           stacked_predictions['prior_stddevs'])
        posterior = tfd.MultivariateNormalDiag(stacked_inferred['posterior_mus'],
                                               stacked_inferred['posterior_stddevs'])
        return beliefs, prior, posterior

    def inference_step(self, batch):
        beliefs, prior, posterior = self.observe_sequence(batch)
        kl = tf.reduce_mean(tf.reduce_sum(tfd.kl_divergence(posterior, prior), 1))
        features = tf.concat([beliefs['stochastic'],
                              beliefs['deterministic']], -1)
        reconstructed = self._observation_decoder(features)
        log_p_observations = tf.reduce_mean(tf.reduce_sum(
            reconstructed.log_prob(batch['observation'][:, 1:]), 1))
        horizon = tf.cast(tf.shape(batch['observation'])[1], tf.float32) - 1.0
        loss = -log_p_observations + self._kl_scale * tf.maximum(self._free_nats * horizon, kl)
        return loss, kl, log_p_observations, reconstructed

    def generate_sequence(self, initial_belief, horizon, actor=None, actions=None):
        sequence_features = tf.TensorArray(tf.float32, horizon)
        features = tf.concat([initial_belief['stochastic'], initial_belief['deterministic']], -1)
        belief = initial_belief
        seeds = tf.cast(self._rng.make_seeds(horizon), tf.int32)
        for t in tf.range(horizon):
            action = actor(tf.stop_gradient(features)
                           ).sample() if actions is None else actions[:, t]
            embeddings = tf.squeeze(self._observation_encoder(
                self._observation_decoder(features[:, None, ...]).mode()), 1)
            _, belief = self.predict(action, belief, embeddings, seeds[:, t])
            features = tf.concat([belief['stochastic'], belief['deterministic']], -1)
            sequence_features = sequence_features.write(t, features)
        return tf.transpose(sequence_features.stack(), [1, 0, 2])


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
