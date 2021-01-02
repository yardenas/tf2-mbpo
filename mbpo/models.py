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
        self._f = tf.keras.layers.GRUCell(deterministic_size)
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
        pass

    def _predict(self, prev_stochastic, prev_action, prev_deterministic, seed):
        d_t_z_t_1 = tf.concat([prev_action, prev_stochastic], -1)
        d_t, _ = self._f(d_t_z_t_1, prev_deterministic)
        prior_mu, prior_stddev = tf.split(self._prior_decoder(d_t), 2, -1)
        prior_stddev = tf.math.softplus(prior_stddev)
        prior = tfd.MultivariateNormalDiag(prior_mu, prior_stddev)
        z_t = prior.sample(seed=seed)
        return prior, z_t, d_t

    def _correct(self, prev_deterministic, current_embeddings, seed=None):
        # TODO (yarden): use also prev_stochastic here
        posterior_mu, posterior_stddev = tf.split(
            self._posterior_decoder(tf.concat([prev_deterministic, current_embeddings], -1)), 2, -1)
        posterior_stddev = tf.math.softplus(posterior_stddev)
        posterior = tfd.MultivariateNormalDiag(posterior_mu, posterior_stddev)
        z_t = posterior.sample(seed=seed)
        return posterior, z_t

    def reset(self, batch_size, training=False):
        initial = {'stochastic': tf.zeros([batch_size, self._stochastic_size], tf.float32),
                   'deterministic': self._f.get_initial_state(None, batch_size, tf.float32)}
        if not training:
            self._current_belief = initial
        return initial

    def _observe_sequence(self, batch):
        embeddings = self._observation_encoder(batch['observation'])
        prev_embeddings = embeddings[:, :-1]
        horizon = tf.shape(prev_embeddings)[1]
        actions = batch['action']
        current_embeddings = embeddings[:, 1:]
        inferred = {'stochastics': tf.TensorArray(tf.float32, horizon),
                    'deterministics': tf.TensorArray(tf.float32, horizon),
                    'prior_mus': tf.TensorArray(tf.float32, horizon),
                    'prior_stddevs': tf.TensorArray(tf.float32, horizon),
                    'posterior_mus': tf.TensorArray(tf.float32, horizon),
                    'posterior_stddevs': tf.TensorArray(tf.float32, horizon)}
        seeds = tf.cast(self._rng.make_seeds(horizon), tf.int32)
        state = self.reset(tf.shape(actions)[0])
        d_t = state['deterministic']
        z_t = state['stochastic']
        for t in tf.range(horizon):
            predict_seed, correct_seed = tfp.random.split_seed(seeds[:, t], 2, "observe")
            prior, _, d_t = self._predict(z_t, actions[:, t], d_t, predict_seed)
            posterior, z_t = self._correct(
                d_t, current_embeddings[:, t], correct_seed)
            inferred['stochastics'] = inferred['stochastics'].write(t, z_t)
            inferred['deterministics'] = inferred['deterministics'].write(t, d_t)
            inferred['prior_mus'] = inferred['prior_mus'].write(t, prior.mean())
            inferred['prior_stddevs'] = inferred['prior_stddevs'].write(t, prior.stddev())
            inferred['posterior_mus'] = inferred['posterior_mus'].write(
                t, posterior.mean())
            inferred['posterior_stddevs'] = inferred['posterior_stddevs'].write(
                t, posterior.stddev())
        stacked_inferred = {k: tf.transpose(v.stack(), [1, 0, 2]) for k, v in inferred.items()}
        beliefs = {'stochastic': stacked_inferred['stochastics'],
                   'deterministic': stacked_inferred['deterministics']}
        prior = tfd.MultivariateNormalDiag(stacked_inferred['prior_mus'],
                                           stacked_inferred['prior_stddevs'])
        posterior = tfd.MultivariateNormalDiag(stacked_inferred['posterior_mus'],
                                               stacked_inferred['posterior_stddevs'])
        return beliefs, prior, posterior

    def inference_step(self, batch):
        beliefs, prior, posterior = self._observe_sequence(batch)
        kl = tf.reduce_mean(tfd.kl_divergence(posterior, prior))
        features = tf.concat([beliefs['stochastic'],
                              beliefs['deterministic']], -1)
        reconstructed = self._observation_decoder(features)
        log_p_observations = tf.reduce_mean(reconstructed.log_prob(batch['observation'][:, 1:]))
        log_p_rewards = tf.reduce_mean(
            self._reward_decoder(features).log_prob(batch['reward']))
        log_p_terminals = tf.reduce_mean(
            self._terminal_decoder(features).log_prob(batch['terminal']))
        loss = self._kl_scale * tf.maximum(
            self._free_nats, kl) - log_p_observations - log_p_rewards - log_p_terminals
        return loss, kl, log_p_observations, log_p_rewards, log_p_terminals, reconstructed, beliefs

    def generate_sequence(self, initial_belief, horizon, actor=None, actions=None,
                          log_sequences=False):
        sequence_features = tf.TensorArray(tf.float32, horizon)
        sequence_decoded = []
        features = tf.concat([initial_belief['stochastic'], initial_belief['deterministic']], -1)
        seeds = tf.cast(self._rng.make_seeds(horizon), tf.int32)
        d_t = initial_belief['deterministic']
        z_t = initial_belief['stochastic']
        for t in range(horizon):
            predict_seed, action_seed = tfp.random.split_seed(seeds[:, t], 2, "generate")
            action = actor(tf.stop_gradient(features)).sample(
                seed=action_seed) if actions is None else actions[:, t]
            _, z_t, d_t = self._predict(z_t, action, d_t, predict_seed)
            features = tf.concat([z_t, d_t], -1)
            sequence_features = sequence_features.write(t, features)
        stacked_features = tf.transpose(sequence_features.stack(), [1, 0, 2])
        stacked_sequence = self._observation_decoder(
            stacked_features).mode() if log_sequences else None
        return stacked_features, self._reward_decoder(
            stacked_features).mode(), self._terminal_decoder(
            stacked_features).mean(), stacked_sequence


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
