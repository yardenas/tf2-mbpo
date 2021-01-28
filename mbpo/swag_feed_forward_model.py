import tensorflow as tf
from tensorflow_probability import distributions as tfd

import mbpo.building_blocks as blocks
import mbpo.world_models as world_models
from mbpo.swag import SWAG


class SwagFeedForwardModel(world_models.BayesianWorldModel):
    def __init__(self, config, logger, observation_shape,
                 reward_layers=1, terminal_layers=1):
        super().__init__(config, logger)
        self._optimizer = SWAG(
            tf.optimizers.Adam(
                config.model_learning_rate,
                clipnorm=config.grad_clip_norm),
            2000, 5)
        self._n_step_loss = config.n_step_loss
        self._posterior_samples = config.posterior_samples
        self._shape = observation_shape
        self._type = config.observation_type
        self._stochastic_size = config.stochastic_size
        self._encoder = blocks.encoder(config.observation_type,
                                       observation_shape, 3, config.units)
        self._posterior_decoder = tf.keras.Sequential(
            [tf.keras.layers.Dense(config.units, tf.nn.relu) for _ in range(1)] +
            [tf.keras.layers.Dense(2 * config.stochastic_size)])
        self._prior_decoder = tf.keras.Sequential(
            [tf.keras.layers.Dense(config.units, tf.nn.relu) for _ in range(1)] +
            [tf.keras.layers.Dense(2 * config.stochastic_size)])
        observation_type = 'image_logits' if config.observation_type in \
                                             ['rgb_image', 'binary_image'] else 'dense_logits'
        self._decoder = blocks.decoder(observation_type, observation_shape,
                                       3, config.units)
        self._reward_decoder = blocks.DenseDecoder((), reward_layers, config.units, tf.nn.relu)
        self._terminal_decoder = blocks.DenseDecoder(
            (), terminal_layers, config.units, tf.nn.relu, 'bernoulli')

    def _update_beliefs(self, prev_action, current_observation):
        pass

    @tf.function
    def _generate_sequences_posterior(self, initial_belief, horizon, actor, actions, log_sequences):
        samples_reconstructed = []
        for _ in range(self._posterior_samples):
            self._optimizer.sample_and_assign(1.0, self.trainable_variables)
            _, _, decoded, z = self._unroll_sequence(initial_belief['deterministic'],
                                                     horizon, actions=actions)
            next_observation, _, _ = self._to_distributions(z, decoded, actions, False)
            samples_reconstructed.append(next_observation.mean())
        stacked_all = tf.stack(samples_reconstructed, 0)
        return {'stochastic': stacked_all, 'deterministic': stacked_all}, stacked_all

    @tf.function
    def _reconstruct_sequences_posterior(self, batch):
        samples_reconstructed = []
        next_observations = batch['observation'][:, 1:]
        initial_observation = batch['observation'][:, 0]
        actions = batch['action']
        for i in range(self._posterior_samples):
            self._optimizer.sample_and_assign(1.0, self.trainable_variables)
            _, _, decoded, z = self._unroll_sequence(initial_observation,
                                                     tf.shape(next_observations)[1],
                                                     actions,
                                                     next_observations)
            next_observation, _, _ = self._to_distributions(z, decoded, actions, False)
            samples_reconstructed.append(next_observation.mean())
        stacked_all = tf.stack(samples_reconstructed, 0)
        return stacked_all, {'stochastic': stacked_all, 'deterministic': stacked_all}

    def _unroll_sequence(self, initial_observation, horizon, actions, next_observations=None,
                         stop_gradient=True):
        inferred = {'prior_mus': tf.TensorArray(tf.float32, horizon),
                    'prior_stddevs': tf.TensorArray(tf.float32, horizon),
                    'posterior_mus': tf.TensorArray(tf.float32, horizon),
                    'posterior_stddevs': tf.TensorArray(tf.float32, horizon),
                    'decoded': tf.TensorArray(tf.float32, horizon),
                    'stochastics': tf.TensorArray(tf.float32, horizon)}
        observation = initial_observation
        for t in range(horizon):
            action = actions[:, t]
            if next_observations is None:
                prior, decoded, z = self._generation_step(observation, action)
                observation = self._observation_dist(decoded).mode()
            else:
                prior, posterior, decoded, z = self._inference_step(
                    observation,
                    next_observations[:, t], action)
                inferred['posterior_mus'] = inferred['posterior_mus'].write(t, posterior.mean())
                inferred['posterior_stddevs'] = inferred['posterior_stddevs'].write(
                    t, posterior.stddev())
                observation = next_observations[:, t]
            observation = tf.stop_gradient(observation) if stop_gradient else observation
            inferred['decoded'] = inferred['decoded'].write(t, decoded)
            inferred['stochastics'] = inferred['stochastics'].write(t, z)
            inferred['prior_mus'] = inferred['prior_mus'].write(t, prior.mean())
            inferred['prior_stddevs'] = inferred['prior_stddevs'].write(t, prior.stddev())
        stacked = {k: tf.transpose(v.stack(),
                                   [1, 0, 2]) for k, v in inferred.items()
                   if k not in ['decoded', 'posterior_mus', 'posterior_stddevs']}
        stacked['decoded'] = tf.transpose(inferred['decoded'].stack(), [1, 0, 2, 3, 4])
        prior = tfd.MultivariateNormalDiag(stacked['prior_mus'],
                                           stacked['prior_stddevs'])
        if next_observations is not None:
            stacked['posterior_mus'] = tf.transpose(inferred['posterior_mus'].stack(), [1, 0, 2])
            stacked['posterior_stddevs'] = \
                tf.transpose(inferred['posterior_stddevs'].stack(), [1, 0, 2])
            posterior = tfd.MultivariateNormalDiag(
                stacked['posterior_mus'],
                stacked['posterior_stddevs'])
        else:
            posterior = None
        return prior, posterior, stacked['decoded'], stacked['stochastics']

    def _inference_step(self, observation, next_observation, action):
        obs_cat = tf.concat([observation[:, None, ...], next_observation[:, None, ...]], 1)
        obs_embd, next_obs_embd = tf.split(self._encoder(obs_cat), 2, 1)
        inputs = tf.concat([obs_embd, action], -1)
        x = self._posterior_decoder(tf.concat([inputs, next_obs_embd], -1))
        posterior_mean, posterior_stddev = tf.split(x, 2, -1)
        posterior_stddev = tf.math.softplus(posterior_stddev)
        posterior = tfd.MultivariateNormalDiag(posterior_mean, posterior_stddev)
        z = posterior.sample()
        decoded = self._decoder(z)
        prior_mean, prior_stddev = tf.split(self._prior_decoder(inputs), 2, -1)
        prior_stddev = tf.math.softplus(prior_stddev)
        prior = tfd.MultivariateNormalDiag(prior_mean, prior_stddev)
        return prior, posterior, decoded, z

    def _generation_step(self, observation, action):
        obs_embd = tf.squeeze(self._encoder(observation[:, None, ...]))
        inputs = tf.concat([obs_embd, action], -1)
        prior_mean, prior_stddev = tf.split(self._prior_decoder(inputs), 2, -1)
        prior_stddev = tf.math.softplus(prior_stddev)
        prior = tfd.MultivariateNormalDiag(prior_mean, prior_stddev)
        z = prior.sample()
        decoded = self._decoder(z)
        return prior, decoded, z

    def _to_distributions(self, stochastics, decoded, action, with_rewards_terminals=True):
        observation_dist = self._observation_dist(decoded)
        cat = tf.concat([stochastics, action], -1)
        if with_rewards_terminals:
            reward_dist = self._reward_decoder(cat)
            terminal_dist = self._terminal_decoder(cat)
        else:
            # A better idea is a flag that reshapes the rewards and terminals to sequence if needed
            # instead of just ignoring them.
            reward_dist = None
            terminal_dist = None
        return observation_dist, reward_dist, terminal_dist

    def _observation_dist(self, logits):
        if self._type == 'rgb_image':
            observation_dist = tfd.Independent(tfd.Normal(logits, 1.0), len(self._shape))
        elif self._type == 'binary_image':
            observation_dist = tfd.Independent(tfd.Bernoulli(
                logits=logits, dtype=tf.float32), len(self._shape))
        else:
            raise RuntimeError("Output type is wrong.")
        return observation_dist

    @tf.function
    def _training_step(self, batch, log_sequences):
        observations, next_observations, actions, rewards, terminals = \
            self._make_training_step_data(batch)
        with tf.GradientTape() as model_tape:
            if self._n_step_loss:
                prior, posterior, decoded, z = self._unroll_sequence(
                    observations[:, 0], tf.shape(next_observations)[1],
                    actions=actions, next_observations=next_observations)
            else:
                prior, posterior, decoded, z = self._inference_step(
                    observations, next_observations, actions)
            next_observation, reward, terminal = self._to_distributions(z, decoded, actions)
            log_p_observations = tf.reduce_mean(next_observation.log_prob(next_observations))
            log_p_rewards = tf.reduce_mean(reward.log_prob(rewards))
            log_p_terminals = tf.reduce_mean(terminal.log_prob(terminals))
            kl = tf.reduce_mean(tfd.kl_divergence(posterior, prior))
            loss = tf.maximum(3.0, kl) - log_p_observations - log_p_rewards - log_p_terminals
        grads = model_tape.gradient(loss, self.trainable_variables)
        self._optimizer.apply_gradients(zip(grads, self.trainable_variables))
        self._logger['observation_log_p'].update_state(-log_p_observations)
        self._logger['rewards_log_p'].update_state(-log_p_rewards)
        self._logger['terminals_log_p'].update_state(-log_p_terminals)
        self._logger['world_model_loss'].update_state(loss)
        self._logger['kl'].update_state(kl)
        self._logger['world_model_grads'].update_state(tf.linalg.global_norm(grads))
        sequences = tf.reshape(next_observation.mean(),
                               tf.shape(batch['observation'][:, 1:])) if log_sequences else None
        return None, sequences

    def _make_training_step_data(self, batch):
        if self._n_step_loss:
            observations = batch['observation'][:, :-1]
            next_observations = batch['observation'][:, 1:]
            actions = batch['action']
            rewards = batch['reward']
            terminals = batch['terminal']
        else:
            # We should actually shuffle the data to ensure that it is i.i.d but...(?)
            observations = tf.reshape(batch['observation'][:, :-1], (-1,) + self._shape)
            next_observations = tf.reshape(batch['observation'][:, 1:], (-1,) + self._shape)
            actions = tf.reshape(batch['action'], [-1, tf.shape(batch['action'])[-1]])
            rewards = tf.reshape(batch['reward'], [-1, ])
            terminals = tf.reshape(batch['terminal'], [-1, ])
        return observations, next_observations, actions, rewards, terminals
