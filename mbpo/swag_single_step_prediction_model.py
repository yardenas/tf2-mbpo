import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd

import mbpo.building_blocks as blocks
import mbpo.world_models as world_models
from mbpo.swag import SWAG


class SwagSingleStepPredictionModel(world_models.BayesianWorldModel):
    def __init__(self, config, logger, observation_shape,
                 units, seed, reward_layers=1, terminal_layers=1):
        super().__init__(config, logger)
        self._optimizer = SWAG(
            tf.optimizers.Adam(
                config.model_learning_rate,
                clipnorm=config.grad_clip_norm),
            5000, 5)
        self._posterior_samples = config.posterior_samples
        self._shape = observation_shape
        self._type = config.observation_type
        self._input_shape = (None,) + observation_shape[:-1] + observation_shape[-1]
        self._encoder = [tf.keras.layers.Conv2D(32, 4, activation=tf.nn.relu, strides=2,
                                                input_shape=self._input_shape),
                         tf.keras.layers.Conv2D(2 * 32, 4, activation=tf.nn.relu, strides=2),
                         tf.keras.layers.Conv2D(4 * 32, 4, activation=tf.nn.relu, strides=2),
                         tf.keras.layers.Conv2D(8 * 32, 4, activation=tf.nn.relu, strides=2)]
        self._decoder = [
            tf.keras.layers.Conv2DTranspose(4 * 32, 5, activation=tf.nn.relu, strides=2),
            tf.keras.layers.Conv2DTranspose(2 * 32, 5, activation=tf.nn.relu, strides=2),
            tf.keras.layers.Conv2DTranspose(32, 6, activation=tf.nn.relu, strides=2),
            tf.keras.layers.Conv2DTranspose(self._shape[-1], 6, strides=2)]
        self._reward_decoder = blocks.DenseDecoder((), reward_layers, units, tf.nn.relu)
        self._terminal_decoder = blocks.DenseDecoder(
            (), terminal_layers, units, tf.nn.relu, 'bernoulli')
        self._rng = tf.random.Generator.from_seed(seed)

    def __call__(self, prev_embeddings, prev_action, current_observation):
        pass

    def _encode(self, observation):
        x = self._encoder[0](observation)
        x = skip_1 = self._encoder[1](x)
        x = self._encoder[2](x)
        x = skip_2 = self._encoder[2](x)
        x = self._encoder[3](x)
        return x, skip_1, skip_2

    def _decode(self, encoded, action, skip_1, skip_2):
        x = tf.concat([encoded, action], -1)
        x = self._decoder[0](x)
        x = tf.concat([x, skip_1], -1)
        x = self._decoder[1](x)
        x = tf.concat([x, skip_2], -1)
        x = self._decoder[2](x)
        x = self._decoder[3](x)
        return x

    def _forward(self, observation, action):
        encoded, skip_1, skip_2 = self._encode(observation)
        decoded = self._decode(encoded, action, skip_1, skip_2)
        if self._type == 'rgb_image':
            dist = tfd.Independent(tfd.Normal(decoded, 1.0), len(self._shape))
        elif self._type == 'binary_image':
            dist = tfd.Independent(tfd.Bernoulli(decoded, dtype=tf.float32), len(self._shape))
        else:
            raise RuntimeError("Output type is wrong.")
        cat = tf.concat([encoded, action], -1)
        return dist, self._reward_decoder(cat), self._terminal_decoder(cat)

    def _update_beliefs(self, prev_embeddings, prev_action, current_observation):
        pass

    def _generate_sequences_posterior(self, initial_belief, horizon,
                                      seed, actor, actions, log_sequences):
        samples_reconstructed = []
        for _ in range(self._posterior_samples):
            self._optimizer.sample_and_assign(1.0, self._model.trainable_variables)
            sequence = self._unroll_sequence(initial_belief, horizon, actor=actor, actions=actions)
            samples_reconstructed.append(sequence)
        return None, tf.stack(samples_reconstructed, 0)

    def _reconstruct_sequences_posterior(self, batch):
        samples_reconstructed = []
        next_observations = batch['observation'][1:]
        initial_observation = batch['observation'][0]
        actions = batch['action']
        for i in range(self._posterior_samples):
            self._optimizer.sample_and_assign(1.0, self.trainable_variables)
            sequence = self._unroll_sequence(initial_observation,
                                             tf.shape(next_observations)[1],
                                             next_observations, actions=actions)
            samples_reconstructed.append(sequence)
            mse = tf.reduce_mean(tf.math.squared_difference(next_observations, sequence))
            self._logger['test_observation_' + str(i)].update_state(mse)
        stacked_all = tf.stack(samples_reconstructed, 1)
        return stacked_all, {'stochastic': stacked_all, 'deterministic': stacked_all}

    def _unroll_sequence(self, initial_observation, horizon, next_observations=None, actor=None,
                         actions=None):
        sequence_decoded = []
        observation = initial_observation
        seeds = tf.cast(self._rng.make_seeds(horizon), tf.int32)
        for t in range(horizon):
            predict_seed, action_seed = tfp.random.split_seed(seeds[:, t], 2, "unroll_sequence")
            action = actor(tf.stop_gradient(observation)
                           ).sample(seed=action_seed) if actions is None else actions[:, t]
            observation_dist, _, _ = self._forward(observation, action)
            prediction = observation_dist.mode()
            observation = prediction if \
                next_observations is None else next_observations[:, t]
            sequence_decoded.append(prediction)
        stacked_sequence = tf.stack(sequence_decoded, 1)
        return stacked_sequence

    def _training_step(self, batch, log_sequences):
        # We should actually shuffle the data to ensure that it is i.i.d but...(?)
        observations = tf.reshape(batch['observation'][:-1], self._input_shape)
        next_observations = tf.reshape(batch['observation'][1:], self._input_shape)
        actions = tf.reshape(batch['action'], self._cat_images)
        rewards = tf.reshape(batch['reward'], self._cat_images)
        terminals = tf.reshape(batch['terminal'], self._cat_images)
        with tf.GradientTape() as model_tape:
            next_observation, reward, terminal = self._forward(observations, actions)
            log_p_observations = next_observation.log_prob(next_observations)
            log_p_rewards = reward.log_prob(rewards)
            log_p_terminals = terminal.log_prob(terminals)
            loss = -log_p_observations - log_p_rewards - log_p_terminals
        grads = model_tape.gradient(loss, self.trainable_variables)
        self._optimizer.apply_gradients(zip(grads, self.trainable_variables))
        self._logger['observation_log_p'].update_state(-log_p_observations)
        self._logger['rewards_log_p'].update_state(-log_p_rewards)
        self._logger['terminals_log_p'].update_state(-log_p_terminals)
        self._logger['world_model_loss'].update_state(loss)
        self._logger['world_model_grads'].update_state(tf.linalg.global_norm(grads))
        sequences = tf.reshape(next_observation.mode(),
                               tf.shape(next_observations)) if log_sequences else None
        return None, sequences
