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
        self._encoder = [tf.keras.layers.Conv2D(32, 4, activation=tf.nn.relu, strides=2,
                                                input_shape=(None,) + observation_shape),
                         tf.keras.layers.Conv2D(2 * 32, 4, activation=tf.nn.relu, strides=2),
                         tf.keras.layers.Conv2D(4 * 32, 4, activation=tf.nn.relu, strides=2),
                         tf.keras.layers.Conv2D(8 * 32, 4, activation=tf.nn.relu, strides=2),
                         tf.keras.layers.GlobalAveragePooling2D()]
        self._decoder = [
            tf.keras.layers.Dense(32 * 32),
            tf.keras.layers.Conv2DTranspose(4 * 32, 2, activation=tf.nn.relu, strides=2),
            tf.keras.layers.Conv2DTranspose(2 * 32, 5, activation=tf.nn.relu, strides=2),
            tf.keras.layers.Conv2DTranspose(32, 2, activation=tf.nn.relu, strides=2),
            tf.keras.layers.Conv2DTranspose(16, 4, activation=tf.nn.relu, strides=2),
            tf.keras.layers.Conv2DTranspose(self._shape[-1], 6, strides=2)]
        self._reward_decoder = blocks.DenseDecoder((), reward_layers, config.units, tf.nn.relu)
        self._terminal_decoder = blocks.DenseDecoder(
            (), terminal_layers, config.units, tf.nn.relu, 'bernoulli')

    def _encode(self, observation):
        x = self._encoder[0](observation)
        x = skip_1 = self._encoder[1](x)
        x = self._encoder[2](x)
        x = skip_2 = self._encoder[3](x)
        x = self._encoder[4](x)
        return x, skip_1, skip_2

    def _decode(self, encoded, action, skip_1, skip_2):
        x = tf.concat([encoded, action], -1)
        x = self._decoder[0](x)
        x = tf.reshape(x, [-1, 1, 1, 32 * 32])
        x = self._decoder[1](x)
        x = tf.concat([x, skip_2], -1)
        x = self._decoder[2](x)
        x = self._decoder[3](x)
        x = tf.concat([x, skip_1], -1)
        x = self._decoder[4](x)
        x = self._decoder[5](x)
        return x

    def _forward(self, observation, action):
        encoded, skip_1, skip_2 = self._encode(observation)
        decoded_res = self._decode(encoded, action, skip_1, skip_2)
        decoded = decoded_res + tf.stop_gradient(observation)
        return encoded, decoded

    def _to_distributions(self, encoded, decoded, action, with_rewards_terminals=True):
        if self._type == 'rgb_image':
            observation_dist = tfd.Independent(tfd.Normal(decoded, 1.0), len(self._shape))
        elif self._type == 'binary_image':
            observation_dist = tfd.Independent(tfd.Bernoulli(
                decoded, dtype=tf.float32), len(self._shape))
        else:
            raise RuntimeError("Output type is wrong.")
        if with_rewards_terminals:
            cat = tf.concat([encoded, action], -1)
            reward_dist = self._reward_decoder(cat)
            terminal_dist = self._terminal_decoder(cat)
        else:
            reward_dist = None
            terminal_dist = None
        return observation_dist, reward_dist, terminal_dist

    def _update_beliefs(self, prev_action, current_observation):
        pass

    @tf.function
    def _generate_sequences_posterior(self, initial_belief, horizon, actor, actions, log_sequences):
        samples_reconstructed = []
        for _ in range(self._posterior_samples):
            self._optimizer.sample_and_assign(1.0, self.trainable_variables)
            _, _, sequence = self._unroll_sequence(initial_belief['deterministic'], horizon,
                                                   actions=actions)
            samples_reconstructed.append(sequence)
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
            _, _, sequence = self._unroll_sequence(initial_observation,
                                                   tf.shape(next_observations)[1],
                                                   actions,
                                                   next_observations)
            samples_reconstructed.append(sequence)
            mse = tf.reduce_mean(tf.math.squared_difference(next_observations, sequence))
            self._logger['test_observation_' + str(i)].update_state(mse)
        stacked_all = tf.stack(samples_reconstructed, 0)
        return stacked_all, {'stochastic': stacked_all, 'deterministic': stacked_all}

    @tf.function
    def _unroll_sequence(self, initial_observation, horizon, actions, next_observations=None,
                         stop_gradient=True):
        encoded_sequece = tf.TensorArray(tf.float32, horizon)
        decoded_sequece = tf.TensorArray(tf.float32, horizon)
        sequence_observations = tf.TensorArray(tf.float32, horizon)
        observation = initial_observation
        for t in range(horizon):
            action = actions[:, t]
            encoded, decoded = self._forward(observation, action)
            observation_dist, _, _ = self._to_distributions(
                encoded, decoded, action, False)
            prediction = observation_dist.mode()
            observation = prediction if \
                next_observations is None else next_observations[:, t]
            observation = tf.stop_gradient(observation) if stop_gradient else observation
            encoded_sequece = encoded_sequece.write(t, encoded)
            decoded_sequece = decoded_sequece.write(t, decoded)
            sequence_observations = sequence_observations.write(t, observation_dist.mean())
        stacked_sequence = tf.transpose(sequence_observations.stack(), [1, 0, 2, 3, 4])
        all_encoded = tf.transpose(encoded_sequece.stack(), [1, 0, 2])
        all_decoded = tf.transpose(decoded_sequece.stack(), [1, 0, 2, 3, 4])
        return all_encoded, all_decoded, stacked_sequence

    @tf.function
    def _training_step(self, batch, log_sequences):
        observations, next_observations, actions, rewards, terminals = \
            self._make_training_step_data(batch)
        with tf.GradientTape() as model_tape:
            if self._n_step_loss:
                encoded, decoded, _ = self._unroll_sequence(
                    observations[:, 0], tf.shape(next_observations)[1], actions=actions)
            else:
                encoded, decoded = self._forward(observations, actions)
            next_observation, reward, terminal = self._to_distributions(
                encoded, decoded, actions)
            log_p_observations = tf.reduce_mean(next_observation.log_prob(next_observations))
            log_p_rewards = tf.reduce_mean(reward.log_prob(rewards))
            log_p_terminals = tf.reduce_mean(terminal.log_prob(terminals))
            loss = -log_p_observations - log_p_rewards - log_p_terminals
        grads = model_tape.gradient(loss, self.trainable_variables)
        self._optimizer.apply_gradients(zip(grads, self.trainable_variables))
        self._logger['observation_log_p'].update_state(-log_p_observations)
        self._logger['rewards_log_p'].update_state(-log_p_rewards)
        self._logger['terminals_log_p'].update_state(-log_p_terminals)
        self._logger['world_model_loss'].update_state(loss)
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
