import tensorflow as tf

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
            1000,
            5)
        self._encoder = [tf.keras.layers.Conv2D(32, 4, activation=tf.nn.relu, strides=2,
                                                input_shape=(None,) + observation_shape),
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

    def _forward(self, observation_sequence, action_sequence):
        x = observation_sequence
        x = self._encoder[0](x)
        x = skip_1 = self._encoder[1](x)
        x = self._encoder[2](x)
        x = skip_2 = self._encoder[2](x)
        x = self._encoder[3](x)
        x = tf.concat([x, action_sequence], -1)
        x = self._decoder[0](x)
        x = tf.concat([x, skip_1], -1)
        x = self._decoder[1](x)
        x = tf.concat([x, skip_2], -1)
        x = self._decoder[2](x)
        return self._decoder[3](x)

    def _update_beliefs(self, prev_embeddings, prev_action, current_observation):
        pass

    def _generate_sequences_posterior(self, initial_belief, horizon, seed, actor,
                                      actions, log_sequences):
        pass

    def _reconstruct_sequences_posterior(self, batch):
        pass

    def _training_step(self, batch, log_sequences):
        with tf.GradientTape() as model_tape:
            loss, kl, log_p_observations, log_p_reward, \
            log_p_terminals, reconstructed, beliefs = self._forward(batch[''])
            self._logger['observation_log_p'].update_state(-log_p_observations)
            self._logger['rewards_log_p'].update_state(-log_p_reward)
            self._logger['terminals_log_p'].update_state(-log_p_terminals)
            self._logger['kl'].update_state(kl)
        grads = model_tape.gradient(loss, self._model.trainable_variables)
        self._optimizer.apply_gradients(zip(grads, self._model.trainable_variables))
        self._logger['world_model_loss'].update_state(loss)
        self._logger['world_model_grads'].update_state(tf.linalg.global_norm(grads))
        return_reconstructed = None if not log_sequences else reconstructed.mode()
        return beliefs, return_reconstructed
