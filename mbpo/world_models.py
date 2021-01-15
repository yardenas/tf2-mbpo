import tensorflow as tf

import mbpo.utils as utils


class BayesianWorldModel(tf.Module):
    def __init__(self, config, logger):
        super().__init__()
        self._logger = logger
        self._config = config

    def __call__(self, prev_action, current_observation):
        current_beliefs = self._update_beliefs(
            prev_action, current_observation)
        return tf.reduce_mean(current_beliefs, 0)

    def generate_sequences_posterior(self, initial_belief, horizon, actor=None, actions=None,
                                     log_sequences=False, step=None):
        sequences_posterior, reconstructed_sequences_posterior = self._generate_sequences_posterior(
            initial_belief, horizon, actor, actions, log_sequences)
        reconstructed_sequences = None
        if log_sequences:
            reconstructed_sequences = tf.reduce_mean(reconstructed_sequences_posterior, 0)
            self._logger.log_video(utils.make_video(reconstructed_sequences[:4],
                                                    self._config.observation_type), step,
                                   name='generation_reconstructed_sequence')
        return {k: tf.reduce_mean(
            v, 0) for k, v in sequences_posterior.items()}, reconstructed_sequences

    def reconstruct_sequences_posterior(self, batch):
        reconstructed_sequences_posterior, beliefs = self._reconstruct_sequences_posterior(batch)
        return tf.reduce_mean(reconstructed_sequences_posterior, 0), {k: tf.reduce_mean(
            v, 0) for k, v in beliefs.items()}

    def train(self, batch, log_sequences=False, step=None):
        train_posterior_beliefs, reconstructed_sequences = self._training_step(batch, log_sequences)
        if log_sequences:
            self._logger.log_video(utils.make_video(reconstructed_sequences[:4],
                                                    self._config.observation_type), step,
                                   name='train_reconstructed_sequence')
            self._logger.log_video(utils.make_video(batch['observation'][:4],
                                                    self._config.observation_type),
                                   step, name='train_true_sequence')
        return train_posterior_beliefs

    def _update_beliefs(self, prev_action, current_observation):
        raise NotImplementedError

    def _generate_sequences_posterior(self, initial_belief, horizon, actor,
                                      actions, log_sequences):
        raise NotImplementedError

    def _reconstruct_sequences_posterior(self, batch):
        raise NotImplementedError

    def _training_step(self, batch, log_sequences):
        raise NotImplementedError
