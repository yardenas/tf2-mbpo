import tensorflow as tf

from mbpo import models
from mbpo import utils as utils


class BayesianWorldModel(tf.Module):
    def __init__(self, config, logger):
        super().__init__()
        self._logger = logger
        self._config = config
        self._rng = tf.random.Generator.from_seed(config.seed)
        self._optimizer = tf.keras.optimizers.Adam(
            learning_rate=self._config.model_learning_rate, clipnorm=self._config.grad_clip_norm,
            epsilon=1e-5)

    def __call__(self, prev_embeddings, prev_action, current_observation):
        current_beliefs, current_embeddiengs = self._update_beliefs(
            prev_embeddings, prev_action, current_observation)
        return tf.reduce_mean(current_beliefs, 0), tf.reduce_mean(current_embeddiengs, 0)

    def generate_sequences_posterior(self, initial_belief, horizon, actor=None, actions=None):
        sequences_posterior = self._generate_sequences_posterior(
            initial_belief, horizon, self._rng.make_seeds(), actor=actor, actions=actions)
        return tf.reduce_mean(sequences_posterior, 0)

    def reconstruct_sequences_posterior(self, batch):
        sequences_posterior = self._reconstruct_sequences_posterior(batch)
        return tf.reduce_mean(sequences_posterior, 0)

    def train(self, batch, log_observed_sequences=False):
        return self._training_step(batch, log_observed_sequences)

    def _log_reconstructed(self, true_observations, reconstructed, name=''):
        sequence = reconstructed.mode()
        if self._config.observation_type in ['binary_image', 'binary_image']:
            self._logger.log_video(tf.transpose(sequence[:3], [0, 1, 4, 2, 3]).numpy(),
                                   'reconstructed_sequence_' + name)
            self._logger.log_video(tf.transpose(true_observations[:3], [0, 1, 4, 2, 3]).numpy(),
                                   'true_sequence_' + name)

    def _update_beliefs(self, prev_embeddings, prev_action, current_observation):
        raise NotImplementedError

    def _generate_sequences_posterior(self, initial_belief, horizon, seed, actor=None,
                                      actions=None):
        raise NotImplementedError

    def _reconstruct_sequences_posterior(self, batch):
        raise NotImplementedError

    def _training_step(self, batch, log_observed_sequences):
        raise NotImplementedError


class EnsembleWorldModel(BayesianWorldModel):
    def __init__(self, config, logger, observation_shape):
        super().__init__(config, logger)
        self._ensemble = [models.WorldModel(
            config.observation_type,
            observation_shape,
            config.stochastic_size,
            config.deterministic_size,
            config.units,
            config.seed)
            for _ in range(config.posterior_samples)]

    # @tf.function
    def _update_beliefs(self, prev_embeddings, prev_action, current_observation):
        beliefs = []
        embeddings = []
        for model in self._ensemble:
            belief, embedding = model(prev_embeddings, prev_action, current_observation)
            beliefs.append(belief)
            embeddings.append(embedding)
        return tf.stack(beliefs, 0), tf.stack(embeddings, 0)

    # @tf.function
    def _generate_sequences_posterior(self, initial_belief, horizon, seed, actor=None,
                                      actions=None):
        ensemble_rollouts = {'features': [],
                             'rewards': [],
                             'terminals': []}
        for model in self._ensemble:
            features, rewards, terminals = model.generate_sequence(
                initial_belief, self._config.horizon, actor=self._actor, actions=actions)
            ensemble_rollouts['features'].append(features)
            ensemble_rollouts['reward'].append(rewards)
            ensemble_rollouts['terminal'].append(terminals)
        return {k: tf.stack(v, 0) for k, v in ensemble_rollouts.items()}

    def _reconstruct_sequences_posterior(self, batch):
        ensemble_rollouts = []
        for i, model in enumerate(self._ensemble):
            loss, kl, log_p_observations, log_p_reward, \
            log_p_terminals, reconstructed, beliefs = model.inference_step(batch)
            self._logger['test_dynamics_' + str(i) + '_log_p'].update_state(-log_p_observations)
            self._logger['test_rewards_' + str(i) + '_log_p'].update_state(-log_p_reward)
            self._logger['test_terminals_' + str(i) + '_log_p'].update_state(-log_p_terminals)
            self._logger['test_kl_' + str(i)].update_state(kl)
            ensemble_rollouts.append(reconstructed.mode())
        return tf.stack(ensemble_rollouts, 0)

    # @tf.function
    def _training_step(self, batch, log_observed_sequences):
        bootstraped_batches = {k: utils.split_batch(v, self._config.posterior_samples)
                               for k, v in batch.items()}
        batches = [{k: v[i] for k, v in bootstraped_batches.items()} for i in
                   range(len(bootstraped_batches['observation']))]
        ensemble_beliefs = []
        parameters = []
        loss = 0.0
        with tf.GradientTape() as model_tape:
            for i, (model_batch, model) in enumerate(zip(batches, self._ensemble)):
                loss, kl, log_p_observations, log_p_reward, \
                log_p_terminals, reconstructed, beliefs = model.inference_step(model_batch)
                ensemble_beliefs.append(beliefs)
                parameters += model.trainable_variables
                self._logger['dynamics_' + str(i) + '_log_p'].update_state(-log_p_observations)
                self._logger['rewards_' + str(i) + '_log_p'].update_state(-log_p_reward)
                self._logger['terminals_' + str(i) + '_log_p'].update_state(-log_p_terminals)
                self._logger['kl_' + str(i)].update_state(kl)
        grads = model_tape.gradient(loss, parameters)
        self._optimizer.apply_gradients(zip(grads, parameters))
        if log_observed_sequences:
            self._log_reconstructed(model_batch['observation'],
                                    reconstructed, str(len(self._ensemble)))
        self._logger['world_model_total_loss'].update_state(loss)
        self._logger['world_model_grads'].update_state(tf.linalg.global_norm(grads))
        return {k: tf.concat(
            [belief[k] for belief in ensemble_beliefs], 0) for k in ensemble_beliefs[0].keys()}
