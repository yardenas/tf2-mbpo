import tensorflow as tf

from mbpo import models
from mbpo import utils as utils
from mbpo import world_models


class EnsembleWorldModel(world_models.BayesianWorldModel):
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
        self._optimizer = tf.keras.optimizers.Adam(
            learning_rate=self._config.model_learning_rate, clipnorm=self._config.grad_clip_norm)

    @tf.function
    def _update_beliefs(self, prev_embeddings, prev_action, current_observation):
        beliefs = []
        embeddings = []
        for model in self._ensemble:
            belief, embedding = model(prev_embeddings, prev_action, current_observation)
            beliefs.append(belief)
            embeddings.append(embedding)
        return tf.stack(beliefs, 0), tf.stack(embeddings, 0)

    @tf.function
    def _generate_sequences_posterior(self, initial_belief, horizon, seed, actor,
                                      actions, log_sequences):
        ensemble_rollouts = {'features': [],
                             'rewards': [],
                             'terminals': []}
        ensemble_reconstructed = []
        for model in self._ensemble:
            features, rewards, terminals, reconstructed = model.generate_sequence(
                initial_belief, horizon, actor=actor, actions=actions,
                log_sequences=log_sequences)
            ensemble_rollouts['features'].append(features)
            ensemble_rollouts['rewards'].append(rewards)
            ensemble_rollouts['terminals'].append(terminals)
            ensemble_reconstructed.append(reconstructed)
        return_reconstructed = None if not log_sequences else tf.stack(ensemble_reconstructed, 0)
        return {k: tf.stack(v, 0) for k, v in ensemble_rollouts.items()}, return_reconstructed

    @tf.function
    def _reconstruct_sequences_posterior(self, batch):
        ensemble_reconstructed = []
        ensemble_beliefs = []
        for i, model in enumerate(self._ensemble):
            loss, kl, log_p_observations, log_p_reward, \
            log_p_terminals, reconstructed, beliefs = model.inference_step(batch)
            self._logger['test_dynamics_' + str(i) + '_log_p'].update_state(-log_p_observations)
            self._logger['test_rewards_' + str(i) + '_log_p'].update_state(-log_p_reward)
            self._logger['test_terminals_' + str(i) + '_log_p'].update_state(-log_p_terminals)
            self._logger['test_kl_' + str(i)].update_state(kl)
            ensemble_reconstructed.append(reconstructed.mode())
            ensemble_beliefs.append(beliefs)
        return tf.stack(ensemble_reconstructed, 0), {k: tf.stack(
            [belief[k] for belief in ensemble_beliefs], 0) for k in ensemble_beliefs[0].keys()}

    @tf.function
    def _training_step(self, batch, log_sequences):
        bootstraped_batches = {k: utils.split_batch(v, self._config.posterior_samples)
                               for k, v in batch.items()}
        batches = [{k: v[i] for k, v in bootstraped_batches.items()} for i in
                   range(len(bootstraped_batches['observation']))]
        ensemble_reconstructed = []
        ensemble_beliefs = []
        total_loss = 0.0
        for i, (model_batch, model) in enumerate(zip(batches, self._ensemble)):
            with tf.GradientTape() as model_tape:
                loss, kl, log_p_observations, log_p_reward, \
                log_p_terminals, reconstructed, beliefs = model.inference_step(model_batch)
                total_loss += loss
                if log_sequences:
                    ensemble_reconstructed.append(reconstructed.mode())
                ensemble_beliefs.append(beliefs)
                self._logger['observation_' + str(i) + '_log_p'].update_state(-log_p_observations)
                self._logger['rewards_' + str(i) + '_log_p'].update_state(-log_p_reward)
                self._logger['terminals_' + str(i) + '_log_p'].update_state(-log_p_terminals)
                self._logger['kl_' + str(i)].update_state(kl)
            grads = model_tape.gradient(loss, model.trainable_variables)
            self._optimizer.apply_gradients(zip(grads, model.trainable_variables))
        self._logger['world_model_total_loss'].update_state(total_loss)
        return_reconstructed = None if not log_sequences else tf.concat(ensemble_reconstructed, 0)
        return {k: tf.concat(
            [belief[k] for belief in ensemble_beliefs],
            0) for k in ensemble_beliefs[0].keys()}, return_reconstructed
