import tensorflow as tf

import mbpo.models as models
from mbpo.swag import SWAG
from mbpo.world_models import BayesianWorldModel


# https://github.com/wjmaddox/swa_gaussian/blob/b172d93278fdb92522c8fccb7c6bfdd6f710e4f0
# /experiments/train/train.py#L183
class LearningRateScheduler(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, init_lr, terminal_lr, warmup_steps):
        super(LearningRateScheduler, self).__init__()
        self._warmup_steps = tf.cast(warmup_steps, tf.float32)
        self._init_lr = init_lr
        self._terminal_lr = terminal_lr

    def get_config(self):
        pass

    def __call__(self, step):
        t = tf.cast(step, tf.float32) / self._warmup_steps
        lr_ratio = self._terminal_lr / self._init_lr
        if t <= 0.5:
            factor = 1.0
        elif t <= 0.9:
            factor = 1.0 - (1.0 - lr_ratio) * (t - 0.5) / 0.4
        else:
            factor = lr_ratio
        return self._init_lr * factor


class SwagWorldModel(BayesianWorldModel):
    def __init__(self, config, logger, observation_shape):
        super(SwagWorldModel, self).__init__(config, logger)
        self._optimizer = SWAG(
            tf.optimizers.Adam(
                config.model_learning_rate,
                clipnorm=config.grad_clip_norm),
            1000,
            5)
        self._model = models.WorldModel(
            config.observation_type,
            observation_shape,
            config.stochastic_size,
            config.deterministic_size,
            config.units,
            config.seed)
        self._posterior_samples = config.posterior_samples

    @tf.function
    def _update_beliefs(self, prev_embeddings, prev_action, current_observation):
        beliefs = []
        embeddings = []
        for _ in range(self._posterior_samples):
            self._optimizer.sample_and_assign(1.0, self._model.trainable_variables)
            belief, embedding = self._model(prev_embeddings, prev_action, current_observation)
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
        for _ in range(self._posterior_samples):
            self._optimizer.sample_and_assign(1.0, self._model.trainable_variables, True)
            features, rewards, terminals, reconstructed = self._model.generate_sequence(
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
        for i in range(self._posterior_samples):
            self._optimizer.sample_and_assign(1.0, self._model.trainable_variables, True)
            loss, kl, log_p_observations, log_p_reward, \
            log_p_terminals, reconstructed, beliefs = self._model.inference_step(batch)
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
        with tf.GradientTape() as model_tape:
            loss, kl, log_p_observations, log_p_reward, \
            log_p_terminals, reconstructed, beliefs = self._model.inference_step(batch)
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
