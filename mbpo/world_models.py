import tensorflow as tf
import tensorflow_probability as tfp

from mbpo import models
from mbpo import utils as utils


class BayesianWorldModel(tf.Module):
    def __init__(self, config, logger, actor):
        super().__init__()
        self._logger = logger
        self._config = config
        self._actor = actor
        self._rng = tf.random.Generator.from_seed(config.seed)
        self._optimizer = tf.keras.optimizers.Adam(
            learning_rate=self._config.model_learning_rate, clipnorm=self._config.grad_clip_norm,
            epsilon=1e-5)

    def __call__(self, observations, actions=None):
        posterior_rollouts = {'next_observation': [],
                              'reward': [],
                              'terminal': []}
        posterior_seeds = tf.cast(self._rng.make_seeds(self._config.posterior_samples), tf.int32)
        for i in range(self._config.posterior_samples):
            for k, v in self._posterior_sample(
                    observations, actions, posterior_seeds[:, i]).items():
                posterior_rollouts[k].append(v)

        # def marginalize_models(key, value):
        #     stacked = tf.stack(value, 0)
        #     if key == 'terminal':
        #         return tfp.stats.percentile(stacked, 50.0, 0)
        #     else:
        #         return tf.reduce_mean(stacked, 0)

        # marginalized_trajectory = {k: tf.reduce_mean(
        #         #     tf.stack(v, 0), 0) for k, v in posterior_rollouts.items()}
        # Everything beyond the first terminal state is also terminal
        # TODO (yarden): this might really destroy backpropagation stuff? if it doesn't work,
        #  predict the probability of continue and use that as discount
        # not_terminals = tf.math.cumprod(1.0 - marginalized_trajectory['terminal'], 1)
        # terminals_idxs = tf.reduce_sum(not_terminals, 1)
        # return {k: tf.where(not_terminals,
        #                     v,
        #                     v[:, terminals_idxs]) for k, v in marginalized_trajectory}
        return {k: tf.reduce_mean(
            tf.stack(v, 0), 0) for k, v in posterior_rollouts.items()}

    def _posterior_sample(self, observation, action, seed):
        raise NotImplementedError

    def gradient_step(self, batch):
        raise NotImplementedError


class EnsembleWorldModel(BayesianWorldModel):
    def __init__(self, config, logger, actor,
                 dynamics_size,
                 reward_layers=1,
                 terminal_layers=1,
                 min_stddev=1e-4,
                 activation=tf.nn.relu):
        super().__init__(config, logger, actor)
        self._ensemble = [models.WorldModel(
            dynamics_size,
            config.cell_size,
            config.dynamics_layers,
            config.units,
            config.seed,
            reward_layers, terminal_layers, min_stddev, activation)
            for _ in range(config.posterior_samples)]

    def _posterior_sample(self, observation, action, seed):
        ensemble_rollouts = {'next_observation': [],
                             'reward': [],
                             'terminal': []}
        obs_seed, shuffle_seed = tfp.random.split_seed(seed, 2, "posterior_sample")
        idxs = tf.random.stateless_uniform([tf.shape(observation)[0]], shuffle_seed, 0,
                                           tf.shape(observation)[0], tf.int32)
        bootstraped_obs = utils.split_batch(tf.gather(observation, idxs),
                                            self._config.posterior_samples)
        bootstraped_acs = utils.split_batch(
            tf.gather(action, idxs),
            self._config.posterior_samples) if action is not None else [None] * len(bootstraped_obs)
        for obs, acs, model in zip(bootstraped_obs, bootstraped_acs, self._ensemble):
            predictions = model(obs, self._config.horizon, actor=self._actor, actions=acs)
            ensemble_rollouts['next_observation'].append(predictions['next_observation'].sample(
                seed=obs_seed))
            ensemble_rollouts['reward'].append(predictions['reward'].mode())
            ensemble_rollouts['terminal'].append(predictions['terminal'].mean())
        return {k: tf.concat(v, 0) for k, v in ensemble_rollouts.items()}

    # @tf.function
    def gradient_step(self, batch):
        bootstraped_batches = {k: utils.split_batch(v, self._config.posterior_samples)
                               for k, v in batch.items()}
        parameters = []
        loss = 0.0
        with tf.GradientTape() as model_tape:
            for i, world_model in enumerate(self._ensemble):
                observations = bootstraped_batches['observation'][i]
                actions = bootstraped_batches['action'][i]
                target_next_observations = bootstraped_batches['next_observation'][i]
                predictions = world_model(
                    observations, tf.shape(actions)[1], actions,
                    next_observations=target_next_observations)
                log_p_dynamics = tf.reduce_mean(
                    predictions['next_observation'].log_prob(target_next_observations))
                target_rewards = bootstraped_batches['reward'][i]
                log_p_reward = tf.reduce_mean(predictions['reward'].log_prob(target_rewards))
                target_terminals = bootstraped_batches['terminal'][i]
                log_p_terminals = tf.reduce_mean(predictions['terminal'].log_prob(target_terminals))
                loss -= (log_p_dynamics + log_p_reward + log_p_terminals)
                parameters += world_model.trainable_variables
                self._logger['dynamics_' + str(i) + '_log_p'].update_state(-log_p_dynamics)
                self._logger['rewards_' + str(i) + '_log_p'].update_state(-log_p_reward)
                self._logger['terminals_' + str(i) + '_log_p'].update_state(-log_p_terminals)
            grads = model_tape.gradient(loss, parameters)
            self._optimizer.apply_gradients(zip(grads, parameters))
        self._logger['world_model_total_loss'].update_state(loss)
        self._logger['world_model_grads'].update_state(tf.linalg.global_norm(grads))
