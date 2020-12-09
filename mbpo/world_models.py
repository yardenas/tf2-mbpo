import tensorflow as tf
import tensorflow_probability as tfp

from mbpo import models


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
        return self._imagine_rollouts(
            observations,
            actions)

    @tf.function
    def _imagine_rollouts(self, observations, actions=None):
        horizon = self._config.horizon if actions is None else tf.shape(actions)[0]
        rollouts = {'observation': tf.TensorArray(tf.float32, size=horizon),
                    'next_observation': tf.TensorArray(tf.float32, size=horizon),
                    'action': tf.TensorArray(tf.float32, size=horizon),
                    'reward': tf.TensorArray(tf.float32, size=horizon),
                    'terminal': tf.TensorArray(tf.float32, size=horizon)}
        done_rollout = tf.zeros((tf.shape(observations)[0]), dtype=tf.bool)
        observation = observations
        seeds = tf.cast(self._rng.make_seeds(horizon), tf.int32)
        for k in tf.range(horizon if actions is None else tf.shape(actions)[0]):
            action_seeds, obs_seeds = tfp.random.split_seed(seeds[:, k], 2, "imagine_rollouts")
            rollouts['observation'] = rollouts['observation'].write(k, observation)
            action = self._actor(tf.stop_gradient(observation)).sample(
                seed=action_seeds) if actions is None else actions[k, ...]
            rollouts['action'] = rollouts['action'].write(k, action)
            predictions = self._predict_next_step(observation, action)
            # If the rollout is done, we stay at the terminal state, not overriding with a new,
            # possibly valid state.
            observation = tf.where(done_rollout[:, None],
                                   observation,
                                   predictions['next_observation'])
            rollouts['next_observation'] = rollouts['next_observation'].write(k, observation)
            terminal = tf.where(done_rollout,
                                1.0,
                                predictions['terminal'])
            rollouts['terminal'] = rollouts['terminal'].write(k, terminal)
            reward = tf.where(done_rollout,
                              0.0,
                              predictions['reward'])
            rollouts['reward'] = rollouts['reward'].write(k, reward)
            done_rollout = tf.logical_or(
                tf.cast(terminal, tf.bool), done_rollout)

        def standardize_shapes(tensor_array):
            stacked = tensor_array.stack()
            if len(tf.shape(stacked)) == 2:
                return tf.transpose(tf.squeeze(stacked))
            else:
                return tf.transpose(stacked, [1, 0, 2])

        return {k: standardize_shapes(v) for k, v in rollouts.items()}

    def _predict_next_step(self, current_observation, current_action):
        all_predictions = {
            'next_observation': tf.TensorArray(tf.float32,
                                               self._config.posterior_samples),
            'reward': tf.TensorArray(tf.float32, self._config.posterior_samples),
            'terminal': tf.TensorArray(tf.float32, self._config.posterior_samples)}
        model_seeds = tf.cast(self._rng.make_seeds(self._config.posterior_samples), tf.int32)
        # TODO (yarden): check with tf.range() (if posterior sample actually happens after
        #  retracing)
        for i in range(self._config.posterior_samples):
            next_observation, reward, terminal = self._posterior_sample(
                current_observation, current_action, model_seeds[:, i])
            all_predictions['next_observation'] = all_predictions[
                'next_observation'].write(i, next_observation)
            all_predictions['terminal'] = all_predictions['terminal'].write(i, terminal)
            all_predictions['reward'] = all_predictions['reward'].write(i, reward)

        def marginalize_models(key, value):
            stacked = value.stack()
            if key == 'terminal':
                return tfp.stats.percentile(stacked, 50.0, 0)
            else:
                return tf.reduce_mean(stacked, 0)

        return {k: marginalize_models(k, v) for k, v in all_predictions.items()}

    def _posterior_sample(self, observation, action, seed=None):
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
            config.dynamics_layers,
            config.units, reward_layers, terminal_layers, min_stddev, activation)
            for _ in range(config.posterior_samples)]

    @tf.function
    def _posterior_sample(self, observation, action, seed=None):
        next_observation, reward, terminal = [], [], []
        bootstrap_seed, obs_seeds = tfp.random.split_seed(seed, 2, "ensemble_seed")
        idxs = tf.random.stateless_uniform([tf.shape(observation)[0]], bootstrap_seed, 0,
                                           tf.shape(observation)[0], tf.int32)
        bootstraped_obs = self._split_batch(tf.gather(observation, idxs))
        bootstraped_acs = self._split_batch(tf.gather(action, idxs))
        for obs, acs, model in zip(bootstraped_obs, bootstraped_acs, self._ensemble):
            preds = model(obs, acs)
            next_observation.append(preds['next_observation'].sample(seed=seed))
            reward.append(preds['reward'].mode())
            terminal.append(preds['terminal'].mode())
        cat_obs = tf.concat(next_observation, 0)
        cat_rews = tf.concat(reward, 0)
        cat_terms = tf.concat(terminal, 0)
        # This implements TS-1 from PETS.
        return cat_obs, cat_rews, cat_terms

    @tf.function
    def gradient_step(self, batch):
        bootstraped_batches = {k: self._split_batch(v)
                               for k, v in batch.items()}
        parameters = []
        loss = 0.0
        with tf.GradientTape() as model_tape:
            for i, world_model in enumerate(self._ensemble):
                observations = bootstraped_batches['observation'][i]
                target_next_observations = bootstraped_batches['next_observation'][i]
                actions = bootstraped_batches['action'][i]
                target_rewards = bootstraped_batches['reward'][i]
                target_terminals = bootstraped_batches['terminal'][i]
                predictions = world_model(observations, actions)
                log_p_dynamics = tf.reduce_mean(
                    predictions['next_observation'].log_prob(target_next_observations))
                log_p_reward = tf.reduce_mean(predictions['reward'].log_prob(target_rewards))
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

    def _split_batch(self, batch):
        return tf.split(batch,
                        [tf.shape(batch)[0] // self._config.posterior_samples] *
                        self._config.posterior_samples +
                        [tf.shape(batch)[0] % self._config.posterior_samples])
