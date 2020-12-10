import numpy as np
import tensorflow as tf
from tqdm import tqdm

import mbpo.models as models
import mbpo.utils as utils
import mbpo.world_models as world_models
from mbpo.cem_actor import CemActor
from mbpo.replay_buffer import ReplayBuffer


# TODO (yarden): observe gradients of actor - if small maybe vanishing??? - if so, return to
# DDPG/SAC style algorithms (with q-function) or use gru/lstm in dynamics

class MBPO(tf.Module):
    def __init__(self, config, logger, observation_space, action_space):
        super(MBPO, self).__init__()
        self._config = config
        self._logger = logger
        self._training_step = 0
        self._experience = ReplayBuffer(observation_space.shape[0], action_space.shape[0])
        self._warmup_policy = lambda: np.random.uniform(action_space.low, action_space.high)
        self._actor = models.Actor(action_space.shape[0], 3, self._config.units,
                                   seed=self._config.seed)
        self.model = world_models.EnsembleWorldModel(
            self._config, self._logger, self._actor, observation_space.shape[0],
            reward_layers=3, terminal_layers=3, min_stddev=1e-4)
        self._dbug_actor = CemActor(self.model)
        self._actor_optimizer = tf.keras.optimizers.Adam(
            learning_rate=self._config.actor_learning_rate, clipnorm=self._config.grad_clip_norm,
            epsilon=1e-5
        )
        self.critic = models.Critic(
            3, self._config.units, output_regularization=self._config.critic_regularization)
        self._delayed_critic = models.Critic(
            3, self._config.units, output_regularization=self._config.critic_regularization)
        utils.clone_model(self.critic, self._delayed_critic)
        self._critic_optimizer = tf.keras.optimizers.Adam(
            learning_rate=self._config.critic_learning_rate, clipnorm=self._config.grad_clip_norm,
            epsilon=1e-5
        )

    def update_model(self, batch):
        self.model.gradient_step(batch)

    def compute_lambda_values(self, next_observations, rewards, terminals):
        lambda_values = tf.TensorArray(tf.float32, self._config.horizon)
        next_values = self._delayed_critic(next_observations).mode()
        v_lambda = next_values[:, -1] * (1.0 - terminals[:, -1])
        # reverse traversing over data.
        for t in tf.range(start=self._config.horizon - 1, limit=-1, delta=-1):
            td = rewards[:, t] + (1.0 - terminals[:, t]) * (
                    1.0 - self._config.lambda_) * self._config.discount * next_values[:, t]
            v_lambda = td + v_lambda * self._config.lambda_ * self._config.discount
            lambda_values = lambda_values.write(t, v_lambda)
        return tf.transpose(lambda_values.stack())

    @tf.function
    def update_actor_critic(self, observation, model_bootstrap):
        discount = tf.math.cumprod(
            self._config.discount * tf.ones([self._config.horizon]), exclusive=True)
        with tf.GradientTape() as actor_tape:
            imagined_rollouts = self.model(observation, model_bootstrap)
            lambda_values = self.compute_lambda_values(
                imagined_rollouts['next_observation'],
                imagined_rollouts['reward'],
                imagined_rollouts['terminal'])
            actor_loss, actor_grads_norm = self._actor_grad_step(
                lambda_values, discount, actor_tape)
        with tf.GradientTape() as critic_tape:
            critic_loss, critic_grads_norm = self._critic_grad_step(
                lambda_values, imagined_rollouts['observation'], discount, critic_tape)
        self._logger['actor_loss'].update_state(actor_loss)
        self._logger['actor_grads'].update_state(actor_grads_norm)
        self._logger['critic_loss'].update_state(critic_loss)
        self._logger['critic_grads'].update_state(critic_grads_norm)
        self._logger['pi_entropy'].update_state(self._actor(observation).entropy())

    @tf.function
    def update_critic(self, observation, reward, next_observation, terminal):
        with tf.GradientTape() as critic_tape:
            next_values = self._delayed_critic(next_observation[:, None, :]).mode()
            td = reward + (1.0 - terminal) * self._config.discount * next_values
            critic_loss = -tf.reduce_mean(self.critic(observation[:, None, :]).log_prob(
                tf.stop_gradient(td)))
            grads = critic_tape.gradient(critic_loss, self.critic.trainable_variables)
            self._critic_optimizer.apply_gradients(zip(grads, self.critic.trainable_variables))
        self._logger['critic_loss'].update_state(critic_loss)
        self._logger['critic_grads'].update_state(tf.linalg.global_norm(grads))

    def _actor_grad_step(self, lambda_values, discount, actor_tape):
        actor_loss = -tf.reduce_mean(
            tf.reduce_sum(lambda_values * discount, axis=1))
        grads = actor_tape.gradient(actor_loss, self._actor.trainable_variables)
        self._actor_optimizer.apply_gradients(zip(grads, self._actor.trainable_variables))
        return actor_loss, tf.linalg.global_norm(grads)

    def _critic_grad_step(self, lambda_values, observations, discount, critic_tape):
        critic_loss = -tf.reduce_mean(tf.reduce_sum(self.critic(observations).log_prob(
            tf.stop_gradient(lambda_values * discount)), axis=1))
        grads = critic_tape.gradient(critic_loss, self.critic.trainable_variables)
        self._critic_optimizer.apply_gradients(zip(grads, self.critic.trainable_variables))
        return critic_loss, tf.linalg.global_norm(grads)

    @property
    def time_to_update(self):
        return self._training_step and \
               self._training_step % self._config.steps_per_update < self._config.action_repeat

    @property
    def time_to_log(self):
        return self._training_step and \
               self._training_step % self._config.steps_per_log < self._config.action_repeat

    @property
    def warm(self):
        return self._training_step >= self._config.warmup_training_steps

    @property
    def time_to_clone_critic(self):
        return self._training_step and \
               self._training_step % self._config.steps_per_critic_clone < \
               self._config.action_repeat

    def observe(self, transition):
        self._training_step += transition.pop('steps', self._config.action_repeat)
        self._experience.store(transition)

    def __call__(self, observation, training=True):
        scaled_obs = np.clip(
            (observation - self._experience.obs_mean) / self._experience.obs_stddev,
            -10.0, 10.0)
        if training:
            if self.warm:
                # action = self._actor(
                # np.expand_dims(scaled_obs, axis=0).astype(np.float32)).sample().numpy()
                action = self._dbug_actor(tf.constant(scaled_obs, dtype=tf.float32)).numpy()
            else:
                action = self._warmup_policy()
            if self.time_to_update and self.warm:
                print("Updating world model, actor and critic.")
                self._experience.update_statistics()
                for _ in tqdm(range(self._config.update_steps), position=0, leave=True):
                    batch = self._experience.sample(self._config.batch_size,
                                                    filter_goal_mets=self._config.filter_goal_mets)
                    self.update_model(batch)
                    # self.update_actor_critic(
                    #     tf.constant(batch['observation'], dtype=tf.float32),
                    #     random.choice(self.ensemble))
                    self.update_critic(
                        tf.constant(batch['observation'], dtype=tf.float32),
                        tf.constant(batch['reward'], dtype=tf.float32),
                        tf.constant(batch['next_observation'], dtype=tf.float32),
                        tf.constant(batch['terminal'], dtype=tf.float32))
                if self.time_to_clone_critic:
                    utils.clone_model(self.critic, self._delayed_critic)
        else:
            # action = self._actor(
            #     np.expand_dims(scaled_obs, axis=0).astype(np.float32)).mode().numpy()
            action = self._dbug_actor(tf.constant(scaled_obs, dtype=tf.float32)).numpy()
        if self.time_to_log and training and self.warm:
            self._logger.log_metrics(self._training_step)
        return np.clip(action, -1.0, 1.0)
