import numpy as np
import tensorflow as tf


class CemActor(object):
    def __init__(self, imagine_rollouts, ensemble):
        self._ensemble = ensemble
        self._imagine_rollouts = imagine_rollouts

    @tf.function
    def __call__(self, observation):
        action_dim = 1
        mu = tf.zeros((8, action_dim))
        sigma = tf.ones_like(mu)
        best_so_far = tf.zeros(action_dim, dtype=tf.float32)
        best_so_far_score = -np.inf * tf.ones((), dtype=tf.float32)
        for _ in tf.range(10):
            action_sequences = tf.random.normal(
                shape=(150, 8, action_dim),
                mean=mu, stddev=sigma
            )
            # TODO (yarden): average over particles!!!
            action_sequences = tf.clip_by_value(action_sequences, -1.0, 1.0)
            action_sequences_batch = action_sequences
            all_rewards = []
            for model in self._ensemble:
                trajectories = self._imagine_rollouts(
                    tf.broadcast_to(observation,
                                    (action_sequences_batch.shape[0], observation.shape[0])),
                    model,
                    tf.transpose(action_sequences_batch, [1, 0, 2])
                )
                all_rewards.append(tf.reduce_sum(
                    trajectories['reward'] * (1.0 - trajectories['terminal']), axis=1))
            scores = tf.reduce_mean(tf.stack(all_rewards, axis=0), axis=0)
            elite_scores, elite = tf.nn.top_k(scores, 10, sorted=False)
            best_of_elite = tf.argmax(elite_scores)
            if tf.greater(elite_scores[best_of_elite], best_so_far_score):
                best_so_far = action_sequences[elite[best_of_elite], 0, :]
                best_so_far_score = elite_scores[best_of_elite]
            elite_actions = tf.gather(action_sequences, elite, axis=0)
            mean, variance = tf.nn.moments(elite_actions, axes=0)
            mu = mean
            sigma = tf.sqrt(variance)
            if tf.less_equal(tf.reduce_mean(sigma), 0.1):
                break
        return tf.clip_by_value(
            best_so_far + tf.random.normal(best_so_far.shape, stddev=0.01),
            -1.0, 1.0)
