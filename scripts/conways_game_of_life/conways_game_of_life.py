import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd

import mbpo.models as models
import mbpo.utils as utils


# https://github.com/davidsandberg/rl_ssms/blob/master/bouncing_ball_prediction.ipynb
def load_sequence(filename):
    data = np.load(filename)
    return data['batch']


def load_data(data_dir, prefix='train'):
    data = []
    for filename in os.listdir(data_dir):
        file_path = os.path.join(data_dir, filename)
        if filename.startswith(prefix):
            data.append(file_path)
    for file_path in data:
        sequence_batch = load_sequence(file_path)
        yield np.array(sequence_batch, np.float32)[..., None]


def show_sequence(sequence):
    plt.rcParams['figure.figsize'] = [15, 8]
    batch, length, width, height, depth = sequence.shape
    out = np.zeros((batch * height, length * width, depth))
    for x in range(length):
        for y in range(batch):
            out[y * height:(y + 1) * height, x * width:(x + 1) * width, :] = sequence[y, x, :, :, :]
    out[0::height, :, :] = 0.5
    out[:, 0::width, :] = 0.5
    out[1::height, :, :] = 0.5
    out[:, 1::width, :] = 0.5
    plt.imshow(out, cmap=matplotlib.cm.Greys_r)


def make_dataset(dir, prefix='train', repeat=0, shuffle=0, seed=0):
    train_dataset = tf.data.Dataset.from_generator(lambda: load_data(dir, prefix=prefix),
                                                   output_types=np.uint8,
                                                   output_shapes=[None, 50, 64, 64, 1])
    train_dataset = train_dataset.map(lambda data: tf.where(
        utils.preprocess(data) > 0.0, 1.0, 0.0))
    if shuffle:
        train_dataset = train_dataset.shuffle(shuffle, seed, reshuffle_each_iteration=True)
    if repeat:
        train_dataset = train_dataset.repeat(repeat)
    return train_dataset.prefetch(10)


@tf.function
def grad(model, batch):
    with tf.GradientTape() as tape:
        features, prior, posterior = observe_sequence(model, batch)
        kl = tf.reduce_mean(tf.reduce_sum(tfd.kl_divergence(posterior, prior), 1))
        log_p_observations = tf.reduce_mean(
            model._observation_decoder(features).log_prob(batch))
        loss = -log_p_observations + model._kl_scale * tf.maximum(model._free_nats, kl)
    return tape.gradient(loss, model.trainable_variables), loss, log_p_observations, kl


@tf.function
def reconstruct(model, batch):
    features, prior, posterior = observe_sequence(model, batch)
    kl = tf.reduce_mean(tf.reduce_sum(tfd.kl_divergence(posterior, prior), 1))
    log_p_observations = tf.reduce_mean(
        model._observation_decoder(features).log_prob(batch))
    loss = -log_p_observations + model._kl_scale * tf.maximum(model._free_nats, kl)
    return model._observation_decoder(features).mode(), loss


def observe_sequence(model, samples):
    next_observations = samples
    actions = tf.zeros(tf.concat([tf.shape(samples)[:2], [1]], 0))
    horizon = tf.shape(next_observations)[1]
    embeddings = model._observation_encoder(next_observations)
    belief = model.reset(tf.shape(actions)[0], True)
    sequence_results = {'stochastics': tf.TensorArray(tf.float32, horizon),
                        'deterministics': tf.TensorArray(tf.float32, horizon),
                        'prior_mus': tf.TensorArray(tf.float32, horizon),
                        'prior_stddevs': tf.TensorArray(tf.float32, horizon),
                        'posterior_mus': tf.TensorArray(tf.float32, horizon),
                        'posterior_stddevs': tf.TensorArray(tf.float32, horizon)}
    rng = tf.random.Generator.from_seed(0)
    seeds = tf.cast(rng.make_seeds(horizon), tf.int32)
    for t in range(horizon):
        predict_seeds, correct_seeds = tfp.random.split_seed(seeds[:, t], 2, "observe_sequence")
        prior, belief_prediction = model.predict(actions[:, t], belief, seed=predict_seeds)
        posterior, belief = model._correct(embeddings[:, t], belief, belief_prediction, prior,
                                           seed=correct_seeds)
        sequence_results['stochastics'] = sequence_results['stochastics'].write(
            t, belief['stochastic'])
        sequence_results['deterministics'] = sequence_results['deterministics'].write(
            t, belief['deterministic'])
        sequence_results['prior_mus'] = sequence_results['prior_mus'].write(t, prior.mean())
        sequence_results['prior_stddevs'] = sequence_results['prior_stddevs'].write(
            t, prior.stddev())
        sequence_results['posterior_mus'] = sequence_results['posterior_mus'].write(
            t, posterior.mean())
        sequence_results['posterior_stddevs'] = sequence_results['posterior_stddevs'].write(
            t, posterior.stddev())
    stacked = {k: tf.transpose(v.stack(), [1, 0, 2]) for k, v in sequence_results.items()}
    features = tf.concat([stacked['stochastics'], stacked['deterministics']], -1)
    prior = tfd.MultivariateNormalDiag(stacked['prior_mus'], stacked['prior_stddevs'])
    posterior = tfd.MultivariateNormalDiag(stacked['posterior_mus'], stacked['posterior_stddevs'])
    return features, prior, posterior


def main():
    tf.random.set_seed(0)
    np.random.seed(0)
    model = models.WorldModel('binary_image', (64, 64, 1), 30, 300, 256, 0)
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=5e-4, clipnorm=100, epsilon=1e-5)
    train_dataset = make_dataset('data', repeat=5, shuffle=3000)
    train_loss = tf.keras.metrics.Mean()
    for i, batch in enumerate(train_dataset):
        show_sequence(batch[:3, ::5, ...])
        grads, loss, log_p_obs, total_kl = grad(model, batch)
        train_loss.update_state(loss)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        if (i % 50) == 0:
            print("Loss: {}\nObs log probs: {}\nKL: {}".format(
                train_loss.result(), log_p_obs, total_kl))
    test_dataset = make_dataset('data', 'test')
    test_elbo = tf.keras.metrics.Mean()
    for i, batch in enumerate(test_dataset):
        reconstructed_sequence, elbo = reconstruct(model, batch)
        test_elbo.update_state(elbo)
        if (i % 50) == 0:
            print("Test ELBO: {}".format(test_elbo.result()))
            show_sequence(reconstructed_sequence[:3, ::5, ...])
            plt.show()
            show_sequence(batch[:3, ::5, ...])
            plt.show()


if __name__ == '__main__':
    main()
