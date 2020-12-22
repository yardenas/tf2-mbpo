import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
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


@tf.function
def grad(model, batch):
    with tf.GradientTape() as tape:
        next_observations = batch
        actions = tf.zeros(tf.concat([tf.shape(batch)[:2], [1]], 0))
        horizon = tf.shape(next_observations)[1]
        embeddings = model._observation_encoder(next_observations)
        belief = model.reset(tf.shape(actions)[0], True)
        loss = 0.0
        total_kl = 0.0
        stochastics = tf.TensorArray(tf.float32, horizon)
        deterministics = tf.TensorArray(tf.float32, horizon)
        for t in range(horizon):
            prior, belief_prediction = model.predict(actions[:, t], belief)
            posterior, belief = model._correct(embeddings[:, t], belief, belief_prediction, prior)
            stochastics = stochastics.write(t, belief['stochastic'])
            deterministics = deterministics.write(t, belief['deterministic'])
            kl = tf.reduce_mean(tfd.kl_divergence(posterior, prior))
            total_kl += kl
        stocs = tf.transpose(stochastics.stack(), [1, 0, 2])
        deters = tf.transpose(deterministics.stack(), [1, 0, 2])
        features = tf.concat([stocs, deters], -1)
        log_p_observations = tf.reduce_mean(
            model._observation_decoder(features).log_prob(next_observations))
        loss -= log_p_observations + model._kl_scale * tf.maximum(model._free_nats, total_kl)
    return tape.gradient(loss, model.trainable_variables), loss, log_p_observations, total_kl


def main():
    train_dataset = tf.data.Dataset.from_generator(lambda: load_data('data'),
                                                   output_types=np.uint8,
                                                   output_shapes=[None, 50, 64, 64, 1])
    train_dataset = train_dataset.map(lambda data: tf.where(
        utils.preprocess(data) > 0.0, 1.0, 0.0))
    n_epochs = 5
    seed = 0
    train_dataset = train_dataset.shuffle(3000, seed, reshuffle_each_iteration=True)
    train_dataset = train_dataset.repeat(n_epochs)
    train_dataset = train_dataset.prefetch(10)
    model = models.WorldModel('binary_image', (64, 64, 1), 30, 300, 256, 0)
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=5e-4, clipnorm=100, epsilon=1e-5)
    for i, batch in enumerate(train_dataset):
        grads, loss, log_p_obs, total_kl = grad(model, batch)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        if (i % 50) == 0:
            print("Loss: {}\nObs log probs: {}\nKL: {}".format(loss, log_p_obs, total_kl))


if __name__ == '__main__':
    main()
