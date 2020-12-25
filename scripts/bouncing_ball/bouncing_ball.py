import collections
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd

import mbpo.models as models
import mbpo.utils as utils

rng = tf.random.Generator.from_seed(0)


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
        for i in range(sequence_batch.shape[0]):
            yield {'observation': np.array(sequence_batch[i], np.float32)[..., None],
                   'action': np.zeros([sequence_batch[i].shape[0], 1], np.float32)}


def show_sequence(sequence, figname=None):
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
    if figname is not None:
        plt.savefig(figname)


def make_dataset(dir, prefix='train', repeat=0, shuffle=0, seed=0, batch_size=16):
    dataset = tf.data.Dataset.from_generator(lambda: load_data(dir, prefix=prefix),
                                             output_types={'observation': np.float32,
                                                           'action': np.float32},
                                             output_shapes={'observation': [50, 64, 64, 1],
                                                            'action': [50, 1]})
    dataset = dataset.map(lambda data: {k: tf.where(
        utils.preprocess(v) > 0.0, 1.0, 0.0) for k, v in data.items()})
    if shuffle:
        dataset = dataset.shuffle(shuffle, seed, reshuffle_each_iteration=True)
    if repeat:
        dataset = dataset.repeat(repeat)
    dataset = dataset.prefetch(1024)
    dataset = dataset.batch(batch_size)
    return dataset


@tf.function
def inference_step(model, batch, reconstruct=False):
    with tf.GradientTape() as tape:
        loss, kl, log_p_observations, reconstructed = model.inference_step(batch)
    reconstructed_sequences = reconstructed.mode() if reconstruct else None
    return tape.gradient(
        loss, model.trainable_variables), loss, log_p_observations, kl, reconstructed_sequences


@tf.function
def reconstruct_sequence(model, batch):
    beliefs, prior, posterior = model.observe_sequence(batch)
    kl = tf.reduce_mean(tf.reduce_sum(tfd.kl_divergence(posterior, prior), 1))
    features = tf.concat([beliefs['stochastic'],
                          beliefs['deterministic']], -1)
    reconstructed = model._observation_decoder(features)
    log_p_observations = tf.reduce_mean(tf.reduce_sum(
        reconstructed.log_prob(batch['observation'][:, 1:]), 1))
    horizon = tf.cast(tf.shape(batch['observation'])[1], tf.float32) - 1.0
    loss = -log_p_observations + model._kl_scale * tf.maximum(model._free_nats * horizon, kl)
    return reconstructed.mode(), loss, {'stochastic': beliefs['stochastic'][:, -1],
                                        'deterministic': beliefs['deterministic'][:, -1]}


@tf.function
def generate_sequence(model, initial_belief):
    horizon = 50
    actions = tf.zeros([tf.shape(initial_belief['stochastic'])[0], horizon, 1], tf.float32)
    features = model.generate_sequence(initial_belief, horizon, actions=actions)
    return model._observation_decoder(features).mode()


def main():
    tf.random.set_seed(0)
    np.random.seed(0)
    model = models.WorldModel('binary_image', (64, 64, 1), 30, 200, 400, 0)
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=5e-5, clipnorm=100)
    train_dataset = make_dataset('dataset', repeat=1, shuffle=0)
    config = collections.namedtuple('Config', ['log_dir'])('results_5e_5')
    logger = utils.TrainingLogger(config)
    # for i, batch in enumerate(train_dataset):
    #     reconstruct = (i % 100) == 0
    #     grads, loss, log_p_obs, total_kl, reconstructed_sequence = inference_step(
    #         model, batch, reconstruct)
    #     logger['loss'].update_state(loss)
    #     logger['log_probs'].update_state(log_p_obs)
    #     logger['kl'].update_state(total_kl)
    #     optimizer.apply_gradients(zip(grads, model.trainable_variables))
    #     if (i % 50) == 0:
    #         logger.log_metrics(i)
    #     if reconstruct:
    #         logger.log_video(tf.transpose(reconstructed_sequence[:3], [0, 1, 4, 2, 3]).numpy(), i,
    #                          "reconstructed_sequence")
    #         logger.log_video(tf.transpose(batch['observation'][:3], [0, 1, 4, 2, 3]).numpy(), i,
    #                          "true_sequence")
    test_dataset = make_dataset('dataset', 'test')
    for i, batch in enumerate(test_dataset):
        reconstructed_sequence, elbo, last_belief = reconstruct_sequence(model, batch)
        logger['test_elbo'].update_state(elbo)
        if (i % 50) == 0:
            print("Test ELBO: {}".format(logger['test_elbo'].result()))
            logger.log_video(tf.transpose(reconstructed_sequence[:3], [0, 1, 4, 2, 3]).numpy(), i,
                             "test_reconstructed_sequence")
            logger.log_video(tf.transpose(batch['observation'][:3], [0, 1, 4, 2, 3]).numpy(), i,
                             "test_true_sequence")
            generated_sequence = generate_sequence(model, last_belief)
            logger.log_video(tf.transpose(generated_sequence[:3], [0, 1, 4, 2, 3]).numpy(), i,
                             "test_genereated_sequence")
    print("Done!")


if __name__ == '__main__':
    main()
