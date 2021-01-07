import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from mbpo.ensemble_world_model import EnsembleWorldModel
from mbpo.swag_world_model import SwagWorldModel
from mbpo.swag_single_step_prediction_model import SwagSingleStepPredictionModel
import mbpo.utils as utils
import scripts.train as train_utils

rng = tf.random.Generator.from_seed(0)


# https://github.com/davidsandberg/rl_ssms/blob/master/bouncing_ball_prediction.ipynb
def load_sequence(filename):
    data = np.load(filename)
    return data['batch']


def load_data(data_dir, prefix='train', stack_observation=1):
    data = []
    for filename in os.listdir(data_dir):
        file_path = os.path.join(data_dir, filename)
        if filename.startswith(prefix):
            data.append(file_path)
    for file_path in data:
        sequence_batch = load_sequence(file_path)
        for i in range(0, sequence_batch.shape[0], stack_observation):
            observation = np.array(
                sequence_batch[i:i + stack_observation],
                np.float32).transpose([-1, 1, 2, 3, 0]).squeeze(0) if stack_observation > 1 else \
                np.array(sequence_batch[i], np.float32)[..., None]
            yield {'observation': observation,
                   'action': np.zeros([sequence_batch[i].shape[0] - 1, 1], np.float32),
                   'reward': np.zeros([sequence_batch[i].shape[0] - 1, ], np.float32),
                   'terminal': np.zeros([sequence_batch[i].shape[0] - 1, ], np.float32)}


def show_sequence(sequence, figname=None):
    plt.rcParams['figure.figsize'] = [15, 8]
    batch, length, width, height, depth = sequence.shape
    out = np.zeros((batch * height, length * width, depth))
    for x in range(length):
        for y in range(batch):
            out[y * height:(y + 1) * height, x * width:(x + 1) * width, :] = sequence[y, x, :,
                                                                             :, :]
    out[0::height, :, :] = 0.5
    out[:, 0::width, :] = 0.5
    out[1::height, :, :] = 0.5
    out[:, 1::width, :] = 0.5
    plt.imshow(out, cmap=matplotlib.cm.Greys_r)
    if figname is not None:
        plt.savefig(figname)


def make_dataset(dir, prefix='train', repeat=0, shuffle=0, seed=0, batch_size=16,
                 stack_observations=1):
    dataset = tf.data.Dataset.from_generator(lambda: load_data(dir, prefix, stack_observations),
                                             output_types={'observation': np.float32,
                                                           'action': np.float32,
                                                           'reward': np.float32,
                                                           'terminal': np.float32},
                                             output_shapes={'observation': [50, 64, 64,
                                                                            stack_observations],
                                                            'action': [49, stack_observations],
                                                            'reward': [49],
                                                            'terminal': [49]})
    dataset = dataset.map(lambda data: {k: tf.where(
        utils.preprocess(v) > 0.0, 1.0, 0.0) for k, v in data.items()})
    if shuffle:
        dataset = dataset.shuffle(shuffle, seed, reshuffle_each_iteration=True)
    if repeat:
        dataset = dataset.repeat(repeat)
    dataset = dataset.prefetch(1024)
    dataset = dataset.batch(batch_size)
    return dataset


def main():
    tf.random.set_seed(0)
    np.random.seed(0)
    config_dict = train_utils.define_config()
    config_dict['observation_type'] = 'binary_image'
    config_dict['model_learning_rate'] = 5e-4
    config_dict['grad_clip_norm'] = 100.0
    config_dict['posterior_samples'] = 5
    config_dict['seed'] = 0
    config_dict['log_dir'] = 'results_ensemble'
    config = train_utils.make_config(config_dict)
    logger = utils.TrainingLogger(config)
    model = SwagSingleStepPredictionModel(config, logger, (64, 64, 1))
    train_dataset = make_dataset('dataset', repeat=2, shuffle=5000,
                                 batch_size=16)
    global_step = 0
    for i, batch in enumerate(train_dataset):
        reconstruct = (i % 100) == 0
        model.train(batch, reconstruct, step=i)
        if (i % 50) == 0:
            logger.log_metrics(i)
        global_step = i
    horizon = 50
    test_dataset = make_dataset('dataset', 'test')
    for i, batch in enumerate(test_dataset):
        actions = tf.zeros([tf.shape(batch['action'])[0], horizon, 1])
        posterior_reconstructed_sequence, beliefs = model.reconstruct_sequences_posterior(batch)
        last_belief = {'stochastic': beliefs['stochastic'][:, -1],
                       'deterministic': beliefs['deterministic'][:, -1]}
        if (i % 50) == 0:
            logger.log_video(tf.transpose(
                posterior_reconstructed_sequence[:4], [0, 1, 4, 2, 3]).numpy(), i + global_step,
                             "test_reconstructed_sequence")
            logger.log_video(tf.transpose(
                batch['observation'][:4], [0, 1, 4, 2, 3]).numpy(), i + global_step,
                             "test_true_sequence")
            model.generate_sequences_posterior(
                last_belief, 50, actions=actions, log_sequences=True, step=i + global_step)
            logger.log_metrics(global_step)
    print("Done!")


if __name__ == '__main__':
    main()
