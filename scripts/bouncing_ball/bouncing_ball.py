import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import mbpo.utils as utils
import scripts.train as train_utils
from mbpo.swag_feed_forward_model import SwagFeedForwardModel
from mbpo.swag_world_model import SwagWorldModel

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
        for i in range(0, sequence_batch.shape[0]):
            observation = np.array(
                sequence_batch[i], np.float32).squeeze() \
                .reshape([-1, stack_observation, 64, 64]) \
                .transpose([0, 2, 3, 1]) if \
                stack_observation > 1 else np.array(sequence_batch[i], np.float32)[..., None]
            yield {'observation': observation,
                   'action': np.zeros([observation.shape[0] - 1, 1], np.float32),
                   'reward': np.zeros([observation.shape[0] - 1, ], np.float32),
                   'terminal': np.zeros([observation.shape[0] - 1, ], np.float32)}


def make_dataset(dir, prefix='train', repeat=0, shuffle=0, seed=0, batch_size=16,
                 stack_observations=1):
    assert (50 % stack_observations) == 0, 'Should be divisable by 50.'
    horizon_length = 50 // stack_observations
    dataset = tf.data.Dataset.from_generator(lambda: load_data(dir, prefix, stack_observations),
                                             output_types={'observation': np.float32,
                                                           'action': np.float32,
                                                           'reward': np.float32,
                                                           'terminal': np.float32},
                                             output_shapes={'observation': [horizon_length, 64, 64,
                                                                            stack_observations],
                                                            'action': [horizon_length - 1, 1],
                                                            'reward': [horizon_length - 1],
                                                            'terminal': [horizon_length - 1]})
    dataset = dataset.map(lambda data: {k: tf.where(
        utils.preprocess(v) > 0.0, 1.0, 0.0) for k, v in data.items()})
    if shuffle:
        dataset = dataset.shuffle(shuffle, seed, reshuffle_each_iteration=True)
    if repeat:
        dataset = dataset.repeat(repeat)
    dataset = dataset.prefetch(1024)
    dataset = dataset.batch(batch_size)
    return dataset


def compare_ground_truth_generated(ground_truth, reconstructed, generated,
                                   reconstruct_skip=2, generate_skip=4, name=''):
    warmup_length = reconstructed.shape[1]
    generation_length = generated.shape[1]
    assert ground_truth.shape[1] == warmup_length + generation_length
    fig = plt.figure(figsize=(15, 4), constrained_layout=True)
    spec = fig.add_gridspec(ncols=2, nrows=1, width_ratios=[reconstruct_skip, generate_skip],
                            height_ratios=[1])
    ax1 = fig.add_subplot(spec[0, 0])
    input_ = np.stack([ground_truth[0, :warmup_length:reconstruct_skip],
                       reconstructed[0, ::reconstruct_skip]], 0)
    show_sequences(input_, ax1)
    ax2 = fig.add_subplot(spec[0, 1], sharey=ax1)
    generation = np.stack([ground_truth[0, warmup_length::generate_skip],
                           generated[0, ::generate_skip]], 0)
    show_sequences(generation, ax2)
    plt.setp(ax2.get_yticklabels(), visible=False)
    _, _, width, height, _ = ground_truth.shape
    warmup_stamps = np.arange(1, warmup_length, reconstruct_skip)
    ax1.set_yticks([height / 2, height * 3 / 2])
    ax1.set_yticklabels(['True', 'Predicted'])
    ax1.set_xticks(np.arange(1, len(warmup_stamps) * 2, 2) * width / 2)
    ax1.set_xticklabels(warmup_stamps)
    ax1.set_title('Warmup')
    generation_stamps = np.arange(1, generation_length, generate_skip) + warmup_stamps[-1]
    ax2.set_xticks(np.arange(1, len(generation_stamps) * 2, 2) * width / 2)
    ax2.set_xticklabels(generation_stamps)
    ax2.tick_params(axis='y', which='both', left=False, right=False)
    ax2.set_title('Generation')
    plt.savefig(name)


def show_sequences(sequence, ax):
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
    ax.imshow(out, cmap=matplotlib.cm.Greys_r)


def choose_model(model_name):
    if model_name == 'RSSM':
        return SwagWorldModel
    elif model_name == 'FeedForward':
        return SwagFeedForwardModel
    else:
        raise RuntimeError('Wrong model name provided')


def main():
    tf.random.set_seed(0)
    np.random.seed(0)
    config_dict = train_utils.define_config()
    config_dict['observation_type'] = 'binary_image'
    config_dict['model_learning_rate'] = 5e-5
    config_dict['seed'] = 0
    config_dict['log_dir'] = 'results_ensemble'
    config_dict['n_step_loss'] = False
    config_dict['model_name'] = 'FeedForward'
    config_dict['stack_observations'] = 1
    config = train_utils.make_config(config_dict)
    logger = utils.TrainingLogger(config)
    model = choose_model(config.model_name)(config, logger, (64, 64, config.stack_observations))
    train_dataset = make_dataset('dataset', repeat=2, shuffle=5000,
                                 batch_size=16, stack_observations=config.stack_observations)
    global_step = 0
    for i, batch in enumerate(train_dataset):
        reconstruct = (i % 100) == 0
        model.train(batch, reconstruct, step=i)
        if (i % 50) == 0:
            logger.log_metrics(i)
        global_step = i
    test_dataset = make_dataset('dataset', 'test', stack_observations=config.stack_observations)
    for i, batch in enumerate(test_dataset):
        sequence_length = tf.shape(batch['observation'])[1]
        conditioning_length = sequence_length // 5
        horizon = sequence_length - conditioning_length
        actions = tf.zeros([tf.shape(batch['action'])[0], horizon, 1])
        posterior_reconstructed_sequence, beliefs = model.reconstruct_sequences_posterior(batch)
        last_belief = {'stochastic': beliefs['stochastic'][:, conditioning_length],
                       'deterministic': beliefs['deterministic'][:, conditioning_length]}
        if (i % 50) == 0:
            logger.log_video(utils.standardize_video(posterior_reconstructed_sequence[:4],
                                                     config.observation_type), i + global_step,
                             "test_reconstructed_sequence")
            logger.log_video(utils.standardize_video(batch['observation'][:4],
                                                     config.observation_type),
                             i + global_step, "test_true_sequence")
            _, reconstructed = model.generate_sequences_posterior(
                last_belief, horizon, actions=actions, log_sequences=True, step=i + global_step)
            logger.log_metrics(global_step)
            compare_ground_truth_generated(
                utils.standardize_video(batch['observation'], config.observation_type, False),
                utils.standardize_video(posterior_reconstructed_sequence[:, :conditioning_length],
                                        config.observation_type, False),
                utils.standardize_video(reconstructed, config.observation_type, False),
                name=config.log_dir + '/results_' + str(i) + '.svg')
    logger.log_metrics(i)
    print("Done!")


if __name__ == '__main__':
    main()
