import matplotlib
import matplotlib.pyplot as plt
import numpy as np


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
