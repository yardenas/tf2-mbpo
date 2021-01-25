import matplotlib.pyplot as plt
import numpy as np


def show_sequences(sequence, ax, color=(1.0, 1.0, 1.0), alpha=1.0, label=None):
    batch, length, width, height, depth = sequence.shape
    out = np.zeros((batch * height, length * width, 3))
    for x in range(length):
        for y in range(batch):
            rgb = sequence[y, x, :, :, :] * np.array(color)
            out[y * height:(y + 1) * height, x * width:(x + 1) * width, :] = rgb
    out[0::height, :, :] = 1.0
    out[:, 0::width, :] = 1.0
    out[1::height, :, :] = 1.0
    out[:, 1::width, :] = 1.0
    extent = 0, length * width, 0, batch * height
    ax.imshow(out, alpha=alpha, extent=extent)
    if label:
        ax.plot(0, 0, 'o', c=color, label=str(label), markersize=0,
                markeredgecolor='black')


def funky_multivariate_normal_pdf(pos, mu, sigma, radius):
    n = mu.shape[0]
    sigma_det = sigma ** 2
    sigma_inv = np.eye(n) / sigma
    N = np.sqrt((2 * np.pi) ** n * sigma_det)
    r = (pos - mu) / np.linalg.norm(pos - mu, axis=2, keepdims=True) * radius
    fac = np.einsum('...k,kl,...l->...', pos - mu - r, sigma_inv, pos - mu - r)
    mask = np.einsum('...k,kl,...l->...', pos - mu, np.eye(n), pos - mu) < radius ** 2
    fac = np.where(mask, 0.0, fac)
    return np.exp(-fac / 2) / N


def compare_ground_truth_generated(ground_truth, reconstructed, generated,
                                   reconstruct_skip=2, generate_skip=4, name=''):
    warmup_length = reconstructed.shape[1]
    generation_length = generated.shape[1]
    assert ground_truth.shape[1] == warmup_length + generation_length
    fig = plt.figure(figsize=(11, 3.5), constrained_layout=True)
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
    ax1.set_yticklabels(['Predict', 'True'])
    ax1.set_xticks(np.arange(1, len(warmup_stamps) * 2, 2) * width / 2)
    ax1.set_xticklabels(warmup_stamps)
    ax1.set_title('Warmup')
    generation_stamps = np.arange(1, generation_length, generate_skip) + warmup_stamps[-1]
    ax2.set_xticks(np.arange(1, len(generation_stamps) * 2, 2) * width / 2)
    ax2.set_xticklabels(generation_stamps)
    ax2.tick_params(axis='y', which='both', left=False, right=False)
    ax2.set_title('Generation')
    plt.savefig(name)


def compare_ground_truth_generated_2(ground_truth, generated, skip=4, name=''):
    fig = plt.figure(figsize=(11, 3))
    ax = fig.add_subplot()
    show_sequences(generated[:2, ::skip], ax, label='Generated')
    show_sequences(ground_truth[:2, ::skip], ax, color=(1.0, 0.0, 0.0), alpha=0.4,
                   label='Ground Truth')
    lgnd = ax.legend(*ax.get_legend_handles_labels(),
                     loc='center', ncol=2, bbox_to_anchor=(0.5, -0.35))
    for handle in lgnd.legendHandles:
        handle._legmarker.set_markersize(6)
    _, length, width, height, _ = ground_truth.shape
    stamps = np.arange(1, length, skip)
    ax.set_xticks(np.arange(1, len(stamps) * 2, 2) * width / 2)
    ax.set_xticklabels(stamps)
    ax.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
    ax.set_xlabel('Time horizon')
    fig.tight_layout()
    fig.subplots_adjust(top=1.15)
    plt.savefig(name)
