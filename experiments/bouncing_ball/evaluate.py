import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from experiments.bouncing_ball.utils import show_sequences


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

    # compare_ground_truth_generated(
    #     utils.standardize_video(batch['observation'], config.observation_type, False),
    #     utils.standardize_video(posterior_reconstructed_sequence[:, :conditioning_length],
    #                             config.observation_type, False),
    #     utils.standardize_video(reconstructed, config.observation_type, False),
    #     name=config.log_dir + '/results_' + str(i) + '.svg')


# https://github.com/wjmaddox/swa_gaussian/blob/b172d93278fdb92522c8fccb7c6bfdd6f710e4f0
# /experiments/uncertainty/save_calibration_curves.py#L26
def calibration_curve(npz_arr):
    outputs, labels = npz_arr["predictions"], npz_arr["targets"]
    if outputs is None:
        out = None
    else:
        num_bins = 20
        confidences = np.max(outputs, 1)
        step = (confidences.shape[0] + num_bins - 1) // num_bins
        bins = np.sort(confidences)[::step]
        if confidences.shape[0] % step != 1:
            bins = np.concatenate((bins, [np.max(confidences)]))
        # bins = np.linspace(0.1, 1.0, 30)
        predictions = np.argmax(outputs, 1)
        bin_lowers = bins[:-1]
        bin_uppers = bins[1:]
        accuracies = predictions == labels
        xs = []
        ys = []
        zs = []
        ece = 0.0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = (confidences > bin_lower) * (confidences < bin_upper)
            prop_in_bin = in_bin.mean()
            if prop_in_bin > 0:
                accuracy_in_bin = accuracies[in_bin].mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
                xs.append(avg_confidence_in_bin)
                ys.append(accuracy_in_bin)
                zs.append(prop_in_bin)
        xs = np.array(xs)
        ys = np.array(ys)
        zs = np.array(zs)
        out = {"confidence": xs, "accuracy": ys, "p": zs, "ece": ece}
    return out


def evaluate(ground_truth, reconstructed, generated):
    warmup_length = reconstructed.shape[1]
    generation_length = generated.shape[1]
    assert ground_truth.shape[1] == warmup_length + generation_length
    gt_generation_interval_ravel = tf.reshape(
        tf.convert_to_tensor(ground_truth[:, warmup_length:]), [-1, 1])
    generated_ravel = tf.reshape(
        tf.convert_to_tensor(generated), [-1, 1])
    bce = tf.keras.metrics.BinaryCrossentropy()
    bce.update_state(gt_generation_interval_ravel,
                     generated_ravel)
    nll = bce.result().numpy()
    acc = tf.keras.metrics.BinaryAccuracy()
    acc.update_state(gt_generation_interval_ravel,
                     generated_ravel)
    accuracy = acc.result().numpy()
    return {'nll': nll,
            'accuracy': accuracy}
