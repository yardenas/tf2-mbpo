import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import experiments.bouncing_ball.utils as utils


# https://github.com/wjmaddox/swa_gaussian/blob/b172d93278fdb92522c8fccb7c6bfdd6f710e4f0
# /experiments/uncertainty/save_calibration_curves.py#L26
def calibration_curve(predictions, labels, num_bins):
    predictions = predictions.ravel()
    predictions = np.stack([1.0 - predictions, predictions], 1)
    labels = labels.ravel()
    confidences = np.max(predictions, 1)
    step = (confidences.shape[0] + num_bins - 1) // num_bins
    bins = np.sort(confidences)[::step]
    if confidences.shape[0] % step != 1:
        bins = np.concatenate((bins, [np.max(confidences)]))
    predictions = np.argmax(predictions, 1)
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
    out = {"calib_confidence": xs, "calib_accuracy": ys, "p": zs, "ece": ece}
    return out


def evaluate(predictions, labels):
    gt_generation_interval_ravel = tf.reshape(
        tf.convert_to_tensor(labels), [-1, 1])
    generated_ravel = tf.reshape(
        tf.convert_to_tensor(predictions), [-1, 1])
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


def summarize_experiment(experiment_runs, args):
    experiment_results = []
    for i, run in enumerate(experiment_runs):
        print('Analyzing experiment: {}'.format(run))
        npz_array = np.load(run)
        predictions, labels = npz_array['predictions'], npz_array['targets']
        utils.compare_ground_truth_generated_2(
            labels, predictions,
            name=os.path.join(args.path, 'results_' + str(i) + '.svg'))
        evaluations = evaluate(predictions, labels)
        calibration_metrics = calibration_curve(predictions, labels, args.num_bins)
        evaluations.update(calibration_metrics)
        experiment_results.append(evaluations)
    all_metrics = {k: np.array(
        [metric[k] for metric in experiment_results]) for k in experiment_results[0].keys()}

    def compute_statistics(metric):
        median = np.median(metric, 0)
        l_percentile = np.percentile(metric, 5, 0)
        u_percentile = np.percentile(metric, 95, 0)
        return {'median': median,
                'l_percentile': l_percentile,
                'u_percentile': u_percentile}

    statistics = {k: compute_statistics(v) for k, v in all_metrics.items() if k not in
                  ['calib_confidence', 'calib_accuracy', 'p']}
    for k, v in statistics.items():
        print('{}: {}'.format(k, v))

    fig = plt.figure(figsize=(11 / 3, 3.5))
    ax = fig.add_subplot()
    for result in experiment_results:
        ax.plot(result['calib_confidence'],
                result['calib_accuracy'], 'o', ls='-', c='C0', alpha=0.3)
    ax.set_xlabel('Confidence')
    ax.set_ylabel('Accuracy')
    ax.set_title(args.title_name)
    ax.autoscale(False)
    ax.plot([0, 1], [0, 1], 'k--')
    ax.grid(True)
    fig.tight_layout()
    plt.savefig(os.path.join(args.path, args.title_name) + '.svg')


def collect_experiment_runs(path):
    experiment_runs = []
    for root, _, filenames in os.walk(path):
        experiment_runs += [os.path.join(root, run) for run in filenames
                            if run == 'test_results.npz']
    return experiment_runs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, required=True)
    parser.add_argument('--num_bins', type=int, default=40)
    parser.add_argument('--title_name', required=True)
    args = parser.parse_args()
    experiment_runs = collect_experiment_runs(args.path)
    summarize_experiment(experiment_runs, args)
    print("Done!")


if __name__ == '__main__':
    main()
