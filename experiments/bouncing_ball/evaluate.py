import argparse

import numpy as np
import tensorflow as tf


# https://github.com/wjmaddox/swa_gaussian/blob/b172d93278fdb92522c8fccb7c6bfdd6f710e4f0
# /experiments/uncertainty/save_calibration_curves.py#L26
def calibration_curve(predictions, labels, num_bins):
    if predictions is None:
        out = None
    else:
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
        out = {"confidence": xs, "accuracy": ys, "p": zs, "ece": ece}
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, required=True)
    parser.add_argument('--num_bins', type=int, default=20)
    args = parser.parse_args()
    npz_array = np.load(args.path)
    predictions, labels = npz_array["predictions"], npz_array["targets"]
    evalutaion = evaluate(predictions, labels)
    print("Done!")


if __name__ == '__main__':
    main()
