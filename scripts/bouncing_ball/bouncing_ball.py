import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


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
        yield np.array(sequence_batch, np.float32)


def show_sequence(sequence):
    plt.rcParams['figure.figsize'] = [15, 8]
    batch, length, width, height, depth = sequence.shape
    out = np.zeros((batch * height, length * width, depth))
    for x in range(length):
        for y in range(batch):
            out[y * height:(y + 1) * height, x * width:(x + 1) * width, :] = sequence[y, x, :, :, :]
    out[0::height, :, :] = 1.0
    out[:, 0::width, :] = 1.0
    out[1::height, :, :] = 1.0
    out[:, 1::width, :] = 1.0
    plt.imshow(out, cmap=matplotlib.cm.Greys_r)


def train():
    pass


def main():
    train_data = load_data("data/bouncing_balls_ds0p1")
    a = next(train_data)


if __name__ == '__main__':
    main()
