import unittest

import numpy as np


class TestLambdaComputation(unittest.TestCase):
    def test_lambda(self):
        values = np.array([-0.2996, 0.31, -0.02774, -0.05212, 0.2573, -0.04715,
                           -0.199, 0.10864, 0.1254, -0.07367, -0.0657, -0.1157,
                           -0.3335, -0.2048, 0.1471])
        horizon = values.shape[0]
        rewards = np.array([-0.2603, 0.07007, 0.2067, 0.253, 0.1948, 0.2153,
                            0.1962, 0.3525, -0.002075, -0.3308, 0.3296, -0.3474,
                            -0.2815, -0.10004, -0.5107])
        terminals = np.zeros_like(values)
        lambda_ = 0.95
        discount = 0.99
        v_lambda = values[-1] * (1.0 - terminals[-1])
        next_values = values[1:]
        values_lambda = np.empty(horizon - 1, np.float32)
        for t in range(horizon - 2, -1, -1):
            td = rewards[t] + (1.0 - terminals[t]) * (1.0 - lambda_) * discount * next_values[t]
            v_lambda = td + v_lambda * lambda_ * discount
            values_lambda[t] = v_lambda
        gt_values = np.array([0.5723, 0.868, 0.8496, 0.686, 0.4468, 0.27,
                              0.0687, -0.1411, -0.5312, -0.5586, -0.239, -0.598,
                              -0.2488, 0.04565])
        self.assertTrue(np.allclose(gt_values, values_lambda, 1.e-3, 1.e-3))


if __name__ == '__main__':
    unittest.main()
