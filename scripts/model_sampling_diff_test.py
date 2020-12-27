import tensorflow as tf


class LinearScaler(tf.Module):
    def __init__(self, weight):
        super().__init__()
        self.w = tf.Variable(weight, trainable=True)

    def __call__(self, inputs):
        return self.w * inputs


def sample_posterior(model, mu, sigma):
    # w = tf.random.normal([], mu, sigma)
    model.w.assign_add(1.0, use_locking=True)


def step(prev_state, policy, model):
    action = policy(tf.stop_gradient(prev_state))
    return model(action + prev_state)


def main():
    mu = tf.Variable(0.0, trainable=False)
    sigma = tf.Variable(1.0, trainable=False)
    model = LinearScaler(4.0)
    policy = LinearScaler(6.0)
    horizon = 2
    posteriors = 1
    with tf.GradientTape(persistent=True) as tape:
        values_mean = 0.0
        for _ in range(posteriors):
            sample_posterior(model, mu, sigma)
            # Assuming that the value of a state is just the state itself (in real life,
            # this mapping
            # as done by another NN).
            next_state = tf.constant(1.0)
            value = 0.0
            for _ in range(horizon):
                next_state = step(next_state, policy, model)
                value += next_state
            values_mean += value / posteriors
    grad = tape.gradient(values_mean, policy.w)
    print("Grad is: {}".format(grad))


if __name__ == '__main__':
    main()
