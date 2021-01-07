import warnings

import tensorflow as tf
from tensorflow_addons.optimizers import SWA
import tensorflow_probability as tfp


class SWAG(SWA):
    def __init__(self,
                 optimizer,
                 start_averaging=0,
                 average_period=10,
                 max_num_models=20,
                 var_clamp=1e-30,
                 **kwargs):
        super(SWAG, self).__init__(
            optimizer,
            start_averaging,
            average_period,
            "SWAG", **kwargs)
        self._set_hyper("max_num_models", max_num_models)
        self._set_hyper("var_clamp", var_clamp)

    def _create_slots(self, var_list):
        self._optimizer._create_slots(var_list=var_list)
        max_num_models = self._get_hyper("max_num_models", tf.int32)
        for var in var_list:
            self.add_slot(var, "mean")
            self.add_slot(var, "mean_squared")
            numel = tf.size(var)
            self.add_slot(var, "cov_mat_sqrt", initializer=tf.zeros([max_num_models, numel]))

    def _resource_apply_dense(self, grad, var):
        train_op = self._optimizer._resource_apply_dense(grad, var)
        mean_op, mean_squared_op, cov_mat_sqrt_op = self._apply_mean_op(train_op, var)
        return tf.group(train_op, mean_op, mean_squared_op, cov_mat_sqrt_op)

    def _resource_apply_sparse(self, grad, var, indices):
        train_op = self._optimizer._resource_apply_sparse(grad, var, indices)
        mean_op, mean_squared_op, cov_mat_sqrt_op = self._apply_mean_op(train_op, var)
        return tf.group(train_op, mean_op, mean_squared_op, cov_mat_sqrt_op)

    def _resource_apply_sparse_duplicate_indices(self, grad, var, indices):
        train_op = self._optimizer._resource_apply_sparse_duplicate_indices(
            grad, var, indices
        )
        mean_op, mean_squared_op, cov_mat_sqrt_op = self._apply_mean_op(train_op, var)
        return tf.group(train_op, mean_op, mean_squared_op, cov_mat_sqrt_op)

    def _apply_mean_op(self, train_op, var):
        mean_var = self.get_slot(var, "mean")
        averae_squared_var = self.get_slot(var, "mean_squared")
        cov_mat_sqrt_var = self.get_slot(var, "cov_mat_sqrt")
        return self._mean_op(
            var, mean_var, averae_squared_var, cov_mat_sqrt_var)

    @tf.function
    def _mean_op(self, var, mean_var, mean_squared_var, cov_mat_sqrt_var):
        mean_period = self._get_hyper("average_period", tf.dtypes.int64)
        start_averaging = self._get_hyper("start_averaging", tf.dtypes.int64)
        # number of times snapshots of weights have been taken (using max to
        # avoid negative values of num_snapshots).
        num_snapshots = tf.math.maximum(
            tf.cast(0, tf.int64),
            tf.math.floordiv(self.iterations - start_averaging, mean_period),
        )
        # The mean update should happen iff two conditions are met:
        # 1. A min number of iterations (start_averaging) have taken place.
        # 2. Iteration is one in which snapshot should be taken.
        checkpoint = start_averaging + num_snapshots * mean_period
        if self.iterations >= start_averaging and self.iterations == checkpoint:
            # https://github.com/wjmaddox/swa_gaussian/blob/b172d93278fdb92522c8fccb7c6bfdd6f710e4f0
            # /swag/posteriors/swag.py#L148
            num_snapshots = tf.cast(num_snapshots, tf.float32)
            den = (num_snapshots + 1.0)
            mean_value = (mean_var * num_snapshots + var) / den
            mean_squared_value = (mean_squared_var * num_snapshots + var ** 2) / den
            cov_mat_sqrt_var_roll = tf.roll(cov_mat_sqrt_var, 1, 0)
            cov_mat_sqrt_var_roll = tf.concat([tf.reshape((var - mean_value), [1, -1]),
                                               cov_mat_sqrt_var_roll[1:]], 0)
            return mean_var.assign(
                mean_value, use_locking=self._use_locking), mean_squared_var.assign(
                mean_squared_value,
                use_locking=self._use_locking), cov_mat_sqrt_var.assign(
                cov_mat_sqrt_var_roll, use_locking=self._use_locking)
        return mean_var, mean_squared_var, cov_mat_sqrt_var

    @tf.function
    def sample_and_assign(self, scale, var_list, override_start=False, seed=None):
        start_averaging = self._get_hyper("start_averaging", tf.dtypes.int64)
        mean_period = self._get_hyper("average_period", tf.dtypes.int64)
        num_snapshots = tf.math.maximum(
            tf.cast(0, tf.int64),
            tf.math.floordiv(self.iterations - start_averaging, mean_period),
        )
        max_num_models = self._get_hyper("max_num_models", tf.int64)
        if ((self.iterations >= start_averaging and num_snapshots >= max_num_models)
                or override_start):
            assign_ops = []
            for var in var_list:
                try:
                    assign_ops.append(
                        var.assign(
                            self.sample(scale, var, seed), use_locking=self._use_locking))
                except Exception as e:
                    warnings.warn("Unable to assign sample to {} : {}".format(var, e))
            tf.group(assign_ops)

    @tf.function(experimental_relax_shapes=True)
    def sample(self, scale, var, seed):
        var_seed, cov_seed = tfp.random.split_seed(seed, 2, "observe")
        max_num_models = self._get_hyper("max_num_models", tf.float32)
        mean = self.get_slot(var, "mean")
        squared_mean = self.get_slot(var, "mean_squared")
        cov_mat_sqrt = self.get_slot(var, "cov_mat_sqrt")
        var_clamp = self._get_hyper("var_clamp", tf.float32)
        variance = tf.maximum(squared_mean - mean ** 2, var_clamp)
        var_sample = tf.math.sqrt(variance) * tf.random.normal(tf.shape(variance), seed=var_seed)
        cov_sample = tf.linalg.matmul(
            cov_mat_sqrt,
            tf.random.normal([tf.shape(cov_mat_sqrt)[0], 1], seed=cov_seed),
            transpose_a=True) / ((max_num_models - 1) ** 0.5)
        rand_sample = var_sample + tf.reshape(cov_sample, tf.shape(var_sample))
        scale_sqrt = scale ** 0.5
        sample = (mean + scale_sqrt * rand_sample)
        return sample


def unflatten_like(tensor, tensor_list):
    output = []
    i = 0
    for item in tensor_list:
        n = tf.size(item)
        output.append(tf.reshape(tensor[:, i:i + n], tf.shape(item)))
        i += n
    return output
