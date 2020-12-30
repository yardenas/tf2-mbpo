import warnings

import tensorflow as tf
from tensorflow_addons.optimizers import SWA


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
            self.add_slot(var, "average_squared")
            numel = tf.size(var)
            self.add_slot(var, "cov_mat_sqrt", initializer=tf.zeros([max_num_models, numel]))

    def _resource_apply_dense(self, grad, var):
        train_op = self._optimizer._resource_apply_dense(grad, var)
        average_op, average_squared_op, cov_mat_sqrt_op = self._apply_average_op(train_op, var)
        return tf.group(train_op, average_op, average_squared_op, cov_mat_sqrt_op)

    def _resource_apply_sparse(self, grad, var, indices):
        train_op = self._optimizer._resource_apply_sparse(grad, var, indices)
        average_op, average_squared_op, cov_mat_sqrt_op = self._apply_average_op(train_op, var)
        return tf.group(train_op, average_op, average_squared_op, cov_mat_sqrt_op)

    def _resource_apply_sparse_duplicate_indices(self, grad, var, indices):
        train_op = self._optimizer._resource_apply_sparse_duplicate_indices(
            grad, var, indices
        )
        average_op, average_squared_op, cov_mat_sqrt_op = self._apply_average_op(train_op, var)
        return tf.group(train_op, average_op, average_squared_op, cov_mat_sqrt_op)

    def _apply_average_op(self, train_op, var):
        average_var = self.get_slot(var, "average")
        averae_squared_var = self.get_slot(var, "average_squared")
        cov_mat_sqrt_var = self.get_slot(var, "cov_mat_sqrt")
        return self._average_op(
            var, average_var, averae_squared_var, cov_mat_sqrt_var)

    @tf.function
    def _average_op(self, var, average_var, average_squared_var, cov_mat_sqrt_var):
        average_period = self._get_hyper("average_period", tf.dtypes.int64)
        start_averaging = self._get_hyper("start_averaging", tf.dtypes.int64)
        # number of times snapshots of weights have been taken (using max to
        # avoid negative values of num_snapshots).
        num_snapshots = tf.math.maximum(
            tf.cast(0, tf.int64),
            tf.math.floordiv(self.iterations - start_averaging, average_period),
        )
        # The average update should happen iff two conditions are met:
        # 1. A min number of iterations (start_averaging) have taken place.
        # 2. Iteration is one in which snapshot should be taken.
        checkpoint = start_averaging + num_snapshots * average_period
        if self.iterations >= start_averaging and self.iterations == checkpoint:
            # https://github.com/wjmaddox/swa_gaussian/blob/b172d93278fdb92522c8fccb7c6bfdd6f710e4f0
            # /swag/posteriors/swag.py#L148
            num_snapshots = tf.cast(num_snapshots, tf.float32)
            den = (num_snapshots + 1.0)
            average_value = (average_var * num_snapshots + var) / den
            average_squared_value = (average_squared_var * num_snapshots + var ** 2) / den
            cov_mat_sqrt_var_roll = tf.roll(cov_mat_sqrt_var, 1, 0)
            cov_mat_sqrt_var_roll = tf.concat([tf.reshape((var - average_value), [1, -1]),
                                               cov_mat_sqrt_var_roll[1:]], 0)
            return average_var.assign(
                average_value, use_locking=self._use_locking), average_squared_var.assign(
                average_squared_value,
                use_locking=self._use_locking), cov_mat_sqrt_var.assign(
                cov_mat_sqrt_var_roll, use_locking=self._use_locking)
        return average_var, average_squared_var, cov_mat_sqrt_var

    @tf.function
    def sample_and_assign_average_vars(self, scale, var_list, override_start=False):
        start_averaging = self._get_hyper("start_averaging", tf.dtypes.int64)
        average_period = self._get_hyper("average_period", tf.dtypes.int64)
        num_snapshots = tf.math.maximum(
            tf.cast(0, tf.int64),
            tf.math.floordiv(self.iterations - start_averaging, average_period),
        )
        max_num_models = self._get_hyper("max_num_models", tf.int64)
        if not ((self.iterations < start_averaging or num_snapshots < max_num_models)
                and not override_start):
            assign_ops = []
            for var in var_list:
                try:
                    assign_ops.append(
                        var.assign(
                            self.sample(scale, var), use_locking=self._use_locking))
                except Exception as e:
                    warnings.warn("Unable to assign sample to {} : {}".format(var, e))
            tf.group(assign_ops)

    @tf.function(experimental_relax_shapes=True)
    def sample(self, scale, var):
        max_num_models = self._get_hyper("max_num_models", tf.float32)
        average = self.get_slot(var, "average")
        squared_average = self.get_slot(var, "average_squared")
        cov_mat_sqrt = self.get_slot(var, "cov_mat_sqrt")
        var_clamp = self._get_hyper("var_clamp", tf.float32)
        variance = tf.maximum(squared_average - average ** 2, var_clamp)
        var_sample = tf.math.sqrt(variance) * tf.random.normal(tf.shape(variance))
        cov_sample = tf.linalg.matmul(
            cov_mat_sqrt,
            tf.random.normal([tf.shape(cov_mat_sqrt)[0], 1]),
            transpose_a=True) / ((max_num_models - 1) ** 0.5)
        rand_sample = var_sample + tf.reshape(cov_sample, tf.shape(var_sample))
        scale_sqrt = scale ** 0.5
        sample = (average + scale_sqrt * rand_sample)
        return sample


def unflatten_like(tensor, tensor_list):
    output = []
    i = 0
    for item in tensor_list:
        n = tf.size(item)
        output.append(tf.reshape(tensor[:, i:i + n], tf.shape(item)))
        i += n
    return output
