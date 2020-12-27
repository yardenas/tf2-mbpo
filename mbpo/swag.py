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
        for var in var_list:
            self.add_slot(var, "average")
            self.add_slot(var, "average_squared")
            self.add_slot(var, "cov_mat_sqrt")

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
        return self.average_op(
            var, (average_var, averae_squared_var, cov_mat_sqrt_var))

    @tf.function
    def average_op(self, var, average_var):
        average_var, average_squared_var, cov_mat_sqrt_var = average_var
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
            diff = tf.reshape((var - average_value), [-1, 1])
            cat = tf.concat([cov_mat_sqrt_var, tf.transpose(diff)], 0)
            max_num_models = self._get_hyper("max_num_models", tf.int64)
            if (num_snapshots + 1) > max_num_models:
                cat = cat[1:, :]
            return average_var.assign(
                average_value, use_locking=self._use_locking), average_squared_var.assign(
                average_squared_value,
                use_locking=self._use_locking), cov_mat_sqrt_var.assign(
                cat, use_locking=self._use_locking)
        return average_var, average_squared_var, cov_mat_sqrt_var

    @tf.function
    def sample_and_assign_average_vars(self, scale, var_list, override_start=False):
        start_averaging = self._get_hyper("start_averaging", tf.dtypes.int64)
        if self.iterations < start_averaging and not override_start:
            return tf.no_op
        samples = self.sample(scale, var_list)
        assign_ops = []
        for sample, var in zip(samples, var_list):
            try:
                assign_ops.append(
                    var.assign(
                        sample, use_locking=self._use_locking))
            except Exception as e:
                warnings.warn("Unable to assign sample sto {} : {}".format(var, e))
        return tf.group(assign_ops)

    @tf.function
    def sample(self, scale, var_list):
        average_list = []
        squared_average_list = []
        cov_mat_sqrt_list = []
        max_num_models = self._get_hyper("max_num_models", tf.float32)
        for var in var_list:
            average_list.append(self.get_slot(var, "average"))
            squared_average_list.append(self.get_slot(var, "average_squared"))
            cov_mat_sqrt_list.append(self.get_slot(var, "cov_mat_sqrt"))
        average = tf.reshape(tf.concat(average_list, 0), [-1, 1])
        squared_average = tf.reshape(tf.concat(squared_average_list, 0), [-1, 1])
        var_clamp = self._get_hyper("var_clamp", tf.float32)
        variance = tf.clip_by_value(squared_average - average ** 2, var_clamp)
        var_sample = tf.math.sqrt(variance) * tf.random.normal(tf.shape(variance))
        cov_mat_sqrt = tf.concat(cov_mat_sqrt_list, 1)
        cov_sample = tf.linalg.matmul(
            tf.transpose(cov_mat_sqrt),
            tf.random.normal(tf.shape(cov_mat_sqrt)[0])) / ((max_num_models - 1) ** 0.5)
        rand_sample = var_sample + cov_sample
        scale_sqrt = scale ** 0.5
        sample = (average + scale_sqrt * rand_sample)[None, ...]
        return unflatten_like(sample, average_list)


def unflatten_like(tensor, tensor_list):
    output = []
    i = 0
    for item in tensor_list:
        n = tf.size(item)
        output.append(tf.reshape(tensor[:, i:i + n], tf.shape(item)))
        i += n
    return output
