from submissions.tf_circle_fit.fit_features import *
import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe

_n_lookahead = 50
_n_burn_in = 500

tf.enable_eager_execution()


class Ptolemy(object):
    def __init__(self, n_epi=2):
        # The variables
        self.order_a = 1.
        self.order_w = 0.2
        self.order_p = 3.

        self.a = tfe.Variable(
            tf.random_normal(shape=(1, n_epi),
                             mean=self.order_a,
                             stddev=self.order_a),
            dtype=tf.float32)
        self.w = tfe.Variable(
            tf.random_normal(shape=(1, n_epi),
                             mean=self.order_w,
                             stddev=self.order_w),
            dtype=tf.float32)
        self.p = tfe.Variable(
            tf.random_normal(shape=(1, n_epi),
                             mean=self.order_p,
                             stddev=self.order_p),
            dtype=tf.float32)
        self.freeze_parameters()

    def freeze_parameters(self,
                          mask=[[1, 1, 1], [0, 0, 1]],
                          pars=[[1., 0.28284271, 3.14159265],
                                [2., 0.28284271, 0.]]):
        unit = np.array([1, 1])

        mask = np.array(mask)
        pars = np.array(pars)

        def assign(x, i):
            x.assign(x * (unit - mask[:, i]) + pars[:, i] * mask[:, i])

        assign(self.a, 0)
        assign(self.w, 1)
        assign(self.p, 2)

    def __call__(self, t):
        # The formula
        x = tf.matmul(self.a,
                      tf.cos(tf.matmul(a=self.w, b=[t], transpose_a=True) +
                             tf.transpose(self.p)))
        y = tf.matmul(self.a,
                      tf.sin(tf.matmul(a=self.w, b=[t], transpose_a=True) +
                             tf.transpose(self.p)))
        phi = tf.atan2(y, x)
        return phi

    def loss(self, predicted_y, desired_y):
        return tf.reduce_mean(tf.square(predicted_y - desired_y))

    def train(self, inputs, outputs, learning_rate):
        with tf.GradientTape() as t:
            current_loss = self.loss(self(inputs), outputs)
        d = t.gradient(current_loss,
                       [self.a, self.w, self.p])
        self.a.assign_sub(learning_rate * self.order_a * d[0])
        self.w.assign_sub(learning_rate * self.order_w * d[1])
        self.p.assign_sub(learning_rate * self.order_p * d[2])
        # self.a.assign_sub(learning_rate * d[0])
        # self.w.assign_sub(learning_rate * d[1])
        # self.p.assign_sub(learning_rate * d[2])
        self.freeze_parameters()

    def test_parameters(self, aa, ww, X):
        z = np.ndarray(shape(len(aa), len(ww)))
        for a, w in np.meshgrid(axis_a, axis_w):
            self.freeze_parameters([[1, 1, 1], [1, 1, 1]],
                                   [[1., 0.28284271, 3.14159265],
                                    [a, w, 0.]])
            y = X[0, -500:]
            x = np.arange(0., len(y))
            z[i, j] = self.loss(self(x), y).numpy()
        return z
