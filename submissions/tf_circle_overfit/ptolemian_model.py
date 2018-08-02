from submissions.tf_circle_fit.quick_features import *
import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe

_n_lookahead = 50
_n_burn_in = 500

tf.enable_eager_execution()


class Ptolemy(object):
    def __init__(self, n_epi=2):
        # The variables
        self.order = np.ones(shape=(n_epi * 3, ))
        self.mask = np.zeros(shape=(n_epi * 3, ))
        self.c = tfe.Variable(
            tf.random_normal(shape=(1, n_epi * 3),
                             mean=self.order,
                             stddev=self.order),
            dtype=tf.float32)
        self.unit = np.ones(shape=(n_epi * 3,))

    def freeze_parameters(self,
                          mask=np.array([1, 1, 1,
                                         0, 0, 1])):
        self.mask = mask

    def assign_parameters(self,
                          pars=np.array([1., 0.28284271, 3.14159265,
                                         2., 0.28284271, 0.])):
        print("pars : ", pars)
        print("mask : ", self.mask)
        self.c.assign(self.c * (self.unit - self.mask) +
                      pars * self.mask)

    def __call__(self, times):
        # The formula
        x = tf.matmul(self.c[:, 0::3],
                      tf.cos(tf.matmul(a=self.c[:, 1::3],
                                       b=[times], transpose_a=True) +
                             tf.transpose(self.c[:, 2::3])))
        y = tf.matmul(self.c[:, 0::3],
                      tf.sin(tf.matmul(a=self.c[:, 1::3],
                                       b=[times], transpose_a=True) +
                             tf.transpose(self.c[:, 2::3])))
        phi = tf.atan2(y, x)
        return phi

    def loss(self, predicted_y, desired_y):
        difference = tf.reduce_mean(tf.square(predicted_y - desired_y))
        l1_penalty = tf.multiply(0.1, tf.reduce_sum(tf.abs(self.c[:, 0::3])))
        return difference + l1_penalty

    def train(self, inputs, outputs, rate):
        with tf.GradientTape() as t:
            current_loss = self.loss(self(inputs), outputs)
        d = t.gradient(current_loss,
                       self.c)
        d -= d * self.mask
        self.assign_parameters(self.c - d * rate)

    def test_parameters(self, aa, ww, X):
        z = np.ndarray(shape(len(aa), len(ww)))
        for a, w in np.meshgrid(axis_a, axis_w):
            self.freeze_parameters([1, 1, 1, 1, 1, 1],
                                   [1., 0.28284271, 3.14159265,
                                    a, w, 0.])
            y = X[0, -500:]
            x = np.arange(0., len(y))
            z[i, j] = self.loss(self(x), y).numpy()
        return z
