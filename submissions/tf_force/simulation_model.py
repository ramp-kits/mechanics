from submissions.tf_force.quick_features import *
import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe

_n_lookahead = 50
_n_burn_in = 500

tf.enable_eager_execution()


class Simulation(object):
    def __init__(self, n_var=6):
        # The variables
        self.order = np.ones(shape=(n_var, ))
        self.mask = np.zeros(shape=(n_var, ))
        self.c = tfe.Variable(
            tf.random_normal(shape=(n_var, ),
                             mean=self.order,
                             stddev=self.order),
            dtype=tf.float32)
        self.unit = np.ones(shape=(n_var,))

    def freeze_parameters(self,
                          mask=np.array([1, 1, 1,
                                         0, 0, 1])):
        self.mask = mask

    def assign_parameters(self,
                          pars=np.array([0, 50,
                                         0., np.sqrt(0.02),
                                         1., 2.])):
        print("pars : ", pars)
        print("mask : ", self.mask)
        self.c.assign(self.c * (self.unit - self.mask) +
                      pars * self.mask)
  
        n_max = 500
        self.n = n_max

    def force(self, x):
        # binary interactions defined here
        f = np.zeros(shape=x.shape)
        # for i, p in enumerate(x):
        #     f[i, :] = 0
        #     for ii, pp in enumerate(x):
        #         rel_pos = p - pp
        #         f[i] += - self.g * \
        #             rel_pos / pow(np.linalg.norm(rel_pos),
        #                           1. + self.k)
        f += - self.g * \
            x / pow(np.linalg.norm(x),
                    1. + self.k)
        return f

    def transform(self, x):
        xx = x[1] * np.array([np.cos(x[0]) - np.sin(x[0]),
                              np.cos(x[0]) + np.sin(x[0])])
        return xx

    def propagate(self, x, v):
        x += v
        v += self.force(x)
        return x

    def inverse_transform(self, x):
        xx = np.array([np.arctan2(x[1], x[0]),
                       np.sqrt(np.dot(x, x))])
        return xx

    def __call__(self, times):
        # The formula

        self.phi = self.c[0]
        self.r = self.c[1]

        self.v = np.array([self.c[2], self.c[3]])
        self.g = self.c[4]
        self.k = self.c[5]

        self.r0 = self.r / 2
        self.v0 = self.v / np.sqrt(2)

        output = []
        x = self.transform([self.phi, self.r])
        x0 = self.transform([self.phi, self.r0])

        for step in times:
            x = self.propagate(x, self.v)
            x0 = self.propagate(x0, self.v0)
            output.append(self.inverse_transform(x - x0)[0])
        return output

    def loss(self, predicted_y, desired_y):
        return tf.reduce_mean(tf.square(predicted_y - desired_y))

    def train(self, inputs, outputs, rate):
        with tf.GradientTape() as t:
            current_loss = self.loss(self(inputs), outputs)
        d = t.gradient(current_loss,
                       [self.c])
        print("d : ", d)
        # d -= d * self.mask
        # self.assign_parameters(self.c - d * rate)
        self.c.assign_sub(d * rate)
