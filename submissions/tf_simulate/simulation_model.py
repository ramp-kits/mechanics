import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe


tf.enable_eager_execution()


class Simulation(object):
    def __init__(self, n_var=6):
        # The variables
        self.order = np.ones(shape=(n_var, ))
        self.mask = np.zeros(shape=(n_var, ))
        self.x0 = tfe.Variable(
            tf.random_normal(shape=(n_var, 1),
                             mean=self.order,
                             stddev=self.order),
            dtype=tf.float32)

        self.t = tfe.Variable(
            tf.random_normal(shape=(n_var, n_var),
                             mean=self.order,
                             stddev=self.order),
            dtype=tf.float32)

        self.p = tfe.Variable(
            tf.random_normal(shape=(n_var, n_var),
                             mean=self.order,
                             stddev=self.order),
            dtype=tf.float32)

        first_column = np.ones(shape=(10, n_var))
        first_column[:, 1:] = 0

        self.time_series = tf.constant(first_column, dtype=tf.float32)
        self.unit = np.ones(shape=(n_var,))

    def freeze_parameters(self,
                          mask=np.array([1, 1, 1,
                                         0, 0, 1])):
        self.mask = mask

    # def assign_parameters(self,
    #                       pars=np.array([0, 50,
    #                                      0., np.sqrt(0.02),
    #                                      1., 2.])):
    #     print("pars : ", pars)
    #     print("mask : ", self.mask)
    #     self.c.assign(self.c * (self.unit - self.mask) +
    #                   pars * self.mask)

    #     n_max = 500
    #     self.n = n_max

    def transform(self, x):
        xx = tf.matmul(self.t, x, transpose_b=False)
        return xx

    def propagate(self, x):
        x = tf.matmul(self.p, x, transpose_b=False)
        return x

    def inverse_transform(self, x):
        xx = tf.matmul(tf.matrix_inverse(self.t), x, transpose_b=False)
        return xx

    def __call__(self, times):
        # The formula

        return tf.matmul(self.transform(self.x0),
                         self.time_series, transpose_b=True)
        # output = []
        # x = self.transform(self.x0)
        # # HERE
        # for step in times:
        #     x = self.propagate(x)
        #     output.append(self.inverse_transform(x)[0, 0])
        # return output

    def loss(self, predicted_y, desired_y):
        print("predicted : ", predicted_y)
        print("desired_y : ", desired_y)
        return tf.reduce_mean(tf.square(predicted_y - desired_y))

    def train(self, inputs, outputs, rate):
        with tf.GradientTape() as t:
            current_loss = self.loss(self(inputs), outputs)
        dx0, dt, dp = t.gradient(current_loss,
                                 [self.x0, self.t, self.p])
        print("dx0 : ", dx0)
        self.x0.assign_sub(dx0 * rate)
        print("dt : ", dt)
        self.t.assign_sub(dt * rate)
        print("dp : ", dp)
        # self.p.assign_sub(dp * rate)
        # d -= d * self.mask
        # self.assign_parameters(self.c - d * rate)
