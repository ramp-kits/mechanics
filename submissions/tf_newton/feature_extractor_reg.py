import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

import tensorflow as tf
import tensorflow.contrib.eager as tfe

label_names = np.array(['A', 'B', 'C', 'D'])

_n_lookahead = 100
_n_burn_in = 20

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

    def __call__(self, n_steps):
        # The formula
        self.phi = self.c[0]

        self.r = self.c[1]
        self.v = np.array([self.c[2], self.c[3]])
        self.g = self.c[4]
        self.k = self.c[5]

        self.phi0 = 0
        self.r0 = self.r / 2
        self.v0 = self.v / np.sqrt(2)

        output = []
        x = self.transform([self.phi, self.r])
        x0 = self.transform([self.phi0, self.r0])

        print("n_steps : ", n_steps)
        for step in np.arange(n_steps, dtype=int):
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


class FeatureExtractor(object):

    def __init__(self):
        self.n_feat = 6
        self.params = np.array([])
        self.window = 0
        self.really_fit = True
        self.models = []
        n = 12
        for i in range(5):
            self.models.append(Simulation(n))
            self.models[i].assign_parameters(
                pars=np.array([1.] * n))

        self.n_components = 2
        self.n_estimators = 2
        self.clf = Pipeline([
            ('pca', PCA(n_components=self.n_components)),
            ('clf', RandomForestClassifier(
                n_estimators=self.n_estimators, random_state=42))
        ])

    def fit(self, X_df, y):
        pass

    def transform(self, X_df):
        n = X_df.shape[0]
        X = np.zeros(shape=(n, 1))

        # This is where each time series is
        # transoformed to be represented by a formula
        # with optimized parameters

        X_system = X_df.values[:, -4:]
        X_phis = X_df.values[:, 0: -4]

        self.c = np.array([])
        self.params = [0]

        for i in range(n):
            self.y_all = X_phis[i, :]
            self.y_to_fit = self.y_all
            model = self.models[np.argmax(X_system[i])]

            self.fit_length = X_phis.shape[1]
            self.y_to_fit = X_phis[i, -self.fit_length:]

            n_steps = _n_burn_in + _n_lookahead

            # cannot train at the moment
            model.train(n_steps, self.y_to_fit, rate=0.01)

            self.c = model.c.numpy()

            X[i][0] = model(n_steps)[-1]
        return X
