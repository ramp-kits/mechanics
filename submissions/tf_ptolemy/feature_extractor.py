import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

import tensorflow as tf
import tensorflow.contrib.eager as tfe
import pdb


model_labels = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}

_n_lookahead = 50.
_n_burn_in = 500


try:
    tf.enable_eager_execution()
except ValueError:
    pass


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
        l1_penalty = tf.multiply(0.1, tf.reduce_sum(
            tf.abs(self.c[:, 0::3] / self.c[:, 0])))
        return difference + l1_penalty

    def train(self, inputs, outputs, rate):
        with tf.GradientTape() as t:
            current_loss = self.loss(self(inputs), outputs)
        d = t.gradient(current_loss,
                       self.c)
        d -= d * self.mask
        self.assign_parameters(self.c - d * rate)


class FeatureExtractor(object):

    def __init__(self):
        self.n_models = 5
        self.n_components = 10
        self.n_estimators = 10
        self.clf = Pipeline([
            ('pca', PCA(n_components=self.n_components)),
            ('clf', RandomForestClassifier(
                n_estimators=self.n_estimators, random_state=42))
        ])

        self.n_feat = 5
        self.params = np.array([])
        self.window = 0
        self.really_fit = True
        self.models = []
        n = 20
        for i in range(5):
            self.models.append(Ptolemy(n))
            self.models[i].assign_parameters(
                pars=np.array([1., 0.28284271, 3.14159265] * n))

        self.n_components = 2
        self.n_estimators = 2
        self.clf = Pipeline([
            ('pca', PCA(n_components=self.n_components)),
            ('clf', RandomForestClassifier(
                n_estimators=self.n_estimators, random_state=42))
        ])

    def fit(self, X_df, y):
        X_phis = X_df.values[:, 0: 200]
        y_model = y[:, 0]
        self.clf.fit(X_phis, y_model)

    def transform(self, X_df):
        n = X_df.shape[0]
        X = np.zeros(shape=(n, 7), dtype=object)

        # This is where each time series is
        # transformed to be represented by a formula
        # with optimized parameters
        X_phis = X_df.values[:, 0: 200]

        self.c = np.array([])
        self.params = [0]

        predicted_model = self.clf.predict(X_phis)
        predicted_prob = self.clf.predict_proba(X_phis)
        for i in range(n):
            model = self.models[model_labels[predicted_model[i]]]

            self.fit_length = X_phis.shape[1]
            self.y_to_fit = X_phis[i, -self.fit_length:]
            if(self.really_fit):
                epochs = range(100)
                for epoch in epochs:
                    times = np.arange(0., len(self.y_to_fit))
                    model.train(times, self.y_to_fit, rate=0.01)

            self.c = model.c.numpy()

            X[i, 0] = model([_n_burn_in + _n_lookahead]).numpy()
            X[i, 1] = predicted_model[i]
            X[i, 2:7] = predicted_prob[i, :]

            # for p in range(len(model.c.numpy()[0])):
            #    X[i][p + 2] = model.c.numpy()[0][p]
        return X
