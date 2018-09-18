import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

import tensorflow as tf
import tensorflow.contrib.eager as tfe

label_names = np.array(['A', 'B', 'C', 'D'])

_n_lookahead = 50.
_n_burn_in = 500

tf.enable_eager_execution()


def find_local_extrema(x):
    n = 3
    maxima = np.array([])
    minima = np.array([])
    loops = np.array([])

    p_prev = np.zeros(n)
    d_prev = np.zeros(n)
    c_prev = np.zeros(n)

    for i, p in enumerate(x):
        d = p - p_prev[0]
        if(d < - np.pi):
            loops = np.append(loops, i)
            d = d + 2. * np.pi

        if(d > np.pi):
            d = d - 2. * np.pi
        c = d - d_prev[0]

        if((d * d_prev[0]) < 0):
            if(c < 0):
                maxima = np.append(maxima, i)
            else:
                minima = np.append(minima, i)

        p_prev = np.append(p, p_prev[:n])
        d_prev = np.append(d, d_prev[:n])
        c_prev = np.append(c, c_prev[:n])
#    print(loops)
    return maxima, minima, loops

def qualitative_features(x):
    "Fitting features..."
    a = np.array([1., 1.])
    w = np.array([1., 1.])
    p = np.array([1., 1.])

    nc = 1
    c = np.array([])

    # Find retrogrades
    maxima, minima, loops = find_local_extrema(x)

    # Find frequencies
    if(len(loops) < 1):
        nc = 1
        c = np.array([[1., 0.28284271, 3.14159265]])
    else:
        dist = loops[-1] - loops[0]
        nloop = len(loops) - 1
        if(dist > 0 & nloop > 0):
            w[0] = 2. * np.pi / (dist / nloop)
        else:
            nc = 1
            c = np.array([1., 0.28284271, 3.14159265])
        dist = maxima[-1] - maxima[0]
        nloop = len(maxima) - 1
        if(nloop > 0):
            w[1] = 2. * np.pi / (dist / nloop) + w[0]
            nc = 2

        # Find phases
        p[0] = loops[0] * w[0] - np.pi
        p[1] = (maxima[0]) * w[1] - p[0]

        # Find amplitudes
        a[0] = 1.
        a[1] = 0.5
        c = np.array([a[0], w[0], p[0], a[1], w[1], p[1]])
    return nc, c


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


class FeatureExtractor(object):

    def __init__(self):
        self.n_feat = 6
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

        X_feat = np.ndarray(shape=(n, self.n_feat))
        for i in range(n):
            self.y_all = X_phis[i, :]
            if(False):
                maxima, minima, loops = find_local_extrema(self.y_all)
                X_feat[i] = [len(maxima), len(minima), len(loops),
                             maxima[0], minima[0], loops[0]]

        for i in range(n):
            self.y_all = X_phis[i, :]
            self.y_to_fit = self.y_all
            model = self.models[np.argmax(X_system[i])]

            nc, cc = qualitative_features(self.y_all)
            self.c = cc
            self.fit_length = X_phis.shape[1]
            self.y_to_fit = X_phis[i, -self.fit_length:]
            if(self.really_fit):
                epochs = range(100)
                for epoch in epochs:
                    # cs = np.append(cs, self.model.c.numpy(), axis=0)
                    times = np.arange(0., len(self.y_to_fit))
                    # model_result = model(times)
                    # print("model result : ", model_result)
                    # current_loss = model.loss(model_result, self.y_to_fit)
                    model.train(times, self.y_to_fit, rate=0.01)
                    # print('Model : %d Epoch %2d: w=%s loss=%2.5f' %
                    #      (X_model[i], epoch,
                    #       str(model.c.numpy()),
                    #       current_loss))
            self.c = model.c.numpy()
            X[i][0] = model([_n_burn_in + _n_lookahead]).numpy()
            # X[i][1] = X_model[i]
            # for p in range(len(model.c.numpy()[0])):
            #    X[i][p + 2] = model.c.numpy()[0][p]
        return X
