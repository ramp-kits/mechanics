from sklearn.base import BaseEstimator
from scipy.optimize import minimize
from submissions.tf_circle_fit.fit_features import *
import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe

_n_lookahead = 50
_n_burn_in = 500

tf.enable_eager_execution()


class Ptolemy(object):
    def __init__(self):
        # Initialize variable to (5.0, 0.0)
        # In practice, these should be initialized to random values.
        # self.c = tfe.Variable(np.array([51, 0.02, 3., 99, 0.01, 0.01]))
        self.a = tfe.Variable([[51., 99.]], dtype=tf.float32)
        self.w = tfe.Variable([[0.02, 0.01]], dtype=tf.float32)
        self.p = tfe.Variable([[2., 0.01]], dtype=tf.float32)

        # self.a0 = tfe.Variable(1.)
        # self.w0 = tfe.Variable(3.)
        # self.p0 = tfe.Variable(0.01)

        # self.a1 = tfe.Variable(0.02)
        # self.w1 = tfe.Variable(0.9)
        # self.p1 = tfe.Variable(0.01)

    def __call__(self, t):
        # a_array = self.c[0::3]
        # w_array = self.c[1::3]
        # p_array = self.c[2::3]
        print("t is : ", t)
        x = 0.
        y = 0.
        phi = 0.
        x = tf.matmul(self.a,
                      tf.cos(tf.matmul(a=self.w, b=[t], transpose_a=True) +
                             tf.transpose(self.p)))
        phi = tf.atan2(y, x)
        return phi


def loss(predicted_y, desired_y):
    return tf.reduce_mean(tf.square(predicted_y - desired_y))


def train(model, inputs, outputs, learning_rate):
    with tf.GradientTape() as t:
        current_loss = loss(model(inputs), outputs)
    d = t.gradient(current_loss,
                   [model.a, model.w, model.p])
    model.a.assign_sub(learning_rate * d[0])
    model.w.assign_sub(learning_rate * d[1])
    model.p.assign_sub(learning_rate * d[2])


class Regressor(BaseEstimator):
    def __init__(self):
        self.c = np.array([51, 0.02, 3., 99, 0.01, 0.01])
        self.really_fit = True
        self.model = Ptolemy()
        pass

    def epicycle_error(self, c):
        a, w, p = decode_parameters(c)
        phi_epi = [f_phi(a, w, p, i) for i in range(self.fit_length)]
        return np.sum((np.unwrap(self.y_to_fit - phi_epi))**2)

    def epicycle_error_independent(self, c_ind):
        c = np.append([1.], c_ind)
        return self.epicycle_error(c)

    def epicycle_error_phase(self, c_phase):
        c = np.array(
            [self.c[0], self.c[1], c_phase[0],
             self.c[3], self.c[4], c_phase[1]]
        )
        return self.epicycle_error(c)

    def fit(self, X, y):
        print("Fitting the series, shape : ", X.shape)
#        print(X)
        # Maybe this will be where the mechanics will be determined
        # Formulas, main parameters etc.
        self.fit_length = X.shape[1]
        self.c = np.array([])
#       self.c = np.ndarray((0, 6))
#        for i in range(X.shape[0]):
        for i in [0]:
            self.y_to_fit = X[i, -self.fit_length:]
            cc = fit_features(self.y_to_fit)
            self.c = cc
            if(self.really_fit):
                #    c_ind = cc[1:]
                # cs = np.ndarray(shape=(0, 6))
                epochs = range(10)
                for epoch in epochs:
                    # cs = np.append(cs, self.model.c.numpy(), axis=0)
                    times = np.arange(0., len(self.y_to_fit))
                    model_result = self.model(times)
                    print("model result : ", model_result)
                    current_loss = loss(model_result, self.y_to_fit)

                    train(self.model, times, self.y_to_fit, learning_rate=0.1)
                    print('Epoch %2d: w=%s loss=%2.5f' %
                          (epoch, str(self.model.w), current_loss))
            self.c[0] = self.model.a[0, 0].numpy()
            self.c[1] = self.model.w[0, 0].numpy()
            self.c[2] = self.model.p[0, 0].numpy()
            self.c[3] = self.model.a[0, 1].numpy()
            self.c[4] = self.model.w[0, 1].numpy()
            self.c[5] = self.model.p[0, 1].numpy()

            # self.c = np.append([1.], res.x)
#            self.c = np.concatenate(self.c, np.append([1.], res.x), axis=1)

    def predict(self, X):
        # For the time being, this is where everything happens
        # This will be where the parameters will be fit
        # Phases etc.

        self.fit_length = 20
        n_predictions = X.shape[0]
        print("n_predictions : ", n_predictions)
        y = np.zeros(n_predictions)
        for i in range(n_predictions):
            print("Predicting ", i)
            self.y_to_fit = X[i, -self.fit_length:]
#            cc = fit_features(self.y_to_fit)
            cc = self.c
            c_ind = cc[2::3]
            p_bnds = (0., np.pi)
            bnds = (p_bnds, p_bnds)
#            print("c_ind : ", c_ind)
            a, w, p = decode_parameters(self.c)
            aa, ww, pp = decode_parameters(cc)
            p = pp
            if(self.really_fit):
                res = minimize(fun=self.epicycle_error_phase,
                               x0=c_ind,
                               method='TNC', tol=1e-8,
                               bounds=bnds,
                               options={
                                   # 'xtol': 0.001,
                                   # 'eps': 0.02,
                                   'maxiter': 10000})
                p = res.x
            y[i] = f_phi(a, w, p, self.fit_length + _n_lookahead)
#        print(y)
        return y.reshape(-1, 1)
