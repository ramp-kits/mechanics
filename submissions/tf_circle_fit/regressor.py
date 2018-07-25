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
        errors = np.ndarray(shape=(0, 3))
        optimizer = tf.train.AdamOptimizer(0.001)

        for i in [0]:
            self.y_to_fit = X[i, -self.fit_length:]
            cc = fit_features(self.y_to_fit)
            self.c = cc
            if(self.really_fit):
                #    c_ind = cc[1:]
                # cs = np.ndarray(shape=(0, 6))
                epochs = range(100)
                for epoch in epochs:
                    # cs = np.append(cs, self.model.c.numpy(), axis=0)
                    times = np.arange(0., len(self.y_to_fit))
                    model_result = self.model(times)
                    # print("model result : ", model_result)
                    current_loss = self.model.loss(model_result, self.y_to_fit)
                    self.model.train(times, self.y_to_fit,
                                     learning_rate=0.01)
                    print('Epoch %2d: w=%s loss=%2.5f' %
                          (epoch, str(self.model.w), current_loss))
                    errors = np.append(errors,
                                       np.array([[self.model.a.numpy()[0, 1],
                                                  self.model.w.numpy()[0, 1],
                                                  current_loss.numpy()]]),
                                       axis=0)
            self.c[0] = self.model.a[0, 0].numpy()
            self.c[1] = self.model.w[0, 0].numpy()
            self.c[2] = self.model.p[0, 0].numpy()
            self.c[3] = self.model.a[0, 1].numpy()
            self.c[4] = self.model.w[0, 1].numpy()
            self.c[5] = self.model.p[0, 1].numpy()

        return errors

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
