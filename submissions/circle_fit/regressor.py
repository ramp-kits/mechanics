from sklearn.base import BaseEstimator
from scipy.optimize import minimize
from submissions.circle_fit.fit_features import *
import numpy as np

_n_lookahead = 50
_n_burn_in = 500


class Regressor(BaseEstimator):
    def __init__(self):
        self.c = np.array([51, 0.02, 3., 99, 0.01, 0.01])
        self.really_fit = True
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
                c_ind = cc[1:]
                a_bnds = (0., 2.)
                w_bnds = (0.001, 0.5)
                p_bnds = (0., np.pi)
                bnds = (w_bnds, p_bnds, a_bnds, w_bnds, p_bnds)
                res = minimize(fun=self.epicycle_error_independent,
                               x0=c_ind,
                               method='TNC', tol=1e-8,
                               bounds=bnds,
                               options={
                                   # 'xtol': 0.001,
                                   # 'eps': 0.02,
                                   'maxiter': 100000})
                self.c = np.append([1.], res.x)
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
