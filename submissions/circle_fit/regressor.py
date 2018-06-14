from sklearn.base import BaseEstimator
import cma

import numpy as np

n_sample = 0


def f_x(a, w, p, t):
    return np.sum(a * np.cos(w * t + p))


def f_y(a, w, p, t):
    return np.sum(a * np.sin(w * t + p))


def f_phi(a, w, p, t):
    return np.arctan2(f_y(a, w, p, t), f_x(a, w, p, t))


class Regressor(BaseEstimator):
    def __init__(self):
        self.c = np.array([51, 0.02, 3., 99, 0.01, 0.01])
        pass

    def epicycle_error(self, c):
        a = c[0::3]
        w = c[1::3]
        p = c[2::3]
        phi_epi = [f_phi(a, w, p, i) for i in range(self.n_sample)]
        return np.sum((np.unwrap(self.y_fit - phi_epi))**2)

    def fit(self, X, y):
        self.n_sample = X.shape[1]
        self.y_fit = X[0]
        self.es = cma.CMAEvolutionStrategy(x0=self.c, sigma0=0.001)
        self.es.optimize(self.epicycle_error)
        self.c = np.array(self.es.result.xbest)

    def predict(self, X):
        y = np.array(f_phi(0, 0, 0, 0))
        y = np.zeros(X.shape[0])
        return y.reshape(-1, 1)
