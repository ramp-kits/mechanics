from sklearn.base import BaseEstimator
import cma

import numpy as np
_n_lookahead = 5
_n_burn_in = 100


def f_x(a, w, p, t):
    return np.sum(a * np.cos(w * t + p))


def f_y(a, w, p, t):
    return np.sum(a * np.sin(w * t + p))


def f_phi(a, w, p, t):
    return np.arctan2(f_y(a, w, p, t), f_x(a, w, p, t))


def decode_parameters(c):
    a = c[0::3]
    w = c[1::3]
    p = c[2::3]
    return a, w, p


class Regressor(BaseEstimator):
    def __init__(self):
        self.c = np.array([51, 0.02, 3., 99, 0.01, 0.01])
        pass

    def epicycle_error(self, c):
        a, w, p = decode_parameters(c)
        phi_epi = [f_phi(a, w, p, i) for i in range(_n_burn_in)]
        return np.sum((np.unwrap(self.y_to_fit - phi_epi))**2)

    def fit(self, X, y):
        print("Fitting the series, shape : ", X.shape)
        print(X)
        # Maybe this will be where the mechanics will be determined
        # Formulas, main parameters etc.
        if(False):
            for i in range(X.shape[0]):
                self.y_to_fit = X[i]
                print("y_to_fit is : ", self.y_to_fit)
                self.es = cma.CMAEvolutionStrategy(x0=self.c, sigma0=0.001)
                self.es.optimize(self.epicycle_error)
                self.c = np.array(self.es.result.xbest)

    def predict(self, X):
        # For the time being, this is where everything happens
        # This will be where the parameters will be fit
        # Phases etc.
        n_predictions = X.shape[0]
        y = np.zeros(n_predictions)
        for i in range(n_predictions):
            self.y_to_fit = X[i]
            print("y_to_fit is : ", self.y_to_fit)
            self.es = cma.CMAEvolutionStrategy(x0=self.c, sigma0=0.001)
            self.es.optimize(self.epicycle_error)
            self.c = np.array(self.es.result.xbest)
            a, w, p = decode_parameters(self.c)
            y[i] = f_phi(a, w, p, _n_burn_in + _n_lookahead)

        return y.reshape(-1, 1)

# NEXT WEEK'S TASK: 
# EVEN SIMPLIFY : VIEW FROM SUN!!!
# CHEAT ON THE FORMULA FIT, determine variables from retrogrades
# ONLY FIT PHASE IN PREDICT
# TEST SLIGHTLY PERTURBED
# COMPARE TO LSTM



