from sklearn.base import BaseEstimator
import numpy as np
from submissions.tf_circle_fit.fit_features import *
from submissions.tf_circle_fit.ptolemian_model import Ptolemy

_n_lookahead = 50.
_n_burn_in = 500


class Regressor(BaseEstimator):
    def __init__(self):
        self.c = np.array([51, 0.02, 3., 99, 0.01, 0.01])
        self.really_fit = True
        self.model = Ptolemy()
        pass

    def fit(self, X, y):
        pass

    def predict(self, X):
        self.fit_length = 20
        n_predictions = X.shape[0]
        print("n_predictions : ", n_predictions)
        y = np.zeros(n_predictions)
        for i in range(n_predictions):
            a, w, p = decode_parameters(X[i])
            self.model.freeze_parameters(
                mask=np.ones(shape=(2, 3)),
                pars=X[i].reshape(2, 3))
            # y[i] = f_phi(a, w, p, self.fit_length + _n_lookahead)
            y[i] = self.model([_n_lookahead]).numpy()
        return y.reshape(-1, 1)
