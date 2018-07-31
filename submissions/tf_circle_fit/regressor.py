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
        self.models = []
        for i in range(3):
            self.models.append(Ptolemy(i))

        self.models[1].assign_parameters(
            pars=np.array([1., 0.28284271, 3.14159265]))

        self.models[2].assign_parameters(
            pars=np.array([1., 0.28284271, 3.14159265,
                           2., 0.28284271, 0.]))

    def fit(self, X, y):
        pass

    def predict(self, X):
        self.fit_length = 20
        n_predictions = X.shape[0]
        print("n_predictions : ", n_predictions)
        y = np.zeros(n_predictions)
        for i in range(n_predictions):
            model_type = int(X[i, 0])
            model = self.models[model_type]
            model_parameters = X[i, 1: len(model.c.numpy()[0]) + 1]
            print("model_parameters : ", model_parameters)
            model.assign_parameters(
                pars=model_parameters)
            # y[i] = f_phi(a, w, p, self.fit_length + _n_lookahead)
            y[i] = model([_n_burn_in + _n_lookahead]).numpy()
        return y.reshape(-1, 1)
