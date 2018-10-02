from sklearn.base import BaseEstimator
# import numpy as np
# from submissions.tf_circle_fit.fit_features import *
# from submissions.tf_circle_fit.ptolemian_model import Ptolemy


class Regressor(BaseEstimator):
    def __init__(self):
        pass

    def fit(self, X, y):
        pass

    def predict(self, X):
        return X[:, 0].reshape(-1, 1)

