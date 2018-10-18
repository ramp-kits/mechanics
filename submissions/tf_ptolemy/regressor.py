from sklearn.base import BaseEstimator


class Regressor(BaseEstimator):
    def __init__(self):
        pass

    def fit(self, X, y):
        pass

    def predict(self, X):
        return X[:, 0].reshape(-1, 1).astype(float)
