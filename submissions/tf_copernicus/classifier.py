from sklearn.base import BaseEstimator


class Classifier(BaseEstimator):
    def __init__(self):
        pass

    def fit(self, X, y):
        pass

    def predict(self, X):
        return(X[:, 1]).reshape(-1, 1).astype(str)

    def predict_proba(self, X):
        return(X[:, 2:7]).reshape(-1, 5)
