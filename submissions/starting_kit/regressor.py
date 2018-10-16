from sklearn.ensemble import GradientBoostingRegressor
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator
import numpy as np


class Regressor(BaseEstimator):
    def __init__(self):
        self.n_components = 10
        self.n_estimators = 40
        self.learning_rate = 0.2
        self.list_model = ['A', 'B', 'C', 'D', 'E', 'F']

        self.dict_reg = {}
        for mod in self.list_model:
            self.dict_reg[mod] = Pipeline([
                ('pca', PCA(n_components=self.n_components)),
                ('reg', GradientBoostingRegressor(
                    n_estimators=self.n_estimators,
                    learning_rate=self.learning_rate,
                    random_state=42))
            ])

    def fit(self, X, y):
        for i, mod in enumerate(self.list_model):
            ind_mod = np.where(np.argmax(X[:, -len(self.list_model):],
                                         axis=1) == i)[0]
            X_mod = X[ind_mod, 0: -len(self.list_model)]
            y_mod = y[ind_mod]
            if(len(y_mod) > 0):
                self.dict_reg[mod].fit(X_mod, y_mod)

    def predict(self, X):
        y_pred = np.zeros(X.shape[0])
        for i, mod in enumerate(self.list_model):
            ind_mod = np.where(np.argmax(X[:, -len(self.list_model):],
                                         axis=1) == i)[0]
            X_mod = X[ind_mod, 0: -len(self.list_model)]
            if(len(X_mod) > 0):
                y_pred[ind_mod] = self.dict_reg[mod].predict(X_mod)
        return y_pred
