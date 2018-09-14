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
        self.list_molecule = ['A', 'B', 'C', 'D']

        self.dict_reg = {}
        for mol in self.list_molecule:
            self.dict_reg[mol] = Pipeline([
                ('pca', PCA(n_components=self.n_components)),
                ('reg', GradientBoostingRegressor(
                    n_estimators=self.n_estimators,
                    learning_rate=self.learning_rate,
                    random_state=42))
            ])

    def fit(self, X, y):
        for i, mol in enumerate(self.list_molecule):
            ind_mol = np.where(np.argmax(X[:, -4:], axis=1) == i)[0]
            X_mol = X[ind_mol, 0: -4]
            y_mol = y[ind_mol]
            if(len(y_mol) > 0):
                print("X_mol : ", X_mol)
                print("y_mol : ", y_mol)
                self.dict_reg[mol].fit(X_mol, y_mol)

    def predict(self, X):
        y_pred = np.zeros(X.shape[0])
        print("X : ", X)
        for i, mol in enumerate(self.list_molecule):
            ind_mol = np.where(np.argmax(X[:, -4:], axis=1) == i)[0]
            X_mol = X[ind_mol, 0: -4]
            print("ind_mol : ", ind_mol)
            print("X_mol : ", X_mol)
            if(len(X_mol) > 0):
                y_pred[ind_mol] = self.dict_reg[mol].predict(X_mol)
        return y_pred
