from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator


class Classifier(BaseEstimator):
    def __init__(self):
        self.clf = RandomForestClassifier(
            n_estimators=10, max_leaf_nodes=10, random_state=42)

    def fit(self, X, y):
        self.clf.fit(X, y)

    def predict_proba(self, X):
        y_pred_proba = self.clf.predict_proba(X)
        return y_pred_proba
