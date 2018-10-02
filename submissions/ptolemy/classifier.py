from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator


class Classifier(BaseEstimator):
    def __init__(self):
        self.n_components = 10
        self.n_estimators = 10
        self.clf = Pipeline([
            ('pca', PCA(n_components=self.n_components)),
            ('clf', RandomForestClassifier(
                n_estimators=self.n_estimators, random_state=42))
        ])

    def fit(self, X, y):
        self.clf.fit(X, y)

    def predict(self, X):
        y_pred = self.clf.predict(X)
        print("y_pred clf : ", y_pred)
        return y_pred

    def predict_proba(self, X):
        y_pred_proba = self.clf.predict_proba(X)
        print("y_pred_proba clf : ", y_pred_proba)
        return y_pred_proba
