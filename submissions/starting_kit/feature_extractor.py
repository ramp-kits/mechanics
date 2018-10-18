import numpy as np


class FeatureExtractor():
    def __init__(self):
        pass

    def fit(self, X_df, y):
        pass

    def transform(self, X_df):
        X_array = X_df.values[:, :-1]
        means = np.mean(X_array, axis=1)
        stds = np.std(X_array, axis=1)
        return np.array([means, stds]).T
