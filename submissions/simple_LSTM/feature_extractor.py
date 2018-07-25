import numpy as np


class FeatureExtractor(object):

    def __init__(self):
        self.params = np.array([])
        self.window = 0
        pass

    def fit(self, X_df, y):
        return True

    def transform(self, X_df):
        return X_df.values
