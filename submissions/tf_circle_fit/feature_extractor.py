import numpy as np
from submissions.circle_fit.fit_features import *


class FeatureExtractor(object):

    def __init__(self):
        self.params = np.array([])
        self.window = 0
        pass

    def fit(self, X_df, y):
        # Greedy:
        # self.window = X_ds.attrs['n_burn_in']
        # or:
        X = X_df.values
        array = X[0, :].reshape(-1, 1)
        self.params = [0]
        c = fit_features(array)
        self.window = 4 * int(2. * np.pi / c[1])
        print("FIT WINDOW DETERMINED : ", self.window)
        return True

    def transform(self, X_df):
        return X_df.values
