import numpy as np
from submissions.circle_fit.fit_features import *
_n_burn_in = 500


class FeatureExtractor(object):

    def __init__(self):
        self.params = np.array([])
        self.window = 0
        pass

    def fit(self, X_ds, y):
        # Greedy:
        # self.window = X_ds.attrs['n_burn_in']
        # or:
        array = X_ds['phi'].values.reshape(-1, 1)
        self.params = [0]
        c = fit_features(array)
        self.window = 4 * int(2. * np.pi / c[1])
        print("FIT WINDOW DETERMINED : ", self.window)
        return True

    def transform(self, X_ds):
        X_array = X_ds['phi'].values.reshape(-1, 1)
        X_ts = np.ndarray(shape=(len(X_array), 0))

        for shift in np.arange(0, _n_burn_in):
            #            print("Preparing series: ", shift)
            X_ts = np.concatenate((
                X_ts,
                np.roll(X_array, -shift)
            ),
                axis=1)
        # This is the range for which features should be provided. Strip
        # the burn-in from the beginning.

        X_ts = X_ts[:, -self.window:]
        print("X_ts valid shape : ", X_ts.shape)
        print("X_ts valid : ", X_ts)

        return X_ts
