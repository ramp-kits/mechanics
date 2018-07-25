import numpy as np


class FeatureExtractor(object):

    def __init__(self):
        pass

    def transform(self, X_ds):
        self.n_sample = 3
        X_array = X_ds['phi'].values.reshape(-1, 1)
        X_ts = np.ndarray(shape=(len(X_array), 0))
        for shift in np.arange(self.n_sample):
            X_ts = np.concatenate((X_ts,
                                   np.roll(X_array, X_ds.n_burn_in - shift)),
                                  axis=1)
        # This is the range for which features should be provided. Strip
        # the burn-in from the beginning.
        valid_range = np.arange(X_ds.n_burn_in, len(X_ds['time']))
        X_ts = X_ts[valid_range]
        return X_ts
