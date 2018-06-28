import numpy as np


class FeatureExtractor(object):

    def __init__(self):
        pass

    def transform(self, X_ds):
        X_array = X_ds['phi'].values.reshape(-1, 1)
        X_ts = np.ndarray(shape=(len(X_array), 0))
        for shift in np.arange(0, X_ds.attrs['n_burn_in']):
            #            print("Preparing series: ", shift)
            X_ts = np.concatenate((
                X_ts,
                np.roll(X_array, -shift)
            ),
                axis=1)
        # This is the range for which features should be provided. Strip
        # the burn-in from the beginning.
        valid_range = np.arange(0, len(X_ds['time']))
#        print("X_ts shape : ", X_ts.shape)
#        print("X_ts : ", X_ts)

        X_ts = X_ts[valid_range]
#        print("X_ts valid shape : ", X_ts.shape)
#        print("X_ts valid : ", X_ts)

        return X_ts
