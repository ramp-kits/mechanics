import numpy as np
import pandas as pd


_n_burn_in = 500


def make_time_series(X_ds, window=_n_burn_in):
    X_array = X_ds['phi'].values.reshape(-1, 1)
    X_ts = np.ndarray(shape=(len(X_array), 0))

    for shift in np.arange(0, _n_burn_in):
        #            print("Preparing series: ", shift)
        X_ts = np.concatenate((
            X_ts,
            np.roll(X_array, -shift, axis=0)
        ),
            axis=1)
    # This is the range for which features should be provided. Strip
    # the burn-in from the beginning.

    X_ts = X_ts[:, -window:]
    print("X_ts valid shape : ", X_ts.shape)
    print("X_ts valid : ", X_ts)
    return pd.DataFrame(X_ts)


def prepare_data():
    data = pd.DataFrame()
    for i_sys, sys in enumerate(["C0",
                                 "FSS0", "FSS1", "FSS5",
                                 "FDS0", "FDS1", "FDS5",
                                 "FG0", "FG1", "FG5",
                                 "FGM0", "FGM1", "FGM5"
                                 ]):
        for planet in range(1, 3):
            filename = "data/phis_sys" + \
                sys + \
                "_planet" + \
                str(planet) + \
                "_nview100_nsim200000.csv"
            df = pd.read_csv(filename)
            df['sys'] = i_sys
            df['planet'] = planet

            data = data.append(df)
    data.to_csv('data_merged.csv')


prepare_data()
