import numpy as np
import pandas as pd
import xarray as xr


_n_lookahead = 100
_n_burn_in = 20
labels = ['A', 'B', 'C', 'D']


def make_time_series(X_array, window=_n_burn_in):
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
    # X_df['phi'] = X_ts
    return pd.DataFrame(X_ts)


def prepare_data():
    data = pd.DataFrame()
    for i_sys, sys in enumerate(["C0",
                                 "FSS0", "FSS1", "FSS5",
                                 # "FDS0", "FDS1", "FDS5",
                                 # "FG0", "FG1", "FG5",
                                 # "FGM0", "FGM1", "FGM5"
                                 ]):
        for planet in range(1, 3):
            filename = "data/phis_sys" + \
                sys + \
                "_planet" + \
                str(planet) + \
                "_nview100_nsim200000.csv"
            df = pd.read_csv(filename)
            df['system'] = labels[i_sys]
            df['planet'] = planet

            data = data.append(df)
    data.to_csv('data_merged.csv', index=False)
    return data


def prepare_data_ts(input_df):
    data_array = input_df['phi'].values[0:-
                                        _n_burn_in -
                                        _n_lookahead:].reshape(-1, 1)

    y_reg_array = input_df['phi'].values[
        _n_burn_in + _n_lookahead:].reshape(-1, 1)
    y_clf_array = input_df['system'].values[
        : -_n_burn_in - _n_lookahead].reshape(-1, 1)

    # Hack for quick testing
    y_clf_array = np.tile(['A', 'B', 'C', 'D'], int(
        len(y_reg_array) / 4)).reshape(-1, 1)

    data_df = make_time_series(data_array)
    data_df['future'] = y_reg_array
    data_df['system'] = y_clf_array
    data_df.to_csv('data_ts_merged.csv', index=False)
    return data_df


data = prepare_data()
prepare_data_ts(data)
