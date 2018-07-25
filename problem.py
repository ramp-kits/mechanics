import os
import numpy as np
import pandas as pd
import rampwf as rw
import xarray as xr

problem_title = \
    'Deducing system mechanics and formulas from limited information'

_n_lookahead = 50
_n_burn_in = 500
_filename = 'test_angles_perfect_circle_nview10_n1M.csv'
_target = 'phi'
# Need better error messages for invalid input parameters
_debug_time_series = False

Predictions = rw.prediction_types.make_regression(
    label_names=[_target])

workflow = rw.workflows.Mechanics()

score_types = [
    rw.score_types.RelativeRMSE(name='rel_rmse', precision=3)
]


# CV implemented here:
def get_cv(X, y):
    n = len(y)
#    train_is = np.arange(0, int(n / 2))
#    test_is = np.arange(int(n / 2), n)
    train_is = np.arange(0, int(n / 2))
    test_is = np.arange(0, int(n / 2))
    yield (train_is, test_is)


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


# Both train and test targets are stripped off the first
# n_burn_in entries
def _read_data(path, filename):
    data_df = pd.read_csv(os.path.join(path, 'data', filename)).loc[::1]
    data_array = data_df.drop(
        ['time'], axis=1).values[0:- _n_burn_in - _n_lookahead:].reshape(-1)

    # x = y to debug look-up times
    if(_debug_time_series):
        data_array = np.arange(0., len(data_array))
    time = data_df['time'].values[0:- _n_burn_in - _n_lookahead:].reshape(-1)
    data_xr = xr.DataArray(
        data_array, coords=[('time', time)], dims=('time'))
    data_ds = xr.Dataset({'phi': data_xr})
    data_ds.attrs = {'n_burn_in': _n_burn_in, 'n_lookahead': _n_lookahead}

    y_array = data_df[_target].values.reshape(-1, 1)
    y_array = y_array[_n_burn_in + _n_lookahead:]

    print("y_array : ", y_array.shape, " : ", y_array)
    print("data_ds : ", data_ds)
    # x = y to debug look-up times
    if(_debug_time_series):
        y_array = np.arange(0., len(data_array)).reshape(-1, 1)
    return data_ds, y_array


def get_train_data(path='.'):
    data_ds, y_array = _read_data(
        path,
        "phis_sysC0_planet1_nview100_nsim200000.csv")
    return make_time_series(data_ds), y_array


def get_test_data(path='.'):
    data_ds, y_array = _read_data(
        path,
        "phis_sysC0_planet1_nview100_nsim200000.csv")
    return make_time_series(data_ds), y_array
