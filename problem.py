import os
import numpy as np
import pandas as pd
import rampwf as rw
import xarray as xr

problem_title =\
    'Prediction of the azimuth of Mars'

_n_lookahead = 5
_n_burn_in = 500
_filename = 'test_angles_perfect_circle_nview10_n1M.csv'
_target = 'phi'
# Need better error messages for invalid input parameters
_debug_time_series = False

Predictions = rw.prediction_types.make_regression(
    label_names=[_target])

# El Nino, a.k.a. [TimeSeries, FeatureExtractor, Regressor]
workflow = rw.workflows.Mechanics(check_sizes=[_n_burn_in + 30], check_indexs=[_n_burn_in + 1])

score_types = [
    rw.score_types.RelativeRMSE(name='rel_rmse', precision=3)
]

# CV implemented here:
cv = rw.cvs.TimeSeries(
    n_cv=2, cv_block_size=0.5, period=1, unit='space_year')
#get_cv = cv


def get_cv(X, y):
    n = len(y)
    train_is = np.arange(0, n - _n_burn_in - _n_lookahead)
    test_is = np.arange(0, n - _n_burn_in - _n_lookahead)
    yield (train_is, test_is)


# Both train and test targets are stripped off the first
# n_burn_in entries
def _read_data(path):
    data_df = pd.read_csv(os.path.join(path, 'data', _filename)).loc[:20000:10]
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
    data_ds, y_array = _read_data(path)
    return data_ds, y_array


def get_test_data(path='.'):
    data_ds, y_array = _read_data(path)
    return data_ds, y_array
