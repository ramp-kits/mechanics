import os
import numpy as np
import pandas as pd
import rampwf as rw
import xarray as xr


problem_title = \
    'Mechanics classification'

_n_lookahead = 100
_n_burn_in = 20
_filename = 'data_ts_merged.csv'
# Need better error messages for invalid input parameters
_debug_time_series = False

# label names for the classification target
_prediction_label_names = ['A', 'B', 'C', 'D']
# the regression target column
_target_column_name_clf = 'system'
# the classification target column
_target_column_name_reg = 'future'

# The first four columns of y_pred will be wrapped in multiclass Predictions.
Predictions_1 = rw.prediction_types.make_multiclass(
    label_names=_prediction_label_names)
# The last column of y_pred will be wrapped in regression Predictions.
# We make a 2D but single-column y_pred (instead of a classical 1D y_pred)
# to make handling the combined 2D y_pred array easier.
Predictions_2 = rw.prediction_types.make_regression(
    label_names=[_target_column_name_reg])
# The combined Predictions is initalized by the list of individual Predictions.
Predictions = rw.prediction_types.make_combined([Predictions_1, Predictions_2])

# The workflow object, named after the RAMP.
# workflow = rw.workflows.Mechanics()
workflow = rw.workflows.DrugSpectra()


# The first score will be applied on the first Predictions
score_type_1 = rw.score_types.ClassificationError(name='err', precision=3)
# The second score will be applied on the second Predictions

# Why RMS doesn't work??
score_type_2 = rw.score_types.RMSE(name='rmse', precision=3)
# score_type_2 = rw.score_types.MARE(name='mare', precision=3)

score_types = [
    # The official score combines the two scores with weights 2/3 and 1/3.
    # To let the score type know that it should be applied on the first
    # Predictions of the combined Predictions' prediction_list, we wrap
    # it into a special MakeCombined score with index 0
    rw.score_types.MakeCombined(score_type=score_type_1, index=0),
    rw.score_types.MakeCombined(score_type=score_type_2, index=1),
    rw.score_types.Combined(
        name='combined', score_types=[score_type_1, score_type_2],
        weights=[2. / 3, 1. / 3], precision=3),
]


# CV implemented here:
def get_cv(X, y):
    # make sure it always has all classes
    train_is = np.array([], dtype=int)
    test_is = np.array([], dtype=int)

    for label in _prediction_label_names:
        y_class = np.where(y[:, 0] == label)[0]
        print(y_class)
        np.random.shuffle(y_class)
        n = len(y_class)
        train_is = np.append(train_is, y_class[:int(n / 2)], axis=0)
        test_is = np.append(test_is, y_class[int(n / 2):], axis=0)

    np.random.shuffle(train_is)
    np.random.shuffle(test_is)
    print("training indices : ", train_is)
    print("test indices : ", test_is)
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
    # X_df['phi'] = X_ts
    return pd.DataFrame(X_ts)


# Both train and test targets are stripped off the first
# n_burn_in entries
def _read_data(path, filename):
    input_df = pd.read_csv(os.path.join(path, 'data', filename)).loc[::1]
    data_df = input_df.drop(['future', 'system'], axis=1)

    y_reg_array = input_df[_target_column_name_reg].values.reshape(-1, 1)
    y_clf_array = input_df[_target_column_name_clf].values.reshape(-1, 1)
    y_array = np.concatenate((y_clf_array,
                              y_reg_array), axis=1)
    print("y_array : ", y_array.shape, " : ", y_array)
    return data_df, y_array

n_sample = 50

def get_train_data(path='.'):
    data_df, y_array = _read_data(
        path,
        _filename)
    # return data_ds[:n_sample], y_array[:n_sample]
    #    "phis_sysFSS0_planet1_nview100_nsim200000.csv"
    return data_df[:n_sample], y_array[:n_sample]


def get_test_data(path='.'):
    data_df, y_array = _read_data(
        path,
        _filename)
    # return data_ds[:n_sample], y_array[:n_sample]
    #    "phis_sysFSS0_planet1_nview100_nsim200000.csv")
    return data_df[:n_sample], y_array[:n_sample]
