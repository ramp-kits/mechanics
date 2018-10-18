import os
import numpy as np
import pandas as pd
import rampwf as rw
from rampwf.score_types.base import BaseScoreType


class Mechanics(object):
    def __init__(self, workflow_element_names=[
            'feature_extractor', 'classifier', 'regressor']):
        self.element_names = workflow_element_names
        self.feature_extractor_workflow = rw.workflows.FeatureExtractor(
            [self.element_names[0]])
        self.classifier_workflow = rw.workflows.Classifier(
            [self.element_names[1]])
        self.regressor_workflow = rw.workflows.Regressor(
            [self.element_names[2]])

    def train_submission(self, module_path, X_df, y_array, train_is=None):
        if train_is is None:
            train_is = slice(None, None, None)
        # Avoid setting with copy warning
        X_train_df = X_df.iloc[train_is].copy()
        y_train_array = y_array[train_is]
        fe = self.feature_extractor_workflow.train_submission(
            module_path, X_train_df, y_train_array)
        X_train_array = self.feature_extractor_workflow.test_submission(
            fe, X_train_df)
        y_train_clf_array = y_train_array[:, 0]
        clf = self.classifier_workflow.train_submission(
            module_path, X_train_array, y_train_clf_array)
        y_train_reg_array = y_train_array[:, 1].astype(float)
        # Concatenating ground truth y_proba (one-hot, derived from labels)
        # to X_train_df.
        # This makes it vulnerable to training sets that don't contain
        # all the classes. So better to use it with stratified CV.
        labels = np.unique(y_array[:, 0])
        label_array = np.zeros((len(y_train_clf_array), len(labels)))
        for i, label in enumerate(labels):
            label_array[:, i] = (y_train_clf_array == label)
        X_train_array = np.hstack([X_train_array, label_array])
        reg = self.regressor_workflow.train_submission(
            module_path, X_train_array, y_train_reg_array)
        # It's a bit ugly that we return the labels here, but I don't see
        # a better solution
        return labels, fe, clf, reg

    def test_submission(self, trained_model, X_df):
        labels, fe, clf, reg = trained_model
        X_test_array = self.feature_extractor_workflow.test_submission(
            fe, X_df)
        y_proba_clf = self.classifier_workflow.test_submission(
            clf, X_test_array)
        label_array = np.zeros((len(y_proba_clf), len(labels)))
        for i, label in enumerate(labels):
            label_array[:, i] = y_proba_clf[:, i]
        X_test_array = np.hstack([X_test_array, label_array])
        y_pred_reg = self.regressor_workflow.\
            test_submission(reg, X_test_array)
        return np.concatenate([y_proba_clf, y_pred_reg.reshape(-1, 1)], axis=1)


problem_title = \
    'Mechanics classification'


_train = 'train.csv'
_test = 'test.csv'

# Need better error messages for invalid input parameters
_debug_time_series = False

# label names for the classification target
_prediction_label_names = ['A', 'B', 'C', 'D', 'E']
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
workflow = Mechanics()


# The first score will be applied on the first Predictions
score_type_1 = rw.score_types.ClassificationError(name='err', precision=3)
# The second score will be applied on the second Predictions


class CyclicRMSE(BaseScoreType):
    is_lower_the_better = True
    minimum = 0.0
    maximum = float('inf')

    def __init__(self, name='rmse', precision=2, periodicity=-1):
        self.name = name
        self.precision = precision
        self.periodicity = -1

    def __call__(self, y_true, y_pred):
        d = y_true - y_pred
        if(self.periodicity > 0):
            d = min(np.mod(d, self.periodicity),
                    np.mod(-d, self.periodicity))
        return np.sqrt(np.mean(np.square(d)))


score_type_2 = CyclicRMSE(name='rmse', precision=3,
                          periodicity=2 * np.pi)

score_types = [
    # The official score combines the two scores with weights 2/3 and 1/3.
    # To let the score type know that it should be applied on the first
    # Predictions of the combined Predictions' prediction_list, we wrap
    # it into a special MakeCombined score with index 0
    rw.score_types.Combined(
        name='combined', score_types=[score_type_1, score_type_2],
        weights=[0.1, 0.9], precision=3),
    rw.score_types.MakeCombined(score_type=score_type_1, index=0),
    rw.score_types.MakeCombined(score_type=score_type_2, index=1),
]


# CV implemented here:

def get_cv(X, y):
    unique_replicates = np.unique(X['distribution'])
    r = np.arange(len(X))
    for replicate in unique_replicates:
        train_is = r[(X['distribution'] != replicate).values]
        test_is = r[(X['distribution'] == replicate).values]
        yield train_is, test_is


# Both train and test targets are stripped off the first
# n_burn_in entries
def _read_data(path, filename):
    input_df = pd.read_csv(os.path.join(path, 'data', filename)).loc[::1]
    data_df = input_df.drop(['future', 'system'], axis=1)

    y_reg_array = input_df[_target_column_name_reg].values.reshape(-1, 1)
    y_clf_array = input_df[_target_column_name_clf].values.reshape(-1, 1)
    y_array = np.concatenate((y_clf_array,
                              y_reg_array), axis=1)
    return data_df, y_array


def get_train_data(path='.'):
    return _read_data(path, _train)


def get_test_data(path='.'):
    return _read_data(path, _test)
