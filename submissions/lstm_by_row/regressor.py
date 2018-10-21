from sklearn.base import BaseEstimator
from keras.layers import Input, Dense
from keras.models import Model
from keras.layers import LSTM
import numpy as np
import pdb

model_labels = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}


class Regressor(BaseEstimator):
    def __init__(self):
        self.models = np.ndarray(shape=(5,), dtype=object)
        self.n_sample = 50
        self.initial_weights = np.ndarray(shape=(5,), dtype=object)

    def fit(self, X, y):
        for i in range(len(self.models)):
            inputs = Input(shape=(self.n_sample, 1),
                           dtype='float', name='main_input')
            layer = LSTM(12)(inputs)
            predictions = Dense(1)(layer)
            self.models[i] = Model(inputs=inputs, outputs=predictions)
            self.models[i].compile(optimizer='adam',
                                   loss='mean_squared_error')
            self.initial_weights[i] = self.models[i].get_weights()

    def predict(self, X):
        n_rows = X.shape[0]
        y_pred = np.zeros(n_rows)

        # This is where each time series is
        # transoformed to be represented by a formula
        # with optimized parameters
        X_phis = X[:, 0: 200]
        X_model = np.argmax(X[:, -len(self.models):], axis=1)

        for i_row in range(n_rows):
            phis = X_phis[i_row, :].reshape(-1)
            model = self.models[X_model[i_row]]
            model.set_weights(self.initial_weights[X_model[i_row]])

            n_time = 10
            target_future = 50
            predict_future = 50
            n_train = 10
            sample_step = 5
            n_steps = len(phis)

            X_ts = np.ndarray(shape=(n_train, n_time, 1))

            for i in range(0, n_train):
                X_ts[i, :, 0] = \
                    phis[i * sample_step:i * sample_step + n_time] / np.pi

            target = phis[n_time + predict_future:
                          n_train * sample_step + n_time + predict_future:
                          sample_step] / np.pi
            inputs = Input(shape=(n_time, 1), dtype='float', name='main_input')
            layer = LSTM(12)(inputs)
            predictions = Dense(1)(layer)
            model = Model(inputs=inputs, outputs=predictions)
            model.compile(optimizer='adam',
                          loss='mean_squared_error')

            model.fit(X_ts, target, epochs=3, batch_size=1, verbose=0)
            x_pred = np.ndarray(shape=(n_steps + target_future, n_time, 1))
            for i in range(0, n_steps - n_time):
                x_pred[i, :, 0] = phis[i:i + n_time] / np.pi

            for i in range(n_steps - n_time, n_steps + target_future):
                x_pred[i, :, 0] = model.predict(
                    x_pred[i - n_time - predict_future]
                    .reshape(1, n_time, 1))[0]

            y_pred[i_row] = x_pred[-1, 0, 0] * np.pi

        return y_pred
