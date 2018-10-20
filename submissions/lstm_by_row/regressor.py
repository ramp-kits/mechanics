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
            n_train = 100
            n_time = 50
            phi_short = np.ndarray(shape=(n_train, n_time, 1))
            for i in range(0, n_train):
                phi_short[i, :, 0] = phis[i:i + n_time] / np.pi
            target = phis[n_time + 50:] / np.pi
            model.fit(phi_short, target, epochs=8, batch_size=1, verbose=2)

            x_pred = np.ndarray(shape=(250, n_time, 1))
            for i in range(0, 150):
                x_pred[i, :, 0] = phis[i:i + n_time] / np.pi

            for i in range(150, 250):
                x_pred[i, :, 0] = model.predict(
                    x_pred[i - n_time - 50].reshape(1, 50, 1))[0]

            y_pred[i_row] = x_pred[249, 0, 0] * np.pi

        return y_pred
