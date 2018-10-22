from sklearn.base import BaseEstimator
from keras.layers import Input, Dense
from keras.models import Model
from keras.layers import LSTM
import numpy as np

model_labels = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}


class Regressor(BaseEstimator):
    def __init__(self):
        self.models = np.ndarray(shape=(5,), dtype=object)
        self.n_sample = 10
        self.initial_weights = np.ndarray(shape=(5,), dtype=object)
        for i in range(len(self.models)):
            inputs = Input(shape=(self.n_sample, 1),
                           dtype='float', name='main_input')
            layer = LSTM(7)(inputs)
            predictions = Dense(1)(layer)
            self.models[i] = Model(inputs=inputs, outputs=predictions)
            self.models[i].compile(optimizer='adam',
                                   loss='mean_squared_error')
            self.initial_weights[i] = self.models[i].get_weights()

    def fit(self, X, y):
        pass

    def predict(self, X):
        n_rows = X.shape[0]
        y_pred = np.zeros(n_rows)

        # This is where each time series is
        # transoformed to be represented by a formula
        # with optimized parameters
        X_phis = X[:, 0: 200]
        X_model = np.argmax(X[:, -len(self.models):], axis=1)

        for i_row in range(n_rows):
            phis = X_phis[i_row, :200].reshape(-1)
            model = self.models[X_model[i_row]]
            model.set_weights(self.initial_weights[X_model[i_row]])

            predict_future = 50
            n_train = 10
            sample_step = 10

            X_ts = np.ndarray(shape=(n_train, self.n_sample, 1))

            for i in range(0, n_train):
                X_ts[i, :, 0] = \
                    phis[i * sample_step:i * sample_step + self.n_sample] \
                    / np.pi

            target = phis[self.n_sample + predict_future:
                          n_train * sample_step + self.n_sample +
                          predict_future:
                          sample_step] / np.pi

            model.fit(X_ts, target, epochs=2, batch_size=1, verbose=0)

            y_pred[i_row] = model.predict(phis[-self.n_sample:]
                                          .reshape(-1, self.n_sample, 1))[0]
        return y_pred
