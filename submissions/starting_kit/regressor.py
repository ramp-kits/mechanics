from sklearn.base import BaseEstimator
from keras.layers import Input, Dense
from keras.models import Model
from keras.layers import LSTM
import numpy as np


class Regressor(BaseEstimator):
    def __init__(self):
        pass

    def fit(self, X, y):
        self.n_sample = X.shape[1]
        inputs = Input(shape=(self.n_sample, 1),
                       dtype='float', name='main_input')
        layer = LSTM(8)(inputs)
        predictions = Dense(1)(layer)
        self.model = Model(inputs=inputs, outputs=predictions)
        self.model.compile(optimizer='adam',
                           loss='mean_squared_error')

        self.model.fit(X.reshape(-1, self.n_sample, 1), y,
                       epochs=1, batch_size=1, verbose=2)

    def predict(self, X):
        y_pred = np.array(self.model.predict(
            X.reshape(-1, self.n_sample, 1))).reshape(-1, 1)
        return y_pred
