from sklearn.base import BaseEstimator
from keras.layers import Input, Dense
from keras.models import Model
from keras.layers import LSTM
import numpy as np

model_labels = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}


class Regressor(BaseEstimator):
    def __init__(self):
        self.models = np.ndarray(shape=(5,), dtype=object)
        pass

    def fit(self, X, y):
        self.n_sample = 200
        inputs = Input(shape=(self.n_sample, 1),
                       dtype='float', name='main_input')
        layer = LSTM(8)(inputs)
        predictions = Dense(1)(layer)
        for i in range(len(self.models)):
            self.models[i] = Model(inputs=inputs, outputs=predictions)
            self.models[i].compile(optimizer='adam',
                                   loss='mean_squared_error')

            self.models[i].fit(X[:, :self.n_sample]
                               .reshape(-1, self.n_sample, 1),
                               y, epochs=1, batch_size=1, verbose=2)

    def predict(self, X):
        y_pred = np.zeros(X.shape[0])
        for i, model in enumerate(self.models):
            ind_mod = np.where(np.argmax(X[:, -len(self.models):],
                                         axis=1) == i)[0]
            X_mod = X[ind_mod, :]
            y_pred[ind_mod] = model.predict(X_mod)
        return y_pred
