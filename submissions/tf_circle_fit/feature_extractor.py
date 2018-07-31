import numpy as np
from submissions.tf_circle_fit.fit_features import *
from submissions.tf_circle_fit.ptolemian_model import Ptolemy

from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator


class FeatureExtractor(object):

    def __init__(self):
        self.n_feat = 6
        self.params = np.array([])
        self.window = 0
        self.really_fit = True
        self.models = []
        for i in range(3):
            self.models.append(Ptolemy(i))
        self.models[2].assign_parameters(
            pars=np.array([[1., 0.28284271, 3.14159265],
                           [2., 0.28284271, 0.]])
        )
        self.n_components = 5
        self.n_estimators = 10
        self.clf = Pipeline([
            ('pca', PCA(n_components=self.n_components)),
            ('clf', RandomForestClassifier(
                n_estimators=self.n_estimators, random_state=42))
        ])

    def fit(self, X_df, y):
        print("======  X_df : ", X_df)
        n = X_df.shape[0]
        # Maybe this will be where the mechanics will be determined
        # Formulas, etc.
        X_feat = np.ndarray(shape=(n, self.n_feat))
        # X_target = X_df.loc(['planet', 'system'], axis=1).values
        y_feat = X_df['system'].values

        X_phis = X_df.drop(['planet', 'system'], axis=1).values

        self.c = np.array([])
        self.params = [0]
        for i in range(n):
            self.y_all = X_phis[i, :]
            print("y_all : ", self.y_all)
            nc, cc = qualitative_features(self.y_all)
            maxima, minima, loops = find_local_extrema(self.y_all)
            X_feat[i] = [len(maxima), len(minima), len(loops),
                         maxima[0], minima[0], loops[0]]
            self.window = 4 * int(2. * np.pi / cc[1])
            self.c = cc
            self.fit_length = X_phis.shape[1]
            self.y_to_fit = X_phis[i, -self.fit_length:]

        self.clf.fit(X_feat, y_feat)

    def transform(self, X_df):
        n = X_df.shape[0]
        X = np.ndarray(shape=(n, 7))

        # This is where each time series is
        # transoformed to be represented by a formula
        # with optimized parameters
        X_phis = X_df.drop(['planet', 'system'], axis=1).values
        self.c = np.array([])
        errors = np.ndarray(shape=(0, 3))
        self.params = [0]

        X_feat = np.ndarray(shape=(n, self.n_feat))
        for i in range(n):
            self.y_all = X_phis[i, :]
            maxima, minima, loops = find_local_extrema(self.y_all)
            X_feat[i] = [len(maxima), len(minima), len(loops),
                         maxima[0], minima[0], loops[0]]

        X_model = self.clf.predict(X_feat)

        for i in range(n):
            self.y_all = X_phis[i, :]
            print("y_all : ", self.y_all)
            nc, cc = qualitative_features(self.y_all)

            self.window = 4 * int(2. * np.pi / cc[1])
            self.c = cc
            self.fit_length = X_phis.shape[1]
            self.y_to_fit = X_phis[i, -self.fit_length:]
            model = self.models[X_model[i]]
            if(self.really_fit):
                epochs = range(100)
                for epoch in epochs:
                    # cs = np.append(cs, self.model.c.numpy(), axis=0)
                    times = np.arange(0., len(self.y_to_fit))
                    model_result = model(times)
                    # print("model result : ", model_result)
                    current_loss = model.loss(model_result, self.y_to_fit)
                    model.train(times, self.y_to_fit,
                                learning_rate=0.01)
                    print('Epoch %2d: w=%s loss=%2.5f' %
                          (epoch, str(model.w), current_loss))

            X[i][0] = X_model[i]
            X[i][1] = model.a[0, 0].numpy()
            X[i][2] = model.w[0, 0].numpy()
            X[i][3] = model.p[0, 0].numpy()
            X[i][4] = model.a[0, 1].numpy()
            X[i][5] = model.w[0, 1].numpy()
            X[i][6] = model.p[0, 1].numpy()
        return X
