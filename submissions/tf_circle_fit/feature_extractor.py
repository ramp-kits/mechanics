import numpy as np
from submissions.circle_fit.fit_features import *
from submissions.tf_circle_fit.ptolemian_model import Ptolemy


class FeatureExtractor(object):

    def __init__(self):
        self.params = np.array([])
        self.window = 0
        self.really_fit = True
        self.model = Ptolemy()

    def fit(self, X_df, y):
        print("======  X_df : ", X_df)
        n = X_df.shape[0]
        X = np.ndarray(shape=(n, 6))

        # Maybe this will be where the mechanics will be determined
        # Formulas, main parameters etc.
        X_phis = X_df.drop(['planet', 'system'], axis=1).values
        self.c = np.array([])
        errors = np.ndarray(shape=(0, 3))
        self.params = [0]

        for i in range(n):
            self.y_all = X_phis[i, :]
            print("y_all : ", self.y_all)
            cc = fit_features(self.y_all)
            self.window = 4 * int(2. * np.pi / cc[1])
            self.c = cc
            self.fit_length = X_phis.shape[1]
            self.y_to_fit = X_phis[i, -self.fit_length:]

            if(self.really_fit):
                #    c_ind = cc[1:]
                # cs = np.ndarray(shape=(0, 6))
                epochs = range(100)
                for epoch in epochs:
                    # cs = np.append(cs, self.model.c.numpy(), axis=0)
                    times = np.arange(0., len(self.y_to_fit))
                    model_result = self.model(times)
                    # print("model result : ", model_result)
                    current_loss = self.model.loss(model_result, self.y_to_fit)
                    self.model.train(times, self.y_to_fit,
                                     learning_rate=0.01)
                    print('Epoch %2d: w=%s loss=%2.5f' %
                          (epoch, str(self.model.w), current_loss))
                    errors = np.append(errors,
                                       np.array([[self.model.a.numpy()[0, 1],
                                                  self.model.w.numpy()[0, 1],
                                                  current_loss.numpy()]]),
                                       axis=0)
            X[i][0] = self.model.a[0, 0].numpy()
            X[i][1] = self.model.w[0, 0].numpy()
            X[i][2] = self.model.p[0, 0].numpy()
            X[i][3] = self.model.a[0, 1].numpy()
            X[i][4] = self.model.w[0, 1].numpy()
            X[i][5] = self.model.p[0, 1].numpy()
        return X

    def transform(self, X_df):
        n = X_df.shape[0]
        X = np.ndarray(shape=(n, 6))

        # Maybe this will be where the mechanics will be determined
        # Formulas, main parameters etc.
        X_phis = X_df.drop(['planet', 'system'], axis=1).values
        self.c = np.array([])
        errors = np.ndarray(shape=(0, 3))
        self.params = [0]

        for i in range(n):
            self.y_all = X_phis[i, :]
            print("y_all : ", self.y_all)
            cc = fit_features(self.y_all)
            self.window = 4 * int(2. * np.pi / cc[1])
            self.c = cc
            self.fit_length = X_phis.shape[1]
            self.y_to_fit = X_phis[i, -self.fit_length:]
            if(self.really_fit):
                #    c_ind = cc[1:]
                # cs = np.ndarray(shape=(0, 6))
                epochs = range(100)
                for epoch in epochs:
                    # cs = np.append(cs, self.model.c.numpy(), axis=0)
                    times = np.arange(0., len(self.y_to_fit))
                    model_result = self.model(times)
                    # print("model result : ", model_result)
                    current_loss = self.model.loss(model_result, self.y_to_fit)
                    self.model.train(times, self.y_to_fit,
                                     learning_rate=0.01)
                    print('Epoch %2d: w=%s loss=%2.5f' %
                          (epoch, str(self.model.w), current_loss))
                    errors = np.append(errors,
                                       np.array([[self.model.a.numpy()[0, 1],
                                                  self.model.w.numpy()[0, 1],
                                                  current_loss.numpy()]]),
                                       axis=0)
            X[i][0] = self.model.a[0, 0].numpy()
            X[i][1] = self.model.w[0, 0].numpy()
            X[i][2] = self.model.p[0, 0].numpy()
            X[i][3] = self.model.a[0, 1].numpy()
            X[i][4] = self.model.w[0, 1].numpy()
            X[i][5] = self.model.p[0, 1].numpy()
        return X
