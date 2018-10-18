import numpy as np


from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from scipy.optimize import minimize

model_labels = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}

_n_lookahead = 50.
_n_burn_in = 500


def f_x(a, w, p, t):
    return np.sum(a * np.cos(w * t + p))


def f_y(a, w, p, t):
    return np.sum(a * np.sin(w * t + p))


def f_phi(a, w, p, t):
    return np.arctan2(f_y(a, w, p, t), f_x(a, w, p, t))


def decode_parameters(c):
    a = c[0::3]
    w = c[1::3]
    p = c[2::3]
    return a, w, p


class FeatureExtractor(object):

    def __init__(self):
        self.n_epicyle = 5
        self.params = np.array([])
        self.window = 0
        self.really_fit = True
        self.models = []
        self.n_models = 5
        self.n_components = 10
        self.n_estimators = 10
        self.clf = Pipeline([
            ('pca', PCA(n_components=self.n_components)),
            ('clf', RandomForestClassifier(
                n_estimators=self.n_estimators, random_state=42))
        ])

    def epicycle_error(self, c):
        a, w, p = decode_parameters(c)
        phi_epi = [f_phi(a, w, p, i) for i in range(200)]
        return np.sum((np.unwrap(self.y_to_fit - phi_epi))**2)

    def epicycle_error_independent(self, c_ind):
        c = np.append([1.], c_ind)
        return self.epicycle_error(c)

    def epicycle_error_phase(self, c_phase):
        c = np.array(
            [self.c[0], self.c[1], c_phase[0],
             self.c[3], self.c[4], c_phase[1]]
        )
        return self.epicycle_error(c)

    def fit(self, X_df, y):
        X_phis = X_df.values[:, 0: 200]
        y_model = y[:, 0]
        self.clf.fit(X_phis, y_model)

    def transform(self, X_df):
        n = X_df.shape[0]
        X = np.zeros(shape=(n, 7), dtype=object)

        # This is where each time series is
        # transoformed to be represented by a formula
        # with optimized parameters

        X_phis = X_df.values[:, 0: 200]
        predicted_model = self.clf.predict(X_phis)
        predicted_prob = self.clf.predict_proba(X_phis)

        self.c = np.array([])
        self.params = [0]

        for i in range(n):
            self.y_to_fit = X_phis[i, :]
#            cc = fit_features(self.y_to_fit)
            cc = np.ones(3 * self.n_epicyle)
            if(self.really_fit):
                a_bnds = (0., 2.)
                w_bnds = (0.001, 0.5)
                p_bnds = (0., np.pi)
                bnds = tuple(map(tuple, np.tile(
                    np.array([a_bnds, w_bnds, p_bnds]), (self.n_epicyle, 1))))
                res = minimize(fun=self.epicycle_error,
                               x0=cc,
                               method='TNC', tol=1e-8,
                               bounds=bnds,
                               options={
                                   # 'xtol': 0.001,
                                   # 'eps': 0.02,
                                   'maxiter': 1000})
                self.c = res.x
            X[i][0] = f_phi(*(decode_parameters(self.c)),
                            t=[_n_burn_in + _n_lookahead])
            X[i, 1] = predicted_model[i]
            X[i, 2:7] = predicted_prob[i, :]

        return X
