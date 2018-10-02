import numpy as np

from scipy.optimize import minimize

label_names = np.array(['A', 'B', 'C', 'D'])

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
        self.n_epicyle = 30
        self.params = np.array([])
        self.window = 0
        self.really_fit = True
        self.models = []

    def epicycle_error(self, c):
        a, w, p = decode_parameters(c)
        phi_epi = [f_phi(a, w, p, i) for i in range(self.fit_length)]
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
        pass

    def transform(self, X_df):
        n = X_df.shape[0]
        X = np.zeros(shape=(n, 1))

        # This is where each time series is
        # transoformed to be represented by a formula
        # with optimized parameters

        X_system = X_df.values[:, -4:]
        X_phis = X_df.values[:, 0: -4]

        self.c = np.array([])
        self.params = [0]

        for i in range(n):
            self.fit_length = X_phis.shape[1]
            self.y_to_fit = X_phis[i, -self.fit_length:]
#            cc = fit_features(self.y_to_fit)
            cc = np.ones(3 * self.n_epicyle)
            if(self.really_fit):
             #               c_ind = cc[1:]
                a_bnds = (0., 2.)
                w_bnds = (0.001, 0.5)
                p_bnds = (0., np.pi)
                bnds = tuple(map(tuple, np.tile(
                    np.array([a_bnds, w_bnds, p_bnds]), (self.n_epicyle, 1))))
                print("bnds : ", bnds)
                res = minimize(fun=self.epicycle_error,
                               x0=cc,
                               method='TNC', tol=1e-8,
                               bounds=bnds,
                               options={
                                   # 'xtol': 0.001,
                                   # 'eps': 0.02,
                                   'maxiter': 100000})
 #               self.c = np.append([1.], res.x)
                self.c = res.x
            X[i][0] = f_phi(*(decode_parameters(self.c)),
                            [_n_burn_in + _n_lookahead])

        return X
