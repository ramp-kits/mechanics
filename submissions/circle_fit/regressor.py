from sklearn.base import BaseEstimator
from scipy.optimize import minimize

import numpy as np
_n_lookahead = 5
_n_burn_in = 200


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


def find_local_extrema(x):
    n = 3
    maxima = np.array([])
    minima = np.array([])
    loops = np.array([])

    p_prev = np.zeros(n)
    d_prev = np.zeros(n)
    c_prev = np.zeros(n)

    for i, p in enumerate(x):
        d = p - p_prev[0]
        if(d < - np.pi):
            loops = np.append(loops, i)
            d = d + 2. * np.pi

        if(d > np.pi):
            d = d - 2. * np.pi
        c = d - d_prev[0]

        if((d * d_prev[0]) < 0):
            if(c < 0):
                maxima = np.append(maxima, i)
            else:
                minima = np.append(minima, i)

        p_prev = np.append(p, p_prev[:n])
        d_prev = np.append(d, d_prev[:n])
        c_prev = np.append(c, c_prev[:n])
#    print(loops)
    return maxima, minima, loops


def fit_features(x):
    "Fitting features..."
    a = np.array([1., 1.])
    w = np.array([1., 1.])
    p = np.array([1., 1.])

    # Find retrogrades
    maxima, minima, loops = find_local_extrema(x)

    # Find frequencies
    dist = loops[-1] - loops[0]
    nloop = len(loops) - 1
    if(nloop > 0):
        w[0] = 2. * np.pi / (dist / nloop)
    else:
        print("Failed to get frequency")

    dist = maxima[-1] - maxima[0]
    nloop = len(maxima) - 1
    if(nloop > 0):
        w[1] = 2. * np.pi / (dist / nloop) + w[0]
    else:
        print("Failed to get frequency")

    # Find phases
    p[0] = loops[0] * w[0] - np.pi
    p[1] = (maxima[0]) * w[1] - p[0]

    # Find amplitudes
    a[0] = 1.
    a[1] = 0.5

    c = np.array([a[0], w[0], p[0], a[1], w[1], p[1]])
    return c


class Regressor(BaseEstimator):
    def __init__(self):
        self.c = np.array([51, 0.02, 3., 99, 0.01, 0.01])
        self.really_fit = True
        pass

    def epicycle_error(self, c):
        a, w, p = decode_parameters(c)
        phi_epi = [f_phi(a, w, p, i) for i in range(_n_burn_in)]
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

    def fit(self, X, y):
        print("Fitting the series, shape : ", X.shape)
#        print(X)
        # Maybe this will be where the mechanics will be determined
        # Formulas, main parameters etc.

        self.c = np.array([])
#       self.c = np.ndarray((0, 6))
#        for i in range(X.shape[0]):
        for i in [0]:
            self.y_to_fit = X[i]
            cc = fit_features(self.y_to_fit)
            self.c = cc
            if(self.really_fit):
                c_ind = cc[1:]
                a_bnds = (0., 2.)
                w_bnds = (0.001, 0.5)
                p_bnds = (0., np.pi)
                bnds = (w_bnds, p_bnds, a_bnds, w_bnds, p_bnds)
                res = minimize(fun=self.epicycle_error_independent,
                               x0=c_ind,
                               method='TNC', tol=1e-8,
                               bounds=bnds,
                               options={
                                   # 'xtol': 0.001,
                                   # 'eps': 0.02,
                                   'maxiter': 100000})
                self.c = np.append([1.], res.x)
#            self.c = np.concatenate(self.c, np.append([1.], res.x), axis=1)

    def predict(self, X):
        # For the time being, this is where everything happens
        # This will be where the parameters will be fit
        # Phases etc.
        n_predictions = X.shape[0]
        print("n_predictions : ", n_predictions)
        y = np.zeros(n_predictions)
        for i in range(n_predictions):
            self.y_to_fit = X[i]
            cc = fit_features(self.y_to_fit)
            c_ind = cc[2::3]
            p_bnds = (0., np.pi)
            bnds = (p_bnds, p_bnds)
#            print("c_ind : ", c_ind)
            a, w, p = decode_parameters(self.c)
            aa, ww, pp = decode_parameters(cc)
            p = pp
            if(self.really_fit):
                res = minimize(fun=self.epicycle_error_phase,
                               x0=c_ind,
                               method='TNC', tol=1e-8,
                               bounds=bnds,
                               options={
                                   # 'xtol': 0.001,
                                   # 'eps': 0.02,
                                   'maxiter': 10000})
                p = res.x
            y[i] = f_phi(a, w, p, _n_burn_in + _n_lookahead)
#        print(y)
        return y.reshape(-1, 1)


# NEXT WEEK'S TASK:
# ONLY FIT PHASE IN PREDICT
# TEST SLIGHTLY PERTURBED
# COMPARE TO LSTM
