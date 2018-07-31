import numpy as np
import tensorflow as tf

import numpy as np


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
