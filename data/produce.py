import numpy as np
import pandas as pd
from Simulation import *
import pdb


_n_lookahead = 50
_n_burn_in = 200

n_data = _n_burn_in + _n_lookahead * 2 + 1  # 251
n_view = 100

n_steps = n_data * n_view


def generate(i_cv=0, random_seed=0, n_sim=100):
    np.random.seed(random_seed)

    systems = ["FS", "P", "M", "FD", "FS3", "FG", "H"]
    labels = ["A", "B", "C", "D", "E", "F"]
    planet = [1, 4, 3, 1, 1, 1]
    velocity = [1., 1., 1., 0.001, 1., 1., 1.]
    distance = [1., 1., 1., 2., 1., 1., 1.]

    sun = Body([0, 0, 0], [0, 0, 0], i=-1, m=100000)

    mmm = [0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001]

    data_phis = pd.DataFrame()
    data_x = pd.DataFrame()
    data_y = pd.DataFrame()
    data_ref_x = pd.DataFrame()
    data_ref_y = pd.DataFrame()

    x_sim = pd.read_csv("cvs.csv").values.reshape(-1, 5, 2)
    dv_sim = pd.read_csv("dvs.csv").values.reshape(-1, 5)

    for i_sim in range(n_sim):
        # i_sys = np.random.randint(0, 4)
        i_sys = i_sim % 6
        name = systems[i_sys]
        n_body = planet[i_sys] + 1

        s = System(name)
        s.name = name
        s.gravity_power = 2.
        if "3" in name:
            s.gravity_power = 3.
            s.g = 100.

        for i in range(n_body):
            print("Adding body : ", i)
            x = distance[i_sys] * x_sim[i_cv, i] \
                * np.random.uniform(0.5, 1.5)
            dv = velocity[i_sys] * dv_sim[i_cv, i] \
                * np.random.uniform(0.8, 1.2)
            s.add_body(x=x, v=s.random_velocity(
                x, dv), m=mmm[i])
        if("FG" in name):
            s.add_body(x=[0., 0.], v=[0., 0.], m=sun.mass)
        s.add_oscillators()
        s.simulate(n_steps)

        s.view(0)
        x, y, phis = get_rows(s, planet[i_sys])
        x_ref, y_ref, phis_ref = get_rows(s, 0)
        new_row = pd.DataFrame(phis[:, 0:_n_burn_in])
        new_row['system'] = labels[i_sys]
        new_row['cv'] = i_cv

        new_row['future'] = phis[:, _n_burn_in + _n_lookahead]
        data_phis = pd.concat([data_phis, new_row], axis=0)
        data_x = pd.concat([data_x, pd.DataFrame(
            x[:, 0:_n_burn_in])], axis=0)
        data_y = pd.concat([data_y, pd.DataFrame(
            y[:, 0:_n_burn_in])], axis=0)
        data_ref_x = pd.concat([data_ref_x, pd.DataFrame(
            x_ref[:, 0:_n_burn_in])], axis=0)
        data_ref_y = pd.concat([data_ref_y, pd.DataFrame(
            y_ref[:, 0:_n_burn_in])], axis=0)
        del s

    print(data_phis)

    output = "prod_test_cv_" + str(i_cv) + "_seed_" + str(random_seed)
    data_phis.to_csv(output + "_phi.csv",
                     index=False)
    data_x.to_csv(output + "_x.csv",
                  index=False)
    data_y.to_csv(output + "_y.csv",
                  index=False)
    data_ref_x.to_csv(output + "_ref_x.csv",
                      index=False)
    data_ref_y.to_csv(output + "_ref_y.csv",
                      index=False)


def get_rows(s, ip):
    x = s.bodies[ip].history[::n_view, 0].reshape(1, -1)
    y = s.bodies[ip].history[::n_view, 1].reshape(1, -1)
    phis = s.get_phi(ip)[::n_view].reshape(1, -1)
    return x, y, phis


def produce():
    n = 10
    for cv in range(100):
        for seed in range(100):
            generate(cv, seed, n)


produce()
