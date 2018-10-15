import numpy as np
import pandas as pd
from Simulation import *


n_sim = 30000
n_view = 100

n_steps = n_sim / n_view
n_body = 3


def generate(name, power=2.):

    sun = Body([0, 0, 0], [0, 0, 0], i=-1, m=100000)

    s = System()
    s.name = name
    s.gravity_power = power
    mmm = [0.00001, 0.00001, 0.00001]

    # Force simulations
    if "FSS" in name:
        s.force = s.force_single_sun
    if "FDS" in name:
        s.force = s.force_double_sun
    if "FG" in name:
        s.force = s.force_gravity
    if "FC" in name:
        s.force = s.force_single_sun
        x = [[50., 0.], [100., 0.], [200., 0.]]
        for i in range(n_body):
            s.add_body(x=x[i], v=random_velocity(x[i], 0), m=mmm[i])
    else:
        x = [[51., 0.], [105., 0.], [220., 0.]]
        for i in range(n_body):
            s.add_body(x=x[i], v=random_velocity(x[i], 0.01), m=mmm[i])

    if "FG" in name:
        s.add_body(x=[0., 0.], v=[0., 0.], m=sun.mass)

    if "F" in name:
        s.simulate(n_sim)

    # Function simulations
    if "P" in name:
        s.simulate_epicycles(n_sim)
    if "H" in name:
        s.simulate_oscillators(n_sim)
    if "M" in name:
        s.simulate_mixed(n_sim)

    s.view(0, n_view)

    write_system(s)
    return s


def write_system(s):
    for ip, p in enumerate(s.bodies):
        phis = s.get_phi(ip)
        x = p.history[::n_view, 0]
        y = p.history[::n_view, 1]
        z = p.history[::n_view, 2]

        time = np.arange(len(phis)) * n_view
        data_full = pd.DataFrame(
            {'time': time, 'x': x, 'y': y, 'z': z, 'phi': phis})
        data_full.to_csv('positions' +
                         '_sys' +
                         s.name +
                         '_planet' +
                         str(ip) +
                         '_nview' +
                         str(n_view) +
                         '_nsim' +
                         str(n_sim) +
                         '.csv',
                         columns=['time', 'x', 'y', 'z', 'phi'],
                         index=False)
        data_phis = pd.DataFrame({'time': time, 'phi': phis})
        data_phis.to_csv('phis' +
                         '_sys' +
                         s.name +
                         '_planet' +
                         str(ip) +
                         '_nview' +
                         str(n_view) +
                         '_nsim' +
                         str(n_sim) +
                         '.csv',
                         columns=['time', 'phi'],
                         index=False)


def produce(i=0):
    np.random.seed(i)
    generate("P" + str(i))
    generate("H" + str(i))
    generate("M" + str(i))
    generate("FSS" + str(i), 2.)
    generate("FG" + str(i), 2.)
    generate("FC" + str(i), 2.)


produce(0)
