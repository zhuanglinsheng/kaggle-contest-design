"""Small deterministic likelihood fixture, independent of research caches."""
import math
import numpy as np


def count_data():
    return dict(theta=.3, Delta2f=.1, N_Delta=8, T_contest=.8, count_start_idx=3,
                hat_t_i=[2.2, 2.9, 6.1], hat_t_j=[4.1, 6.2],
                hat_y=[.1, .1, .2, .2, -.15, -.15, .05, .05])


def parameter_point():
    return np.array([math.log(10), math.log(12), 0., math.log(4), .1, math.log(100)])
