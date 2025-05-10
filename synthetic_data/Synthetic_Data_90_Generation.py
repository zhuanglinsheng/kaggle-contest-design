# std library
import os
import json
from pathlib import Path
from pprint import pprint
from datetime import datetime, timedelta
from typing import Sequence

# third-party
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from numpy import typing as npt

# local
from synthetic_data import synthetic_data_simulation  # type: ignore

# Capacities & Total Innovation Uncertainty
param_theta = 1.0

param_c_i = 1.2
param_c_j = 1.5
param_sigma = 2.0
param_lambda = 1.0
param_r = 15.0

# Contest Time Duaration
start_time = datetime(2025, 1, 1, 0, 0, 0)
end_time = datetime(2025, 4, 1, 0, 0, 0)
contest_days = (end_time - start_time).days
contest_seconds = int((end_time - start_time).total_seconds())
contest_hours = int(contest_seconds / 3600)
contest_timegrids: list[datetime] = np.arange(start_time, end_time, \
		timedelta(hours=1), dtype=datetime).tolist()
time_unit_2f = 1 / 24


for random_seed in range(10):

	time_grids, i_effort_dynamic, j_effort_dynamic, \
	real_gap_dynamic, perceived_gap_dynamic, observed_gap_dynamic, \
	observed_i_commits, observed_j_commits = synthetic_data_simulation(
			theta=param_theta,
			c_i=param_c_i,
			c_j=param_c_j,
			sigma=param_sigma,
			lamb=param_lambda,
			intensity_effort_ratio=param_r,
			hour_arrival_ub=1.0,
			start_time=start_time,
			end_time=end_time,
			time_unit=timedelta(hours=1),
			time_unit_2f=time_unit_2f,
			approx = True,  # using approximated version by default
			seed_brownian = random_seed + 0,
			seed_poisson = random_seed + 1,
			seed_uniform = random_seed + 2,
		)

	# save the observed data
	wd = os.getcwd()
	wd_synthetic_data = os.path.join(wd, f'data_{contest_days}_{random_seed}.json')
	roundint = lambda x: int(round(x))

	with open(wd_synthetic_data, 'w') as f:
		json.dump({
			'theta': param_theta,
			'ratio': param_r,
			'N_Delta': roundint((end_time - start_time).total_seconds() / 3600),
			'Delta2f': time_unit_2f,
			'Ni': len(observed_i_commits),
			'Nj': len(observed_j_commits),
			'hat_t_i': [(dt - start_time).total_seconds() / 3600 for dt in observed_i_commits],
			'hat_t_j': [(dt - start_time).total_seconds() / 3600 for dt in observed_j_commits],
			'hat_y': observed_gap_dynamic.tolist(),
			'efforts_i': i_effort_dynamic.tolist(),
			'efforts_j': j_effort_dynamic.tolist(),
		}, f, indent=4)

