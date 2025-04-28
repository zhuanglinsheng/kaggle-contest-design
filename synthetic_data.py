import numpy as np
from datetime import datetime, timedelta

import poisson
import ryvkin_model

def synthetic_data_simulation(
		c_i: float,
		c_j: float,
		theta: float,
		sigma: float,
		lamb: float,
		r: float,
		hour_arrival_rate_ub: float,
		start_time: datetime,
		end_time: datetime,
		time_unit: timedelta = timedelta(hours=1),
		y0: float = 0,
		tilde_y0: float = 0,
		*,
		seed_brownian: int = 337,
		seed_poisson: int = 1234,
		seed_uniform: int = 5678
):
	"""
	"""
	## Time discretization
	time_grids = np.arange(start_time, end_time, time_unit)

	## Simulate a path of innovation shocks
	rng_shock = np.random.default_rng(seed=seed_brownian)
	brownian = rng_shock.normal(size=time_grids)

	## Simulate homogeneous Poisson process
	poisson_i = poisson.simulate_poisson_process(start_time, end_time, hour_arrival_rate_ub, seed_poisson)
	poisson_j = poisson.simulate_poisson_process(start_time, end_time, hour_arrival_rate_ub, seed_poisson + 1)

	# Simulate from uniform distribution on [0, 1]
	rng_uniform = np.random.default_rng(seed=seed_uniform)
	uniform_i = rng_uniform.uniform(low=0, high=1, size=len(poisson_i)) # no problem here since
	uniform_j = rng_uniform.uniform(low=0, high=1, size=len(poisson_j)) #     uniform_i != uniform_j

	#
	dynamic_i_effort = np.zeros_like(time_grids, dtype=np.float64)
	dynamic_j_effort = np.zeros_like(time_grids, dtype=np.float64)
	dynamics_gap = np.zeros(shape = len(time_grids), dtype=np.float64)
	i_submissions = []
	j_submissions = []
	gap_t = 0
	for idx_time, time in enumerate(time_grids):
		q_i, q_j = ryvkin_model.get_equilibrium_efforts( \
			gap_t, time, time_grids[-1],
			prize=theta,
			c_i=c_i,
			c_j=c_j,
			innov_uncert=sigma
		)
		tensity_i = r * q_i
		tensity_j = r * q_j

		# Select submission events
		poisson_i_select = [(t, u) for t, u in zip(poisson_i, uniform_i) if time <= t <= time + time_unit]
		poisson_j_select = [(t, u) for t, u in zip(poisson_j, uniform_j) if time <= t <= time + time_unit]

		if len(poisson_i_select) > 0:
			pass

		if len(poisson_j_select) > 0:
			pass

		dynamics_gap[idx_time] = gap_t + (q_i - q_j)
		dynamic_i_effort[idx_time] = q_i
		dynamic_j_effort[idx_time] = q_j
		gap_t = dynamics_gap[idx_time]
	return dynamic_i_effort, dynamic_j_effort, dynamics_gap
