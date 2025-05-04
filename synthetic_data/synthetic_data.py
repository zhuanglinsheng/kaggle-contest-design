"""Generate Synthetic Data
"""

# std library
import sys
import os
from datetime import datetime, timedelta

# third-party
import numpy as np

# local
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from ryvkin import ryvkin_model
from utils import simulate_poisson_process


def synthetic_data_simulation(
		theta: float,
		c_i: float,    # daily cost
		c_j: float,    # daily cost
		sigma: float,  # daily innovation risk
		lamb: float,
		intensity_effort_ratio: float,
		hour_arrival_ub: float,
		start_time: datetime,
		end_time: datetime,
		time_unit: timedelta = timedelta(hours=1),
		time_unit_2f: float = 1 / 24,  # transform `time_unit` to float
		*,
		approx: bool = True,  # by default, we use the approximated version
		seed_brownian: int = 442,
		seed_poisson: int = 1234,
		seed_uniform: int = 5678,
):
	"""
	"""
	# Time discretization
	## not including `end_time`
	time_grids: list[datetime] = np.arange(start_time, end_time, time_unit, dtype=datetime).tolist()

	# Simulation (unit = hour)
	## noises and shocks
	rng_brownian = np.random.default_rng(seed=seed_brownian)
	signal_noises = rng_brownian.normal(size=len(time_grids))
	innovation_shocks = rng_brownian.normal(size=len(time_grids))
	## unthinned poisson
	rng_poisson = np.random.default_rng(seed=seed_poisson)
	rng_uniform = np.random.default_rng(seed=seed_uniform)
	poisson_i = simulate_poisson_process(start_time, end_time, hour_arrival_ub, rng_poisson)
	poisson_j = simulate_poisson_process(start_time, end_time, hour_arrival_ub, rng_poisson)
	uniform_i = rng_uniform.uniform(low=0, high=1, size=len(poisson_i))
	uniform_j = rng_uniform.uniform(low=0, high=1, size=len(poisson_j))

	# Solve equilibrium paths
	i_effort_dynamic = np.zeros_like(time_grids, dtype=np.float64)
	j_effort_dynamic = np.zeros_like(time_grids, dtype=np.float64)

	real_gap_dynamic = np.zeros(shape = 1 + len(time_grids), dtype=np.float64)
	real_gap_t = real_gap_dynamic[0]            # = 0
	perceived_gap_dynamic = np.zeros(shape = 1 + len(time_grids), dtype=np.float64)
	perceived_gap_t = perceived_gap_dynamic[0]  # = 0
	observed_gap_dynamic = np.zeros(shape = 1 + len(time_grids), dtype=np.float64)
	observed_gap_t = observed_gap_dynamic[0]    # = 0

	i_submission_events: list[datetime] = []
	j_submission_events: list[datetime] = []

	for idx_time, (time, shock, noise) in enumerate( \
							zip(time_grids, innovation_shocks, signal_noises)):
		## equation (7)
		q_i, q_j = ryvkin_model.get_equilibrium_efforts( \
			tilde_y=perceived_gap_t,
			t=time,
			T=end_time,
			prize=theta,
			c_i=c_i,
			c_j=c_j,
			sigma=sigma,
			approx=approx
		)
		i_effort_dynamic[idx_time] = q_i  # real-time daily effort
		j_effort_dynamic[idx_time] = q_j  # real-time daily effort

		## y: equation (1)
		expected_d_gap_t = (q_i - q_j) * time_unit_2f
		innovation_shock_t = time_unit_2f**0.5 * sigma * shock
		real_gap_t += expected_d_gap_t + innovation_shock_t
		real_gap_dynamic[idx_time + 1] = real_gap_t

		## tilde_y: equation (11)
		kalman_gain = lamb**0.5 * sigma * (observed_gap_t - perceived_gap_t)
		perceived_gap_t += expected_d_gap_t + kalman_gain * time_unit_2f
		perceived_gap_dynamic[idx_time + 1] = perceived_gap_t

		## Thinning:
		i_submitted, j_submitted = False, False
		accept_rate_i = intensity_effort_ratio * q_i / 24 / hour_arrival_ub
		accept_rate_j = intensity_effort_ratio * q_j / 24 / hour_arrival_ub
		if accept_rate_i > 1 or accept_rate_j > 1:
			print('Warn: `accept_rate_i(j) > 1`, increase `hour_arrival_ub`!!!')
		poisson_i_unthin = [(t, u) for t, u in zip(poisson_i, uniform_i) if time <= t <= time + time_unit]
		for t_i, u_i in poisson_i_unthin:
			if u_i < accept_rate_i:
				i_submission_events.append(t_i)
				i_submitted = True
		poisson_j_unthin = [(t, u) for t, u in zip(poisson_j, uniform_j) if time <= t <= time + time_unit]
		for t_j, u_j in poisson_j_unthin:
			if u_j < accept_rate_j:
				j_submission_events.append(t_j)
				j_submitted = True
		## public leaderboard: equation (10)
		if i_submitted or j_submitted:
			observed_gap_t_noise = noise / (time_unit_2f * lamb)**0.5
			observed_gap_t = real_gap_t + observed_gap_t_noise
		observed_gap_dynamic[idx_time + 1] = observed_gap_t

	return time_grids, i_effort_dynamic, j_effort_dynamic, \
			real_gap_dynamic, perceived_gap_dynamic, observed_gap_dynamic, \
			i_submission_events, j_submission_events
