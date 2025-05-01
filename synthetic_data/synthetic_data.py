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
		time_unit_2f: float = 1/24,  # transform `time_unit` to daily base
		*,
		seed_brownian: int = 37,
		seed_poisson: int = 1234,
		seed_uniform: int = 5678
):
	"""
	"""
	# Time discretization
	time_grids: list[datetime] = np.arange(start_time, end_time, time_unit, dtype=datetime).tolist()

	# Simulation (unit = hour)
	rng_brownian = np.random.default_rng(seed=seed_brownian)
	rng_poisson = np.random.default_rng(seed=seed_poisson)
	rng_uniform = np.random.default_rng(seed=seed_uniform)

	signal_noises = rng_brownian.normal(size=len(time_grids))
	innovation_shocks = rng_brownian.normal(size=len(time_grids))
	poisson_i = simulate_poisson_process(start_time, end_time, hour_arrival_ub, rng_poisson)
	poisson_j = simulate_poisson_process(start_time, end_time, hour_arrival_ub, rng_poisson)
	uniform_i = rng_uniform.uniform(low=0, high=1, size=len(poisson_i))
	uniform_j = rng_uniform.uniform(low=0, high=1, size=len(poisson_j))

	# Solve equilibrium paths
	i_effort_dynamic = np.zeros_like(time_grids, dtype=np.float64)
	j_effort_dynamic = np.zeros_like(time_grids, dtype=np.float64)

	real_gap_dynamic = np.zeros(shape = len(time_grids), dtype=np.float64)
	real_gap_t = 0

	perceived_gap_dynamic = np.zeros(shape = len(time_grids), dtype=np.float64)
	perceived_gap_t = 0

	observed_gap_dynamic = np.zeros(shape = len(time_grids), dtype=np.float64)
	observed_gap_t = 0

	i_submission_events: list[datetime] = []
	j_submission_events: list[datetime] = []

	for idx_time, (time, shock, noise) in enumerate(zip(time_grids, innovation_shocks, signal_noises)):

		## equation (7)
		q_i, q_j = ryvkin_model.get_equilibrium_efforts( \
			tilde_y=perceived_gap_t,
			t=time,
			T=end_time,
			prize=theta,
			c_i=c_i,
			c_j=c_j,
			sigma=sigma
		)
		i_effort_dynamic[idx_time] = q_i  # realtime daily effort
		j_effort_dynamic[idx_time] = q_j  # realtime daily effort

		## tau: equation (8)
		intensity_i = intensity_effort_ratio * q_i  # realtime daily intensity
		intensity_j = intensity_effort_ratio * q_j  # realtime daily intensity

		## y: equation (1)
		expected_d_gap_t = (q_i - q_j) * time_unit_2f
		real_gap_t += expected_d_gap_t + sigma * shock
		real_gap_dynamic[idx_time] = real_gap_t

		## tilde_y: equation (11)
		kalman_gain = lamb**0.5 * sigma * (observed_gap_t - perceived_gap_t)
		perceived_gap_t += expected_d_gap_t + kalman_gain * time_unit_2f
		perceived_gap_dynamic[idx_time] = perceived_gap_t

		## Thinning:
		poisson_i_unthin = [(t, u) for t, u in zip(poisson_i, uniform_i) if time <= t <= time + time_unit]
		poisson_j_unthin = [(t, u) for t, u in zip(poisson_j, uniform_j) if time <= t <= time + time_unit]
		for t_i, u_i in poisson_i_unthin:
			if u_i < intensity_i / 24 / hour_arrival_ub:
				# submission events
				i_submission_events.append(t_i)
				# update public leaderboard: equation (10)
				observed_gap_t = real_gap_dynamic[idx_time] + noise / lamb**0.5
		for t_j, u_j in poisson_j_unthin:
			if u_j < intensity_j / 24 / hour_arrival_ub:
				# submission events
				j_submission_events.append(t_j)
				# update public leaderboard: equation (10)
				observed_gap_t = real_gap_dynamic[idx_time] + noise / lamb**0.5
		observed_gap_dynamic[idx_time] = observed_gap_t

	return time_grids, i_effort_dynamic, j_effort_dynamic, \
			real_gap_dynamic, perceived_gap_dynamic, observed_gap_dynamic, \
			i_submission_events, j_submission_events
