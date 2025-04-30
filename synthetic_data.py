"""
"""

# std library
from datetime import datetime, timedelta

# third-party
import numpy as np

# local
import ryvkin_model


def simulate_poisson_process(
		start_time: datetime,
		end_time: datetime,
		hour_arrival_rate: float,
		rng: np.random.Generator
) -> list[datetime]:
	"""
	"""
	events: list[datetime] = []
	current_t = start_time
	while True:
		hour_duration = rng.exponential(scale=1 / hour_arrival_rate)
		current_t += timedelta(hours=hour_duration)
		if current_t < end_time:
			events.append(current_t)
		else:
			break
	return events

def synthetic_data_simulation(
		c_i: float,    # hourly unit cost
		c_j: float,    # hourly unit cost
		theta: float,
		sigma: float,  # hourly standard deviation
		lamb: float,
		r: float,
		hour_arrival_ub: float,
		start_time: datetime,
		end_time: datetime,
		time_unit: timedelta = timedelta(hours=1),
		time_unit_2f: float = 1/24,
		*,
		seed_brownian: int = 37,
		seed_poisson: int = 1234,
		seed_uniform: int = 5678
):
	"""
	"""
	# Time discretization
	time_grids = np.arange(start_time, end_time, time_unit, dtype=datetime)

	# Simulation
	rng_brownian = np.random.default_rng(seed=seed_brownian)
	rng_poisson = np.random.default_rng(seed=seed_poisson)
	rng_uniform = np.random.default_rng(seed=seed_uniform)
	##
	signal_noises = rng_brownian.normal(size=time_grids.size)
	innovation_shocks = rng_brownian.normal(size=time_grids.size)
	poisson_i = simulate_poisson_process(start_time, end_time, hour_arrival_ub, rng_poisson)
	poisson_j = simulate_poisson_process(start_time, end_time, hour_arrival_ub, rng_poisson)
	uniform_i = rng_uniform.uniform(low=0, high=1, size=len(poisson_i))
	uniform_j = rng_uniform.uniform(low=0, high=1, size=len(poisson_j))

	#
	i_effort_dynamic = np.zeros_like(time_grids, dtype=np.float64)
	j_effort_dynamic = np.zeros_like(time_grids, dtype=np.float64)

	real_gap_dynamic = np.zeros(shape = time_grids.shape, dtype=np.float64)
	real_gap_t = 0

	perceived_gap_dynamic = np.zeros(shape = time_grids.shape, dtype=np.float64)
	perceived_gap_t = 0

	observed_gap_dynamic = np.zeros(shape = time_grids.shape, dtype=np.float64)
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
			innov_uncert=sigma
		)
		i_effort_dynamic[idx_time] = q_i
		j_effort_dynamic[idx_time] = q_j

		## tau: equation (8)
		tensity_i = r * q_i
		tensity_j = r * q_j
		## y: equation (1)
		expected_d_gap_t = (q_i - q_j) * time_unit_2f
		real_gap_t += expected_d_gap_t + sigma * shock
		real_gap_dynamic[idx_time] = real_gap_t
		## tilde_y: equation (11)
		kalman_gain = lamb**0.5 * sigma * (observed_gap_t - perceived_gap_t)
		perceived_gap_t += expected_d_gap_t + kalman_gain * time_unit_2f
		perceived_gap_dynamic[idx_time] = perceived_gap_t

		## Thinning:
		poisson_i_select = [(t, u) for t, u in zip(poisson_i, uniform_i) if time <= t <= time + time_unit]
		poisson_j_select = [(t, u) for t, u in zip(poisson_j, uniform_j) if time <= t <= time + time_unit]
		for t_i, u_i in poisson_i_select:
			if u_i < tensity_i / hour_arrival_ub:
				#print('>>>(i)', u_i, tensity_i, hour_arrival_ub)
				# submission events
				i_submission_events.append(t_i)
				# update public leaderboard: equation (10)
				observed_gap_t = real_gap_dynamic[idx_time] + noise / lamb**0.5
		for t_j, u_j in poisson_j_select:
			if u_j < tensity_j / hour_arrival_ub:
				#print('>>>(j)', u_j, tensity_j, hour_arrival_ub)
				# submission events
				j_submission_events.append(t_j)
				# update public leaderboard: equation (10)
				observed_gap_t = real_gap_dynamic[idx_time] + noise / lamb**0.5
		observed_gap_dynamic[idx_time] = observed_gap_t

	return time_grids, i_effort_dynamic, j_effort_dynamic, \
			real_gap_dynamic, perceived_gap_dynamic, observed_gap_dynamic, \
			i_submission_events, j_submission_events
