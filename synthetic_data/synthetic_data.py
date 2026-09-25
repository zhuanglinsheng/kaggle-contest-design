"""Generate Synthetic Data
"""

# std library
from datetime import datetime, timedelta

# third-party
import numpy as np

from utils import simulate_poisson_process


def steady_state_variance(sigma: float, lamb: float) -> float:
	"""Return the steady-state filtering variance \bar S(lambda)."""
	if sigma <= 0 or lamb <= 0:
		raise ValueError("sigma and lamb must be strictly positive")
	return sigma / lamb**0.5


def leading_order_efforts(
		tilde_y: float,
		t: datetime,
		T: datetime,
		*,
		prize: float,
		c_i: float,
		c_j: float,
		sigma: float,
		lamb: float,
) -> tuple[float, float]:
	"""Evaluate the leading-order effort rule in Theorem 1."""
	remaining_days = (T - t).total_seconds() / (24 * 3600)
	if remaining_days <= 0:
		return 0.0, 0.0
	if prize < 0 or c_i <= 0 or c_j <= 0:
		raise ValueError("prize must be nonnegative and costs must be positive")
	q_t = sigma**2 * remaining_days + steady_state_variance(sigma, lamb)
	kernel = np.exp(-tilde_y**2 / (2 * q_t)) / np.sqrt(2 * np.pi * q_t)
	return prize * kernel / c_i, prize * kernel / c_j


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
		mu_0: float = 0.0,
		approx: bool = True,  # retained for compatibility; PaperJK7 is leading-order
		seed_brownian: int = 442,
		seed_poisson: int = 1234,
		seed_uniform: int = 5678,
		seed_initial: int | None = None,
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
	rng_initial = np.random.default_rng(seed=seed_initial if seed_initial is not None else seed_brownian + 3)
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

	bar_s = steady_state_variance(sigma, lamb)
	real_gap_dynamic = np.zeros(shape = 1 + len(time_grids), dtype=np.float64)
	real_gap_dynamic[0] = rng_initial.normal(loc=mu_0, scale=bar_s**0.5)
	real_gap_t = real_gap_dynamic[0]
	perceived_gap_dynamic = np.zeros(shape = 1 + len(time_grids), dtype=np.float64)
	perceived_gap_dynamic[0] = mu_0
	perceived_gap_t = perceived_gap_dynamic[0]
	observed_gap_dynamic = np.zeros(shape = 1 + len(time_grids), dtype=np.float64)
	observed_gap_dynamic[0] = mu_0
	observed_gap_t = observed_gap_dynamic[0]
	last_observation_step = 0

	i_submission_events: list[datetime] = []
	j_submission_events: list[datetime] = []

	for idx_time, (time, z_shock, z_noise) in enumerate( \
							zip(time_grids, innovation_shocks, signal_noises)):
		## equation (7): leading-order equilibrium effort
		q_i, q_j = leading_order_efforts( \
			perceived_gap_t,
			t=time,
			T=end_time,
			prize=theta,
			c_i=c_i,
			c_j=c_j,
			sigma=sigma,
			lamb=lamb,
		)
		i_effort_dynamic[idx_time] = q_i  # real-time daily effort
		j_effort_dynamic[idx_time] = q_j  # real-time daily effort

		## y: equation (1)
		expected_d_gap_t = (q_i - q_j) * time_unit_2f
		innovation_shock_t = time_unit_2f**0.5 * sigma * z_shock
		real_gap_t += expected_d_gap_t + innovation_shock_t
		real_gap_dynamic[idx_time + 1] = real_gap_t

		## tilde_y: equation (11)
		kalman_rate = lamb**0.5 * sigma  # = lambda \bar{S}
		kalman_weight = -np.expm1(-kalman_rate * time_unit_2f)
		perceived_gap_t += expected_d_gap_t + kalman_weight * (observed_gap_t - perceived_gap_t)
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
			observation_step = idx_time + 1
			readout_interval = (observation_step - last_observation_step) * time_unit_2f
			observed_gap_t_noise = z_noise / (readout_interval * lamb)**0.5
			observed_gap_t = real_gap_t + observed_gap_t_noise
			last_observation_step = observation_step
		observed_gap_dynamic[idx_time + 1] = observed_gap_t

	return time_grids, i_effort_dynamic, j_effort_dynamic, \
			real_gap_dynamic, perceived_gap_dynamic, observed_gap_dynamic, \
			i_submission_events, j_submission_events
