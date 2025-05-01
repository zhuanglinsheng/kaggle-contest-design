"""c_i=1e-08, c_j=1e-08, sigma=6.49612e+38, lambda=1e-08, r=4.43168e+100
"""

# std library
import math
from datetime import datetime, timedelta
from typing import Sequence

# third-party
import numpy as np
from numpy import typing as npt
from scipy import optimize, stats


def aux_fn_gamma(u: float) -> float:
	assert -1 <= u <= 1, 'input out of domain'
	if u == -1:
		return -math.inf
	elif u == 1:
		return math.inf
	else:
		return u / (1 - u**2) + math.atanh(u)

def aux_fn_invgamma(z: float) -> float:
	"""
	"""
	res = lambda u: z - aux_fn_gamma(u)
	sol = optimize.root_scalar(res, bracket=[-1, 1])
	if not sol.converged:
		print('Error...')
		print(sol)
	return sol.root

def aux_fn_invgamma_approx(z: float) -> float:
	return np.arctan(0.856 * z) * 2 / np.pi

def aux_fn_rho(z: float, rho_i: float, rho_j: float, approx: bool) -> float:
	gamma_i = aux_fn_gamma(rho_i)
	gamma_j = aux_fn_gamma(rho_j)

	if math.isinf(gamma_i):
		if gamma_i > 0:
			if math.isinf(gamma_j):
				if gamma_j > 0:  # gamma_i = +inf, gamma_j = +inf
					if z == 0:
						return 0
					elif z > 0:
						return 1
					else:
						return -1
				else:            # gamma_i = +inf, gamma_j = -inf
					return -1
			else:                # gamma_i = +inf, gamma_j is finite
				return 1
		else:
			if math.isinf(gamma_j):
				if gamma_j > 0:  # gamma_i is finite, gamma_j = +inf
					return 1
				else:            # gamma_i is finite, gamma_j = -inf
					return -1
	loc:float = stats.norm.cdf(z) * (gamma_i + gamma_j) - gamma_j  # type:ignore
	if not approx:
		return aux_fn_invgamma(loc)
	else:
		return aux_fn_invgamma_approx(loc)

def get_equilibrium_efforts(
		tilde_y: float,
		t: datetime,
		T: datetime,
		*,
		prize: float,
		c_i: float,    # daily cost
		c_j: float,    # daily cost
		sigma: float,  # daily innovation shock
		approx: bool = False,
		display: bool = False
) -> tuple[float, float]:
	"""Implementing the Equilibrium
	"""
	remaining_hours = (T - t).total_seconds() / 3600
	remaining_days = remaining_hours / 24  # days
	w_i = prize / sigma**2 / c_i
	w_j = prize / sigma**2 / c_j
	rho_i = (math.exp(w_i) + math.exp(-w_j) - 2) / (math.exp(w_i) - math.exp(-w_j))
	rho_j = (math.exp(w_j) + math.exp(-w_i) - 2) / (math.exp(w_j) - math.exp(-w_i))
	tilde_y_stderr = sigma * remaining_days**0.5
	z = tilde_y / tilde_y_stderr
	rho_z = aux_fn_rho(z, rho_i, rho_j, approx)
	gamma_i = aux_fn_gamma(rho_i)
	gamma_j = aux_fn_gamma(rho_j)
	density: float = stats.norm.pdf(
		tilde_y, loc=0, scale=tilde_y_stderr)  # type: ignore
	emplify = sigma**2 / 2 * (gamma_i + gamma_j) * (1 - rho_z**2)
	K_i = emplify * (1 + rho_z)
	K_j = emplify * (1 - rho_z)
	q_i = density * K_i
	q_j = density * K_j
	if display:
		print(w_i, w_j, rho_i, rho_j, gamma_i, gamma_j, z, rho_z, emplify, K_i, K_j)
	return q_i, q_j

def solve_equailibrium_path(
		time_grids: Sequence[datetime] | npt.NDArray[np.datetime64],
		time_unit_2f: float,
		end_time: datetime,
		observed_gap_dynamic: Sequence[float] | npt.NDArray[np.float64],
		*,
		prize: float,
		c_i: float,
		c_j: float,
		innov_uncert: float,
		lamb: float
):
	"""
	"""
	tilde_y = 0

	perceived_gap_dym: list[float] = []
	effort_i_dym: list[float] = []
	effort_j_dym: list[float] = []

	for idx_time, (time, hat_y) in enumerate(zip(time_grids, observed_gap_dynamic)):
		# calculate q_i, q_j
		q_i, q_j = get_equilibrium_efforts(
			tilde_y, time, end_time,
			prize=prize, c_i=c_i, c_j=c_j, sigma=innov_uncert
		)
		perceived_gap_dym.append(tilde_y)
		effort_i_dym.append(q_i)
		effort_j_dym.append(q_j)

		# calculate tilde_y
		kalman_gain = (hat_y - tilde_y) * lamb**0.5 * innov_uncert
		tilde_y += (q_i - q_j + kalman_gain) * time_unit_2f
	return perceived_gap_dym, effort_i_dym, effort_j_dym
