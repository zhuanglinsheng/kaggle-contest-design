"""
"""
import numpy as np
from scipy import optimize, stats
from datetime import datetime
from typing import SupportsFloat

def aux_fn_gamma(u: float) -> float:
	assert -1 <= u <= 1, 'input out of domain'
	if u == -1:
		return -np.inf
	elif u == 1:
		return np.inf
	else:
		return u / (1 - u**2) + np.log((1 + u) / (1 - u)) / 2

def aux_fn_rho(z: float, rho_i: float, rho_j: float) -> float:
	gamma_i = aux_fn_gamma(rho_i)
	gamma_j = aux_fn_gamma(rho_j)

	if np.isinf(gamma_i):
		if gamma_i > 0:
			if np.isinf(gamma_j):
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
			if np.isinf(gamma_j):
				if gamma_j > 0:  # gamma_i is finite, gamma_j = +inf
					return 1
				else:            # gamma_i is finite, gamma_j = -inf
					return -1
	def res(y):
		return aux_fn_gamma(y) - stats.norm.cdf(z) * (gamma_i + gamma_j) + gamma_j

	sol = optimize.root_scalar(res, bracket=(-1, 1))
	if not sol.converged:
		print('Error...')
		print(sol)
	return sol.root

def get_equilibrium_efforts(
		y: float,
		t: SupportsFloat | datetime | np.datetime64,
		T: SupportsFloat | datetime | np.datetime64,
		*,
		prize: float,
		c_i: float,
		c_j: float,
		innov_uncert: float
) -> tuple[float, float]:
	"""Implementing the Equilibrium
	"""
	if isinstance(t, np.datetime64):
		t = t.astype(datetime)
	if isinstance(T, np.datetime64):
		T = T.astype(datetime)
	if isinstance(t, SupportsFloat) and isinstance(T, SupportsFloat):
		remaining_time = float(T) - float(t)
	elif isinstance(t, datetime) and isinstance(T, datetime):
		remaining_time = (T - t).total_seconds() / 3600
	else:
		remaining_time = 0
		ValueError("{t} and {T} should be either float or datatime")

	w_i = prize / innov_uncert**2 / c_i
	w_j = prize / innov_uncert**2 / c_j
	rho_i = (np.exp(w_i) + np.exp(-w_j) - 2) / (np.exp(w_i) - np.exp(-w_j))
	rho_j = (np.exp(w_j) + np.exp(-w_i) - 2) / (np.exp(w_j) - np.exp(-w_i))
	z = y / innov_uncert / remaining_time**0.5
	rho_z = aux_fn_rho(z, rho_i, rho_j)
	gamma_i = aux_fn_gamma(rho_i)
	gamma_j = aux_fn_gamma(rho_j)
	density: float = stats.norm.pdf(y, loc=0, scale=innov_uncert * remaining_time)  # type: ignore
	emplify = innov_uncert**2 / 2 * (gamma_i + gamma_j) * (1 - rho_z**2)
	K_i = emplify * (1 + rho_z)
	K_j = emplify * (1 - rho_z)
	q_i = density * K_i
	q_j = density * K_j
	return q_i, q_j
