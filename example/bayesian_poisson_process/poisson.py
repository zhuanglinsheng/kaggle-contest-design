"""Generate Poisson Process

- Homogeneous Poisson process
- Inhomogeneous Poisson process
"""

# std library
from datetime import datetime, timedelta
from typing import Callable
import random


def simulate_hpp(
		start_time: datetime,
		end_time: datetime,
		hour_arrivals: float,
		rng: random.Random
) -> list[datetime]:
	"""Simulate homogeneous Poisson process
	"""
	events: list[datetime] = []
	current_t = start_time
	while True:
		hour_duration = rng.expovariate(lambd=hour_arrivals)  # lambd = 1 / E(x)
		current_t += timedelta(hours=hour_duration)
		if current_t < end_time:
			events.append(current_t)
		else:
			break
	return events

def simulate_nhpp(
		start_time: datetime,
		end_time: datetime,
		hour_arrivals: Callable[[float], float],
		hour_arrivals_ub: float,
		rng_poisson: random.Random,
		rng_uniform: random.Random,
):
	"""Simulate non-homogeneous Poisson process

	Note:
		Using thinning technique

	Arguments:
		- `hour_arrivals`: a function mapping from [0, 1] to R_+.
		- `hour_arrivals_ub`: the upper bound of `hour_arrivals`.
	"""
	events: list[datetime] = []
	unthinned_events = simulate_hpp( \
			start_time, end_time, hour_arrivals_ub, rng_poisson)
	uniforms = [rng_uniform.uniform(a = 0, b = 1) for _ in unthinned_events]
	total_secs = (end_time - start_time).total_seconds()

	for u, t in zip(uniforms, unthinned_events):
		t_loc = (t - start_time).total_seconds() / total_secs
		t_intensity = hour_arrivals(t_loc)
		if u < t_intensity / hour_arrivals_ub:
			events.append(t)
	return events, unthinned_events, uniforms
