"""
"""

# std library
from datetime import datetime, timedelta
from typing import Callable

# third-party
import numpy as np


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

def simulate_nhpp(
		start_time: datetime,
		end_time: datetime,
		hour_arrivals_ub: float,
		hour_arrivals: Callable[[float], float],
		rng_poisson: np.random.Generator,
		rng_uniform: np.random.Generator,
) -> list[datetime]:
	"""Using thinning technique

	Note:
		hour_arrivals(t) : [0, 1] -> [0, hour_arrivals_ub]
	"""
	events: list[datetime] = []
	unthinned_events = simulate_poisson_process( \
			start_time, end_time, hour_arrivals_ub, rng_poisson)
	uniforms = rng_uniform.uniform(low=0, high=1, size=len(unthinned_events))
	total_secs = (end_time - start_time).total_seconds()

	for u, t in zip(uniforms, unthinned_events):
		t_loc = (t - start_time).total_seconds() / total_secs
		t_intensity = hour_arrivals(t_loc)
		if u < t_intensity / hour_arrivals_ub:
			events.append(t)
	return events
