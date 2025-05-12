"""
"""

# std library
from datetime import datetime, timedelta
from typing import Callable

# third-party
import numpy as np
from scipy.interpolate import interp1d


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

def intensity_interp(
		time_ticks: np.ndarray,
		intensity_sample: np.ndarray,
		ratio: float = 1,
		kind: str = 'linear',
		fill_value = np.nan
) -> Callable[[float], float]:
	"""Using interpolation method to approximate the intensity function.
	"""
	t_floats = time_ticks.astype(float)
	intensity_f = interp1d(t_floats, intensity_sample * ratio, kind=kind, fill_value=fill_value)
	return intensity_f

def intensity_interp_wrapper(
		intensity_f: Callable[[float], float]
) -> Callable[[datetime | np.datetime64], float]:
	"""A wrapper of in the interpolation function `intensity_f`
	"""
	def intensity(x: datetime | np.datetime64) -> float:
		if isinstance(x, datetime):
			x = np.datetime64(x)
		x_float = x.astype(float)
		return intensity_f(x_float)
	return intensity

def intensity_interp_d(
		time_ticks: np.ndarray,
		intensity_sample: np.ndarray,
		ratio: float = 1,
		kind: str = 'linear',
		fill_value = np.nan
) -> Callable[[datetime | np.datetime64], float]:
	"""Using interpolation method to approximate the intensity function.
	"""
	intensity_f = intensity_interp(time_ticks, intensity_sample, ratio, kind, fill_value)
	return intensity_interp_wrapper(intensity_f)
