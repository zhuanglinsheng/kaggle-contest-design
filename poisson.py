"""
"""
from typing import Sequence, Callable
from datetime import datetime, timedelta

import numpy as np
from scipy.interpolate import interp1d

from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib.dates as mdates

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

def simulate_inhomogeneous_poisson_process(
		start_time: datetime,
		end_time: datetime,
		intensity: Callable[[datetime], float],
		hour_arrival_ub: float,
		seed_poisson: int = 1234,
		seed_uniform: int = 1234,
) -> list[datetime]:
	"""Using thinning method
	"""
	rng_poisson = np.random.default_rng(seed_poisson)
	rng_uniform = np.random.default_rng(seed=seed_uniform)

	# Step 1: simulate homogeneous process
	events: list[datetime] = simulate_poisson_process( \
			start_time, end_time, hour_arrival_ub, rng_poisson)

	# Step 2: thinning
	events_inhomo: list[datetime] = []
	n_events = len(events)
	uniforms = rng_uniform.uniform(low=0, high=1, size=n_events)
	for event, u in zip(events, uniforms):
		event_intensity = intensity(event)
		if u < event_intensity / hour_arrival_ub:
			events_inhomo.append(event)
	return events_inhomo

def plot_intensity(
		fn_intensity: Callable[[datetime | np.datetime64], float],
		x_range: tuple[datetime | np.datetime64, datetime | np.datetime64],
		time_delt: timedelta | np.timedelta64,
		figsize = (10, 3)
):
	"""
	"""
	if isinstance(x_range[0], datetime):
		start_time = np.datetime64(x_range[0])
	else:
		start_time = x_range[0]
	if isinstance(x_range[1], datetime):
		end_time = np.datetime64(x_range[1])
	else:
		end_time = x_range[1]
	if isinstance(time_delt, timedelta):
		dt = np.timedelta64(time_delt)
	else:
		dt = time_delt
	current_t = start_time
	time_ticks = []
	intensity_vals = []
	while True:
		current_t += dt
		if current_t < end_time:
			time_ticks.append(current_t)
			intensity_vals.append(fn_intensity(current_t))
		else:
			break
	fig, ax = plt.subplots(figsize=figsize)
	ax.plot(time_ticks, intensity_vals)
	return fig, ax

def plot_poisson_events(
		ax: Axes,
		events: Sequence[float | datetime],
		vline_width: float = 0.01,
		y_tick: float = 0,
		color = 'red'
	) -> Axes:
	"""
	"""
	# hline
	xmin, xmax = ax.get_xlim()
	ax.hlines(y=y_tick, xmin=xmin, xmax=xmax)  # type: ignore

	# vlines
	ax.vlines(x=events,   # type: ignore
			ymin=0 - vline_width / 3,
			ymax=y_tick + vline_width, colors=color)
	return ax
