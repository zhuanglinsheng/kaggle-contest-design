"""
"""
import bisect
from datetime import datetime, timedelta

import pandas as pd

class Leaderboard:
	"""
	"""
	_datetimes: list[datetime]
	_submitor: list[int]
	_ranks: list[list[int]]      # list of `ordered IDs`
	_scores: list[list[float]]   # list of `ordered scores`
	_agents: set[int]

	def __init__(self):
		self._datetimes = []
		self._submitor = []
		self._ranks = []
		self._scores = []
		self._agents = set()

	def first_submission_time(self) -> datetime:
		return self._datetimes[0]

	def last_submission_time(self) -> datetime:
		return self._datetimes[-1]

	def number_of_submissions(self) -> int:
		return len(self._datetimes)

	def refresh(self, time: datetime, id: int, score: float,
			*,
			min_timedelta: timedelta = timedelta(minutes=1)
	) -> None:
		dyn_ranks = []
		dyn_scores = []
		last_submission_time = None
		last_submitor = None
		if len(self._agents) > 0:
			dyn_ranks = self._ranks[-1].copy()
			dyn_scores = self._scores[-1].copy()
			last_submission_time = self._datetimes[-1]
			last_submitor = self._submitor[-1]

		if id in dyn_ranks:
			old_id_index = dyn_ranks.index(id)
			dyn_ranks.pop(old_id_index)
			dyn_scores.pop(old_id_index)

		new_id_index = bisect.bisect_left([-e for e in dyn_scores], -score)
		dyn_ranks.insert(new_id_index, id)
		dyn_scores.insert(new_id_index, score)

		if last_submission_time is not None \
				and id == last_submitor \
				and time < last_submission_time + min_timedelta:
			self._ranks.pop(-1)
			self._scores.pop(-1)
			self._datetimes.pop(-1)

		self._ranks.append(dyn_ranks)
		self._scores.append(dyn_scores)
		self._datetimes.append(time)
		self._submitor.append(id)
		self._agents.add(id)

	def filter(self, ids: list[int]) -> 'Leaderboard':
		filtered_board = Leaderboard()
		for time, submitor, ranks, scores in zip( \
					self._datetimes, self._submitor, self._ranks, self._scores):
			if submitor in ids:
				submitor_rank = ranks.index(submitor)
				submitor_score = scores[submitor_rank]
				filtered_board.refresh(time, submitor, submitor_score)
		return filtered_board

	def display(self, submit_idx: int, nows: int | None = None) -> tuple[datetime | None, pd.DataFrame]:
		"""
		"""
		if submit_idx < 0:  # transform index
			submit_idx += len(self._datetimes)
		if submit_idx >= len(self._datetimes) or submit_idx < 0:  # check
			the_time = None
			the_ranks = []
			the_scores = []
		else:
			if nows is None:
				nows = len(self._datetimes)
			the_time = self._datetimes[submit_idx]
			the_ranks = self._ranks[submit_idx][:nows]
			the_scores = self._scores[submit_idx][:nows]

		submitors_hist = self._submitor[:submit_idx + 1]
		submit_count = []
		participate_days = []
		last_submit_days_ago = []
		current_date = self._datetimes[submit_idx]

		for id in the_ranks:
			submit_from_id = [1 if id == e else 0 for e in submitors_hist]
			first_submit_idx = submitors_hist.index(id)
			first_submit_date = self._datetimes[first_submit_idx]
			last_submit_idx = len(submitors_hist) - 1 - submitors_hist[::-1].index(id)
			last_submit_date = self._datetimes[last_submit_idx]
			submit_count.append(sum(submit_from_id))
			participate_days.append((current_date - first_submit_date).days)
			last_submit_days_ago.append((current_date - last_submit_date).days)

		return the_time, pd.DataFrame({
			'rank': the_ranks,
			'score': the_scores,
			'submit_count': submit_count,
			'participate_days': participate_days,
			'last_submit_days_ago': last_submit_days_ago
		})

	def submission_records_of(self, id: int) -> pd.DataFrame:
		records = pd.DataFrame()
		for time, submitor, ranks, scores in zip(self._datetimes, self._submitor, self._ranks, self._scores):
			if id == submitor:
				id_index = ranks.index(id)
				new_row = pd.DataFrame([{
					'time': time,
					'rank': id_index,
					'score': scores[id_index]
				}])
				records = pd.concat([records, new_row])
		return records

	def real_time_gap_between(self, id_i: int, id_j: int, delta: timedelta) -> pd.DataFrame:
		"""
		"""
		records_i = self.submission_records_of(id_i)
		records_j = self.submission_records_of(id_j)
		start_time: datetime = max(records_i['time'].iloc[0], records_j['time'].iloc[0])
		end_time = self._datetimes[-1] + delta  # after the last submission
		current_time = start_time.replace(minute=0, second=0, microsecond=0) + delta
		data = {
			'time': [],
			'hat_x_i': [],
			'hat_x_j': [],
			'hat_y': [],
		}
		while current_time < end_time:
			current_x_i = records_i.loc[records_i['time'] < current_time, ['score']].iloc[-1].values[0]
			current_x_j = records_j.loc[records_j['time'] < current_time, ['score']].iloc[-1].values[0]
			data['time'].append(current_time)
			data['hat_x_i'].append(current_x_i)
			data['hat_x_j'].append(current_x_j)
			data['hat_y'].append(current_x_i - current_x_j)
			current_time += delta
		return pd.DataFrame(data)

	def get_final_rank(self) -> list[int]:
		return self._ranks[-1].copy()
