import os
import json
import statistics
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

from leaderboard import Leaderboard


def contest_basic_setting(tbl_contests: pd.DataFrame, contest_id: int):
	"""
	"""
	tbl_contest_info = tbl_contests.loc[tbl_contests['Id'] == contest_id]
	# prize
	prize = tbl_contest_info['RewardQuantity'].iloc[0]
	# deadline
	deadline = tbl_contest_info['DeadlineDate'].iloc[0].to_pydatetime()
	deadline = deadline.replace(hour=0, minute=0, second=0, microsecond=0)
	deadline += timedelta(days=1)
	# others
	max_daily_submit = tbl_contest_info['MaxDailySubmissions'].iloc[0]
	percentage = tbl_contest_info['LeaderboardPercentage'].iloc[0]
	return deadline, prize, max_daily_submit, percentage


def contest_basic_submission_info(
		tbl_submissions: pd.DataFrame, contest_id: int
) -> pd.DataFrame:
	"""
	"""
	tbl = tbl_submissions.loc[tbl_submissions['CompetitionId']==contest_id]
	print('>>> How many teams are there in this contest?')
	print(tbl['TeamId'].unique().size)
	print('>>> How many submissions are there in total?')
	print(len(tbl))
	print('>>> List the number of submissions for the most active 5 teams:')
	print(tbl.groupby('TeamId').size().sort_values(ascending=False).head(5))
	return tbl


def randomize_within_day(group: pd.Series, seed: int = 1234):
	rgn = np.random.default_rng(seed)
	seconds = rgn.choice(range(86400), size=len(group), replace=False)
	seconds.sort()
	randomized_times = group + pd.to_timedelta(seconds, unit='s')
	return randomized_times


def leaderboard_fulfill(tbl_contest_submissions, deadline: datetime):
	leaderboard_public = Leaderboard()
	leaderboard_private = Leaderboard()
	tbl_contest_submissions_sorted = tbl_contest_submissions.sort_values(by='SubmissionDate')
	submission_dates = tbl_contest_submissions_sorted['SubmissionDate']
	submission_datetimes = submission_dates.groupby(submission_dates).apply(randomize_within_day).reset_index(level=0, drop=True)
	for idx_row, row in tbl_contest_submissions_sorted.iterrows():
		time = submission_datetimes[idx_row].to_pydatetime()
		if time > deadline:
			break
		team_id = row['TeamId']
		score_pub = row['PublicScore']
		score_pri = row['PrivateScore']
		leaderboard_public.refresh(time, team_id, score_pub)
		leaderboard_private.refresh(time, team_id, score_pri)
	return leaderboard_public, leaderboard_private


def save_contest_data(
		tbl_submissions: pd.DataFrame,
		contest_id: int,
		team_i_id: int, team_j_id: int,
		deadline: datetime, prize: float, max_daily_submit: int, percentage: float
) -> None:
	# json file & utils
	wd = os.getcwd()
	wd_synthetic_data = os.path.join(wd, f'__jsondata__/contest_{contest_id}.json')
	roundint = lambda x: int(round(x))

	# contest setting
	#deadline, prize, max_daily_submit, percentage = contest_basic_setting(contest_id)
	param_theta = prize / 1000
	param_r = 15
	time_unit_2f = 1 / 24
	delta = timedelta(hours=1)

	# observations
	contest_submissions = tbl_submissions.loc[tbl_submissions['CompetitionId']==contest_id]
	leaderboard_pub, leaderboard_pri = leaderboard_fulfill(contest_submissions, deadline)

	# \hat{y}_t
	tbl_hat_y = leaderboard_pub.real_time_gap_between(team_i_id, team_j_id, delta=timedelta(hours=1))
	end_time: datetime = tbl_hat_y['time'].iloc[-1].to_pydatetime() + timedelta(hours=1)
	start_time: datetime = tbl_hat_y['time'].iloc[0].to_pydatetime()
	start_time_commit = start_time - delta
	start_time_contest = start_time - 24 * delta
	observed_gap_dynamic = tbl_hat_y['y'].tolist()
	# normalze \hat{y}_t
	observed_gap_dynamic_std = statistics.stdev(observed_gap_dynamic)
	observed_gap_dynamic = [20 * e / observed_gap_dynamic_std for e in observed_gap_dynamic]
	observed_gap_dynamic_init = [observed_gap_dynamic[0]] * 24
	observed_gap_dynamic = observed_gap_dynamic_init + observed_gap_dynamic

	# y_t
	tbl_y = leaderboard_pri.real_time_gap_between(team_i_id, team_j_id, delta=timedelta(hours=1))
	real_gap_dynamic = tbl_y['y'].tolist()
	# normalze \hat{y}_t
	real_gap_dynamic = [20 * e / observed_gap_dynamic_std for e in real_gap_dynamic]
	real_gap_dynamic_init = [real_gap_dynamic[0]] * 24
	real_gap_dynamic = real_gap_dynamic_init + real_gap_dynamic

	# \hat{t}_{i(j)}
	observed_i_commits = leaderboard_pub.submission_records_of(team_i_id)['time']
	observed_i_commits = observed_i_commits.loc[observed_i_commits > start_time_commit].tolist()
	observed_j_commits = leaderboard_pub.submission_records_of(team_j_id)['time']
	observed_j_commits = observed_j_commits.loc[observed_j_commits > start_time_commit].tolist()

	with open(wd_synthetic_data, 'w') as f:
		json.dump({
			'theta': param_theta,
			'percentage': float(percentage),
			'max_daily_submit': int(max_daily_submit),
			'ratio': param_r,
			'N_Delta': roundint((end_time - start_time_contest).total_seconds() / 3600),
			'Delta2f': time_unit_2f,
			'Ni': len(observed_i_commits),
			'Nj': len(observed_j_commits),
			'hat_t_i': [(dt - start_time_contest).total_seconds() / 3600 for dt in observed_i_commits],
			'hat_t_j': [(dt - start_time_contest).total_seconds() / 3600 for dt in observed_j_commits],
			'hat_y': observed_gap_dynamic,
			'real_y': real_gap_dynamic,
		}, f, indent=4)


min_overlapping_days = 10
first_n_candidates = 20


def player_basic_criterion(player: pd.DataFrame) -> bool:
	# active days >= min
	if player['participate_days'].iloc[0] - player['last_submit_days_ago'].iloc[0] < min_overlapping_days:
		return False
	# total submissions >= min
	return True


def select_2_strongest(
		tbl_submissions: pd.DataFrame,
		contest_id: int, deadline: datetime
) -> tuple[int, int] | None:
	"""
	"""
	tbl = tbl_submissions.loc[tbl_submissions['CompetitionId']==contest_id]
	_, leaderboard_pri = leaderboard_fulfill(tbl, deadline)
	_, pri_final_rank = leaderboard_pri.display(-1, first_n_candidates)
	# determine player i
	player_i_idx = 0
	player_i = pri_final_rank.iloc[[player_i_idx]]
	while not player_basic_criterion(player_i):
		player_i_idx += 1
		if player_i_idx >= len(pri_final_rank):
			return None
		player_i = pri_final_rank.iloc[[player_i_idx]]
	# determine player j
	player_j_idx = 0
	player_j = pri_final_rank.iloc[[player_j_idx]]
	while True:
		if player_j_idx == player_i_idx or not player_basic_criterion(player_j):
			player_j_idx += 1
			if player_j_idx >= len(pri_final_rank):
				return None
			player_j = pri_final_rank.iloc[[player_j_idx]]
			continue
		# overlapp
		start = min(player_i['participate_days'].values[0], player_j['participate_days'].values[0])
		end = max(player_i['last_submit_days_ago'].values[0], player_j['last_submit_days_ago'].values[0])
		if start - end > min_overlapping_days:
			break
		else:
			player_j_idx += 1
			if player_j_idx >= len(pri_final_rank):
				return None
			player_j = pri_final_rank.iloc[[player_j_idx]]
	return player_i['rank'].iloc[0], player_j['rank'].iloc[0]
