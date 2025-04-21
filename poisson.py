"""
"""
import pandas as pd

from matplotlib import pyplot as plt
from matplotlib.axis import Axis
from matplotlib.figure import Figure

from typing import Iterable
from datetime import datetime



def plot_contest_submissions(submissions: pd.DataFrame, max_nteams: int = 5):
	"""
	"""
	submissions.drop_duplicates(subset='SubmissionDate', keep='last', inplace=True)
	contest_time_start = submissions['EnabledDate'].iloc[0]
	contest_time_end = submissions['DeadlineDate'].iloc[0]
	contest_teams_n_submits = submissions.groupby('TeamId').size().sort_values(ascending=False)

	fig = plt.figure(figsize=(10, 5))
	ax = plt.gca()

	idx_team = 0
	y_ticks = ([], [])

	for team in contest_teams_n_submits.index:
		team_submissiondates = submissions.loc[submissions['TeamId'] == team, ['SubmissionDate']]
		if idx_team < max_nteams:
			idx_team += 1
			team_loc = idx_team
			y_ticks[0].append(team_loc)
			y_ticks[1].append(team)
			ax.hlines(y=team_loc, xmax=contest_time_start, xmin=contest_time_end)
			ax.vlines(x=team_submissiondates, ymax=idx_team + 0.1, ymin=idx_team - 0.1)
	ax.set_xlim(contest_time_start, contest_time_end)
	ax.set_yticks(*y_ticks)
	ax.set_ylabel('Team', rotation=0)
	#ax.xaxis.set_label_coords(1.0, -0.05)
	ax.yaxis.set_label_coords(-0.05, 1.0)
	#plt.tight_layout()
	fig.show()
	return ax, fig
