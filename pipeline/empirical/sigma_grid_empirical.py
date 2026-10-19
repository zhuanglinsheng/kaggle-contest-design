"""Apply the fixed simulated-data sigma-prior choice to the ten real inputs."""
import json
from concurrent.futures import ProcessPoolExecutor
from pipeline.empirical import sigma_prior_grid as g
from pipeline.empirical import sigma_lambda_posterior as p
from pipeline.empirical import sigma_lambda_trials as s


def chain(task):return p.chain(task)


if __name__=='__main__':
 selected=json.loads((s.OUT/'sigma_grid_selection.json').read_text())['selected']['variant']
 jobs=[('empirical',str(cid),selected) for cid in s.IDS]
 for job in jobs:p.prepare(*job)
 with ProcessPoolExecutor(max_workers=4) as pool:list(pool.map(chain,[(*job,j,60000) for job in jobs for j in range(1,5)]))
 for job in jobs:p.summarize(*job)
