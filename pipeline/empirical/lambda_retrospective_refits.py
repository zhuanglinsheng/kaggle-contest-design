"""Fresh fits replace three failed importance updates, retaining same targets."""
import json
from concurrent.futures import ProcessPoolExecutor
from pipeline.empirical import lambda_prior_refinement as r
from pipeline.empirical import sigma_lambda_posterior as p


def chain(task):return p.chain(task)


if __name__=='__main__':
 chosen=json.loads((r.OUT/'selection.json').read_text())['selected']['variant']
 failed=[x for x in json.loads((r.OUT/'retrospective.json').read_text()) if not x['passed']]
 jobs=[('recovery',f"{x['contest_id']}_{x['replicate']}",chosen) for x in failed]
 for job in jobs:p.prepare(*job)
 with ProcessPoolExecutor(max_workers=4) as pool:list(pool.map(chain,[(*job,j,60000) for job in jobs for j in range(1,5)]))
 for job in jobs:p.summarize(*job)
