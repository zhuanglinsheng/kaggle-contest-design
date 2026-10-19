"""Confirm the predeclared count-weight sensitivity; not a prior-only fit."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pipeline.empirical import sigma_lambda_posterior as p
from pipeline.empirical import sigma_lambda_trials as s

p.METHODS=p.METHODS+('count_quarter_lambda_external',)


def chain(task):return p.chain(task)


if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=4);args=parser.parse_args()
 jobs=[('recovery',f'{cid}_2','count_quarter_lambda_external') for cid in s.IDS]
 for job in jobs:p.prepare(*job)
 with ProcessPoolExecutor(max_workers=args.workers) as pool:list(pool.map(chain,[(*job,j,30000) for job in jobs for j in range(1,5)]))
 for job in jobs:p.summarize(*job)
