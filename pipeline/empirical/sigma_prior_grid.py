"""Small prespecified sigma-prior grid, selected on exploratory replicates only."""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from pipeline.empirical import sigma_lambda_trials as s
from pipeline.empirical import sigma_lambda_posterior as p

GRID={f'sigma_m{mean:g}_sd{sd:g}':dict(lambda_mean=100.,lambda_sd=2.25,sigma_mean=mean,sigma_sd=sd)
 for mean in (.5,1.,2.) for sd in (.3,.4,.5,.75)}
s.VARIANTS.update(GRID)
p.METHODS=p.METHODS+tuple(GRID)


def job(case):
 for variant in GRID:s.optimize('recovery',case,variant)
 print('SIGMA GRID',case,flush=True)


def select():
 rows=[]
 for variant in GRID:
  values=[]
  for cid in s.IDS:
   for rep in (0,1):
    x=json.loads((s.OUT/'recovery'/f'{cid}_{rep}'/f'map_{variant}.json').read_text())
    values.append([x['absolute_log_error'][k] for k in ('sigma','lambda')])
  values=np.array(values);rows.append(dict(variant=variant,sigma_loss=float(values[:,0].mean()),lambda_loss=float(values[:,1].mean()),maximum_sigma_loss=float(values[:,0].max())))
 # Constraint set before scanning: preserve the lambda improvement, guard tails.
 eligible=[x for x in rows if x['lambda_loss']<=.35 and x['maximum_sigma_loss']<=.75]
 if not eligible:raise RuntimeError('No sigma grid candidate satisfies the fixed guards')
 best=min(eligible,key=lambda x:x['sigma_loss'])
 s.n.save_json(s.OUT/'sigma_grid_selection.json',dict(selected=best,all_candidates=rows,selection_cases=20,
  rule='Minimize sigma mean absolute log error on replicates0/1, constrained lambda loss<=.35 and maximum sigma log error<=.75.',
  validation='New replicate3, not used in this selection. Empirical nonlinear estimates never used in selection.'))
 print('SELECTED',json.dumps(best),flush=True)


def chain(task):return p.chain(task)


if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=('train','posterior'),default='train');args=parser.parse_args()
 if args.stage=='train':
  s.n.save_json(s.OUT/'sigma_grid_design.json',dict(grid=GRID,selection_replicates=[0,1],validation_replicate=3,
   constraints=dict(lambda_mean_absolute_log_error=.35,maximum_sigma_absolute_log_error=.75),source_sha256=s.n.digest(__file__)))
  with ProcessPoolExecutor(max_workers=4) as pool:list(pool.map(job,[f'{cid}_{rep}' for cid in s.IDS for rep in (0,1)]))
  select()
 else:
  selected=json.loads((s.OUT/'sigma_grid_selection.json').read_text())['selected']['variant']
  jobs=[('recovery',f'{cid}_3',variant) for cid in s.IDS for variant in ('baseline','lambda_external',selected)]
  for item in jobs:p.prepare(*item)
  with ProcessPoolExecutor(max_workers=8) as pool:list(pool.map(chain,[(*item,j,30000) for item in jobs for j in range(1,5)]))
  for item in jobs:p.summarize(*item)
