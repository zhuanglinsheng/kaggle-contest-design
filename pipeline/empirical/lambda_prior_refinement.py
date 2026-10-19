"""Refine lambda priors on existing training simulations; freeze before new test.

All methods retain the leading likelihood and original sigma prior. Screening
uses validated posterior draws, including autocorrelation-adjusted importance
checks. Older validation sets are retrospective; realization5 is prospective.
"""
import argparse
import json
import math
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from scipy.special import softmax
from pipeline.empirical import nonlinear_reestimate as n
from pipeline.empirical import sigma_lambda_trials as s
from pipeline.empirical import sigma_lambda_posterior as p
from pipeline.empirical import paired_prior_report as report

OUT=s.OUT/'lambda_refinement'
GRID={f'lambda_m{mean:g}_sd{sd:g}':dict(lambda_mean=mean,lambda_sd=sd)
 for mean in (30.,100.,300.,1000.) for sd in (1.5,2.25,3.)}
s.VARIANTS.update(GRID);p.METHODS=p.METHODS+tuple(GRID)


def screen_one(rep,cid,variant):
 root=s.OUT/'posterior/recovery'/f'{cid}_{rep}'/'lambda_external'
 saved=json.loads((root/'summary.json').read_text())
 if not saved['passed']:raise RuntimeError('Unvalidated source fit')
 paths=[root/f'contest_{cid}/chain_{j}.npz' for j in range(1,5)]
 if saved['chain_sha256']!=[n.digest(path) for path in paths]:raise RuntimeError('Source posterior changed')
 full=[np.load(path)['z'] for path in paths];indices=np.linspace(0,len(full[0])-1,3000,dtype=int)
 z=np.concatenate([x[indices] for x in full]);cfg=GRID[variant]
 ratio=-.5*((z[:,3]-math.log(cfg['lambda_mean']))/cfg['lambda_sd'])**2+.5*((z[:,3]-math.log(100.))/2.25)**2
 weights=softmax(ratio);data=json.loads((s.OUT/'recovery'/f'{cid}_{rep}/data.json').read_text())
 values=report.quantities(z,data);means,sds,q=n.weighted_summary(values,weights);effective=[]
 for k in range(values.shape[1]):
  influence=(len(z)*weights*(values[:,k]-means[k])).reshape(4,-1)
  effective.append(float(min(len(z),n.effective_sample_size(influence)*sds[k]**2/max(influence.var(ddof=1),1e-30))))
 diag=dict(weight_ess=float(1/(weights@weights)),minimum_mean_ess=min(effective),maximum_weight=float(weights.max()))
 passed=diag['weight_ess']>=400 and diag['minimum_mean_ess']>=400 and diag['maximum_weight']<.02
 ans=dict(replicate=rep,contest_id=cid,variant=variant,passed=passed,diagnostics=diag,source_summary_sha256=n.digest(root/'summary.json'))
 if passed:
  summaries={name:dict(mean=float(means[k]),sd=float(sds[k]),q025=float(q[k,0]),median=float(q[k,1]),q975=float(q[k,2])) for k,name in enumerate(report.NAMES)}
  truth=saved['truth'];ans.update(summaries=summaries,truth=truth,
   relative_error={key:abs(summaries[key]['median']/truth[key]-1) for key in ('sigma','lambda')},
   covered={key:summaries[key]['q025']<=truth[key]<=summaries[key]['q975'] for key in ('sigma','lambda')})
 return ans


def design():
 OUT.mkdir(parents=True,exist_ok=True)
 value=dict(grid=GRID,training_replicates=[0,1],retrospective_replicates=[2,3,4],prospective_replicate=5,
  all_other_priors='Original sigma,cost,mu0,r priors retained.',
  choice='Require all20 ESS gates; sigma mean relative error <=1.1 times source; prefer lambda coverage>=19/20 then minimum lambda mean relative error. If none reaches19/20, screen best accuracy with coverage>=18/20 but explicitly mark coverage target unmet.',
  source_sha256=n.digest(__file__))
 path=OUT/'design.json'
 if path.exists() and json.loads(path.read_text())!=value:raise RuntimeError('Frozen lambda refinement design differs')
 n.save_json(path,value)


def select():
 design();baseline=[]
 for rep in (0,1):
  for cid in s.IDS:baseline.append(json.loads((s.OUT/'posterior/recovery'/f'{cid}_{rep}/lambda_external/summary.json').read_text()))
 sigma_guard=1.1*np.mean([f['relative_error']['sigma'] for f in baseline]);fits=[];ranking=[]
 for variant in GRID:
  current=[screen_one(rep,cid,variant) for rep in (0,1) for cid in s.IDS];fits+=current
  item=dict(variant=variant,all_valid=all(f['passed'] for f in current))
  if item['all_valid']:
   item.update(lambda_error=float(np.mean([f['relative_error']['lambda'] for f in current])),
    sigma_error=float(np.mean([f['relative_error']['sigma'] for f in current])),
    maximum_lambda_error=float(max(f['relative_error']['lambda'] for f in current)),covered=sum(f['covered']['lambda'] for f in current))
  ranking.append(item);print('LAMBDA SCREEN',json.dumps(item),flush=True)
 feasible=[r for r in ranking if r['all_valid'] and r['sigma_error']<=sigma_guard and r['covered']>=18]
 preferred=[r for r in feasible if r['covered']>=19]
 if not feasible:raise RuntimeError('No lambda candidate satisfies screening guards')
 best=min(preferred or feasible,key=lambda r:r['lambda_error'])
 selection=dict(selected=best,ranking=ranking,sigma_guard=float(sigma_guard),training_coverage_target_met=bool(preferred),training_cases=20,
  prospective_test='Freeze before generating realization5. No alternative seed is substituted if validation fails.')
 n.save_json(OUT/'training.json',fits);n.save_json(OUT/'selection.json',selection)
 print('FROZEN LAMBDA CHOICE',json.dumps(best),flush=True)


def retrospective():
 chosen=json.loads((OUT/'selection.json').read_text())['selected']['variant']
 fits=[screen_one(rep,cid,chosen) for rep in (2,3,4) for cid in s.IDS]
 n.save_json(OUT/'retrospective.json',fits)
 for rep in (2,3,4):
  current=[f for f in fits if f['replicate']==rep]
  print('RETROSPECTIVE',rep,all(f['passed'] for f in current),
   {k:float(np.mean([f['relative_error'][k] for f in current])) for k in ('sigma','lambda')} if all(f['passed'] for f in current) else '',flush=True)


def chain(task):return p.chain(task)


def fit():
 chosen=json.loads((OUT/'selection.json').read_text())['selected']['variant']
 jobs=[('recovery',f'{cid}_5',variant) for cid in s.IDS for variant in ('lambda_external',chosen)]
 for job in jobs:p.prepare(*job)
 with ProcessPoolExecutor(max_workers=8) as pool:list(pool.map(chain,[(*job,j,30000) for job in jobs for j in range(1,5)]))
 for job in jobs:p.summarize(*job)


if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=('select','retrospective','fit'),default='select');args=parser.parse_args()
 if args.stage=='select':select()
 elif args.stage=='retrospective':retrospective()
 else:fit()
