"""Apply the already-defined simulation interval calibration to lambda.

Uses only realizations0/1 to freeze the coefficient. Realization5 is checked
after freezing; no posterior or point estimate is modified.
"""
import argparse
import json
import math
import numpy as np
from pipeline.empirical import nonlinear_reestimate as n
from pipeline.empirical import sigma_lambda_trials as s
from pipeline.empirical.sigma_interval_calibration import normalized_error,calibrated_interval

OUT=s.OUT/'lambda_refinement'


def read(rep,cid):
 path=s.OUT/'posterior/recovery'/f'{cid}_{rep}'/'lambda_external/summary.json'
 value=json.loads(path.read_text())
 if not value['passed']:raise RuntimeError('Invalid interval source posterior')
 return value,path


def freeze():
 records=[]
 for rep in (0,1):
  for cid in s.IDS:
   fit,path=read(rep,cid);v=fit['summaries']['lambda']
   records.append(dict(replicate=rep,contest_id=cid,score=normalized_error(fit['truth']['lambda'],v['median'],v['q025'],v['q975']),summary_sha256=n.digest(path)))
 rank=min(len(records),math.ceil((len(records)+1)*.95));factor=max(1.,sorted(v['score'] for v in records)[rank-1])
 value=dict(multiplier=factor,training_replicates=[0,1],training_records=records,order_statistic_rank=rank,
  validation_replicate=5,previous_replicates=[2,3,4],rule='Same single log-halfwidth multiplier rule as sigma; all configurations share one coefficient.',
  interpretation='Simulation-calibrated interval, distinct from original95% Bayesian credible interval. Point estimates and posterior unchanged.',script_sha256=n.digest(__file__))
 path=OUT/'lambda_interval_frozen.json'
 if path.exists() and json.loads(path.read_text())!=value:raise RuntimeError('Frozen lambda interval rule changed')
 n.save_json(path,value);print('FROZEN LAMBDA INTERVAL MULTIPLIER',factor,flush=True)


def check():
 value=json.loads((OUT/'lambda_interval_frozen.json').read_text());factor=value['multiplier'];rows=[];metrics={}
 for rep in (0,1,2,3,4,5):
  group=[]
  for cid in s.IDS:
   fit,path=read(rep,cid);v=fit['summaries']['lambda'];truth=fit['truth']['lambda']
   lower,upper=calibrated_interval(v['median'],v['q025'],v['q975'],factor)
   group.append(dict(replicate=rep,contest_id=cid,truth=truth,median=v['median'],mean=v['mean'],original_lower=v['q025'],original_upper=v['q975'],calibrated_lower=lower,calibrated_upper=upper,
    original_covers=v['q025']<=truth<=v['q975'],calibrated_covers=lower<=truth<=upper,original_relative_width=(v['q975']-v['q025'])/truth,calibrated_relative_width=(upper-lower)/truth,summary_sha256=n.digest(path)))
  metrics[str(rep)]=dict(original_covered=sum(x['original_covers'] for x in group),calibrated_covered=sum(x['calibrated_covers'] for x in group),
   original_median_relative_width=float(np.median([x['original_relative_width'] for x in group])),calibrated_median_relative_width=float(np.median([x['calibrated_relative_width'] for x in group])))
  rows+=group
 n.write_csv(OUT/'lambda_interval_comparison.csv',rows);n.save_json(OUT/'lambda_interval_metrics.json',metrics)
 print(json.dumps(metrics,indent=2),flush=True)


if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=('freeze','check'),default='freeze');args=parser.parse_args()
 if args.stage=='freeze':freeze()
 else:check()
