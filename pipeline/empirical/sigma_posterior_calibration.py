"""Check whether posterior-based sigma calibration improves on MAP screening.

The same grid was fixed before replicate3. Select using replicate2 only and
then evaluate replicate3; this is an additional, retrospective split check,
not a fresh blinded validation. Reweight exact leading posteriors only where
all reported means have sufficient effective sample size.
"""
import json
import math
import numpy as np
from scipy.special import softmax
from pipeline.empirical import nonlinear_reestimate as n
from pipeline.empirical import sigma_lambda_trials as s
from pipeline.empirical import sigma_prior_grid as g
from pipeline.empirical import paired_prior_report as report


def evaluate(rep,cid,variant):
 root=s.OUT/'posterior/recovery'/f'{cid}_{rep}'/'lambda_external'
 saved=json.loads((root/'summary.json').read_text())
 if not saved['passed']:raise RuntimeError('Source posterior is not validated')
 paths=[root/f'contest_{cid}/chain_{j}.npz' for j in range(1,5)]
 if saved['chain_sha256']!=[n.digest(path) for path in paths]:raise RuntimeError('Source chains changed')
 full=[np.load(path)['z'] for path in paths]
 indices=np.linspace(0,len(full[0])-1,3000,dtype=int)
 z=np.concatenate([chain[indices] for chain in full]);cfg=g.GRID[variant]
 logratio=-.5*((z[:,2]-math.log(cfg['sigma_mean']))/cfg['sigma_sd'])**2+.5*((z[:,2]-math.log(2))/.5)**2
 weights=softmax(logratio);data=json.loads((s.OUT/'recovery'/f'{cid}_{rep}/data.json').read_text())
 values=report.quantities(z,data);means,sds,q=n.weighted_summary(values,weights)
 effective=[]
 for k in range(values.shape[1]):
  influence=(len(z)*weights*(values[:,k]-means[k])).reshape(4,-1)
  effective.append(float(min(len(z),n.effective_sample_size(influence)*sds[k]**2/max(influence.var(ddof=1),1e-30))))
 diag=dict(weight_ess=float(1/(weights@weights)),minimum_mean_ess=min(effective),maximum_weight=float(weights.max()))
 passed=diag['weight_ess']>=400 and diag['minimum_mean_ess']>=400 and diag['maximum_weight']<.02
 ans=dict(replicate=rep,contest_id=cid,variant=variant,passed=passed,diagnostics=diag,
  source_summary_sha256=n.digest(root/'summary.json'),source_chain_sha256=saved['chain_sha256'])
 if passed:
  summaries={name:dict(mean=float(means[k]),sd=float(sds[k]),q025=float(q[k,0]),median=float(q[k,1]),q975=float(q[k,2])) for k,name in enumerate(report.NAMES)}
  truth=saved['truth'];ans.update(summaries=summaries,truth=truth,
   absolute_log_error={key:abs(math.log(summaries[key]['median']/truth[key])) for key in ('sigma','lambda')},
   relative_error={key:abs(summaries[key]['median']/truth[key]-1) for key in ('sigma','lambda')})
 return ans


def run():
 dest=s.OUT/'posterior_sigma_calibration';dest.mkdir(parents=True,exist_ok=True)
 design=dict(grid=g.GRID,training_replicate=2,validation_replicate=3,
  criterion='Minimize mean absolute log sigma error, requiring all10 ESS gates and mean absolute log lambda error<=.35.',
  status='Retrospective supplementary split check, not a new untouched confirmation; grid itself predates replicate3.')
 n.save_json(dest/'design.json',design);training=[];ranking=[]
 for variant in g.GRID:
  fits=[evaluate(2,cid,variant) for cid in s.IDS];training+=fits
  valid=all(f['passed'] for f in fits)
  item=dict(variant=variant,all_valid=valid)
  if valid:item.update(sigma_loss=float(np.mean([f['absolute_log_error']['sigma'] for f in fits])),lambda_loss=float(np.mean([f['absolute_log_error']['lambda'] for f in fits])))
  ranking.append(item)
  print('POSTERIOR CALIBRATION',json.dumps(item),flush=True)
 eligible=[v for v in ranking if v['all_valid'] and v['lambda_loss']<=.35]
 best=min(eligible,key=lambda v:v['sigma_loss'])
 n.save_json(dest/'selection.json',dict(selected=best,ranking=ranking))
 validation=[evaluate(3,cid,best['variant']) for cid in s.IDS]
 n.save_json(dest/'training.json',training);n.save_json(dest/'validation.json',validation)
 metrics={}
 for rep,fits in [(2,[f for f in training if f['variant']==best['variant']]),(3,validation)]:
  if all(f['passed'] for f in fits):
   metrics[str(rep)]={key:dict(mean_absolute_log_error=float(np.mean([f['absolute_log_error'][key] for f in fits])),mean_absolute_relative_error=float(np.mean([f['relative_error'][key] for f in fits]))) for key in ('sigma','lambda')}
 n.save_json(dest/'metrics.json',metrics);print('CALIBRATED',best,metrics,flush=True)


if __name__=='__main__':run()
