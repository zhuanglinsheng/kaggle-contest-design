"""Resolve failed chain exploration with fixed pilot mixtures, same targets."""
import json
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from pipeline.empirical import nonlinear_reestimate as n
from pipeline.empirical import sigma_lambda_trials as s
from pipeline.empirical import sigma_lambda_posterior as p
from pipeline.empirical.sigma_lambda_refine import curvature


def prepare(cid,variant):
 p.configure('empirical',str(cid),variant);path=n.OUT/f'proposal_{cid}.json';old=json.loads(path.read_text())
 if old.get('failed_chain_repair'):return
 data,_,_,_=p.load(cid);target=p.Target(data)
 z=np.concatenate([np.load(n.OUT/f'contest_{cid}/chain_{j}.npz')['z'] for j in range(1,5)])
 threshold=np.median(z[:,2]);groups=[z[z[:,2]<=threshold],z[z[:,2]>threshold]]
 components=[dict(mean=g.mean(0).tolist(),covariance=(np.cov(g,rowvar=False)+np.eye(6)*1e-6).tolist()) for g in groups]
 maps=json.loads((s.OUT/'empirical'/str(cid)/f'map_{variant}.json').read_text())
 for mode in maps['runs']:
  point=np.array(mode['z'])
  if any(np.linalg.norm(point-np.array(c['mean']))<.5 for c in components):continue
  components.append(dict(mean=point.tolist(),covariance=curvature(target,point).tolist()))
 n.learn_proposal(cid)
 proposal=json.loads(path.read_text());proposal['mixture_components']=[dict(weight=1/len(components),**c) for c in components]
 proposal['failed_chain_repair']=dict(pilot_split='log_sigma median',threshold=float(threshold),components=len(components),
  note='Computational proposal only; estimator priors, likelihood and data unchanged.')
 n.save_json(path,proposal)


def chain(task):return p.chain(task)


if __name__=='__main__':
 jobs=[]
 for cid in s.IDS:
  for variant in p.METHODS[1:]:
   path=s.OUT/'posterior/empirical'/str(cid)/variant/'summary.json'
   if not json.loads(path.read_text())['passed']:jobs.append(('empirical',str(cid),variant))
 for _,case,variant in jobs:prepare(int(case),variant)
 with ProcessPoolExecutor(max_workers=4) as pool:list(pool.map(chain,[(*job,j,60000) for job in jobs for j in range(1,5)]))
 for job in jobs:p.summarize(*job)
