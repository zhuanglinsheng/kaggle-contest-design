"""Audit secondary empirical modes using unchanged estimator targets."""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from pipeline.empirical import nonlinear_reestimate as n
from pipeline.empirical import sigma_lambda_posterior as p
from pipeline.empirical import sigma_lambda_trials as s


def curvature(target,z):
 step=np.array([.005,.005,.005,.005,.001,.005]);H=np.zeros((6,6));f=-target.posterior(z)
 for j in range(6):
  e=np.eye(6)[j]*step[j];H[j,j]=(-target.posterior(z+e)-2*f-target.posterior(z-e))/step[j]**2
  for k in range(j):
   v=np.eye(6)[k]*step[k]
   H[j,k]=H[k,j]=(-target.posterior(z+e+v)+target.posterior(z+e-v)+target.posterior(z-e+v)-target.posterior(z-e-v))/(4*step[j]*step[k])
 val,vec=np.linalg.eigh(H);return (vec*(1/np.maximum(val,.05)))@vec.T


def prepare(cid):
 variant='activity_lambda_external';p.configure('empirical',str(cid),variant)
 path=n.OUT/f'proposal_{cid}.json';old=json.loads(path.read_text())
 if old.get('secondary_mode_audit'):return
 data,_,_,_=p.load(cid);target=p.Target(data)
 z=np.concatenate([np.load(n.OUT/f'contest_{cid}/chain_{j}.npz')['z'] for j in range(1,5)])
 components=[dict(mean=z.mean(0).tolist(),covariance=(np.cov(z,rowvar=False)+np.eye(6)*1e-6).tolist())]
 maps=json.loads((s.OUT/'empirical'/str(cid)/f'map_{variant}.json').read_text())
 for mode in maps['runs']:
  point=np.array(mode['z'])
  if any(np.linalg.norm(point-np.array(c['mean']))<.5 for c in components):continue
  components.append(dict(mean=point.tolist(),covariance=curvature(target,point).tolist()))
 n.learn_proposal(cid)
 proposal=json.loads(path.read_text());proposal['mixture_components']=[dict(weight=1/len(components),**c) for c in components]
 proposal['secondary_mode_audit']=dict(mode_components=len(components),purpose='Explore every distinct optimized mode including lower posterior peaks. Exact target unchanged.')
 n.save_json(path,proposal)


def chain(task):return p.chain(task)


if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--contests',type=int,nargs='+',default=[8076,5390,3929]);args=parser.parse_args()
 for cid in args.contests:prepare(cid)
 jobs=[('empirical',str(cid),'activity_lambda_external') for cid in args.contests]
 with ProcessPoolExecutor(max_workers=4) as pool:list(pool.map(chain,[(*job,j,60000) for job in jobs for j in range(1,5)]))
 for job in jobs:p.summarize(*job)
