"""Posterior confirmation of prespecified simple sigma/lambda improvements."""
import argparse
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from pipeline.empirical import sigma_lambda_trials as s
from pipeline.empirical import nonlinear_reestimate as n
from pipeline.empirical import paired_prior_report as rep

BASE_LOAD=n.load_baseline
METHODS=('baseline','lambda_external','activity_lambda_external')


class Target(s.Target):
 def __init__(self,data,*args,**kwargs):super().__init__(data,os.environ['SIGMA_LAMBDA_VARIANT'])


def load(cid):
 dataset=os.environ['SIGMA_LAMBDA_DATASET'];case=os.environ['SIGMA_LAMBDA_CASE']
 path=s.OUT/dataset/case/'data.json';data=json.loads(path.read_text())
 _,chains,z,provenance=BASE_LOAD(cid)
 return data,chains,z,dict(**provenance,trial_dataset=dataset,trial_case=case,trial_data_sha256=n.digest(path),
  variant=os.environ['SIGMA_LAMBDA_VARIANT'],trial_script_sha256=n.digest(s.__file__),posterior_script_sha256=n.digest(__file__))


def configure(dataset,case,variant):
 os.environ['SIGMA_LAMBDA_DATASET']=dataset;os.environ['SIGMA_LAMBDA_CASE']=str(case);os.environ['SIGMA_LAMBDA_VARIANT']=variant
 s.kernel();n.SOURCE=s.OUT/'weighted_leading_kernel.cpp';n.Likelihood=Target;n.kernel=s.kernel;n.load_baseline=load
 n.OUT=s.OUT/'posterior'/dataset/str(case)/variant;n.OUT.mkdir(parents=True,exist_ok=True)


def prepare(dataset,case,variant):
 cid=int(str(case).split('_')[0]);configure(dataset,case,variant)
 path=n.OUT/f'proposal_{cid}.json'
 if path.exists() and (n.OUT/'target.json').exists():return
 ans=s.optimize(dataset,case,variant);z=np.array(ans['best']['z']);data,_,_,provenance=load(cid)
 target=Target(data);step=np.array([.005,.005,.005,.005,.001,.005]);H=np.zeros((6,6));f=-target.posterior(z)
 for j in range(6):
  e=np.eye(6)[j]*step[j]
  H[j,j]=(-target.posterior(z+e)-2*f-target.posterior(z-e))/step[j]**2
  for k in range(j):
   v=np.eye(6)[k]*step[k]
   H[j,k]=H[k,j]=(-target.posterior(z+e+v)+target.posterior(z+e-v)+target.posterior(z-e+v)-target.posterior(z-e-v))/(4*step[j]*step[k])
 eigen,vec=np.linalg.eigh(H);cov=(vec*(1/np.maximum(eigen,.05)))@vec.T
 # A broad second component has full support; neither changes target density.
 components=[dict(weight=.7,mean=z.tolist(),covariance=cov.tolist()),dict(weight=.3,mean=z.tolist(),covariance=(cov*3).tolist())]
 n.save_json(path,dict(contest_id=cid,kernel_sha256=n.digest(n.SOURCE),mode_z=z.tolist(),covariance=cov.tolist(),mixture_components=components,
  raw_hessian_eigenvalues=eigen.tolist(),configuration=provenance))
 n.save_json(n.OUT/'target.json',dict(prior_mean=target.mean.tolist(),prior_sd=target.sd.tolist(),
  count_weight=s.VARIANTS[variant].get('count_weight',1.),**provenance))


def chain(task):
 dataset,case,variant,j,samples=task;configure(dataset,case,variant)
 cid=int(str(case).split('_')[0]);replicate=int(str(case).split('_')[1]) if '_' in str(case) else 0
 n.sample_mixture_gaussian(cid,j,1500,samples,.025,64,281000+cid+replicate*10000+METHODS.index(variant)*100000,pilot=True)


def summarize(dataset,case,variant):
 configure(dataset,case,variant);cid=int(str(case).split('_')[0]);data,_,_,provenance=load(cid)
 paths=[n.OUT/f'contest_{cid}/chain_{j}.npz' for j in range(1,5)]
 z=np.array([np.load(path)['z'] for path in paths]);flat=z.reshape(-1,6)
 physical=rep.quantities(flat,data).reshape(4,z.shape[1],-1)
 # Decisions here concern sigma/lambda; retain diagnostics for every quantity.
 diagnostics={name:dict(rank_rhat=n.rank_rhat(physical[:,:,k]),bulk_ess=n.rank_ess(physical[:,:,k])[0],tail_ess=n.rank_ess(physical[:,:,k])[1]) for k,name in enumerate(rep.NAMES)}
 passed=all(v['rank_rhat']<=1.01 and min(v['bulk_ess'],v['tail_ess'])>=400 for v in diagnostics.values())
 means,sds,q=n.weighted_summary(physical.reshape(-1,physical.shape[2]),np.ones(len(flat))/len(flat))
 summaries={name:dict(mean=float(means[k]),sd=float(sds[k]),q025=float(q[k,0]),median=float(q[k,1]),q975=float(q[k,2])) for k,name in enumerate(rep.NAMES)}
 result=dict(passed=passed,diagnostics=diagnostics,configuration=provenance,chain_sha256=[n.digest(path) for path in paths])
 if passed:result['summaries']=summaries
 truthpath=s.OUT/dataset/str(case)/'truth.json'
 if passed and truthpath.exists():
  truth=json.loads(truthpath.read_text())['truth'];result['truth']=truth
  result['absolute_log_error']={key:abs(math.log(summaries[key]['median']/truth[key])) for key in ('sigma','lambda')}
  result['relative_error']={key:abs(summaries[key]['median']/truth[key]-1) for key in ('sigma','lambda')}
  result['truth_in_95_interval']={key:summaries[key]['q025']<=truth[key]<=summaries[key]['q975'] for key in ('sigma','lambda')}
 n.save_json(n.OUT/'summary.json',result)
 print('POSTERIOR',dataset,case,variant,passed,'minESS',min(min(v['bulk_ess'],v['tail_ess']) for v in diagnostics.values()),flush=True)


if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--dataset',choices=('empirical','recovery'),default='recovery');parser.add_argument('--variants',nargs='+',default=list(METHODS))
 parser.add_argument('--samples',type=int,default=30000);parser.add_argument('--workers',type=int,default=8);parser.add_argument('--stage',choices=('run','report'),default='run')
 parser.add_argument('--contests',type=int,nargs='+',default=list(s.IDS));args=parser.parse_args()
 cases=[str(cid) if args.dataset=='empirical' else f'{cid}_2' for cid in args.contests]
 jobs=[(args.dataset,case,variant) for case in cases for variant in args.variants]
 if args.stage=='run':
  for job in jobs:prepare(*job)
  with ProcessPoolExecutor(max_workers=args.workers) as pool:list(pool.map(chain,[(*job,j,args.samples) for job in jobs for j in range(1,5)]))
 for job in jobs:summarize(*job)
