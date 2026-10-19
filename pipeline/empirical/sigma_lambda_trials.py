"""Small leading-estimator experiments focused on sigma and lambda.

Empirical nonlinear estimates are references, not truth. Two fixed synthetic
realizations per ten prespecified configurations provide actual recovery tests.
MAP screening is explicitly exploratory; retained candidates require posterior
sampling before any publication replacement.
"""
import argparse
import ctypes
import json
import math
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.special import softmax
from pipeline.empirical import nonlinear_reestimate as n
from pipeline.empirical import estimation_detail_audit as a
from synthetic_data.synthetic_data import synthetic_data_simulation

OUT=n.ROOT/'results/validation/sigma_lambda_trials'
IDS=(4699,13333,4657,6322,8076,23249,4407,5390,4366,3929)
BASE_SD=[.75,.75,.5,.75,1.,.5]
VARIANTS={
 'baseline':{},
 'r_weak':dict(r_sd=1.),
 'r_activity':dict(r_mean=80.,r_sd=1.5),
 'joint_scale_weak':dict(cost_sd=1.25,r_sd=1.),
 'noise_wide':dict(sigma_sd=1.,lambda_sd=1.5),
 'noise_external':dict(sigma_mean=.5,sigma_sd=1.25,lambda_mean=100.,lambda_sd=2.25),
 'activity_external':dict(cost_sd=1.25,r_mean=80.,r_sd=1.5,sigma_mean=.5,sigma_sd=1.25,lambda_mean=100.,lambda_sd=2.25),
 'count_quarter':dict(count_weight=.25),
 'score_only':dict(count_weight=0.),
 'lambda_external':dict(lambda_mean=100.,lambda_sd=2.25),
 'activity_lambda_external':dict(r_mean=80.,r_sd=1.5,lambda_mean=100.,lambda_sd=2.25),
 'count_quarter_lambda_external':dict(count_weight=.25,lambda_mean=100.,lambda_sd=2.25),
}
LIB=None


def kernel():
 global LIB
 if LIB is not None:return LIB
 OUT.mkdir(parents=True,exist_ok=True)
 original=n.ROOT/'pipeline/empirical/nonlinear_fast_kernel.cpp'
 source=original.read_text()
 replacements={
 'const double* held,const int* counts_i':'const double* held,const double* feedback,double count_weight,double* components,const int* counts_i',
 'resvar=steady,ll=0;':'resvar=steady,ll=0,count_ll=0,score_ll=0;',
 'll+=counts_i[i]*':'count_ll+=counts_i[i]*',
 'if(i>=count_start) ll-=ii+jj;':'if(i>=count_start) count_ll-=ii+jj;',
 'll+=-.5*(std::log(2*pi*v)+innovation*innovation/v);':'score_ll+=-.5*(std::log(2*pi*v)+innovation*innovation/v);',
 'weight*(held[i]-perceived)':'weight*(feedback[i]-perceived)',
 'if(!std::isfinite(ll)) *status=-4;':'ll=score_ll+count_weight*count_ll; components[0]=score_ll;components[1]=count_ll;\n  if(!std::isfinite(ll)) *status=-4;',
 }
 for before,after in replacements.items():
  if source.count(before)!=1:raise RuntimeError(('Kernel replacement count',before,source.count(before)))
  source=source.replace(before,after)
 path=OUT/'weighted_leading_kernel.cpp'
 if path.exists() and path.read_text()!=source:raise RuntimeError('Trial kernel changed')
 if not path.exists():path.write_text(source)
 binary=OUT/(n.digest(path)[:20]+'.so')
 if not binary.exists():
  temp=binary.with_suffix(f'.{os.getpid()}.tmp')
  subprocess.run(['c++','-O3','-std=c++17','-shared','-fPIC',str(path),'-o',str(temp)],check=True)
  temp.replace(binary)
 lib=ctypes.CDLL(str(binary));arr=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
 ints=np.ctypeslib.ndpointer(dtype=np.int32,flags='C_CONTIGUOUS')
 lib.empirical_likelihood.argtypes=[ctypes.c_int]+[ctypes.c_double]*9+[arr,arr,ctypes.c_double,arr,ints,ints,ctypes.c_int,ctypes.c_void_p,ctypes.c_int,ctypes.c_int,ctypes.c_double,ctypes.POINTER(ctypes.c_int)]
 lib.empirical_likelihood.restype=ctypes.c_double;LIB=lib
 return lib


def prior_config(data,variant):
 cfg=VARIANTS[variant]
 mean=np.array([0.,0.,math.log(cfg.get('sigma_mean',2.)),math.log(cfg.get('lambda_mean',1.)),data['hat_y'][0],math.log(cfg.get('r_mean',10.))])
 sd=np.array(BASE_SD);sd[:2]=cfg.get('cost_sd',.75)
 sd[2]=cfg.get('sigma_sd',.5);sd[3]=cfg.get('lambda_sd',.75);sd[5]=cfg.get('r_sd',.5)
 return mean,sd


class Target:
 def __init__(self,data,variant):
  self.data=data;self.variant=variant;self.lib=kernel();self.calls=0
  self.mean,self.sd=prior_config(data,variant)
  self.held=np.ascontiguousarray(data['hat_y'],dtype=np.float64)
  self.feedback=np.ascontiguousarray(data.get('feedback_gap',data['hat_y']),dtype=np.float64)
  self.counts=[]
  for key in ('hat_t_i','hat_t_j'):
   counts=np.zeros(data['N_Delta'],dtype=np.int32)
   for t in data[key]:counts[max(1,math.ceil(t))-1]+=1
   self.counts.append(counts)
 def prior(self,z):
  if not np.isfinite(z).all() or np.any(z[[0,1,2,3,5]]<math.log(.01)):return -np.inf
  return float(-.5*np.sum(((z-self.mean)/self.sd)**2)-np.log(self.sd).sum()-3*math.log(2*math.pi))
 def components(self,z):
  ci,cj,sig,lam=np.exp(z[:4]);d=self.data;status=ctypes.c_int();parts=np.zeros(2)
  value=self.lib.empirical_likelihood(d['N_Delta'],d['Delta2f'],d['T_contest'],d['theta'],ci,cj,sig,lam,float(z[4]),float(z[5]),
   self.held,self.feedback,VARIANTS[self.variant].get('count_weight',1.),parts,*self.counts,d['count_start_idx']-1,None,0,64,.025,ctypes.byref(status))
  self.calls+=1
  if status.value:return -np.inf,parts
  return value,parts
 def posterior(self,z):
  prior=self.prior(z)
  return prior+self.components(z)[0] if np.isfinite(prior) else -np.inf


def truth_point(cid):
 path=a.NONLINEAR/f'posterior_grid_{cid}.npz';grid=np.load(path)
 weights=softmax(grid['fine']-grid['base']);_,sd,q=n.weighted_summary(grid['z'],weights)
 center=q[:,1];index=np.argmin(np.sum(((grid['z']-center)/np.maximum(sd,1e-8))**2,axis=1))
 return grid['z'][index].copy(),dict(reference_posterior_sha256=n.digest(path),reference_grid_sha256=n.digest(path.with_suffix('.json')))


def prepare(replicates=(0,1)):
 OUT.mkdir(parents=True,exist_ok=True);kernel()
 design=dict(contests=IDS,variants=VARIANTS,recovery_replicates=list(replicates),seeds='271000+100*contest_id+replicate*10, then +1/+2/+3 for Poisson/uniform/initial',
  fixed_parameters='Joint representative from validated original nonlinear posterior. Truth only for newly generated data, not empirical data.',
  split='Replicates0/1 exploratory, inspected before adding lambda-only candidates. Replicate2 is a new held-out realization for posterior confirmation. Ten configurations are a targeted stress sample, not representative of73.',
  interpretation='Count weighting is generalized-Bayes sensitivity, not a prior-only modification; score_only drops count likelihood but retains readout times and leading drift.',
  metric='Mean absolute log error for sigma and lambda, separately and jointly; also relative median errors and coverage after validated sampling.',
  source_sha256=n.digest(__file__),simulator_sha256=n.digest(n.ROOT/'synthetic_data/synthetic_data.py'))
 n.save_json(OUT/('design_round2.json' if (OUT/'design.json').exists() else 'design.json'),design)
 for cid in IDS:
  data,_,z,provenance=n.load_baseline(cid);folder=OUT/'empirical'/str(cid);folder.mkdir(parents=True,exist_ok=True)
  n.save_json(folder/'data.json',data);n.save_json(folder/'provenance.json',provenance)
  truthz,reference=truth_point(cid);truth=a.physical(truthz);truth['r']=math.exp(truthz[5])
  # Generate once per fixed configuration; seeds and candidates never rerolled.
  if all((OUT/'recovery'/f'{cid}_{rep}/data.json').exists() for rep in replicates):continue
  n.SOURCE=n.ROOT/'pipeline/empirical/nonlinear_fast_kernel.cpp'
  evaluator=n.Likelihood(data,.00625,1024);policy,dy=evaluator.policy(truthz)
  finer,fdy=n.Likelihood(data,.003125,2048).policy(truthz)
  sigma=truth['sigma'];T=data['T_contest'];scale=sigma/math.sqrt(T)
  def interpolate(t,y,pol,step):
   pos=y/(sigma*math.sqrt(T))/step+(pol.shape[1]-1)/2
   if pos<0 or pos>=pol.shape[1]-1:raise RuntimeError('Simulation outside policy domain')
   j=int(pos);f=pos-j;time=min(t/T,1)*(pol.shape[0]-1);k=min(int(time),pol.shape[0]-2);tf=time-k
   return scale*((1-tf)*((1-f)*pol[k,j]+f*pol[k,j+1])+tf*((1-f)*pol[k+1,j]+f*pol[k+1,j+1]))
  checks=[(t*T,y*sigma*math.sqrt(T)) for t in (0,.25,.5,.75,.9) for y in (-2,-1,0,1,2)]
  coarse=np.array([interpolate(t,y,policy,dy) for t,y in checks]);fine=np.array([interpolate(t,y,finer,fdy) for t,y in checks])
  error=float(np.max(np.abs(coarse-fine))/max(np.max(fine),1e-20))
  if error>.01:raise RuntimeError(f'Simulation policy grid discrepancy: {cid} {error}')
  leading_peak=data['theta']/min(truth['c_i'],truth['c_j'])/math.sqrt(2*math.pi*sigma/math.sqrt(truth['lambda']))
  bound=max(1.,1.01*truth['r']*max(float(policy.max())*scale,leading_peak)/24)
  for rep in replicates:
   target=OUT/'recovery'/f'{cid}_{rep}';target.mkdir(parents=True,exist_ok=True)
   if (target/'data.json').exists():continue
   seed=271000+100*cid+rep*10;start=datetime(2026,1,1)
   result=synthetic_data_simulation(theta=data['theta'],c_i=truth['c_i'],c_j=truth['c_j'],sigma=sigma,lamb=truth['lambda'],intensity_effort_ratio=truth['r'],hour_arrival_ub=bound,
    start_time=start,end_time=start+timedelta(hours=data['N_Delta']),mu_0=truth['mu_0'],effort_policy=lambda t,y:interpolate(t,y,policy,dy),
    seed_brownian=seed,seed_poisson=seed+1,seed_uniform=seed+2,seed_initial=seed+3)
   times,mi,mj,latent,perceived,public,ei,ej=result
   sim=dict(theta=data['theta'],Delta2f=data['Delta2f'],N_Delta=len(times),count_start_idx=1,T_contest=T,Ni=len(ei),Nj=len(ej),
    hat_t_i=[(t-start).total_seconds()/3600 for t in ei],hat_t_j=[(t-start).total_seconds()/3600 for t in ej],hat_y=public[1:].tolist(),feedback_gap=public[:-1].tolist())
   n.save_json(target/'data.json',sim);n.save_json(target/'truth.json',dict(truth=truth,seed=seed,grid_discrepancy=error,**reference))
   np.savez_compressed(target/'paths.npz',latent=latent,public=public,perceived=perceived,mi=mi,mj=mj)
   print('GENERATED',cid,rep,len(ei),len(ej),error,flush=True)


def optimize(dataset,cid,variant):
 folder=OUT/('empirical' if dataset=='empirical' else 'recovery')/str(cid)
 dest=folder/f'map_{variant}.json'
 if dest.exists():return json.loads(dest.read_text())
 data=json.loads((folder/'data.json').read_text());target=Target(data,variant)
 original_cid=int(str(cid).split('_')[0]);_,_,old,_=n.load_baseline(original_cid)
 start=n.representative_point(old);start[4]=data['hat_y'][0]
 # Neither optimization start uses the synthetic truth.
 alternate=target.mean.copy();alternate[2]=math.log(.3);alternate[3]=math.log(100.)
 third=start.copy();third[2]+=math.log(2);third[5]+=math.log(4)
 bounds=[(math.log(.01)+1e-7,12)]*4+[(None,None),(math.log(.01)+1e-7,12)]
 runs=[]
 for point in (start,alternate,third):
  result=minimize(lambda z:-target.posterior(z),point,method='L-BFGS-B',bounds=bounds,options=dict(maxiter=500,ftol=1e-11,gtol=1e-6))
  runs.append(dict(z=result.x.tolist(),logposterior=-float(result.fun),converged=bool(result.success),message=str(result.message)))
 best=max(runs,key=lambda v:v['logposterior']);value,parts=target.components(np.array(best['z']))
 ans=dict(dataset=dataset,case=str(cid),variant=variant,best=best,runs=runs,score_loglik=float(parts[0]),count_loglik=float(parts[1]),calls=target.calls,
  data_sha256=n.digest(folder/'data.json'),note='Exploratory MAP, not a posterior mean/median or validated estimate.')
 if (folder/'truth.json').exists():
  truth=json.loads((folder/'truth.json').read_text())['truth'];ans['truth']=truth
  ans['absolute_log_error']={key:abs(best['z'][k]-math.log(truth[key])) for k,key in ((2,'sigma'),(3,'lambda'))}
 n.save_json(dest,ans);return ans


def map_job(payload):
 dataset,cid=payload
 for variant in VARIANTS:
  ans=optimize(dataset,cid,variant)
  print('MAP',dataset,cid,variant,ans['best']['converged'],flush=True)


def scan():
 tasks=[('empirical',cid) for cid in IDS]+[('recovery',f'{cid}_{rep}') for cid in IDS for rep in (0,1)]
 with ProcessPoolExecutor(max_workers=4) as pool:list(pool.map(map_job,tasks))


if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=('prepare','scan'),default='prepare');args=parser.parse_args()
 if args.stage=='prepare':prepare((2,))
 else:scan()
