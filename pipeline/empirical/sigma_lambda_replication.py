"""Generate new confirmation data without changing retained trial designs."""
import argparse
import json
import math
from datetime import datetime,timedelta
import numpy as np
from pipeline.empirical import sigma_lambda_trials as s
from pipeline.empirical import nonlinear_reestimate as n
from synthetic_data.synthetic_data import synthetic_data_simulation


def generate(rep):
 n.SOURCE=n.ROOT/'pipeline/empirical/nonlinear_fast_kernel.cpp'
 for cid in s.IDS:
  folder=s.OUT/'recovery'/f'{cid}_{rep}';folder.mkdir(parents=True,exist_ok=True)
  if (folder/'data.json').exists():continue
  data,_,_,_=n.load_baseline(cid);z,reference=s.truth_point(cid);truth=s.a.physical(z);truth['r']=math.exp(z[5])
  sigma=truth['sigma'];T=data['T_contest'];scale=sigma/math.sqrt(T)
  policy,dy=n.Likelihood(data,.00625,1024).policy(z)
  def at(t,y):
   pos=y/(sigma*math.sqrt(T))/dy+(policy.shape[1]-1)/2
   if not 0<=pos<policy.shape[1]-1:raise RuntimeError('Simulation outside policy domain')
   j=int(pos);f=pos-j;time=t/T*(policy.shape[0]-1);k=min(int(time),policy.shape[0]-2);tf=time-k
   return scale*((1-tf)*((1-f)*policy[k,j]+f*policy[k,j+1])+tf*((1-f)*policy[k+1,j]+f*policy[k+1,j+1]))
  peak=data['theta']/min(truth['c_i'],truth['c_j'])/math.sqrt(2*math.pi*sigma/math.sqrt(truth['lambda']))
  bound=max(1.,1.01*truth['r']*max(float(policy.max())*scale,peak)/24)
  seed=271000+100*cid+rep*10;start=datetime(2026,1,1)
  result=synthetic_data_simulation(theta=data['theta'],c_i=truth['c_i'],c_j=truth['c_j'],sigma=sigma,lamb=truth['lambda'],intensity_effort_ratio=truth['r'],hour_arrival_ub=bound,
   start_time=start,end_time=start+timedelta(hours=data['N_Delta']),mu_0=truth['mu_0'],effort_policy=at,seed_brownian=seed,seed_poisson=seed+1,seed_uniform=seed+2,seed_initial=seed+3)
  times,mi,mj,latent,perceived,public,ei,ej=result
  sim=dict(theta=data['theta'],Delta2f=data['Delta2f'],N_Delta=len(times),count_start_idx=1,T_contest=T,Ni=len(ei),Nj=len(ej),
   hat_t_i=[(t-start).total_seconds()/3600 for t in ei],hat_t_j=[(t-start).total_seconds()/3600 for t in ej],hat_y=public[1:].tolist(),feedback_gap=public[:-1].tolist())
  n.save_json(folder/'data.json',sim);n.save_json(folder/'truth.json',dict(truth=truth,seed=seed,simulation_grid='Same checked .00625/1024 policy as preceding replicates',**reference))
  np.savez_compressed(folder/'paths.npz',latent=latent,public=public,perceived=perceived,mi=mi,mj=mj)
  print('CONFIRMATION GENERATED',cid,rep,len(ei),len(ej),flush=True)


if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--replicate',type=int,default=3);args=parser.parse_args();generate(args.replicate)
