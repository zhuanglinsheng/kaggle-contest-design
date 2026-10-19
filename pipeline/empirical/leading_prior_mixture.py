"""Frozen pilot-informed proposal; retain the leading statistical target."""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from pipeline.empirical import nonlinear_reestimate as n
from pipeline.empirical import fast_leading_prior as f


def learn(cid,profile):
    f.configure(profile,'empirical')
    path=n.OUT/f'proposal_{cid}.json'
    old=json.loads(path.read_text())
    if old.get('leading_mixture'):return
    z=np.concatenate([np.load(n.OUT/f'contest_{cid}/chain_{j}.npz')['z'] for j in range(1,5)])
    threshold=float(np.median(z[:,5]))
    groups=[z[z[:,5]<=threshold],z[z[:,5]>threshold]]
    n.learn_proposal(cid)
    proposal=json.loads(path.read_text())
    proposal['mixture_components']=[dict(weight=len(g)/len(z),mean=g.mean(0).tolist(),
        covariance=(np.cov(g,rowvar=False)+np.eye(6)*1e-6).tolist()) for g in groups]
    proposal['leading_mixture']=dict(coordinate='log_r',threshold=threshold,
        note='Sampling proposal only; prior and likelihood unchanged.')
    n.save_json(path,proposal)


def chain(task):
    cid,profile,j=task;f.configure(profile,'empirical')
    n.sample_mixture_gaussian(cid,j,2000,30000,.025,64,251100+cid,pilot=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--contest',type=int,default=5390)
    parser.add_argument('--profile',default='scale_weak');args=parser.parse_args()
    learn(args.contest,args.profile)
    with ProcessPoolExecutor(max_workers=4) as pool:
        list(pool.map(chain,[(args.contest,args.profile,j) for j in range(1,5)]))
    f.summarize(args.contest,args.profile,'empirical')
