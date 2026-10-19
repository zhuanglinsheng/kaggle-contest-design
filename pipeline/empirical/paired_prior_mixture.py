"""Improve nonlinear tail exploration under the unchanged common prior."""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from pipeline.empirical import nonlinear_reestimate as n
from pipeline.empirical import paired_prior_improvement as p


def learn(cid,profile,generation):
    p.configure(profile)
    proposal=n.OUT/f'proposal_{cid}.json'
    current=json.loads(proposal.read_text())
    if 'mixture_components' in current and generation==current.get('mixture_generation',1):return
    folder=n.OUT/f'contest_{cid}'
    z=np.concatenate([np.load(folder/f'chain_{j}.npz')['z'] for j in range(1,5)])
    threshold=float(np.median(z[:,2]));groups=[z[z[:,2]<=threshold],z[z[:,2]>threshold]]
    if min(map(len,groups))<100:raise RuntimeError('Insufficient pilot coverage')
    n.learn_proposal(cid,generation)
    current=json.loads(proposal.read_text())
    current['mixture_components']=[dict(weight=len(g)/len(z),mean=g.mean(axis=0).tolist(),
       covariance=(np.cov(g,rowvar=False)+np.eye(6)*1e-6).tolist()) for g in groups]
    current['mixture_split']=dict(coordinate='log_sigma',threshold=threshold,
       purpose='Frozen sampling proposal only. Exact target, support and priors unchanged.',
       pilot_counts=list(map(len,groups)))
    current['mixture_generation']=generation
    n.save_json(proposal,current)
    print('COMMON PRIOR MIXTURE LEARNED',cid,profile,flush=True)


def chain(task):
    cid,profile,j,warmup,samples=task;p.configure(profile)
    n.sample_mixture_gaussian(cid,j,warmup,samples,.0125,256,231100+cid,pilot=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--contest',type=int,default=4699)
    parser.add_argument('--contests',type=int,nargs='+')
    parser.add_argument('--profile',choices=p.PROFILES,default='scale_weak')
    parser.add_argument('--stage',choices=('learn','run','grid'),default='run')
    parser.add_argument('--generation',type=int,default=1);parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--warmup',type=int,default=1000);parser.add_argument('--samples',type=int,default=6000)
    args=parser.parse_args()
    for cid in args.contests or [args.contest]:
        if args.stage in ('learn','run'):learn(cid,args.profile,args.generation)
        if args.stage=='run':
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                list(pool.map(chain,[(cid,args.profile,j,args.warmup,args.samples) for j in range(1,5)]))
        if args.stage in ('run','grid'):
            p.configure(args.profile);n.posterior_grid(cid,draws=2048,workers=args.workers)
