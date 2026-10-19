"""Learn a fixed nonlinear recovery proposal without changing its posterior."""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from pipeline.empirical import nonlinear_reestimate as n
from pipeline.empirical import activity_nonlinear_recovery as a


def learn(generation):
    a.configure();proposal=n.OUT/'proposal_4657.json';current=json.loads(proposal.read_text())
    if 'mixture_components' in current and current.get('mixture_generation',1)==generation:return
    folder=n.OUT/'contest_4657';z=np.concatenate([np.load(folder/f'chain_{j}.npz')['z'] for j in range(1,5)])
    # The weak absolute-effort direction is chiefly the common cost/r scale.
    threshold=float(np.median(z[:,5]));groups=[z[z[:,5]<=threshold],z[z[:,5]>threshold]]
    n.learn_proposal(4657,generation)
    current=json.loads(proposal.read_text());current['mixture_components']=[dict(weight=len(g)/len(z),
       mean=g.mean(axis=0).tolist(),covariance=(np.cov(g,rowvar=False)+np.eye(6)*1e-6).tolist()) for g in groups]
    current['mixture_generation']=generation
    current['mixture_split']=dict(coordinate='log_r',threshold=threshold,
        purpose='Fixed computational proposal only; activity prior and likelihood unchanged.')
    n.save_json(proposal,current)


def chain(task):
    j,warmup,samples=task;a.configure()
    n.sample_mixture_gaussian(4657,j,warmup,samples,.0125,256,241100+4657,pilot=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--generation',type=int,default=1)
    parser.add_argument('--workers',type=int,default=4);parser.add_argument('--warmup',type=int,default=1000)
    parser.add_argument('--samples',type=int,default=6000);parser.add_argument('--stage',choices=('learn','run','grid'),default='run')
    args=parser.parse_args()
    if args.stage in ('learn','run'):learn(args.generation)
    if args.stage=='run':
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            list(pool.map(chain,[(j,args.warmup,args.samples) for j in range(1,5)]))
    if args.stage in ('run','grid'):
        a.configure();n.posterior_grid(4657,draws=2048,workers=args.workers);a.summarize()
