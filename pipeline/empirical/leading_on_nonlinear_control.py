"""Test the improved leading estimator against preserved nonlinear-generated data."""
import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pipeline.empirical import fast_leading_prior as f
from pipeline.empirical import paired_prior_recovery as r
from pipeline.empirical import nonlinear_reestimate as n
from pipeline.empirical import paired_prior_improvement as p

BASE_CONFIGURE=f.configure


def configure(profile,kind='recovery'):
    BASE_CONFIGURE(profile,'recovery')
    os.environ['KAGGLE_RECOVERY_POLICY']='nonlinear'
    n.OUT=p.OUT/'recovery_4657/nonlinear'/profile/'leading_fast'
    n.OUT.mkdir(parents=True,exist_ok=True)


def chain(task):
    j,profile,warmup,samples=task;configure(profile)
    n.sample(4657,j,warmup,samples,.025,64,211100+4657,pilot=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=1)
    parser.add_argument('--samples',type=int,default=30000);parser.add_argument('--warmup',type=int,default=2000)
    args=parser.parse_args()
    first=json.loads((r.FIXTURE/'leading_control/truth.json').read_text())['truth']
    second=json.loads((r.FIXTURE/'nonlinear/truth.json').read_text())['truth']
    if first!=second:raise RuntimeError('The two preserved controls have different truths')
    f.configure=configure
    f.prepare(4657,'activity_scale','recovery')
    n.save_json(n.OUT/'control_design.json',dict(policy_generating_data='nonlinear',policy_used_in_estimation='leading',
       common_prior='activity_scale',truth=second,script_sha256=n.digest(__file__),
       purpose='Separate prior-scale improvement from structural effort approximation. Different realized paths preclude subtracting estimates as a general bias decomposition.'))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(chain,[(j,'activity_scale',args.warmup,args.samples) for j in range(1,5)]))
    f.summarize(4657,'activity_scale','recovery')
