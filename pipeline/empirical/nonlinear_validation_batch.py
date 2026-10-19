"""Resume top-ten sampling refinement or posterior-grid validation.

Run only when another batch driver is not active. Child processes retain the
individual estimator's immutable settings, provenance and checkpoints.
"""
from __future__ import annotations
import argparse,json,os,subprocess,sys,time
from pathlib import Path
import numpy as np
from pipeline.empirical import nonlinear_reestimate as n


def sampling_diagnostics(cid):
    folder=n.OUT/f'contest_{cid}'
    if not all((folder/f'chain_{j}.npz').exists() and (folder/f'diagnostics_{j}.json').exists() for j in range(1,5)):
        return None
    z=np.array([np.load(folder/f'chain_{j}.npz')['z'] for j in range(1,5)])
    data,_,_,_=n.load_baseline(cid)
    v=n.physical_draws(z.reshape(-1,6),data['theta']).reshape(4,z.shape[1],9)
    r=max(n.rank_rhat(v[:,:,k]) for k in range(9))
    ess=min(e for k in range(9) for e in n.rank_ess(v[:,:,k]))
    return dict(maximum_rank_rhat=r,minimum_ess=ess,passed=r<1.01 and ess>=400)


def run(stage,workers):
    n.OUT=n.ROOT/'results/estimation/nonlinear_top10/fast_v1'
    n.SOURCE=Path(n.__file__).with_name('nonlinear_fast_kernel.cpp')
    ids=[a['contest_id'] for a in json.loads((n.OUT/'selection.json').read_text())['selected']]
    waiting=set(ids);running={};finished=[];failed=[]
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',OMP_NUM_THREADS='1')
    start=time.monotonic()
    while waiting or running:
        for key,(process,log) in list(running.items()):
            code=process.poll()
            if code is not None:
                log.close();del running[key]
                result=dict(contest_id=key[0],chain=key[1],exit_code=code)
                finished.append(result)
                if code:failed.append(result);waiting.discard(key[0])
                print('FINISHED',result,flush=True)
        for cid in sorted(waiting):
            if any(key[0]==cid for key in running):continue
            diagnostics=sampling_diagnostics(cid)
            folder=n.OUT/f'contest_{cid}'
            if stage=='refine':
                if diagnostics is None:
                    if not (folder/'settings_1.json').exists():
                        failed.append(dict(contest_id=cid,error='Start the four pilot chains before refinement'));waiting.remove(cid);continue
                    if running:continue
                    commands=[]
                    for chain in range(1,5):
                        config=json.loads((folder/f'settings_{chain}.json').read_text())
                        sampler={'prior-screen':'sample-prior','mixture-prior-screen':'sample-mixture','mixture-gaussian-screen':'sample-mixture-gaussian'}.get(config.get('sampler_mode'),'sample')
                        command=[sampler,'--contest',str(cid),'--chain',str(chain)]
                        for flag in ['warmup','samples','dy','nt','seed']:command+=['--'+flag,str(config[flag])]
                        if config.get('pilot_requires_posterior_grid_validation'):command+=['--pilot']
                        commands.append((chain,command))
                elif diagnostics['passed']:
                    waiting.remove(cid);print('CONVERGED',cid,diagnostics,flush=True);continue
                else:
                    if running:continue
                    archive=n.OUT/'learned_proposal_pilots'/f'contest_{cid}'
                    if not archive.exists():
                        proposal=n.OUT/f'proposal_{cid}.json'
                        if not proposal.exists():
                            legacy=n.OUT.parent/'implicit_v1'/f'proposal_{cid}.json'
                            old=json.loads(legacy.read_text())
                            n.save_json(proposal,dict(contest_id=cid,kernel_sha256=n.digest(n.SOURCE),mode_z=old['mode_z'],covariance=old['covariance'],source_proposal=old,proposal_origin='Legacy proposal transferred; exact Metropolis acceptance retains the current target.'))
                        n.learn_proposal(cid)
                        commands=[(chain,['sample','--contest',str(cid),'--chain',str(chain),'--warmup','400','--samples','2400','--dy','.0125','--nt','256','--seed',str(20261011+cid*10+100000),'--pilot']) for chain in range(1,5)]
                    else:
                        commands=[(chain,['extend','--contest',str(cid),'--chain',str(chain),'--extra-samples','2000']) for chain in range(1,5)]
                folder.mkdir(parents=True,exist_ok=True)
            else:
                if diagnostics is None or not diagnostics['passed']:
                    failed.append(dict(contest_id=cid,error='Complete sampling refinement before grid validation'));waiting.remove(cid);continue
                if len(running)>=workers:break
                commands=[(0,['posterior-grid','--contest',str(cid)])]
                waiting.remove(cid)
            for chain,command in commands:
                log_path=(folder/f'batch_refinement_{chain}.log') if stage=='refine' else n.OUT/f'grid_{cid}.log'
                log=log_path.open('a')
                args=[sys.executable,'-u','-m','pipeline.empirical.nonlinear_reestimate',*command,'--backend','fast']
                running[(cid,chain)]=(subprocess.Popen(args,cwd=n.ROOT,env=env,stdout=log,stderr=subprocess.STDOUT),log)
            print('STARTED',stage,cid,flush=True)
            if stage=='refine':break
        n.save_json(n.OUT/f'resumable_{stage}_batch.json',dict(status='running' if waiting or running else ('failed' if failed else 'complete'),
            driver_pid=os.getpid(),waiting=sorted(waiting),running=[dict(contest_id=k[0],chain=k[1],pid=p.pid) for k,(p,_) in running.items()],finished=finished,failed=failed,seconds=time.monotonic()-start))
        if waiting or running:time.sleep(5)
    if stage=='validate':
        unresolved=[cid for cid in ids if not (n.OUT/f'posterior_grid_{cid}.json').exists() or json.loads((n.OUT/f'posterior_grid_{cid}.json').read_text()).get('status')!='validated']
        if unresolved:raise RuntimeError(f'Posterior validation unresolved: {unresolved}; inspect per-contest audits')
    if failed:raise RuntimeError(f'Batch failures: {failed}')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['refine','validate']);parser.add_argument('--workers',type=int,default=2)
    args=parser.parse_args();run(args.stage,args.workers)

if __name__=='__main__':main()
