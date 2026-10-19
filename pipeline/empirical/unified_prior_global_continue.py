"""Continue identical MH chains, preserving the completed shorter run.

Only the stopping length changes. The target, proposal, RNG state, warmup,
and retained history remain identical; original artifacts are archived intact.
"""
import argparse
import concurrent.futures
import json
import shutil
from pathlib import Path
from pipeline.empirical import unified_prior_reestimate as u,unified_prior_global_check as g


def main(scenario,cid,mode,samples):
    case=f'{scenario}_{cid}_{mode}';root=u.OUT/'global_crosschecks'/case
    settings=json.loads((root/f'contest_{cid}/settings_1.json').read_text());old=settings['samples']
    if samples<=old:raise RuntimeError('Continuation must increase the retained length')
    archive=u.OUT/'global_crosschecks/archive'/f'{case}_{old}'
    if archive.exists():raise RuntimeError('Archive exists; do not overwrite a completed stage')
    archive.parent.mkdir(parents=True,exist_ok=True)
    root.rename(archive);root.mkdir()
    for name in ('design.json','data.json','target_parity.json',f'proposal_{cid}.json'):shutil.copy2(archive/name,root/name)
    folder=root/f'contest_{cid}';folder.mkdir()
    records=[]
    for j in range(1,5):
        source=archive/f'contest_{cid}'
        cfg=json.loads((source/f'settings_{j}.json').read_text())
        if cfg['samples']!=old:raise RuntimeError('Different original chain lengths')
        cfg['samples']=samples
        u.save(folder/f'settings_{j}.json',cfg)
        shutil.copy2(source/f'checkpoint_{j}.npz',folder/f'checkpoint_{j}.npz')
        records.append(dict(chain=j,original_chain_sha256=u.digest(source/f'chain_{j}.npz'),
                            original_settings_sha256=u.digest(source/f'settings_{j}.json')))
    u.save(root/'continuation.json',dict(from_samples=old,to_samples=samples,original_stage=str(archive),
        unchanged_proposal_sha256=u.digest(root/f'proposal_{cid}.json'),original_chains=records,source_sha256=u.digest(__file__),
        method='Continue from saved RNG/checkpoint with unchanged posterior, proposal and frozen post-warmup kernel.'))
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:
        list(pool.map(g.chain,[(case,cid,j,samples) for j in range(1,5)]))
    g.summarize(case,cid)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('scenario');parser.add_argument('contest_id',type=int);parser.add_argument('--mode',choices=('full','early'),default='full');parser.add_argument('--samples',type=int,default=60000);args=parser.parse_args();main(args.scenario,args.contest_id,args.mode,args.samples)
