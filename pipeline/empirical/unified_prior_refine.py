"""Long NUTS refinement for cases that failed the unchanged acceptance gates."""
import argparse
import json
import subprocess
import os
import numpy as np
from pipeline.empirical import unified_prior_reestimate as u,unified_prior_batch as batch


def run(cid,mode):
    case=u.OUT/'estimation'/mode/f'contest_{cid}'
    if (case/'accepted.json').exists():return
    config=json.loads((case/'configuration.json').read_text())
    data=config['inputs']
    previous=case/'attempt3'
    chains=[u.e.read_columns(previous/f'chain_{j}.csv') for j in range(1,5)]
    sigma=np.concatenate([c['sigma'] for c in chains])
    flat={name:np.concatenate([c[name] for c in chains]) for name in u.e.PARAMETERS}
    starts=[]
    for q in (.2,.5,.8,.98):
        index=int(np.argmin(np.abs(np.log(sigma)-np.quantile(np.log(sigma),q))))
        starts.append({name:float(flat[name][index]) for name in u.e.PARAMETERS})
    for attempt,(warmup,samples,delta) in enumerate(((4000,32000,.99),(6000,64000,.995)),1):
        folder=case/f'long_refinement{attempt}';folder.mkdir(parents=True,exist_ok=True)
        settings=dict(warmup=warmup,samples=samples,adapt_delta=delta,max_depth=13,
                      source_sha256=u.digest(__file__),batch_sha256=u.digest(batch.__file__),original_configuration=config)
        signature=folder/'settings.json'
        if signature.exists() and json.loads(signature.read_text())!=settings:raise RuntimeError('Changed long-run configuration')
        u.save(signature,settings);u.save(folder/'data.json',data)
        paths=[folder/f'chain_{j}.csv' for j in range(1,5)];processes=[]
        for j,path in enumerate(paths,1):
            u.save(folder/f'init_{j}.json',starts[j-1])
            if u.e.csv_complete(path,samples):continue
            log=(folder/f'chain_{j}.log').open('w')
            cmd=[str(u.MODEL),'sample',f'num_warmup={warmup}',f'num_samples={samples}','adapt',f'delta={delta}',
                 'algorithm=hmc','engine=nuts','max_depth=13','metric=dense_e','random',f'seed={20301015+cid+(100000 if mode=="early" else 0)}',f'id={j}',
                 'data',f'file={folder/"data.json"}','output',f'file={path}','refresh=500','sig_figs=18',f'init={folder/f"init_{j}.json"}']
            processes.append((subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,OPENBLAS_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1')),log))
        codes=[]
        for p,h in processes:codes.append(p.wait());h.close()
        if any(codes):raise RuntimeError('Long NUTS chain failed')
        row,diagnostics=batch.diagnose(cid,mode,paths,data)
        result=dict(summary=row,diagnostics=diagnostics,configuration=config,attempt=f'long_refinement{attempt}',
                    chain_paths=[str(p) for p in paths],chain_sha256=[u.digest(p) for p in paths])
        u.save(folder/'diagnostics.json',result)
        print('LONG REFINEMENT',cid,mode,attempt,diagnostics['passed'],
              max(v['rank_rhat'] for v in diagnostics['quantities'].values()),
              min(min(v['bulk_ess'],v['tail_ess']) for v in diagnostics['quantities'].values()),flush=True)
        if diagnostics['passed']:
            u.save(case/'accepted.json',result);u.assemble();return
    raise RuntimeError('Long refinement still fails the unchanged gates')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('contest_id',type=int);parser.add_argument('--mode',choices=('full','early'),default='full');args=parser.parse_args();run(args.contest_id,args.mode)
