"""Run the frozen prior batch with variance reduction for lambda means."""
import argparse
import concurrent.futures
import csv
import json
from pathlib import Path
from pipeline.empirical import unified_prior_reestimate as u
from pipeline.empirical import unified_prior_moments as rb

RAW_DIAGNOSE=u.diagnose


def diagnose(cid,mode,paths,data):
    row,diagnostics=RAW_DIAGNOSE(cid,mode,paths,data)
    eligible=(row['divergences']==0 and diagnostics['saturated_transitions']==0
              and min(diagnostics['energy_bfmi'])>=.3
              and all(v['rank_rhat']<=1.01 and min(v['bulk_ess'],v['tail_ess'])>=400
                      for v in diagnostics['quantities'].values()))
    if eligible:
        row,diagnostics=rb.augment(cid,mode,paths,data,row,diagnostics)
    diagnostics['batch_protocol_sha256']=u.digest(__file__)
    return row,diagnostics


def fit(task):
    u.diagnose=diagnose
    cid,mode,pilot=task
    path=u.OUT/'estimation'/mode/f'contest_{cid}/accepted.json'
    if path.exists():
        saved=json.loads(path.read_text())
        if saved['diagnostics'].get('batch_protocol_sha256')!=u.digest(__file__):
            paths=[Path(p) for p in saved['chain_paths']]
            data=json.loads((paths[0].parent/'data.json').read_text())
            row,diagnostics=diagnose(cid,mode,paths,data)
            if not diagnostics['passed']:
                raise RuntimeError('Previously accepted chain requires further moment refinement')
            saved['summary']=row;saved['diagnostics']=diagnostics
            u.save(path,saved)
    return u.fit_one(task)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--contests',type=int,nargs='+')
    parser.add_argument('--no-outputs',action='store_true')
    args=parser.parse_args()
    design=dict(prior='log(lambda) ~ N(log100,2.25^2); all other priors unchanged',
                source_sha256=u.digest(__file__),moments_source_sha256=u.digest(rb.__file__),runner_sha256=u.digest(u.__file__),
                posterior='Same four-chain NUTS target and samples; conditional lambda mean integration only reduces Monte Carlo variation.',
                gates='rank Rhat<=1.01, bulk/tail ESS>=400 for primitives and w, zero divergences/depth saturations, BFMI>=.3, sigma/lambda mean relative MCSE<=3%, quadrature agreement<=0.1%.',
                supports='Original .01 numerical lower bounds and no upper bounds; quadrature finite tail domain verified analytically and by doubled resolution.')
    signature=u.OUT/'batch_protocol.json'
    if signature.exists() and json.loads(signature.read_text())!=design:
        raise RuntimeError('Frozen batch protocol changed')
    u.save(signature,design)
    manifest=u.MAIN_ESTIMATION_DIR/'accepted_chain_sources.csv'
    tasks=sorted((int(row['contest_id']),row['mode'],False) for row in csv.DictReader(manifest.open()))
    if args.contests:tasks=[t for t in tasks if t[0] in args.contests]
    failures=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures={pool.submit(fit,t):t for t in tasks}
        for future in concurrent.futures.as_completed(futures):
            try:future.result()
            except Exception as error:
                failures.append(dict(task=futures[future],error=str(error)))
                print('FAILED',futures[future],error,flush=True)
            u.assemble()
            u.save(u.OUT/'batch_failures.json',failures)
    if failures:raise RuntimeError(f'{len(failures)} fits failed; preserve and refine')
    if not args.no_outputs and not args.contests:
        from pipeline.empirical.unified_prior_outputs import main as outputs
        outputs()


if __name__=='__main__':main()
