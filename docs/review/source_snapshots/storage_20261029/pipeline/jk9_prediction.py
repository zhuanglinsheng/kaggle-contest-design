"""Portable prediction reconstruction from recovered original sources.

summary reaggregates historical scores; baselines recomputes A/B from inputs.
Neither operation certifies missing historical C/D integration diagnostics.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from scipy.special import gammaln, logsumexp
from pipeline.publication import ROOT, digest, write_json
from pipeline.jk9 import RELEASE
from pipeline.prediction import baselines as b
from pipeline.prediction.posterior import counts_for

EVIDENCE = RELEASE / 'evidence/prediction'


def verify_recovered_sources():
    manifest=json.loads((EVIDENCE/'source_recovery_manifest.json').read_text())
    for row in manifest['files']:
        payload=(ROOT/row['path']).read_bytes()
        blob=hashlib.sha1(b'blob '+str(len(payload)).encode()+b'\0'+payload).hexdigest()
        if hashlib.sha256(payload).hexdigest()!=row['sha256'] or blob!=row['git_blob']:
            raise ValueError(f"Recovered source identity changed: {row['path']}")
    return len(manifest['files'])


def splits(contest=None, split=None):
    rows = json.loads((EVIDENCE / 'splits.json').read_text())['records']
    return [r for r in rows if (contest is None or r['contest_id'] == contest)
            and (split is None or r['split'] == split)]


def raw_input(row):
    path = ROOT / row['input_snapshot']
    if digest(path) != row['input_sha256']:
        raise ValueError(f'Changed input: {path}')
    return json.loads(path.read_text())


def summary(records, replicates=500000):
    """Original paired contest bootstrap, including original random-number order."""
    if replicates <= 0 or replicates % 50:
        raise ValueError('Replicates must be a positive multiple of 50')
    result = dict(replicates=replicates, seed_base=20261025+1904, folds={},
                  scope='Reaggregation of original score records; integration certificates are separate')
    for fold, split in enumerate(('primary_hour', 'complete_day')):
        selected = [r for r in records if r['split'] == split]
        lookup = {(r['model'], r['contest_id']): r for r in selected}
        cases = sorted({r['contest_id'] for r in selected})
        if len(cases) != 73 or len(lookup) != len(selected) or len(selected) != 292:
            raise ValueError(f'Expected 73 distinct contests x 4 models: {split}')
        contrasts = [('D','C'), ('D','A'), ('D','B'), ('B','A')]
        diff = np.array([[lookup[a,c]['mean_joint_logscore']-lookup[b_,c]['mean_joint_logscore']
                          for a,b_ in contrasts] for c in cases])
        rng = np.random.default_rng(result['seed_base']+fold*1000000)
        samples = [diff[rng.integers(0,73,size=(replicates//50,73))].mean(axis=1) for _ in range(50)]
        draws = np.concatenate(samples)
        ci = np.quantile(draws, [.025,.975], axis=0)
        batchci = np.array([np.quantile(x,[.025,.975],axis=0) for x in np.array_split(draws,10)])
        endpoint = batchci.std(axis=0,ddof=1)/np.sqrt(10)
        result['folds'][split] = dict(contests=73, means={m:float(np.mean([
            lookup[m,c]['mean_joint_logscore'] for c in cases])) for m in 'ABCD'},
            contrasts={a+'-'+b_:dict(mean=float(diff[:,j].mean()),interval=ci[:,j].tolist(),
                endpoint_mcse=endpoint[:,j].tolist()) for j,(a,b_) in enumerate(contrasts)})
    return result


def baseline(row):
    raw = raw_input(row); N = raw['N_Delta']; dt = raw['Delta2f']
    cut = row['cutoff_hour']; start = raw['count_start_hour']; cnt = counts_for(raw)
    train = cnt[:,start:cut]; test = cnt[:,cut:]; ai,aj = train.sum(axis=1)+.5; A = ai+aj
    z = np.log((N-np.arange(start,cut))/N); future = np.log((N-np.arange(cut,N))/N)
    Z = float((train[0]+train[1])@z)
    scoreA,_,_ = b.forecast(np.array([0.]),np.array([math.log(.5+(cut-start)*dt)]),
                            np.array([0.]),ai,aj,future,test,dt)
    previous = None; converged = False
    # Original 64..512 grid, with declared further refinements if required.
    for order in (64,128,256,512,1024,2048):
        beta,den,lw,diag = b.quadrature_posterior(A,Z,z,dt,order)
        scoreB,_,_ = b.forecast(beta,den,lw,ai,aj,future,test,dt)
        if previous is not None:
            converged = (np.max(abs(scoreB-previous[0]))<1e-5
                and abs(diag['log_integral']-previous[1])<1e-8
                and abs(diag['mean']-previous[2])<1e-7)
            if converged: break
        previous = scoreB.copy(),diag['log_integral'],diag['mean']
    logZ,err0 = b.log_integral(A,Z,z,dt)
    errors=[]
    for j in sorted(set(np.argsort(scoreB)[:3].tolist()+[len(future)//2])):
        ki,kj = test[:,j]; K = int(ki+kj)
        C = gammaln(ai+ki)-gammaln(ai)-gammaln(ki+1)+gammaln(aj+kj)-gammaln(aj)-gammaln(kj+1)
        num,err = b.log_integral(A+K,Z+K*future[j],z,dt,extra_z=float(future[j]))
        errors.append(float(scoreB[j]-(C+K*math.log(dt)+num-logZ)))
    passed = bool(converged and max(map(abs,errors))<1e-4)
    return dict(**row, A_mean_joint_logscore=float(scoreA.mean()),B_mean_joint_logscore=float(scoreB.mean()),
                order=order,converged=bool(converged),independent_score_errors=errors,
                training_integral_relative_error=err0,numerical_score_check_passed=passed)


def posterior_score(row, accepted_path, model):
    """Original 32 chain-block score diagnostic; rare hours remain unresolved.

    Do not reuse historical tail replacements with a newly sampled posterior.
    Parameter convergence alone never accepts the predictive integration.
    """
    from pipeline.prediction.posterior import predictions
    from pipeline.empirical.estimate import read_columns
    from pipeline.config import resolve_manifest_path
    accepted_path = Path(accepted_path).resolve()
    accepted = json.loads(accepted_path.read_text())
    config = accepted['configuration']; raw = raw_input(row)
    if (not accepted['diagnostics']['passed'] or config['model'] != model
            or config['training_cutoff'] != row['cutoff_hour']
            or config['raw_input_sha256'] != row['input_sha256']):
        raise ValueError('Accepted fit must match model, frozen input and training prefix')
    source = ROOT/'pipeline/models'/('free_kernel.stan' if model=='C' else 'leading_order.stan')
    if digest(source) != config['source_model_sha256']:
        raise ValueError('Fit uses a different model source')
    paths = accepted['chain_paths']; hashes = accepted['chain_sha256']
    if len(paths) != 4 or len(hashes) != 4:
        raise ValueError('Expected four hash-checked chains')
    groups=[]; squares=[]; sizes=[]; halves=[]
    for path, expected in zip(paths,hashes,strict=True):
        path=resolve_manifest_path(path,accepted_path)
        if digest(path)!=expected: raise ValueError(f'Changed chain: {path}')
        col=read_columns(path); n=len(col['sigma'])
        if n<8 or n%8: raise ValueError('Original block diagnostic needs eight equal blocks per chain')
        for j, index in enumerate(np.array_split(np.arange(n),8)):
            ll,*_=predictions(raw,{k:v[index] for k,v in col.items()},row['cutoff_hour'],free=model=='C')
            groups.append(logsumexp(ll,axis=0)-math.log(len(index)))
            squares.append(logsumexp(2*ll,axis=0)-math.log(len(index)))
            sizes.append(len(index));halves.append(j//4)
    if len(set(sizes))!=1: raise ValueError('Original diagnostic requires equal chain-block sizes')
    groups=np.asarray(groups);lw=np.log(np.asarray(sizes)/sum(sizes))
    lp=logsumexp(groups+lw[:,None],axis=0)
    ess=sum(sizes)*np.exp(2*lp-logsumexp(np.asarray(squares)+lw[:,None],axis=0))
    influence=np.sum(np.exp(groups-lp[None,:])-1,axis=1)/len(lp)
    mcse=float(np.std(influence,ddof=1)/math.sqrt(32))
    half=[float((logsumexp(groups[np.asarray(halves)==h],axis=0)-math.log(16)).mean()) for h in (0,1)]
    rare=[row['cutoff_hour']+j+1 for j in range(len(lp)) if ess[j]<400 or lp[j]<-20]
    passed=bool(not rare and mcse<=.001 and abs(half[0]-half[1])<=max(.002,4*mcse))
    return dict(**row,model=model,accepted_sha256=digest(accepted_path),
        mean_joint_logscore=float(lp.mean()),mean_logscore_batch_influence_mcse=mcse,
        chain_stratified_half_score_difference=abs(half[0]-half[1]),
        minimum_raw_probability_weight_ess=float(ess.min()),unrepaired_rare_probability_hours=rare,
        mean_score_numerical_screen_passed=passed,
        scope='Fresh fixed-training-posterior score; failed integration stays unresolved; no historical overrides reused')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['summary','baselines','fit','score','integrate','plan'])
    p.add_argument('--contest',type=int);p.add_argument('--split',choices=['primary_hour','complete_day'])
    p.add_argument('--output',default='.artifacts/reproduction/prediction')
    p.add_argument('--model',choices=['C','D'],default='D')
    p.add_argument('--accepted',type=Path)
    p.add_argument('--budgets',type=int,nargs='+',default=[32768,65536,131072,262144,524288,1048576])
    p.add_argument('--seed-offset',type=int,default=0)
    args=p.parse_args(); output=ROOT/args.output; output.mkdir(parents=True,exist_ok=True)
    if args.action=='summary':
        verify_recovered_sources()
        recovered=json.loads((EVIDENCE/'recovered_scores.json').read_text())
        result=summary(recovered['records']); write_json(output/'historical_summary.json',result)
        print(json.dumps(result,indent=2));return
    if args.action=='plan':
        print(json.dumps(splits(args.contest,args.split),indent=2));return
    if args.action in ('score','integrate') and (args.accepted is None or args.contest is None or args.split is None):
        p.error('score/integrate requires --accepted, --contest and --split')
    results=[];baseline_records=[]
    for row in splits(args.contest,args.split):
        if args.action=='integrate':
            from pipeline.prediction.importance import integrate
            result=integrate(row,args.accepted,args.model,output/f"importance_{args.model}_{row['contest_id']}_{row['split']}",
                             args.budgets,args.seed_offset)
            results.append(result['passed']);continue
        if args.action=='fit':
            from pipeline.jk9_fit import fit
            fit(dict(row,scenario='main',mode='early'),output,model_kind=args.model,
                cutoff=row['cutoff_hour'],split=row['split'])
            continue
        result=baseline(row) if args.action=='baselines' else posterior_score(row,args.accepted,args.model)
        if args.action=='baselines':baseline_records.append(result)
        passed=result.get('numerical_score_check_passed',result.get('mean_score_numerical_screen_passed'))
        results.append(passed)
        write_json(output/f"{args.action}_{args.model if args.action=='score' else 'AB'}_{row['contest_id']}_{row['split']}.json",result)
        print(row['contest_id'],row['split'],passed,flush=True)
    if baseline_records:
        references=json.loads((EVIDENCE/'recovered_scores.json').read_text())['records']
        audit=[]
        for r in baseline_records:
            errors={m:r[m+'_mean_joint_logscore']-next(x['mean_joint_logscore'] for x in references
                if x['contest_id']==r['contest_id'] and x['split']==r['split'] and x['model']==m) for m in 'AB'}
            audit.append(dict(contest_id=r['contest_id'],split=r['split'],differences_from_original=errors,
                numerical_score_check_passed=r['numerical_score_check_passed'],quadrature_order=r['order']))
        write_json(output/'baseline_rerun_audit.json',dict(records=audit,all_146_completed=len(audit)==146,
            all_passed=all(results),source_sha256={str(p.relative_to(ROOT)):digest(p) for p in
                [Path(__file__).resolve(),ROOT/'pipeline/prediction/baselines.py']},
            scope='Fresh A/B quadrature and independent adaptive checks; comparison against recovered original contest score means'))
    if not all(results):
        raise RuntimeError('Unresolved baseline integration; retained diagnostics, no ranking acceptance')


if __name__=='__main__':main()
