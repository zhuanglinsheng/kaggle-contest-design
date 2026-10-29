"""Resume independent weak-feedback candidate repairs without changing the paper.

Run as a module. Saves source-signed raw mesh results separately from the old
classifications. Passing a candidate audit never assigns a parameter-cell class.
"""
from __future__ import annotations

from pipeline.storage import work_path

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path

import pandas as pd

from pipeline.config import ROOT
from pipeline.theory.joint_design_conservative_audit import solve


SOURCE=work_path(ROOT/'results/theory/numerical_joint_plane_moderate_dense/verified_policy_samples.csv')
OUT=work_path(ROOT/'results/theory/numerical_joint_plane_conservative_repair')


def audit_one(job):
    case,nodes,stretch,max_calls=job
    results=[]
    for n in nodes:
        try:
            result=solve(case['sigma'],case['alpha'],case['T'],case['lambda'],
                         nodes=n,stretch=stretch,rtol=2e-7,atol=2e-9,
                         max_calls=max_calls)
        except (RuntimeError,ValueError,FloatingPointError) as exc:
            result=dict(nodes=n,error=str(exc),mesh_checks_passed=False)
        results.append(result)
    # A failed finest mesh cannot be replaced by an earlier successful one.
    accepted=results[-2:] if all(r.get('mesh_checks_passed') for r in results[-2:]) else []
    record=dict(case=case,meshes=results,
                supported_scope='Candidate mesh/payoff checks; no global classification certificate')
    if len(accepted)>=2:
        delta=abs(accepted[-1]['B']-accepted[-2]['B'])
        record.update(two_mesh_checks_passed=True,B=accepted[-1]['B'],
                      estimated_output_error=3*delta,
                      refinement_difference=delta)
    else:
        record['two_mesh_checks_passed']=False
    return record


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--workers',type=int,default=3)
    parser.add_argument('--nodes',type=int,nargs='+',default=[801,1601,3201])
    parser.add_argument('--stretch',type=float,default=18.)
    parser.add_argument('--max-calls',type=int,default=12000)
    parser.add_argument('--min-precision',type=float,default=0.,help='Exclusive lower precision bound')
    parser.add_argument('--max-precision',type=float,default=.01)
    args=parser.parse_args()
    if not 0<=args.min_precision<args.max_precision:
        parser.error('Require 0 <= min-precision < max-precision')
    if len(args.nodes)<2 or args.nodes!=sorted(set(args.nodes)):
        parser.error('Provide at least two distinct, increasing mesh sizes')
    OUT.mkdir(parents=True,exist_ok=True)
    digest=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    signature=dict(input_sha256=digest(SOURCE),solver_sha256=digest(Path(__file__).with_name('joint_design_conservative_audit.py')),
                   runner_sha256=digest(Path(__file__)),nodes=args.nodes,
                   stretch=args.stretch,max_calls=args.max_calls,min_precision=args.min_precision,max_precision=args.max_precision)
    key=hashlib.sha256(json.dumps(signature,sort_keys=True).encode()).hexdigest()[:12]
    folder=OUT/key
    folder.mkdir(exist_ok=True)
    (folder/'configuration.json').write_text(json.dumps(signature,indent=2)+'\n')
    data=pd.read_csv(SOURCE)
    failed=data[data.B.isna() & (data['lambda']>args.min_precision) & (data['lambda']<=args.max_precision)]
    failed=failed.sort_values(['lambda','sigma'],ascending=[False,True])
    pending=[];records=[]
    for row in failed.to_dict('records'):
        case={k:float(row[k]) for k in ['sigma','alpha','T','lambda']}
        name=hashlib.sha256(json.dumps(case,sort_keys=True).encode()).hexdigest()[:16]+'.json'
        path=folder/name
        if path.exists():
            records.append(json.loads(path.read_text()))
        else:
            pending.append((path,(case,args.nodes,args.stretch,args.max_calls)))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures={pool.submit(audit_one,job):path for path,job in pending}
        for future in as_completed(futures):
            result=future.result();records.append(result)
            futures[future].write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
            print(json.dumps(dict(completed=len(records),total=len(failed),case=result['case'],
                  two_mesh_checks_passed=result['two_mesh_checks_passed'],
                  meshes=[{k:r[k] for k in ['nodes','B','global_residual','mesh_checks_passed','error'] if k in r} for r in result['meshes']])),flush=True)
    summary=dict(candidates=len(records),two_mesh_checks_passed=sum(r['two_mesh_checks_passed'] for r in records),
                 unresolved=sum(not r['two_mesh_checks_passed'] for r in records),
                 note='Original classifications and paper figures unchanged. Output-error estimates are mesh diagnostics, not rigorous PDE bounds.')
    (folder/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary),flush=True)


if __name__=='__main__':
    main()
