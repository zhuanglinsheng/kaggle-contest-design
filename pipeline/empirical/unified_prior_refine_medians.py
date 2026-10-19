"""Keep the long NUTS refinement; use the declared primary-median gates."""
import argparse
import json
import time
from pipeline.empirical import unified_prior_refine as refinement
from pipeline.empirical import unified_prior_medians as medians
from pipeline.empirical import unified_prior_auxiliary as auxiliary

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('contest_id',type=int);parser.add_argument('--mode',choices=('full','early'),default='full');parser.add_argument('--scenario',choices=auxiliary.SCENARIOS);args=parser.parse_args()
    if args.scenario:
        refinement.u.OUT=auxiliary.BASE_OUT/'auxiliary'/args.scenario
        refinement.u.assemble=auxiliary.assemble
    refinement.batch.diagnose=medians.diagnose
    case=refinement.u.OUT/'estimation'/args.mode/f'contest_{args.contest_id}'
    candidate=case/'global_mh_candidate.json' if (case/'global_mh_candidate.json').exists() else case/'metric_candidate.json'
    if candidate.exists() and not (case/'accepted.json').exists():
        value=json.loads(candidate.read_text())
        if not value['diagnostics']['passed'] or value['diagnostics'].get('median_protocol_sha256')!=refinement.u.digest(medians.__file__):raise RuntimeError('Invalid metric candidate')
        if any(refinement.u.digest(p)!=h for p,h in zip(value['chain_paths'],value['chain_sha256'])):raise RuntimeError('Changed metric candidate')
        for _ in range(30):
            if (case/'accepted.json').exists():break
            time.sleep(1)
        if not (case/'accepted.json').exists():refinement.u.save(case/'accepted.json',value)
        refinement.u.assemble()
    else:
        refinement.run(args.contest_id,args.mode)
