"""Run frozen JK9 nonlinear controllers in an isolated, relocatable checkout.

Historical records stay byte-for-byte unchanged. A runtime adapter resolves
chain paths and records the original provenance before invoking the controllers.
Fresh results retain unresolved comparisons; old repair certificates are not
implicitly applied to a different run.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path

from pipeline.publication import ROOT, digest, write_json
from pipeline.jk9 import RELEASE, TABLE, relocated
from pipeline.jk9_fit import plan
from pipeline.config import resolve_manifest_path


def load_controller(name):
    spec=importlib.util.spec_from_file_location('jk9_'+name,TABLE/(name+'.py'))
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def runtime_accepted(cid, output, accepted=None):
    """Resolve/check all four chains; never rewrite the original accepted record."""
    if accepted is None:
        entry=next(r for r in json.loads((RELEASE/'accepted_fits.json').read_text())['fits'] if r['contest_id']==cid)
        original=ROOT/entry['accepted_path']
        if digest(original)!=entry['accepted_sha256']:
            raise ValueError('Original accepted record changed')
        record=json.loads(original.read_text())
        paths=[ROOT/p for p in entry['chain_paths']]
        if record['chain_sha256']!=entry['chain_sha256']:
            raise ValueError('Index and accepted chain identities differ')
    else:
        original=Path(accepted).resolve();record=json.loads(original.read_text())
        paths=[resolve_manifest_path(p,original) for p in record['chain_paths']]
    if not record['diagnostics']['passed'] or len(paths)!=4 or len(record['chain_sha256'])!=4:
        raise ValueError('Expected four diagnostically accepted chains')
    for path,expected in zip(paths,record['chain_sha256'],strict=True):
        if digest(path)!=expected:raise ValueError(f'Changed chain: {path}')
    raw=plan('main','full',cid)[0]
    if (record['configuration'].get('model','D') != 'D'
            or record['configuration'].get('source_model_sha256') != digest(ROOT/'pipeline/models/leading_order.stan')):
        raise ValueError('Design calculations require the retained leading-order posterior target')
    if record['configuration']['raw_input_sha256']!=raw['input_sha256']:
        raise ValueError('Posterior and frozen main input do not match')
    if record['configuration'].get('training_cutoff',raw_input_N(raw))!=raw_input_N(raw):
        raise ValueError('Design calculations require the full-window posterior')
    record=dict(record,chain_paths=[str(p.resolve()) for p in paths],
                runtime_source=dict(path=str(original),sha256=digest(original),
                    scope='Ephemeral path adapter; original accepted bytes preserved'))
    destination=output/'runtime_posteriors'/f'{cid}.json'
    if destination.exists() and json.loads(destination.read_text())!=record:
        raise ValueError('Runtime input changed; use a new output directory')
    write_json(destination,record)
    return destination


def raw_input_N(row):
    return json.loads((ROOT/row['input_snapshot']).read_text())['N_Delta']


def controllers(output):
    batch=load_controller('design_batch');batch.root=ROOT;batch.report=output;batch.out=output
    refine=load_controller('refine_prizes')
    # RefinedCase inherits its own imported batch.Case; configure that instance.
    refine.batch.root=ROOT;refine.batch.report=output;refine.batch.out=output;refine.HERE=output
    assemble=load_controller('assemble_refined');assemble.HERE=output
    fixed=load_controller('fixed_precision_table');fixed.HERE=output;fixed.DEST=output/'fixed_precision'
    return batch,refine,assemble,fixed


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['plan','quadrature','screen','refine','assemble','fixed','all'])
    pick=p.add_mutually_exclusive_group(required=True)
    pick.add_argument('--contest',type=int);pick.add_argument('--all',action='store_true')
    p.add_argument('--accepted',type=Path,help='Fresh full-window accepted.json; one contest only')
    p.add_argument('--accepted-root',type=Path,help='Fresh fit output root containing main/full/<cid>/accepted.json')
    p.add_argument('--output',type=Path,default=Path('tmp/runs/design'))
    args=p.parse_args()
    if args.accepted and (args.contest is None or args.accepted_root):p.error('--accepted requires one contest and no --accepted-root')
    output=args.output.resolve()
    if args.accepted:args.accepted=args.accepted.resolve()
    if args.accepted_root:args.accepted_root=args.accepted_root.resolve()
    if output==TABLE or TABLE in output.parents:p.error('Use a fresh output directory outside historical evidence')
    rows=plan('main','full',args.contest)
    if args.stage=='plan':
        print(json.dumps(dict(contests=[r['contest_id'] for r in rows],output=str(output),
            stages=['screen','refine','assemble','fixed'],cache_mode=os.getenv('JK9_CACHE_MODE','sqlite'),
            scope='New nonlinear calculations; no old comparator certificate is transplanted'),indent=2));return
    output.mkdir(parents=True,exist_ok=True)
    # Frozen controllers use source-relative AST/hash checks and a few relative
    # source paths. Anchor these to the checkout, independently of caller cwd.
    os.chdir(ROOT)
    from pipeline.jk9 import main as jk9_main
    jk9_main(['prepare-inputs'])
    batch,refine,assemble,fixed=controllers(output)
    for row in rows:
        cid=row['contest_id']; folder=output/'contests'/str(cid)
        accepted=args.accepted or (args.accepted_root/'main/full'/str(cid)/'accepted.json' if args.accepted_root else None)
        runtime=runtime_accepted(cid,output,accepted)
        stages=['screen','refine','assemble','fixed'] if args.stage=='all' else [args.stage]
        for stage in stages:
            if stage=='quadrature':batch.Case(cid,runtime)
            elif stage=='screen':
                if (folder/'result.json').exists():raise FileExistsError('Screen already completed; select refine/assemble/fixed to resume')
                batch.Case(cid,runtime).run()
            elif stage=='refine':refine.refine_case(folder/'result.json')
            elif stage=='assemble':assemble.assemble(folder)
            elif stage=='fixed':
                if not (folder/'quadrature.json').exists():batch.Case(cid,runtime)
                fixed.compute(cid)
            print(cid,stage,'completed; inspect retained numerical acceptance flags',flush=True)


if __name__=='__main__':main()
