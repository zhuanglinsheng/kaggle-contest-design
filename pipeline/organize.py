"""Move intermediate work products into tmp/ without deleting or compressing.

The release lock is the initial authoritative allowlist. Source/method files,
release records and explicitly retained final summaries stay outside tmp/.
Directory renames preserve large point-cache trees without rewriting each file.
"""
import argparse
import json
from pathlib import Path
import os

from pipeline.storage import ROOT,TMP
from pipeline.publication import write_json, load, local

RELEASE=ROOT/'results/releases/paperjk9-20261029'
RELOCATION_MANIFEST=ROOT/'docs/storage_layout.json'


def protected_files():
    # Current publication files and the historical evidence lock are both retained.
    locked=json.loads((RELEASE/'manifest.json').read_text())['files']
    selector=ROOT/'current_release.toml'
    if selector.exists():
        current=local(ROOT,load(ROOT)['manifest'])
        locked={**locked,**json.loads(current.read_text())['files']}
    keep={ROOT/p for p in locked}
    for p in (ROOT/'results/releases').rglob('*'):
        if p.is_file():keep.add(p)
    for group in ('richer_steady','initialization_comparison'):
        for p in (ROOT/'results/synthetic'/group).rglob('*'):
            if p.is_file() and (p.name.endswith('summary.json') or p.name in ('design.json','comparison.json')):keep.add(p)
    for folder in ('docs/review','results','archive/research'):
        for base,dirs,files in os.walk(ROOT/folder):
            dirs[:]=[d for d in dirs if d not in {'__pycache__','build','coarse_cache','fine_cache','point_cache','fine_recovery','cross_coordinate_recovery'}]
            for name in files:
                p=Path(base)/name
                if (p.suffix in {'.py','.stan','.cpp'} or p.name=='.gitignore'
                        or (p.suffix=='.md' and folder!='results')):keep.add(p)
                if folder=='docs/review' and p.suffix=='.txt' and not p.name.startswith(('log_','chain_')):keep.add(p)
                if folder=='archive/research' and p.suffix=='.txt':keep.add(p)
                if folder=='docs/review' and p.parent.name=='cleanup_20261027' and p.name in {'manifest.json','verification.json'}:keep.add(p)
    keep.add(ROOT/'results/README.md')
    return {p for p in keep if p.exists() and p.is_file()}


def relocation_plan():
    keep=protected_files();parents={p for f in keep for p in f.parents}
    moves=[]
    def add(source,destination,role):
        if source.exists():moves.append(dict(source=source.relative_to(ROOT).as_posix(),
            destination=destination.relative_to(ROOT).as_posix(),role=role))
    def walk(folder):
        if not folder.exists():return
        for p in sorted(folder.iterdir()):
            if p.is_symlink():raise ValueError(f'Symlink requires explicit classification: {p}')
            if p.is_dir() and (p in parents or (TMP/'history'/p.relative_to(ROOT)).exists()):walk(p)
            elif p in keep:continue
            else:add(p,TMP/'history'/p.relative_to(ROOT),'superseded_or_intermediate_work')
    for folder in ('docs/review','results','archive/research'):walk(ROOT/folder)
    for old,new,role in [
        ('.cache','tmp/cache/legacy_dotcache','cache_and_build'),
        ('.artifacts/caches','tmp/cache_archives','legacy_cache_archive'),
        ('.artifacts/reproduction','tmp/runs','regenerated_work'),
        ('.artifacts/recovered_prediction','tmp/recovered/prediction_scores','historical_hourly_scores'),
        ('.artifacts/recovered_prediction_bundle','tmp/recovered/prediction_bundle','historical_candidate_chains'),
        ('metakaggle/__jsondata__','tmp/inputs/main','reconstructed_working_inputs'),
        ('metakaggle/__jsondata_robust23__','tmp/inputs/robustness','reconstructed_working_inputs')]:add(ROOT/old,ROOT/new,role)
    for folder in ('pipeline/models','metakaggle','synthetic_data'):
        for p in (ROOT/folder).iterdir():
            if p.is_file() and (p.suffix=='.hpp' or (not p.suffix and p.name!='LICENSE')):
                add(p,TMP/'build/legacy'/p.relative_to(ROOT),'compiled_model')
    for p in (ROOT/'paper').iterdir():
        if p.is_file() and (p.suffix in {'.aux','.log','.out','.fls','.fdb_latexmk','.blg','.toc'} or p.name.endswith('.synctex.gz')):
            add(p,TMP/'latex/legacy'/p.name,'latex_auxiliary')
    # Bytecode is work product too. Avoid traversing cache forests or environments.
    occupied={ROOT/r['source'] for r in moves if (ROOT/r['source']).is_dir()}
    for base,dirs,files in os.walk(ROOT):
        dirs[:]=[d for d in dirs if d not in {'.git','.venv','.venv-jk9','tmp'} and Path(base)/d not in occupied]
        if '__pycache__' in dirs:
            p=Path(base)/'__pycache__';add(p,TMP/'cache/old_bytecode'/p.relative_to(ROOT),'python_bytecode');dirs.remove('__pycache__')
    return moves


def apply(moves):
    for row in moves:
        source=ROOT/row['source'];destination=ROOT/row['destination']
        if not source.exists():raise FileNotFoundError(source)
        if destination.exists():raise FileExistsError(f'Never merge or overwrite work products: {destination}')
    for row in moves:
        source=ROOT/row['source'];destination=ROOT/row['destination']
        destination.parent.mkdir(parents=True,exist_ok=True);source.rename(destination)
    old=ROOT/'.artifacts'
    if old.exists() and not any(old.iterdir()):old.rmdir()
    return len(moves)


def audit():
    from pipeline import config
    from pipeline.jk9_fit import plan
    moves=relocation_plan()
    if moves:
        raise ValueError(f'{len(moves)} unclassified work-product roots remain outside tmp/: {moves[:3]}')
    for path in (config.MAIN_DATA_DIR,config.ROBUSTNESS_DATA_DIR,config.RESULTS_DIR,config.CACHE_DIR,
                 config.EMPIRICAL_MODEL,config.SYNTHETIC_SINGLE_MODEL,config.SYNTHETIC_POOLED_MODEL):
        if not path.is_relative_to(TMP):raise ValueError(f'Work output escapes tmp/: {path}')
    index=json.loads((RELEASE/'accepted_fits.json').read_text())
    if len(index['fits'])!=73:raise ValueError('73 publication posterior records required')
    for fit in index['fits']:
        if fit['accepted_path'].startswith('tmp/') or not (ROOT/fit['accepted_path']).is_file():
            raise ValueError('Formal posterior provenance must be retained outside tmp/')
        if not all(p.startswith('tmp/') for p in fit['chain_paths']):
            raise ValueError('Generated posterior chains must be in tmp/')
    rows=plan('main','full')
    print(f'PASS: no classified intermediates outside tmp/; {len(rows)} frozen main inputs and 73 formal posterior records remain retained.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    choices=parser.add_mutually_exclusive_group()
    choices.add_argument('--apply',action='store_true')
    choices.add_argument('--audit',action='store_true')
    args=parser.parse_args()
    if args.audit:audit();return
    moves=relocation_plan()
    if args.apply:
        count=apply(moves)
        prior=json.loads(RELOCATION_MANIFEST.read_text()) if RELOCATION_MANIFEST.exists() else dict(moves=[])
        prior.update(schema_version=1,scope='Role-based storage relocation; rename only, no deletion, compression, model changes or scientific rerun',
            tmp_policy='All regenerable runs, chains, hourly scores, caches, builds and superseded work are local under tmp/. Retained current inputs, source and final evidence remain outside tmp/.',
            moves=prior['moves']+moves)
        write_json(RELOCATION_MANIFEST,prior)
        print(f'Moved {count} work-product roots to tmp/; manifest: docs/storage_layout.json')
    else:print(json.dumps(dict(moves=moves,roots=len(moves)),indent=2))


if __name__=='__main__':main()
