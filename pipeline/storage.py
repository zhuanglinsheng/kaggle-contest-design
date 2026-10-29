"""Repository storage roles: retained evidence versus disposable work products.

Only explicit publication commands promote outputs into results/. Computation,
compilation, reconstructed working inputs and caches belong below tmp/.
"""
from pathlib import Path
import json
import shutil

ROOT = Path(__file__).resolve().parents[1]
TMP = ROOT / 'tmp'
RUNS = TMP / 'runs'
CACHE = TMP / 'cache'
BUILD = TMP / 'build'
HISTORY = TMP / 'history'
WORK_RESULTS = TMP / 'work' / 'results'


def historical_path(value):
    """Locate moved historical files without modifying frozen record bytes."""
    path=Path(value)
    if not path.is_absolute():path=ROOT/path
    if path.exists():return path
    try:relative=path.relative_to(ROOT)
    except ValueError:
        parts=path.parts
        for marker in ('docs','results','metakaggle','synthetic_data'):
            if marker in parts:
                path=ROOT/Path(*parts[parts.index(marker):]);relative=path.relative_to(ROOT);break
        else:raise ValueError(f'Unrecognized historical path: {value}')
    candidate=HISTORY/relative
    if candidate.exists():return candidate
    # Some retained source snapshots and reconstructed inputs have role-specific
    # destinations rather than the generic historical-work prefix.
    inventory=ROOT/'docs/storage_layout.json'
    if inventory.exists():
        moves=json.loads(inventory.read_text())['moves']
        for move in sorted(moves,key=lambda r:len(r['source']),reverse=True):
            try:tail=relative.relative_to(move['source'])
            except ValueError:continue
            destination=ROOT/move['destination']/tail
            if destination.exists():return destination
    return path


def work_path(value):
    """Fresh work output corresponding to a historical results/... path."""
    value=Path(value)
    if value.is_absolute():value=value.relative_to(ROOT)
    return TMP/'work'/value


def stage_stan(source):
    """Stage source beside its platform build, never beside maintained code."""
    source=Path(source).resolve()
    relative=source.relative_to(ROOT)
    if relative.parts[0]=='tmp':return source
    destination=BUILD/'models'/relative
    destination.parent.mkdir(parents=True,exist_ok=True)
    if not destination.exists() or destination.read_bytes()!=source.read_bytes():
        shutil.copy2(source,destination)
    return destination


def compile_stan(source):
    from cmdstanpy import CmdStanModel
    return CmdStanModel(stan_file=str(stage_stan(source)),cpp_options={'PRECOMPILED_HEADERS':'false'})
