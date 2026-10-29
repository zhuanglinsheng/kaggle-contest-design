"""Losslessly compact numerical caches; keep archive inventories outside Git.

Only known cache directories are selected. Results, posterior chains, seeds and
source code are never selected. Each archive is independently read and checked
before any source directory is removed. Interrupted archives never authorize
deletion. Restore verifies the archive and refuses to overwrite changed files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tarfile

ROOT = Path(__file__).resolve().parents[1]
CACHE_NAMES = {'coarse_cache', 'fine_cache', 'point_cache', 'fine_recovery',
               'cross_coordinate_recovery'}
BASE = Path('tmp/history/docs/review/PaperJK9/nonlinear_table_rebuild_20261025/contests')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def record_hash(total, name, content_hash):
    total.update(name.encode() + b'\0' + content_hash.encode() + b'\n')


def inventory(archive):
    total = hashlib.sha256()
    count = size = 0
    with tarfile.open(archive, 'r|gz') as tar:
        for member in tar:
            name = Path(member.name)
            if name.is_absolute() or '..' in name.parts or not member.isfile():
                raise ValueError(f'Unsafe archive member: {member.name}')
            h = hashlib.sha256()
            with tar.extractfile(member) as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b''):
                    h.update(block)
            record_hash(total, member.name, h.hexdigest())
            count += 1
            size += member.size
    return dict(files=count, bytes=size, tree_sha256=total.hexdigest())


def save(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def pack(root, directories, archive, *, prune=False):
    root, archive = Path(root).resolve(), Path(archive).resolve()
    directories = sorted(Path(p).resolve() for p in directories)
    for folder in directories:
        folder.relative_to(root)
        if folder.is_symlink() or not folder.is_dir():
            raise ValueError(f'Not a real cache directory: {folder}')
        if archive.is_relative_to(folder):
            raise ValueError('Archive must be outside source directories')
    manifest = archive.with_suffix(archive.suffix + '.json')
    if archive.exists() or manifest.exists():
        raise FileExistsError(archive)
    archive.parent.mkdir(parents=True, exist_ok=True)
    temporary = archive.with_suffix(archive.suffix + '.partial')
    total = hashlib.sha256()
    count = size = 0
    with tarfile.open(temporary, 'w|gz', compresslevel=1) as tar:
        for folder in directories:
            for current, subdirs, files in os.walk(folder):
                subdirs.sort()
                if any((Path(current) / d).is_symlink() for d in subdirs):
                    raise ValueError('Symlink in cache')
                for filename in sorted(files):
                    path = Path(current) / filename
                    if path.is_symlink() or not path.is_file():
                        raise ValueError(f'Not a regular cache file: {path}')
                    name = path.relative_to(root).as_posix()
                    info = tar.gettarinfo(str(path), arcname=name)
                    # Read the same bytes once for the archive and fingerprint.
                    with path.open('rb') as stream:
                        content = stream.read()
                    import io
                    tar.addfile(info, io.BytesIO(content))
                    record_hash(total, name, hashlib.sha256(content).hexdigest())
                    count += 1
                    size += len(content)
    expected = dict(files=count, bytes=size, tree_sha256=total.hexdigest())
    if inventory(temporary) != expected:
        raise ValueError('Archive failed independent byte verification')
    temporary.replace(archive)
    result = dict(schema_version=1, archive=archive.name,
                  archive_sha256=digest(archive), archive_bytes=archive.stat().st_size,
                  roots=[p.relative_to(root).as_posix() for p in directories],
                  **expected, originals_removed=False)
    save(manifest, result)
    if prune:
        for folder in directories:
            shutil.rmtree(folder)
        result['originals_removed'] = True
        save(manifest, result)
    return result


def restore(root, archive):
    root, archive = Path(root).resolve(), Path(archive)
    manifest = json.loads(archive.with_suffix(archive.suffix + '.json').read_text())
    if digest(archive) != manifest['archive_sha256']:
        raise ValueError('Archive checksum mismatch')
    if inventory(archive) != {k: manifest[k] for k in ('files', 'bytes', 'tree_sha256')}:
        raise ValueError('Archive content mismatch')
    with tarfile.open(archive, 'r|gz') as tar:
        for member in tar:
            target = root / member.name
            target.resolve().relative_to(root)
            content = tar.extractfile(member).read()
            if target.exists():
                if target.is_symlink() or target.read_bytes() != content:
                    raise FileExistsError(f'Refusing to overwrite: {target}')
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(target.suffix + '.restore-tmp')
            temporary.write_bytes(content)
            temporary.replace(target)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['pack', 'restore'])
    parser.add_argument('--store', type=Path, default=ROOT / 'tmp/cache_archives')
    parser.add_argument('--prune', action='store_true')
    parser.add_argument('--contest', type=int)
    args = parser.parse_args()
    # Keep the existing computation controllers stopped during compaction.
    import fcntl
    locks = []
    for name in ('design_controller.lock', 'refinement_controller.lock'):
        path = ROOT / BASE.parent / name
        if path.exists():
            handle = path.open('r')
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            locks.append(handle)
    if args.action == 'restore':
        archives = sorted(args.store.glob(f'{args.contest or "*"}.tar.gz'))
        if not archives:
            raise FileNotFoundError('No matching cache archives')
        for archive in archives:
            meta = json.loads(archive.with_suffix(archive.suffix+'.json').read_text())
            destination = ROOT if all(p.startswith('tmp/') for p in meta['roots']) else ROOT/'tmp/history'
            result = restore(destination, archive)
            print(f'Restored {archive.name}: {result["files"]} files', flush=True)
        return
    folders = sorted((ROOT / BASE).glob(str(args.contest) if args.contest else '*'))
    for folder in folders:
        directories = [p for p in folder.iterdir() if p.is_dir() and p.name in CACHE_NAMES]
        if not directories:
            continue
        result = pack(ROOT, directories, args.store / f'{folder.name}.tar.gz', prune=args.prune)
        print(f'{folder.name}: {result["files"]} files, {result["bytes"]} -> '
              f'{result["archive_bytes"]} bytes; pruned={result["originals_removed"]}', flush=True)


if __name__ == '__main__':
    main()
