"""Preserve recoverable historical prediction blobs outside the Git worktree index.

This reads Git objects only. It never changes refs, stages files, runs garbage
collection or rewrites history. Recovered blob identities are checked twice.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from pipeline.publication import ROOT,write_json

TREES={
 'prediction_reconstruction_20261025':'d48028fb5f2f9f35d44f65dcc421d8e56f045212',
 'prediction_precision_20261025':'57b092b12d21d3dfe1c12938dc1ce92a2564dadb',
 'prediction_score_snapshots_20261025':'be9862f1669961d46664ca9e0e451117c3678de6',
}


def recover(output,inventory_only=False):
    output=Path(output).resolve();rows=[];objects={}
    for name,tree in TREES.items():
        listing=subprocess.check_output(['git','ls-tree','-r',tree],cwd=ROOT,text=True)
        for line in listing.splitlines():
            metadata,path=line.split('\t',1);mode,kind,blob=metadata.split()
            if kind!='blob' or Path(path).suffix not in {'.csv','.txt','.stan','.cpp','.py'}:continue
            rows.append(dict(original_path=name+'/'+path,git_tree=tree,git_blob=blob))
            objects[blob]=None
    if inventory_only:return dict(files=rows,unique_blobs=len(objects),restored=False)
    folder=output/'objects';folder.mkdir(parents=True,exist_ok=True)
    proc=subprocess.Popen(['git','cat-file','--batch'],cwd=ROOT,stdin=subprocess.PIPE,stdout=subprocess.PIPE)
    try:
        for blob in objects:
            proc.stdin.write((blob+'\n').encode());proc.stdin.flush()
            header=proc.stdout.readline().decode().split()
            if len(header)!=3 or header[1]!='blob':raise ValueError(f'Unavailable historical blob: {blob}')
            size=int(header[2]);payload=proc.stdout.read(size)
            if len(payload)!=size or proc.stdout.read(1)!=b'\n':raise ValueError('Truncated Git object stream')
            if hashlib.sha1(b'blob '+str(size).encode()+b'\0'+payload).hexdigest()!=blob:raise ValueError('Git identity mismatch')
            sha=hashlib.sha256(payload).hexdigest();destination=folder/blob
            if destination.exists():
                if hashlib.sha256(destination.read_bytes()).hexdigest()!=sha:raise ValueError(f'Refusing to replace changed recovery: {destination}')
            else:destination.write_bytes(payload)
            objects[blob]=dict(sha256=sha,bytes=size,artifact_path=str(destination.relative_to(output)))
    finally:
        proc.stdin.close();proc.wait()
    for row in rows:row.update(objects[row['git_blob']])
    result=dict(files=rows,unique_blobs=len(objects),unique_bytes=sum(r['bytes'] for r in objects.values()),
        restored=True,scope='Content inventory of recovered text, chain and score files; local bundle root supplied separately; fitting and integration records are produced by their calculation stages')
    write_json(output/'manifest.json',result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ROOT/'tmp/recovered/prediction_bundle')
    p.add_argument('--inventory-only',action='store_true');args=p.parse_args()
    result=recover(args.output,args.inventory_only)
    print(json.dumps({k:v for k,v in result.items() if k!='files'},indent=2))


if __name__=='__main__':main()
