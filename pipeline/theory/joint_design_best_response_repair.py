"""Reproduce independent best-response audits for the six repaired candidates.

Run with python -m pipeline.theory.joint_design_best_response_repair. Raw
outputs and exact source snapshots are stored separately from paper inputs.
"""

from pipeline.storage import work_path
import json,hashlib,concurrent.futures
from pathlib import Path
from pipeline.theory.joint_design_best_response_audit import audit
from pipeline.config import ROOT
SOURCE=work_path(ROOT/'results/theory/numerical_joint_plane_conservative_repair/8a63cc9c1fc1')
OUT=work_path(ROOT/'results/theory/numerical_joint_plane_best_response_repair')
def job(case):
 results=[]
 for n,steps in [(1601,2000),(3201,2000),(3201,4000)]:
  try:r=audit(case['sigma'],case['alpha'],case['T'],case['lambda'],nodes=n,steps=steps)
  except Exception as exc:r={'error':str(exc)}
  results.append(dict(r,nodes=n,steps=steps))
 return dict(case=case,meshes=results)
def main():
 signature={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'pipeline/theory/joint_design_best_response_audit.py',ROOT/'pipeline/theory/joint_design_conservative_audit.py',Path(__file__)]}
 key=hashlib.sha256(json.dumps(signature,sort_keys=True).encode()).hexdigest()[:12]
 folder=OUT/key;folder.mkdir(parents=True,exist_ok=True);(folder/'configuration.json').write_text(json.dumps(signature,indent=2))
 cases=[d['case'] for p in SOURCE.glob('*.json') if 'case' in (d:=json.loads(p.read_text()))]
 for source in [ROOT/'pipeline/theory/joint_design_best_response_audit.py',ROOT/'pipeline/theory/joint_design_conservative_audit.py',Path(__file__)]:
  (folder/source.name).write_bytes(source.read_bytes())
 records=[]
 with concurrent.futures.ProcessPoolExecutor(max_workers=2) as pool:
  futures=[pool.submit(job,c) for c in cases]
  for future in concurrent.futures.as_completed(futures):
   r=future.result();records.append(r)
   name=hashlib.sha256(json.dumps(r['case'],sort_keys=True).encode()).hexdigest()[:16]
   (folder/(name+'.json')).write_text(json.dumps(r,indent=2,allow_nan=False))
   print(json.dumps({'completed':len(records),'case':r['case'],'diagnostics':[{k:m[k] for k in ['nodes','steps','deviation_gain','max_deviation_gain','max_subgame_deviation_gain','error'] if k in m} for m in r['meshes']]}),flush=True)
 valid=[r for r in records if all('error' not in m for m in r['meshes'])]
 summary=dict(cases=len(records),completed_cases=len(valid),
  maximum_finest_subgame_deviation_gain=max((r['meshes'][-1]['max_subgame_deviation_gain'] for r in valid),default=None),
  maximum_finest_deviation_gain=max((r['meshes'][-1]['max_deviation_gain'] for r in valid),default=None),
  maximum_audit_time_refinement_change=max((abs(r['meshes'][-1]['max_deviation_gain']-r['meshes'][-2]['max_deviation_gain']) for r in valid),default=None),
  all_spatial_deviation_gains_reduced=all(r['meshes'][1]['max_deviation_gain']<r['meshes'][0]['max_deviation_gain'] for r in valid),
  global_classification_verified=False,results=records)
 (folder/'summary.json').write_text(json.dumps(summary,indent=2))
 print('SAVED',folder,flush=True)
if __name__=='__main__':main()
