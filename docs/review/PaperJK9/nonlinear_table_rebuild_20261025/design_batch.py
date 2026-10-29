"""Resumable nonlinear table calculation; keep original Bayesian posteriors.

Screening and multilevel validation both solve the original nonlinear game.
No first-order output or prize inversion enters any quality calculation.
"""
from pathlib import Path
import concurrent.futures as futures
import ast,csv,hashlib,json,math,os,time,traceback
import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from pipeline.config import MAIN_DATA_DIR
from pipeline import solver_cache
from pipeline.empirical.counterfactual import PRIZE_DIVISOR
from pipeline.empirical.nonlinear_design import Draw,Design,Resolution,checked_output
from pipeline.empirical.nonlinear_value_stream import checked_output
from pipeline.empirical.nonlinear_design_similarity import SimilarityOutput
root=Path.cwd();report=root/'docs/review/PaperJK9';out=report/'nonlinear_table_rebuild_20261025';out.mkdir(exist_ok=True);DELTAS=(.1,.25,.5,1.,2.,5.,10.,25.,100.);TIMES=(.1,.25,.5,.75,1.,1.25,1.5,2.);PRIZES=(.125,.25,.5,1.,2.,4.,8.,16.,32.,64.,128.)
ids=sorted({int(r['contest_id']) for r in csv.DictReader((root/'results/reestimation/lambda100_sd225/estimation/accepted_chain_sources.csv').open()) if r['mode']=='full'})
def save(p,d):
 if solver_cache.is_cache(p):
  solver_cache.put(p,d);return
 p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(d,indent=2,allow_nan=False));tmp.replace(p)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def validation_protocol():
 tree=ast.parse(Path(__file__).read_text());node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Case');methods={n.name:n for n in node.body if isinstance(n,ast.FunctionDef)};selected=[]
 class FailureBookkeeping(ast.NodeTransformer):
  def visit_If(self,node):
   node=self.generic_visit(node)
   return node if node.body else None
  def visit_Return(self,node):
   value=node.value
   if isinstance(value,ast.Call) and isinstance(value.func,ast.Attribute) and isinstance(value.func.value,ast.Name) and value.func.value.id=='self' and value.func.attr=='store_unresolved':return None
   return node
 for name in ['native','fine_output','validate']:
  method=methods[name]
  # The final four validation statements only store/log/return the result.
  # Fingerprint the calculation separately from controller/resume plumbing.
  if name=='validate':
   method.body=method.body[:-4]
   # Returning as soon as a failed draw is found changes only failure
   # bookkeeping. Every successful paired calculation remains identical.
   method=FailureBookkeeping().visit(method)
  selected.append(ast.dump(method,include_attributes=False))
 files=['pipeline/empirical/nonlinear_design.py','pipeline/empirical/nonlinear_design_similarity.py','pipeline/theory/nonlinear_similarity_kernel.cpp','pipeline/theory/joint_design_exact.py','pipeline/theory/asymmetric_full_output.py','pipeline/synthetic/nonlinear.py','pipeline/empirical/nonlinear_value_stream.py','pipeline/theory/nonlinear_value_stream.cpp']
 identity=dict(computational_methods=selected,dependencies={f:sha(root/f) for f in files})
 return hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
def available(cid):
 for p in [report/f'reconstruction_20261025/estimation/full/contest_{cid}/accepted.json',out/f'posterior/contest_{cid}/accepted.json']:
  if p.exists():return p
 return None
class Case:
 def __init__(self,cid,accepted):
  self.cid=cid;self.folder=out/f'contests/{cid}';self.folder.mkdir(parents=True,exist_ok=True);self.source=json.loads(accepted.read_text());rawpath=MAIN_DATA_DIR/f'contest_{cid}.json';assert self.source['configuration']['raw_input_sha256']==sha(rawpath) and self.source['diagnostics']['passed'];raw=json.loads(rawpath.read_text());self.observed=Design(float(raw['theta'])/PRIZE_DIVISOR,float(raw['N_Delta'])*float(raw['Delta2f']));self.engine=SimilarityOutput(out/'build');self.groups={k:[] for k in ['screen','coarse','fine']};rng=np.random.default_rng(202610250000+cid)
  for chain,(p,h) in enumerate(zip(self.source['chain_paths'],self.source['chain_sha256'])):
   assert sha(p)==h;df=pd.read_csv(p,comment='#',usecols=['c_i','c_j','sigma','lambda','mu_0'])
   for name,count in [('screen',1),('coarse',32),('fine',4)]:
    # Independent with-replacement random quadrature conditional on the
    # original accepted empirical posterior; preserve within-draw covariance.
    rows=[]
    for index in rng.integers(0,len(df),size=count):
     r=df.iloc[int(index)];rows.append(Draw(float(r.c_i),float(r.c_j),float(r.sigma),float(r['lambda']),float(r.mu_0),chain,int(index)))
    self.groups[name].append(rows)
  self.lambda_ref=float(np.median([d.observed_precision for g in self.groups['coarse'] for d in g]));self.precisions=[None,math.inf,self.lambda_ref*.001,self.lambda_ref*.03,self.lambda_ref*.3,self.lambda_ref*3];self.events=[];self.validated={};self.screen_base=self.screen(self.observed);self.provenance=dict(accepted=str(accepted),accepted_sha256=sha(accepted),original_chain_sha256=self.source['chain_sha256'],seed=202610250000+cid,screen_per_chain=1,coarse_per_chain=32,fine_per_chain=4,method='multilevel original-game nonlinear HJB, original posterior unchanged',source_sha256=sha(Path(__file__)),native_source_sha256=self.engine.source_sha256)
  self.provenance['validation_protocol_sha256']=validation_protocol();draws={k:[[vars(d) for d in g] for g in v] for k,v in self.groups.items()};quad=self.folder/'quadrature.json';progress=self.folder/'validation_progress.json'
  if quad.exists() and progress.exists():
   prior=json.loads(quad.read_text());stored=json.loads(progress.read_text())
   if prior['draws']==draws and prior['provenance']['accepted_sha256']==self.provenance['accepted_sha256'] and stored.get('accepted_sha256')==self.provenance['accepted_sha256'] and stored.get('protocol_sha256')==self.provenance['validation_protocol_sha256']:
    self.validated.update(stored['results']);print('RESUME VALIDATION',cid,len(self.validated),flush=True)
  save(quad,dict(provenance=self.provenance,draws=draws))
 def native(self,d,design):
  key=hashlib.sha256(json.dumps(dict(draw=vars(d),design=design.record(),native=self.engine.source_sha256,spacing=.15,steps=32),sort_keys=True).encode()).hexdigest();p=self.folder/'coarse_cache'/f'{key}.json'
  cached=solver_cache.get(p)
  if cached is not None:return cached
  try:r=dict(resolved=True,**self.engine.evaluate(d,design,spacing=.15,steps=32,full_nodes=2001,quadrature=64))
  except Exception as e:r=dict(resolved=False,error=repr(e))
  save(p,r);return r
 def screen(self,design):
  if design==self.observed:return dict(status='identity',B=None)
  values=[self.native(d,design) for g in self.groups['screen'] for d in g];base=[self.native(d,self.observed) for g in self.groups['screen'] for d in g]
  if not all(v['resolved'] for v in values+base):return dict(status='unresolved')
  return dict(status='evaluated',gap=float(np.mean([v['B']-b['B'] for v,b in zip(values,base)])))
 def cost(self,design,delta):return design.prize/self.observed.prize+delta*(design.horizon/self.observed.horizon)**2
 def propose(self):
  proposals={self.observed};threats=[]
  for multiple in TIMES:
   zero=Design(0.,self.observed.horizon*multiple,None);z=self.screen(zero)
   if z.get('gap',-math.inf)>=0:
    proposals.add(zero);continue
   for precision in self.precisions:
    previous=0.;feasible=None
    for ratio in PRIZES:
     candidate=Design(self.observed.prize*ratio,self.observed.horizon*multiple,precision)
     if candidate==self.observed:ok=True
     else:
      trial=self.screen(candidate);ok=trial.get('gap',-math.inf)>=0
      if trial['status']=='unresolved':threats.append(candidate.record())
     if ok:feasible=ratio;break
     previous=ratio
    if feasible is None:continue
    # Bracket refinement finds one feasible crossing without asserting
    # globally monotone quality or ruling out disconnected feasible islands.
    for _ in range(4):
     mid=(previous+feasible)/2;candidate=Design(self.observed.prize*mid,self.observed.horizon*multiple,precision);trial=self.screen(candidate)
     if trial.get('gap',-math.inf)>=0:feasible=mid
     else:
      if trial['status']=='unresolved':threats.append(candidate.record())
      previous=mid
    for factor in [1.,1.05,1.2]:
     candidate=Design(self.observed.prize*feasible*factor,self.observed.horizon*multiple,precision)
     if self.screen(candidate).get('gap',-math.inf)>=0:proposals.add(candidate)
   save(self.folder/'search_progress.json',dict(status='screening',duration_multiple=multiple,proposals=[d.record() for d in proposals],unresolved_screen_candidates=threats))
   print('SCREEN',self.cid,multiple,'proposals',len(proposals),flush=True)
  self.screen_threats=threats;return proposals
 def fine_output(self,d,design):
  key=hashlib.sha256(json.dumps(dict(draw=vars(d),design=design.record(),native=self.engine.source_sha256,protocol='independent_recovery_v2'),sort_keys=True).encode()).hexdigest();path=self.folder/'fine_recovery'/f'{key}.json'
  cached=solver_cache.get(path)
  if cached is not None:return cached
  first=checked_output(d,design,cache_dir=self.folder/'fine_cache',resolution=Resolution())
  if first['resolved']:return first
  errors=[t['error'] for t in first['trials'] if 'error' in t];valid=[t['B'] for t in first['trials'] if 'B' in t]
  resource_only=bool(errors) and all('memory budget' in e or 'Initial state too close' in e for e in errors)
  if not valid and not resource_only:return first
  trials=[]
  for dx,nt in [(.025,512),(.0125,1024)]:
   try:trials.append(self.engine.evaluate(d,design,spacing=dx,steps=nt,full_nodes=8001,quadrature=160))
   except Exception as e:return dict(resolved=False,error=repr(e),previous=first)
  radius=3*abs(trials[-1]['B']-trials[0]['B'])+trials[-1]['budget'];value=trials[-1]['B']
  if valid:
   difference=abs(value-valid[-1])
   if difference>max(radius,1e-4*max(1.,abs(value))):return dict(resolved=False,error='Independent solvers disagree beyond refinement diagnostics',previous=first,similarity_trials=trials)
   value=valid[-1];radius+=difference
  result=dict(resolved=True,B=value,diagnostic_radius=radius,method='original_HJB_independent_recovery',trials=trials,reference_trials=first['trials']);save(path,result);return result
 def validate(self,design):
  if design==self.observed:return dict(status='identity',passed=True,mean_gap=0.,mcse=0.,numerical_radius=0.)
  key=json.dumps(design.record(),sort_keys=True)
  if key in self.validated:return self.validated[key]
  coarse_by_chain=[];correction_by_chain=[];radii=[];failure=[]
  for chain,draws in enumerate(self.groups['coarse']):
   values=[]
   for d in draws:
    a,b=self.native(d,self.observed),self.native(d,design)
    if a['resolved'] and b['resolved']:values.append(b['B']-a['B'])
    else:
     failure.append(dict(stage='coarse',chain=chain,index=d.index))
     return self.store_unresolved(design,failure)
   coarse_by_chain.append(values)
  for chain,draws in enumerate(self.groups['fine']):
   values=[]
   for d in draws:
    a,b=self.native(d,self.observed),self.native(d,design);fa=self.fine_output(d,self.observed)
    if not fa['resolved']:
     return self.store_unresolved(design,failure+[dict(stage='fine',chain=chain,index=d.index)])
    fb=self.fine_output(d,design)
    if all(v['resolved'] for v in [a,b,fa,fb]):values.append((fb['B']-fa['B'])-(b['B']-a['B']));radii.append(fa['diagnostic_radius']+fb['diagnostic_radius'])
    else:
     failure.append(dict(stage='fine',chain=chain,index=d.index))
     return self.store_unresolved(design,failure)
   correction_by_chain.append(values)
  if failure:result=dict(status='unresolved',passed=False,failed_draws=failure)
  else:
   mean=float(np.mean([np.mean(g) for g in coarse_by_chain])+np.mean([np.mean(g) for g in correction_by_chain]));terms=[float(np.var(g,ddof=1)/len(g)/16) for g in coarse_by_chain+correction_by_chain];variance=sum(terms);denom=sum(v*v/(len(g)-1) for v,g in zip(terms,coarse_by_chain+correction_by_chain));df=variance*variance/denom if denom>0 else 1e9;se=math.sqrt(variance);critical=float(student_t.ppf(.975,df));radius=float(np.mean(radii));lower=mean-critical*se-radius;result=dict(status='evaluated',passed=lower>=0,mean_gap=mean,mcse=se,degrees_freedom=df,critical_value=critical,numerical_radius=radius,lower_diagnostic_margin=lower,scope='conditional original empirical posterior quadrature; numerical radius not a strict continuous bound')
  self.validated[key]=result;save(self.folder/'validation_progress.json',dict(status='validating',results=self.validated,protocol_sha256=self.provenance['validation_protocol_sha256'],accepted_sha256=self.provenance['accepted_sha256']));print('VALIDATE',self.cid,design.record(),result.get('passed'),result.get('mean_gap'),flush=True);return result
 def store_unresolved(self,design,failure):
  result=dict(status='unresolved',passed=False,failed_draws=failure,remaining_draws_not_evaluated=True)
  self.validated[json.dumps(design.record(),sort_keys=True)]=result
  save(self.folder/'validation_progress.json',dict(status='validating',results=self.validated,protocol_sha256=self.provenance['validation_protocol_sha256'],accepted_sha256=self.provenance['accepted_sha256']))
  print('VALIDATE UNRESOLVED',self.cid,design.record(),failure,flush=True)
  return result
 def run(self):
  proposals=self.propose();selections=[];full_choices={d for d in proposals if d.prize==0 or d.precision==math.inf}
  for delta in DELTAS:
   # Find a supported incumbent in each restriction, preserving all failed
   # cheaper proposals. This is best verified among searched proposals.
   full=None
   for d in sorted(full_choices,key=lambda x:self.cost(x,delta)):
    if self.validate(d)['passed']:full=d;break
   selected=self.observed
   for d in sorted(proposals,key=lambda x:self.cost(x,delta)):
    if self.cost(d,delta)>=self.cost(selected,delta):break
    if self.validate(d)['passed']:selected=d;break
   if full is not None and self.cost(full,delta)<self.cost(selected,delta):selected=full
   kind='zero_prize' if selected.prize==0 else 'full_disclosure' if selected.precision==math.inf else 'less_disclosure';addition=None
   if selected.prize>0 and kind=='less_disclosure' and full is not None:addition=100*(1-self.cost(selected,delta)/self.cost(full,delta))
   selections.append(dict(contest_id=self.cid,delta=delta,choice=kind,prize_ratio=selected.prize/self.observed.prize,time_multiplier=selected.horizon/self.observed.horizon,cost_saving_pct=100*(1-self.cost(selected,delta)/(1+delta)),additional_reduction_pct=addition,selected=selected.record(),full_comparator=None if full is None else full.record(),retained_observed=selected==self.observed))
  record=dict(status='completed_verified_incumbents',contest_id=self.cid,provenance=self.provenance,selections=selections,validation=self.validated,screen_unresolved=self.screen_threats,scope='Best verified searched proposals; not a certified global optimum. Original nonlinear game; unchanged original approximate Bayesian posterior.',publication_ready=False);save(self.folder/'result.json',record);return record

def run_case(cid,accepted):
 p=out/f'contests/{cid}/result.json'
 if p.exists():return json.loads(p.read_text())
 return Case(cid,Path(accepted)).run()
if __name__=='__main__':
 import argparse,fcntl
 parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,choices=range(1,7),default=2);args=parser.parse_args()
 handle=(out/'design_controller.lock').open('w');fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB);state=dict(status='running',pid=os.getpid(),completed={},failed={},active={},total=73);pending=set(ids);jobs={};dest=out/'design_status.json'
 state['workers']=args.workers
 with futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
  while pending or jobs:
   while len(jobs)<args.workers:
    ready=next((cid for cid in sorted(pending) if available(cid)),None)
    if ready is None:break
    pending.remove(ready);job=pool.submit(run_case,ready,str(available(ready)));jobs[job]=ready;state['active'][str(ready)]=dict(accepted=str(available(ready)));save(dest,state)
   if not jobs:
    post=out/'posterior_status.json'
    if post.exists() and json.loads(post.read_text())['status'].startswith('completed'):
     for cid in pending:state['failed'][str(cid)]='Original posterior unavailable'
     pending.clear();break
    time.sleep(5);continue
   done,_=futures.wait(jobs,timeout=5,return_when=futures.FIRST_COMPLETED)
   for job in done:
    cid=jobs.pop(job);state['active'].pop(str(cid),None)
    try:r=job.result();state['completed'][str(cid)]=str(out/f'contests/{cid}/result.json')
    except Exception as e:state['failed'][str(cid)]=repr(e);save(out/f'contests/{cid}/error.json',dict(error=repr(e),traceback=traceback.format_exc()))
    print('DESIGN COVERAGE',len(state['completed']),'/73','failed',len(state['failed']),flush=True)
   state['pending_posteriors']=len(pending);state['updated_utc']=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat();save(dest,state)
 state['status']='completed' if len(state['completed'])==73 else 'completed_with_failures';save(dest,state)
