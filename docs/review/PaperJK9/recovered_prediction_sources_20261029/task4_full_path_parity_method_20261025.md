# Task 4：批量路径求值的独立核对

比较8个模型/案例组合，每组3个实际训练后验点的整段预测概率；训练密度另与已核验v2比较。未来公共读数剧烈扰动后训练密度及第一小时预测保持不变。只有通过后才使用批量积分。

```python
import ast,ctypes,json,hashlib,math
from pathlib import Path
import numpy as np
from scipy.special import gammaln,gammaincc,logsumexp
from pipeline.empirical import estimate as e,unified_prior_reestimate as u
p=Path('docs/review/PaperJK9')
def code(f):return Path(f).read_text().partition(chr(10)+chr(96)*3+'python'+chr(10))[2].rpartition(chr(10)+chr(96)*3)[0]
exec(compile(code(p/'task4_predictive_density_v2_parity_method_20261025.md').split('# Load the exact reported Python prediction function')[0],'density-v2','exec'))
root=p/'predictive_integration_20261025';pl=root/'full_path_components.so';pathlib=ctypes.CDLL(str(pl.resolve()));pathlib.batch_forecast_path.argtypes=[ctypes.c_int,ctypes.c_int,arr,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_double,ctypes.c_double,ctypes.c_double,arr,ints,ints,arr,arr];pathlib.batch_forecast_path.restype=None
class FullPathTarget(ReviewTarget):
 def __init__(self,raw,cut,kind):super().__init__(raw,cut,cut,kind)
 def posterior_and_path(self,z):
  z=np.ascontiguousarray(np.atleast_2d(z),dtype=float);assert z.shape[1]==self.dim;d=self.data;N=d['N_Delta'];ll=np.empty(len(z));lp=np.empty((len(z),N-self.cut));pathlib.batch_forecast_path(len(z),self.dim,z,self.cut,N,d['count_start_idx']-1,d['Delta2f'],d['T_contest'],d['theta'],self.held,*self.counts,ll,lp);prior=-.5*np.sum(((z-self.mean)/self.sd)**2,axis=1)-np.log(self.sd).sum()-.5*self.dim*math.log(2*math.pi);support=np.isfinite(z).all(axis=1)&(z[:,[0,1,2,3,5]]>=math.log(.01)).all(axis=1);prior[~support]=-np.inf;return prior+ll,lp
# Verify complete held-out paths with the independently implemented scalar/vector predictor.
src=code(p/'task4_calibrated_posterior_prediction_method_20261025.md');tree=ast.parse(src);env=dict(e=e,np=np,math=math,gammaln=gammaln,gammaincc=gammaincc,logsumexp=logsumexp)
for name in ['counts_for','predictions']:
 fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name);exec(compile(ast.get_source_segment(src,fn),'independent-prediction','exec'),env)
index=json.loads((p/'task4_primary_training_index_20261025.json').read_text());records=[]
for kind in ['C','D']:
 for cid in [3288,3960,4104,13333]:
  entry=next(r for r in index['models'][kind]['entries'] if r['contest_id']==cid);a=json.loads(Path(entry['accepted']).read_text());c=e.read_columns(Path(a['chain_paths'][0]));raw=json.loads(next(Path('results/reestimation/lambda100_sd225/input_snapshots/main').glob(f'contest_{cid}_*.json')).read_text());cut=a['configuration']['training_cutoff'];names=(*e.PARAMETERS,'deadline_power','gap_log_scale') if kind=='C' else e.PARAMETERS;ix=[0,len(c['sigma'])//2,len(c['sigma'])-1];columns={k:c[k][ix] for k in names};z=np.array([columns[k] for k in names]).T.copy();z[:,:4]=np.log(z[:,:4]);target=FullPathTarget(raw,cut,kind);lp,path=target.posterior_and_path(z);ref=env['predictions'](raw,columns,cut,free=(kind=='C'))[0];err=float(np.max(abs(path-ref)));assert err<1e-7
  for hour in [cut,raw['N_Delta']-1]:
   old=ReviewTarget(raw,cut,hour,kind);q,pr=old.posterior_and_prediction(z);density_error=float(np.max(abs(lp-q)));prediction_error=float(np.max(abs(path[:,hour-cut]-pr)));assert density_error<1e-8 and prediction_error<1e-7
  changed=json.loads(json.dumps(raw));changed['hat_y'][cut:]=[1e6]*(len(changed['hat_y'])-cut);q,pr=FullPathTarget(changed,cut,kind).posterior_and_path(z);future_error=float(np.max(abs(lp-q)));first_error=float(np.max(abs(path[:,0]-pr[:,0])));assert future_error==0 and first_error==0;records.append(dict(model=kind,contest_id=cid,draws=3,forecast_hours=path.shape[1],maximum_complete_path_log_probability_difference=err,training_density_difference=density_error,future_training_difference=future_error,current_endpoint_difference=first_error))
result=dict(passed=True,records=records,source_sha256=hashlib.sha256((root/'full_path_components.cpp').read_bytes()).hexdigest(),binary_sha256=hashlib.sha256(pl.read_bytes()).hexdigest(),scope='Full-path backend matches verified training density and independent complete-hour Python predictions, with future-information invariance');(root/'full_path_parity.json').write_text(json.dumps(result,indent=2));print('Full-path target parity passed',len(records),'model/case combinations',flush=True)

```
