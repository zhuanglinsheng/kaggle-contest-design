# Task 4：快速求值器与原目标的独立核对

在log(ci),log(cj),log(sigma),log(lambda),mu0,logr（C另有a/logb）坐标使用完整正确先验与硬下界。原Stan log density在jacobian=false时加前四个log参数的Jacobian，比较差值是否仅为参数无关常数。预测单小时联合计数密度另与原Python逐时评分函数核对。没有以模型相似或采样诊断替代目标密度一致性。

下面ReviewTarget供后续重要性积分使用；四个模型/案例组合、每组四个实际抽样点同时检查训练与当前小时预测。

```python
import ast,ctypes,csv,json,math,hashlib,subprocess
from pathlib import Path
import numpy as np
from scipy.special import gammaln,gammaincc,logsumexp
from pipeline.empirical import estimate as e,unified_prior_reestimate as u
p=Path('docs/review/PaperJK9');folder=p/'predictive_integration_20261025';libpath=folder/'density_components.so';build=json.loads((folder/'build.json').read_text());assert hashlib.sha256(libpath.read_bytes()).hexdigest()==build['binary_sha256']
lib=ctypes.CDLL(str(libpath.resolve()));arr=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS');ints=np.ctypeslib.ndpointer(dtype=np.int32,flags='C_CONTIGUOUS');lib.batch_components.argtypes=[ctypes.c_int,ctypes.c_int,arr,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_double,ctypes.c_double,ctypes.c_double,arr,ints,ints,ctypes.c_int,ctypes.c_int,arr,arr];lib.batch_components.restype=None
class ReviewTarget:
 def __init__(self,raw,cut,hour,kind):
  self.raw=raw;self.data=e.stan_data(raw,'full');self.cut=cut;self.hour=hour;self.kind=kind;self.dim=8 if kind=='C' else 6;assert cut<=hour<self.data['N_Delta'];self.held=np.ascontiguousarray(self.data['hat_y'],dtype=float);self.counts=[]
  for k in ['i','j']:
   ts=np.asarray(self.data['hat_t_'+k]);v=np.bincount(np.maximum(1,np.ceil(ts).astype(int))-1,minlength=self.data['N_Delta'])[:self.data['N_Delta']];self.counts.append(np.ascontiguousarray(v,dtype=np.int32))
  prior=u.prior(self.data);self.mean=np.asarray(prior['prior_mean']+([.5,0.] if kind=='C' else []));self.sd=np.asarray(prior['prior_sd']+([.5,.5] if kind=='C' else []))
 def components(self,z):
  z=np.ascontiguousarray(np.atleast_2d(z),dtype=float);assert z.shape[1]==self.dim;ll=np.empty(len(z));lp=np.empty(len(z));d=self.data;lib.batch_components(len(z),self.dim,z,self.cut,self.hour,d['count_start_idx']-1,d['Delta2f'],d['T_contest'],d['theta'],self.held,*self.counts,int(self.counts[0][self.hour]),int(self.counts[1][self.hour]),ll,lp);return ll,lp
 def posterior_and_prediction(self,z):
  z=np.atleast_2d(np.asarray(z,dtype=float));ll,lp=self.components(z);prior=-.5*np.sum(((z-self.mean)/self.sd)**2,axis=1)-np.log(self.sd).sum()-.5*self.dim*math.log(2*math.pi);support=np.isfinite(z).all(axis=1)&(z[:,[0,1,2,3,5]]>=math.log(.01)).all(axis=1);prior[~support]=-np.inf;return prior+ll,lp
# Load the exact reported Python prediction function, independently of the C++ evaluator.
f=p/'task4_calibrated_posterior_prediction_method_20261025.md';txt=f.read_text().partition(chr(10)+chr(96)*3+'python'+chr(10))[2].rpartition(chr(10)+chr(96)*3)[0];tree=ast.parse(txt);env=dict(e=e,np=np,math=math,gammaln=gammaln,gammaincc=gammaincc,logsumexp=logsumexp)
for name in ['counts_for','predictions']:
 fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name);exec(compile(ast.get_source_segment(txt,fn),str(f),'exec'),env)
checks=folder/'parity_checks';checks.mkdir(exist_ok=True);records=[]
for kind,cid,hour_onebased in [('D',3288,760),('C',3288,760),('D',2454,672),('C',2454,672)]:
 entry=next(r for r in json.loads((p/f'task4_{kind}_median_training_status_20261025.json').read_text())['finished'] if r['contest_id']==cid);a=json.loads(Path(entry['accepted']).read_text());chain=e.read_columns(Path(a['chain_paths'][0]));rawpath=next(Path('results/reestimation/lambda100_sd225/input_snapshots/main').glob(f'contest_{cid}_*.json'));raw=json.loads(rawpath.read_text());data=e.stan_data(raw,'early');cut=data['N_Delta'];data.update(u.prior(data));dp=checks/f'{kind}_{cid}_data.json';dp.write_text(json.dumps(data));target=ReviewTarget(raw,cut,hour_onebased-1,kind);diffs=[]
 for serial,index in enumerate([0,len(chain['sigma'])//3,2*len(chain['sigma'])//3,len(chain['sigma'])-1]):
  names=(*e.PARAMETERS,'deadline_power','gap_log_scale') if kind=='C' else e.PARAMETERS;params={k:float(chain[k][index]) for k in names};point=np.array([params[k] for k in names]);point[:4]=np.log(point[:4]);post,logpred=target.posterior_and_prediction(point);pp=checks/f'{kind}_{cid}_params_{serial}.json';pp.write_text(json.dumps(params));out=checks/f'{kind}_{cid}_lp_{serial}.csv';model=(p/('reconstruction_20261025/model_snapshot/detail_model' if kind=='D' else 'prediction_reconstruction_20261025/C/model_snapshot/free_kernel')).resolve();cmd=[str(model),'log_prob',f'constrained_params={pp}','jacobian=false','data',f'file={dp}','output',f'file={out}','sig_figs=18'];run=subprocess.run(cmd,text=True,capture_output=True);(checks/f'{kind}_{cid}_log_{serial}.txt').write_text(run.stdout+run.stderr);assert run.returncode==0;row=next(csv.DictReader(line for line in out.open() if not line.startswith('#')));stan=float(row['lp__'])+float(point[:4].sum());diffs.append(float(post[0]-stan));one={k:np.array([v]) for k,v in params.items()};prediction=env['predictions'](raw,one,hour_onebased-1,hour_onebased,free=(kind=='C'))[0][0,0];err=float(logpred[0]-prediction);assert abs(err)<1e-8;records.append(dict(model=kind,contest_id=cid,hour=hour_onebased,point_index=index,constant_density_difference=float(post[0]-stan),prediction_log_density_difference=err))
 assert np.ptp(diffs)<1e-7,(kind,cid,diffs)
result=dict(status='completed',all_passed=True,records=records,maximum_within_case_density_difference_after_constant=max(np.ptp([r['constant_density_difference'] for r in records if r['model']==m and r['contest_id']==c]) for m,c in [('D',3288),('C',3288),('D',2454),('C',2454)]),maximum_prediction_log_density_difference=max(abs(r['prediction_log_density_difference']) for r in records),binary_sha256=build['binary_sha256'],scope='Same empirical training density in log-positive coordinates, up to a parameter-independent constant; same proper single-hour joint count density as independent Python scorer');(folder/'parity.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='records'},indent=2))

```
