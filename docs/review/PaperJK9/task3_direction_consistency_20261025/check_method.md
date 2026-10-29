# Task 3：固定参数的方向一致检查

只比较质量变化的符号，不要求两模型质量水平或改善幅度相同，不重估参数、不重新优化设计。六个既有实际联合代表抽样点全部保留；逐点固定成本、创新风险、初始均值和原初始方差。以原设计为基准，分别改变奖金至.9/1.1倍、期限至.9/1.1倍、反馈精度至.5/2倍。精度为正，避开零反馈数值支线；倍数变化不是共同绝对精度政策的后验评估，也不是导数证明。

一阶采用原实现的96/192点求积；非线性采用两层网格。两层差三倍是诊断半径，不是严格误差界。符号暂不明确时预先允许再加密一层；失败明确记录。六点结果不是全后验或全部竞赛的方向认证。已有全披露奖金曲线作为另外的方向证据保留，不因改变任务范围删除反例。

```python
from pathlib import Path
import json,os,math,hashlib
import numpy as np
from pipeline.empirical.fixed_initial_design import Output,beta
p=Path('docs/review/PaperJK9');out=p/'task3_direction_consistency_20261025';ns={};s=(out/'native_output_interface.md').read_text().partition('\n```python\n')[2].rpartition('\n```')[0];exec(compile(s,'native-direction-interface','exec'),ns);native=ns['native_output'];cid=int(os.environ['TASK3_DIRECTION_CONTEST']);point=next(r for r in json.loads((p/'task3_representative_joint_points_20261025.json').read_text())['points'] if r['contest_id']==cid);z0=point['point'];z=np.array([math.log(z0[k]) for k in ['c_i','c_j','sigma','lambda']]+[z0['mu_0']]);draws={k:np.array([z0[k]]) for k in ['c_i','c_j','sigma','lambda','mu_0']};T0=point['observed_duration'];theta0=point['observed_prize'];lambda0=z0['lambda'];S0=point['fixed_initial_variance'];dest=out/f'contest_{cid}.json';assert not dest.exists();state=dict(status='running',contest_id=cid,scope='Fixed actual representative joint draw; original design; finite changes, not posterior average or global design',input=point,records=[],sources={n:hashlib.sha256(Path(n).read_bytes()).hexdigest() for n in ['pipeline/empirical/fixed_initial_design.py',str(out/'resources/nonlinear_output_kernel.cpp'),str(out/'resources/nonlinear_variable_diffusion_kernel.cpp')]})
def save():
 tmp=dest.with_suffix('.json.tmp');tmp.write_text(json.dumps(state,indent=2));tmp.replace(dest)
save()
for variable,multiplier in [('baseline',1.),('prize',.9),('prize',1.1),('duration',.9),('duration',1.1),('precision',.5),('precision',2.)]:
 T=T0*(multiplier if variable=='duration' else 1);theta=theta0*(multiplier if variable=='prize' else 1);precision=lambda0*(multiplier if variable=='precision' else 1);first=[]
 for order in [96,192]:
  model=Output(draws,T,order);assert abs(float(model.initial[0])-S0)<1e-12;first.append(float(beta(draws['mu_0'],model.q)[0]+theta*model.coefficients(precision)[0]))
 r=dict(variable=variable,multiplier=multiplier,T=T,theta=theta,precision=precision,first_order_trials=first,nonlinear_trials=[])
 for dy,nt in [(.0125,1024),(.00625,2048)]:
  try:r['nonlinear_trials'].append(dict(dy=dy,nt=nt,B=native(z,S0,T,theta,precision,dy=dy,nt=nt)))
  except Exception as ex:r['nonlinear_trials'].append(dict(dy=dy,nt=nt,error=repr(ex)))
 state['records'].append(r);save();print('DIRECTION',cid,variable,multiplier,flush=True)
# One predetermined extra level if refinement changes can obscure any sign.
base=state['records'][0]
def radius(r):
 ts=r['nonlinear_trials'];return max(3*abs(ts[-1]['B']-ts[-2]['B']),1e-12*max(abs(ts[-1]['B']),1.))
if all('B' in t for t in base['nonlinear_trials']):
 ambiguous=[r for r in state['records'][1:] if all('B' in t for t in r['nonlinear_trials']) and abs(r['nonlinear_trials'][-1]['B']-base['nonlinear_trials'][-1]['B'])<=radius(r)+radius(base)]
 if ambiguous:
  for r in [base]+ambiguous:
   try:r['nonlinear_trials'].append(dict(dy=.003125,nt=4096,B=native(z,S0,r['T'],r['theta'],r['precision'],dy=.003125,nt=4096)))
   except Exception as ex:r['nonlinear_trials'].append(dict(dy=.003125,nt=4096,error=repr(ex)))
   save()
state['status']='completed';save()
```
