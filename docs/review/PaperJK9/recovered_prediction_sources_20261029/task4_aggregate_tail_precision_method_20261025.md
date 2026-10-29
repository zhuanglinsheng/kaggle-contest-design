# Task 4：降低两例保守平均误差的尾部精度分配

114个稀有小时全部已通过后，两例仍未通过平均误差门槛。其中3960仅尾部MCSE之和/332就达.001380，超过.001，不能把“每小时已通过”说成“全案例已通过”。3509尾部项约.000916，几乎耗尽总误差预算。因此按旧方差及每次递推长度分配新样本，目标使各例尾部项约<=.0005，为普通小时后验积分留出预算。

用a_h=旧MCSE*sqrt(旧N)，计算预算约满足min sum(h*N_h)，约束sum(a_h/sqrt(N_h))<=目标且N_h不小于旧N；按幂次2向上取整，实际结果仍重新核验。这里仅为计算量规划，不是模型设计或真实误差证明。复用旧合格独立预算并新增独立生产预算，原目标/概率/ESS/矩/一致性门槛不变；旧结果存history，新MCSE需更小且与旧概率一致才可替换。两个独立积分同时运行，索引写入用锁串行。最终仍须全案例重评分。

```python
import json,math,hashlib,os,threading,concurrent.futures,traceback
from pathlib import Path
p=Path('docs/review/PaperJK9');status=p/'task4_C_aggregate_tail_precision_status_20261025.json';assert not status.exists();ix=json.loads((p/'task4_tail_probability_index_20261025.json').read_text());scores=json.loads((p/'task4_C_calibrated_forecast_20261025.json').read_text());hours={r['contest_id']:r['forecast_hours'] for r in scores['records']};tasks=[];plans=[]
for cid in [3509,3960]:
 rr=[]
 for key,t in ix['entries'].items():
  if t['model']=='C' and t['contest_id']==cid and t['split']=='primary_hour':
   st=json.loads(Path(t['integral_status']).read_text());old=st['records'][-1];assert st['passed'];rr.append(dict(key=key,entry=t,old_samples=old['samples'],old_mcse=old['logscore_mcse'],amplitude=old['logscore_mcse']*math.sqrt(old['samples']),work_per_draw=t['hour_index']))
 H=hours[cid];goal=.0005*H
 def allocation(lagrange):return [max(r['old_samples'],(lagrange*r['amplitude']/(2*r['work_per_draw']))**(2/3)) for r in rr]
 def sum_error(lagrange):return sum(r['amplitude']/math.sqrt(n) for r,n in zip(rr,allocation(lagrange)))
 hi=1.
 while sum_error(hi)>goal:hi*=2
 lo=0.
 for _ in range(80):
  mid=(lo+hi)/2
  if sum_error(mid)>goal:lo=mid
  else:hi=mid
 predicted=0.;selected=0
 for r,nstar in zip(rr,allocation(hi)):
  n=2**math.ceil(math.log2(max(nstar,r['old_samples'])));predicted+=r['amplitude']/math.sqrt(n)
  if n>r['old_samples']:
   selected+=1;tasks.append(dict(contest_id=cid,hour=r['entry']['hour_index'],base_stage=str(Path(r['entry']['integral_status']).parent),previous_status_sha256=r['entry']['integral_status_sha256'],samples=[n,2*n],desired_mcse=r['amplitude']/math.sqrt(n),old_samples=r['old_samples'],old_mcse=r['old_mcse']))
 plans.append(dict(contest_id=cid,forecast_hours=H,tail_hours=len(rr),old_tail_mcse_component=sum(r['old_mcse'] for r in rr)/H,requested_tail_component=.0005,planned_component=predicted/H,selected_hours=selected))
tasks.sort(key=lambda r:(-r['samples'][0],r['contest_id'],r['hour']));state=dict(status='running',driver_pid=os.getpid(),scope='Same target; variance-based sample allocation reduces conservative sum of tail-hour MCSE, without changing numerical gates',plans=plans,tasks=tasks,finished=[],failed=[],active={},maximum_simultaneous_integrals=2);lock=threading.Lock()
def digest(f):return hashlib.sha256(Path(f).read_bytes()).hexdigest()
def save():
 tmp=status.with_suffix('.json.tmp');tmp.write_text(json.dumps(state,indent=2));tmp.replace(status)
def code(f):return Path(f).read_text().partition(chr(10)+chr(96)*3+'python'+chr(10))[2].rpartition(chr(10)+chr(96)*3)[0]
method=code(p/'task4_fixed_tail_precision_method_20261025.md');verify=code(p/'task4_precision_tail_registration_method_20261025.md');save()
def worker(task):
 cid=task['contest_id'];h=task['hour'];key=f'{cid}:{h}'
 with lock:state['active'][key]=task;save()
 try:
  assert digest(Path(task['base_stage'])/'status.json')==task['previous_status_sha256'];env=dict(KIND='C',CID=cid,HOUR=h,SPLIT='primary_hour',RUN_LABEL='_precision_refine1',BASE_PROPOSAL_STAGE=task['base_stage'],RNG_RUN_OFFSET=155000000,PRODUCTION_BUDGETS=task['samples'],DESIRED_MCSE=task['desired_mcse']);exec(compile(method,'frozen-proposal-tail-precision','exec'),env)
  if not env['state']['passed']:raise RuntimeError('Independent precision budgets not accepted')
  exec(compile(verify,'precision-original-target-and-locked-registry','exec'),dict(KIND='C',CID=cid,HOUR=h,SPLIT='primary_hour',RUN_LABEL='_precision_refine1'))
  with lock:state['finished'].append(dict(task,updated_registry=True));state['active'].pop(key);save()
 except Exception as ex:
  with lock:state['failed'].append(dict(task,reason=repr(ex),traceback=traceback.format_exc()));state['active'].pop(key);save()
  print('TAIL PRECISION ERROR',cid,h,repr(ex),flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(worker,tasks))
current=json.loads((p/'task4_tail_probability_index_20261025.json').read_text());state['actual_tail_components']={str(cid):sum(t['logscore_mcse'] for t in current['entries'].values() if t['model']=='C' and t['contest_id']==cid and t['split']=='primary_hour')/hours[cid] for cid in [3509,3960]};state['status']='completed' if not state['failed'] else 'completed_with_uncertainty';save();print('TAIL AGGREGATE PRECISION ENDED',len(state['finished']),len(state['failed']),state['actual_tail_components'],flush=True)

```
