"""Simulation-calibrate sigma uncertainty without changing posterior estimates.

The original 95% posterior interval is retained. A separate calibrated interval
uses one multiplier fitted on exploratory replicates0/1 only. Replicates2/3 are
retrospective checks; replicate4 is a new prospective check. There is no claim
that empirical parameter truths are known or that10/10 proves95% coverage.
"""
import argparse
import json
import math
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from pipeline.empirical import nonlinear_reestimate as n
from pipeline.empirical import sigma_lambda_trials as s
from pipeline.empirical import sigma_lambda_posterior as p

OUT=s.OUT/'sigma_interval_calibration'
ALPHA=.05


def normalized_error(truth,median,lower,upper):
 if not 0<lower<median<upper:raise ValueError('Ordered positive interval required')
 distance=math.log(truth/median)
 return abs(distance)/(math.log(upper/median) if distance>=0 else math.log(median/lower))


def calibrated_interval(median,lower,upper,multiplier):
 if multiplier<1:raise ValueError('Calibration must not shrink the original interval')
 return (math.exp(math.log(median)-multiplier*math.log(median/lower)),
         math.exp(math.log(median)+multiplier*math.log(upper/median)))


def read(rep,cid):
 folder=s.OUT/'posterior/recovery'/f'{cid}_{rep}'/'lambda_external'
 path=folder/'summary.json';value=json.loads(path.read_text())
 if not value['passed']:raise RuntimeError(f'Invalid source fit {path}')
 paths=[folder/f'contest_{cid}/chain_{j}.npz' for j in range(1,5)]
 if value['chain_sha256']!=[n.digest(path) for path in paths]:raise RuntimeError('Source chains changed')
 return value,path


def design():
 OUT.mkdir(parents=True,exist_ok=True)
 value=dict(method='Single multiplier applied to asymmetric log-scale 95% posterior widths; point estimates unchanged.',
  training_replicates=[0,1],retrospective_replicates=[2,3],prospective_replicate=4,contests=list(s.IDS),alpha=ALPHA,
  selection='For each training fit, compute truth-to-log-median distance divided by the original interval halfwidth in that direction. Use max(1, kth ordered score), k=ceil((n+1)*(1-alpha)), capped at n. With20 cases k=20.',
  interpretation='Additional simulation-calibrated uncertainty interval; not the unmodified95% Bayesian credible interval. Chosen configurations are not an exchangeable random population; no distribution-free coverage guarantee.',
  diagnostics='Report coverage, interval width and point errors together; never adjust multiplier using validation truths.',
  script_sha256=n.digest(__file__))
 path=OUT/'design.json'
 if path.exists() and json.loads(path.read_text())!=value:raise RuntimeError('Frozen calibration design differs')
 n.save_json(path,value)


def chain(task):return p.chain(task)


def fit(replicates):
 jobs=[('recovery',f'{cid}_{rep}','lambda_external') for rep in replicates for cid in s.IDS]
 for job in jobs:p.prepare(*job)
 with ProcessPoolExecutor(max_workers=8) as pool:list(pool.map(chain,[(*job,j,30000) for job in jobs for j in range(1,5)]))
 for job in jobs:p.summarize(*job)


def train():
 design();records=[]
 for rep in (0,1):
  for cid in s.IDS:
   value,path=read(rep,cid);sigma=value['summaries']['sigma']
   score=normalized_error(value['truth']['sigma'],sigma['median'],sigma['q025'],sigma['q975'])
   records.append(dict(replicate=rep,contest_id=cid,score=score,summary_sha256=n.digest(path)))
 rank=min(len(records),math.ceil((len(records)+1)*(1-ALPHA)))
 multiplier=max(1.,sorted(r['score'] for r in records)[rank-1])
 result=dict(multiplier=multiplier,order_statistic_rank=rank,training_cases=len(records),training_records=records,
  note='Validation truth values are never read by train(). No claim that point estimation error is reduced.')
 path=OUT/'frozen_multiplier.json'
 if path.exists() and json.loads(path.read_text())!=result:raise RuntimeError('Frozen multiplier differs')
 n.save_json(path,result);print('FROZEN SIGMA MULTIPLIER',multiplier,flush=True)


def check():
 frozen=json.loads((OUT/'frozen_multiplier.json').read_text());multiplier=frozen['multiplier'];rows=[];metrics={}
 for rep in (0,1,2,3,4):
  selected=[]
  for cid in s.IDS:
   value,path=read(rep,cid);sigma=value['summaries']['sigma'];truth=value['truth']['sigma']
   lower,upper=calibrated_interval(sigma['median'],sigma['q025'],sigma['q975'],multiplier)
   selected.append(dict(replicate=rep,contest_id=cid,truth=truth,median=sigma['median'],
    original_lower=sigma['q025'],original_upper=sigma['q975'],calibrated_lower=lower,calibrated_upper=upper,
    original_covers=bool(sigma['q025']<=truth<=sigma['q975']),calibrated_covers=bool(lower<=truth<=upper),
    original_relative_width=(sigma['q975']-sigma['q025'])/truth,calibrated_relative_width=(upper-lower)/truth,
    point_relative_error=abs(sigma['median']/truth-1),summary_sha256=n.digest(path)))
  metrics[str(rep)]=dict(cases=len(selected),original_covered=sum(r['original_covers'] for r in selected),
   calibrated_covered=sum(r['calibrated_covers'] for r in selected),original_median_relative_width=float(np.median([r['original_relative_width'] for r in selected])),
   calibrated_median_relative_width=float(np.median([r['calibrated_relative_width'] for r in selected])),
   mean_point_relative_error=float(np.mean([r['point_relative_error'] for r in selected])))
  rows+=selected
 n.write_csv(OUT/'coverage.csv',rows);n.save_json(OUT/'metrics.json',metrics)
 lines=['sigma区间校准：保留点估计，单独报告校准后的不确定性','',
  f'共同校准系数k={multiplier:.5f}。只使用探索实现0/1的20个拟合选取，并在新实现4的验证前冻结。所有配置使用同一系数，不逐场调整。',
  '保持原lambda_external后验不变；在log sigma尺度，以后验中位数为中心，将原95%区间左右半宽同乘k。新结果必须标为“模拟校准区间”，不能替换名称后称原95%后验可信区间。',
  '实现0/1为训练；2/3是已有数据的事后检查；4为新生成的前瞻检查。没有按验证结果重新选择k或模拟种子。','',
  '实现 | 原区间覆盖 | 校准后覆盖 | 原相对宽度中位数 | 校准后相对宽度中位数 | 平均点估计相对误差']
 for rep,m in metrics.items():
  lines.append(f"{rep} | {m['original_covered']}/10 | {m['calibrated_covered']}/10 | {m['original_median_relative_width']:.1%} | {m['calibrated_median_relative_width']:.1%} | {m['mean_point_relative_error']:.1%}")
 lines+=['','区间覆盖改善不等于点估计误差下降。该校准利用指定模拟机制下的恢复偏差与不确定性；十个配置并非随机抽取，不能据此保证总体95%覆盖，也不能保证每次新模拟都10/10。',
  '原论文、一阶努力公式、参数后验、lambda区间和既有试验结果均保持不变。若用于论文，应并列报告原后验区间与校准区间及其宽度。']
 (OUT/'report.txt').write_text('\n'.join(lines)+'\n')
 artifacts=['design.json','frozen_multiplier.json','coverage.csv','metrics.json','report.txt']
 n.save_json(OUT/'completion.json',dict(status='completed_simulation_interval_calibration',script_sha256=n.digest(__file__),
  artifact_sha256={name:n.digest(OUT/name) for name in artifacts},
  source_summary_sha256={str(s.OUT/'posterior/recovery'/f"{r['contest_id']}_{r['replicate']}"/'lambda_external/summary.json'):r['summary_sha256'] for r in rows}))
 print(json.dumps(metrics,indent=2),flush=True)


if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=('design','fit_train','train','fit_test','check'),default='design');args=parser.parse_args()
 if args.stage=='design':design()
 elif args.stage=='fit_train':fit((0,1))
 elif args.stage=='train':train()
 elif args.stage=='fit_test':fit((4,))
 else:check()
