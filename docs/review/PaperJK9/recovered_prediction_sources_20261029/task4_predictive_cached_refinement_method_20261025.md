# Task 4：先复用更长的完整同目标链核查预测精度

C2454初始2k/链评分MCSE约.00212，超过.001计算初筛。早期增强均值门槛批次已留下完整8k/链缓存，先重新核对最终参数诊断、C额外参数、输入和四链完整性，再用于独立的预测精度检查，不重新采样。此操作不按预测分数好坏选择链，不改变目标后验或原参数接受记录。

新链索引仅指向更长且通过混合检查的同目标抽样；预测积分是否通过需另执行评分，不因较长样本而自动宣称通过。若没有足够长缓存，后续需增加抽样并做预算加倍比较。

首次调用在诊断包装器缺少math全局依赖时退出，未写入新接受记录；补齐原函数依赖后重新执行，没有改变诊断函数。

```python
import ast,types,json,hashlib,math
from pathlib import Path
import numpy as np
from pipeline.empirical import estimate as e,unified_prior_reestimate as u,nonlinear_reestimate as nonlinear
p=Path('docs/review/PaperJK9');mp=Path('pipeline/empirical/unified_prior_medians.py');mt=mp.read_text();fn=next(n for n in ast.parse(mt).body if isinstance(n,ast.FunctionDef) and n.name=='diagnose');ns={'u':u,'np':np,'RAW_DIAGNOSE':u.diagnose,'__file__':str(mp)};exec(compile(ast.get_source_segment(mt,fn),str(mp),'exec'),ns);med=types.SimpleNamespace(diagnose=ns['diagnose'],__file__=str(mp));train=(p/'task4_final_protocol_training_method_20261025.md').read_text().partition(chr(10)+chr(96)*3+'python'+chr(10))[2].rpartition(chr(10)+chr(96)*3)[0];dfn=next(n for n in ast.parse(train).body if isinstance(n,ast.FunctionDef) and n.name=='diagnose');env=dict(e=e,u=u,np=np,math=math,nonlinear=nonlinear,med=med,MODEL_KIND='C');exec(compile(ast.get_source_segment(train,dfn),'final-protocol-diagnose','exec'),env)
def digest(f):return hashlib.sha256(Path(f).read_bytes()).hexdigest()
case=p/'prediction_reconstruction_20261025/C/early/contest_2454';baseline=json.loads((case/'accepted_median_protocol.json').read_text());folder=case/'attempt3';paths=[folder/f'chain_{j}.csv' for j in range(1,5)]
for f in paths:
 with f.open('rb') as h:h.seek(max(0,f.stat().st_size-4096));assert b'#  Elapsed Time:' in h.read()
 assert e.csv_complete(f,8000)
rawpath=next(Path('results/reestimation/lambda100_sd225/input_snapshots/main').glob('contest_2454_*.json'));data=e.stan_data(json.loads(rawpath.read_text()),'early');assert json.loads((folder/'data.json').read_text())==dict(data,**u.prior(data));row,d=env['diagnose'](2454,paths,data);assert d['passed'];out=case/'accepted_predictive_cached_8000.json';assert not out.exists();result=dict(summary=row,diagnostics=d,chain_paths=[str(f.resolve()) for f in paths],chain_sha256=[digest(f) for f in paths],configuration=baseline['configuration'],attempt='cached_attempt3_8000',scope='Longer complete same-target chains for predictive precision assessment; not selected by forecast performance',superseded_for_predictive_integration_only=str((case/'accepted_median_protocol.json').resolve()));out.write_text(json.dumps(result,indent=2));index=p/'task4_predictive_chain_index_20261025.json';current=json.loads(index.read_text()) if index.exists() else dict(scope='Optional longer validated same-target chains for predictive integration; canonical parameter acceptance unchanged',entries={});current['entries']['C:2454']=dict(accepted=str(out.resolve()),accepted_sha256=digest(out),draws_per_chain=8000,prediction_precision_not_yet_established=True);index.write_text(json.dumps(current,indent=2));print('C2454 cached 4x8000 validated for predictive precision; source chains preserved')

```
