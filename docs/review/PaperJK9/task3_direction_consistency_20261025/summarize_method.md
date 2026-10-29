# 方向结果汇总

按两个模型各自的新减原质量差判号。非线性诊断半径为两端最近两层差绝对值三倍之和；一阶诊断半径同时计入求积加密及浮点尺度。未超过半径判为不确定，不把不同幅度判为失败。独立求解器核验另列，不将单点结果当作后验平均结论。

```python
from pathlib import Path
import json,csv,collections
p=Path('docs/review/PaperJK9/task3_direction_consistency_20261025');rows=[]
def sign(x,e):return 0 if abs(x)<=e else (1 if x>0 else -1)
def nr(r):
 t=r['nonlinear_trials'];return max(3*abs(t[-1]['B']-t[-2]['B']),1e-12*max(1.,abs(t[-1]['B'])))
def fr(r):
 t=r['first_order_trials'];return max(3*abs(t[-1]-t[-2]),1e-12*max(1.,abs(t[-1])))
for f in sorted(p.glob('contest_*.json')):
 d=json.loads(f.read_text());assert d['status']=='completed' and len(d['records'])==7;b=d['records'][0]
 for r in d['records'][1:]:
  fo=r['first_order_trials'][-1]-b['first_order_trials'][-1];fe=fr(r)+fr(b);sf=sign(fo,fe);row=dict(contest_id=d['contest_id'],variable=r['variable'],multiplier=r['multiplier'],first_order_change=fo,first_order_sign=sf,first_order_radius=fe)
  if all('B' in t for s in [b,r] for t in s['nonlinear_trials']):
   nl=r['nonlinear_trials'][-1]['B']-b['nonlinear_trials'][-1]['B'];ne=nr(r)+nr(b);sn=sign(nl,ne);row.update(nonlinear_change=nl,nonlinear_radius=ne,nonlinear_sign=sn,classification='uncertain' if not sf or not sn else ('consistent' if sf==sn else 'opposite'))
  else:row.update(classification='numerically_unresolved')
  rows.append(row)
counts={v:dict(collections.Counter(r['classification'] for r in rows if r['variable']==v)) for v in ['prize','duration','precision']};summary=dict(scope='Six fixed original representative joint draws; finite one-variable changes; no posterior population consistency claim',comparisons=36,counts_by_variable=counts,strict_error_certificate=False,rows=rows,independent_2551_prize_check=json.loads((p/'independent_2551_prize_sign.json').read_text()))
(p/'summary.json').write_text(json.dumps(summary,indent=2));keys=sorted(set().union(*(r.keys() for r in rows)))
with (p/'directions.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
print(json.dumps(counts,indent=2))
```
