# Task 4：日期切分接受记录统一读取

合并原训练、互不重叠的接续批次以及经过哈希检查的同目标补充接受记录；按竞赛去重并排序。每个消费者仍核对输入、训练截止、模型、先验、链SHA及诊断，不把主切分后验混入日期窗口。函数只读，不修改任何接受记录。

```python
def accepted_day_entries(root,model):
 import json,hashlib
 from pathlib import Path
 root=Path(root);entries={}
 for name in [f'task4_{model}_complete_day_training_status_20261025.json',f'task4_{model}_complete_day_resume_status_20261025.json']:
  sf=root/name
  if sf.exists():
   for row in json.loads(sf.read_text())['finished']:entries[row['contest_id']]=row
 sf=root/'task4_complete_day_supplemental_acceptances_20261025.json'
 if sf.exists():
  for row in json.loads(sf.read_text())['entries']:
   if row['model']==model:
    assert hashlib.sha256(Path(row['accepted']).read_bytes()).hexdigest()==row['accepted_sha256'];entries[row['contest_id']]=row
 return [entries[c] for c in sorted(entries)]

```
