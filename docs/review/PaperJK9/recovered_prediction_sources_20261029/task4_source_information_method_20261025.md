# Task 4：源提交回放与初始信息可得性核查

日期：2026-10-25。目的：不把哈希/前缀一致误作信息可得性证明。只读扫描 ZIP 中的 Teams/Submissions，按当前冻结的 73 个竞赛和焦点对回放同一预处理；不重选参赛者，不改 canonical 输入或实现。

核验：提交时刻/计数、初始化窗口一致，公私差距路径误差小于 1e-9；追溯初始化联合读数的两个源提交，确认在预测截止前。保留 UTC+8 旧日期约定，列出不跨同一提交日的备选截止。记录 ZIP 条目 CRC（完整读取时 zipfile 校验）与源代码和输入哈希。

本核查不能恢复真实日内时刻，也不消除事后焦点对选择；它检验构造数据的可复现性和初始化未引用预测期提交。最终官方期限元数据另已读取，须与构造窗口长分开解释。

```python
import csv,json,hashlib,math,time,zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from pipeline.empirical.rebuild_main import rebuild,assign_times,FIELDS
from pipeline.empirical.estimate import early_cutoff
report=Path('docs/review/PaperJK9');entries=list(csv.DictReader((report/'task4_prediction_split_inventory_20261025.csv').open()));ids={int(r['contest_id']) for r in entries};archive=Path('metakaggle/__rawdata__/meta-kaggle.zip');out=report/'task4_source_information_audit_20261025.json'
if out.exists():raise RuntimeError('Existing source audit: inspect before resume')
state=dict(status='reading_source',scope='Read-only replay of current focal-pair hourly inputs from archived source submissions; no canonical inputs changed',preprocessing_source_sha256=hashlib.sha256(Path('pipeline/empirical/rebuild_main.py').read_bytes()).hexdigest(),records=[],actual_intraday_timestamps_observed=False)
def save():
 tmp=out.with_suffix('.json.tmp');tmp.write_text(json.dumps(state,indent=2,allow_nan=False));tmp.replace(out)
save();start=time.monotonic();teammap={};parts=[]
with zipfile.ZipFile(archive) as z:
 state['archive_entries']={n:dict(crc32=z.getinfo(n).CRC,uncompressed_bytes=z.getinfo(n).file_size,compressed_bytes=z.getinfo(n).compress_size) for n in ['Teams.csv','Submissions.csv']}
 with z.open('Teams.csv') as stream:
  for chunk in pd.read_csv(stream,chunksize=500000,usecols=['Id','CompetitionId']):
   chosen=chunk.loc[chunk.CompetitionId.isin(ids)]
   teammap.update(zip(chosen.Id.astype(int),chosen.CompetitionId.astype(int)))
 print('Source teams selected',len(teammap),flush=True)
 teamids=set(teammap);source_rows=0
 with z.open('Submissions.csv') as stream:
  for chunk in pd.read_csv(stream,chunksize=500000,usecols=['Id','TeamId','SubmissionDate','IsAfterDeadline',*FIELDS],dtype={'IsAfterDeadline':'boolean'}):
   source_rows+=len(chunk);keep=chunk.TeamId.isin(teamids)&~chunk.IsAfterDeadline.fillna(False)
   if keep.any():parts.append(chunk.loc[keep].copy())
   if source_rows%5000000==0:print('Source submissions scanned',source_rows,flush=True)
frame=pd.concat(parts,ignore_index=True).drop_duplicates('Id').dropna(subset=FIELDS);frame['CompetitionId']=frame.TeamId.map(teammap);frame['SubmissionDate']=pd.to_datetime(frame.SubmissionDate,format='%m/%d/%Y');state['source_submission_rows_scanned']=source_rows;state['selected_scored_rows']=len(frame);state['status']='replaying';save()
for entry in entries:
 cid=int(entry['contest_id']);rawpath=next(Path('results/reestimation/lambda100_sd225/input_snapshots/main').glob(f'contest_{cid}_*.json'));raw=json.loads(rawpath.read_text());f=frame.loc[frame.CompetitionId==cid];pair=(raw['team_i'],raw['team_j']);record=dict(contest_id=cid,input_sha256=hashlib.sha256(rawpath.read_bytes()).hexdigest())
 try:
  new,_=rebuild(f,raw,pair,raw['time_imputation_seed'],raw['score_metric']);record['counts_match']=bool(new['hat_t_i']==raw['hat_t_i'] and new['hat_t_j']==raw['hat_t_j']);record['initialization_matches']=bool(new['initialization_hours']==raw['initialization_hours'] and new['count_start_hour']==raw['count_start_hour']);record['maximum_public_path_difference']=float(np.max(np.abs(np.asarray(new['hat_y'])-raw['hat_y'])));record['maximum_private_path_difference']=float(np.max(np.abs(np.asarray(new['real_y'])-raw['real_y'])));record['source_replay_passed']=bool(record['counts_match'] and record['initialization_matches'] and record['maximum_public_path_difference']<1e-9 and record['maximum_private_path_difference']<1e-9)
  timed=assign_times(f,raw['time_imputation_seed']);initial_hour=raw['count_start_hour'];endpoint=raw['start_time']+(initial_hour+1)*3600;cut=early_cutoff(raw);cuttime=raw['start_time']+cut*3600;trace=[]
  for team in pair:
   past=timed.loc[(timed.TeamId==team)&(timed.imputed_time<=endpoint)].sort_values('imputed_time').iloc[-1]
   trace.append(dict(team_id=int(team),submission_id=int(past.Id),submission_date=past.SubmissionDate.isoformat(),imputed_hours_from_window_start=float((past.imputed_time-raw['start_time'])/3600),source_before_prediction_cutoff=bool(past.imputed_time<=cuttime),source_before_window_start=bool(past.imputed_time<raw['start_time'])))
  record['initial_source_trace']=trace;record['all_initial_sources_before_cutoff']=all(t['source_before_prediction_cutoff'] for t in trace);record['training_cutoff_hour']=cut;record['earliest_joint_endpoint_hour']=initial_hour+1
  dayboundary=math.floor((cuttime+8*3600)/86400)*86400-8*3600;record['complete_source_day_cutoff_hour']=int(round((dayboundary-raw['start_time'])/3600));record['source_day_cutoff_valid']=bool(raw['count_start_hour']<record['complete_source_day_cutoff_hour']<raw['N_Delta'])
 except Exception as exc:record['error']=str(exc);record['source_replay_passed']=False
 state['records'].append(record);save()
 if len(state['records'])%10==0:print('Source input replay',len(state['records']),'/73',flush=True)
state['status']='completed';state['seconds']=time.monotonic()-start;state['replay_passed_contests']=sum(r['source_replay_passed'] for r in state['records']);state['initial_sources_before_cutoff_contests']=sum(r.get('all_initial_sources_before_cutoff',False) for r in state['records']);save();print(json.dumps({k:v for k,v in state.items() if k not in ('records','archive_entries')},ensure_ascii=False),flush=True)
```
