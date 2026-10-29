# Task 3：后验重建方法与执行快照

> 诊断口径更正（2026-10-25）：以下历史批次使用了较早 runner 的额外均值 MCSE<=3% 门槛。原最终冻结流程实际使用 `unified_prior_medians.py`，长尾均值误差单列，不作为最终接受门槛。本批已经通过的链仍有效；延长抽样不是最终协议的必需条件。后续接续见[最终协议方法](task4_final_protocol_training_method_20261025.md)。保留以下当时的执行源和记录供追溯。


日期：2026-10-25。用户要求仔细搜索仓库与归档，实在找不到则重建。已对 1,872 个 Stan/chain CSV 计算 SHA256，与原审查清单的 1,588 个链哈希比对，无匹配；Git 可访问历史仅保存统一先验的清单和汇总，原始 ZIP 内无后验文件。

先重建已选 6 个代表性竞赛的 full 后验，优先 20270；后续预测任务需另按训练窗口重估，不能用 full 抽样代替训练后验。该批次是完整任务的一阶段，不等于全部 146 次估计已经重建。

直接使用现有 `detail_model.stan` 的逐字节快照与原 runner 的数据构造、先验及诊断函数，不修改原模型或实现。CmdStan 2.36.0 编译时关闭预编译头，构建产物仅在审查重建目录，不改 CmdStan 安装。四链 dense NUTS，沿用 1000/2000、2000/4000、4000/8000 的 warmup/sampling 日程及 .98/.99/.995 目标接受率。本历史重建采用的增强门槛：无发散及树深饱和、每链 BFMI>=.3、所有核查量 rank Rhat<=1.01、bulk/tail ESS>=400，sigma/lambda 相对均值 MCSE<=.03。

新抽样从保存中位数的分散初值启动，不是原抽样字节恢复，也不以逼近旧点估计为接受标准。输入哈希、初值、命令、链、诊断与接受记录单独保存于 `reconstruction_20261025/`，不覆盖既有估计或汇总。此原始链重建是用户本轮明确授权的文件写入例外；论文及实现代码继续冻结。

状态：[status.json](reconstruction_20261025/status.json)。严格诊断未通过前不得用于后续反事实或称作已接受后验。

```python
import csv, hashlib, json, math, os, subprocess, time
from pathlib import Path
import numpy as np
from pipeline.empirical import estimate as estimator
from pipeline.empirical import unified_prior_reestimate as original
root=Path.cwd()
report=root/'docs/review/PaperJK9'
out=report/'reconstruction_20261025'
model=out/'model_snapshot/detail_model'
original_root=root/'results/reestimation/lambda100_sd225'
manifest=original_root/'estimation/accepted_chain_sources.csv'
entries={(int(r['contest_id']),r['mode']):r for r in csv.DictReader(manifest.open())}
summary={(int(r['contest_id']),r['mode']):r for r in csv.DictReader((original_root/'estimation/posterior_summary_final.csv').open())}
cases=[20270,23652,2464,5144,2551,8219]
status_path=out/'status.json'
if status_path.exists():
    raise RuntimeError('Existing reconstruction status: inspect before explicit resume')
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,value):
    p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False));tmp.replace(p)
state=dict(status='running',purpose='Reconstruct same-prior posterior for selected counterfactual checks; later tasks may require other contests/windows',driver_pid=os.getpid(),cases=cases,finished=[],failed=[],current=None,posterior_reconstruction_complete=False,existing_results_overwritten=False)
save(status_path,state)
start=time.monotonic()
for cid in cases:
    entry=entries[(cid,'full')];rawpath=original_root/'input_snapshots/main'/f"contest_{cid}_{entry['input_sha256']}.json"
    assert digest(rawpath)==entry['input_sha256']
    data=estimator.stan_data(json.loads(rawpath.read_text()),'full')
    inputs=dict(data,**original.prior(data));row=summary[(cid,'full')]
    baseline={k:float(row[k+'_median']) for k in ('c_i','c_j','sigma','lambda','mu_0','r')}
    case=out/'estimation/full'/f'contest_{cid}'
    config=dict(contest_id=cid,mode='full',raw_input_sha256=digest(rawpath),source_model_sha256=digest(model.with_suffix('.stan')),executable_sha256=digest(model),original_runner_sha256=digest(Path(original.__file__)),sampler='CmdStan 2.36.0 NUTS',chains=4,prior=original.prior(data),initialization='Dispersed saved same-prior marginal medians; these are starting values, not reused draws',new_draws_not_byte_identical_to_original=True)
    save(case/'configuration.json',config)
    previous=None
    for attempt,(warmup,samples,delta) in enumerate(((1000,2000,.98),(2000,4000,.99),(4000,8000,.995)),1):
        folder=case/f'attempt{attempt}';folder.mkdir(parents=True,exist_ok=True)
        save(folder/'data.json',inputs)
        paths=[folder/f'chain_{j}.csv' for j in range(1,5)]
        processes=[]
        for j,path in enumerate(paths,1):
            if previous is None:
                factor=(.8,1.,1.2,1.5)[j-1]
                init={k:max(.0101,baseline[k]*factor) for k in ('c_i','c_j','sigma','lambda')}
                init['mu_0']=baseline['mu_0']+(j-2.5)*.05
                init['log_r']=math.log(baseline['r'])+(j-2.5)*.1
            else:
                columns=estimator.read_columns(previous[j-1]);init={k:float(np.median(columns[k])) for k in estimator.PARAMETERS}
            init_path=folder/f'init_{j}.json';save(init_path,init)
            log=(folder/f'chain_{j}.log').open('w')
            command=[str(model),'sample',f'num_warmup={warmup}',f'num_samples={samples}','adapt',f'delta={delta}','algorithm=hmc','engine=nuts','max_depth=13','metric=dense_e','random',f'seed={20261025+cid}',f'id={j}','data',f"file={folder/'data.json'}",'output',f'file={path}','refresh=100','sig_figs=18',f'init={init_path}']
            process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,OPENBLAS_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',OMP_NUM_THREADS='1'))
            processes.append((process,log,command))
        state['current']=dict(contest_id=cid,attempt=attempt,warmup=warmup,samples=samples,delta=delta,pids=[x[0].pid for x in processes],commands=[x[2] for x in processes]);save(status_path,state)
        print(f'START {cid} full attempt={attempt}',flush=True)
        while any(x[0].poll() is None for x in processes):
            state['elapsed_seconds']=time.monotonic()-start
            state['current']['process_return_codes']=[x[0].poll() for x in processes]
            save(status_path,state);time.sleep(2)
        codes=[x[0].returncode for x in processes]
        for _,log,_ in processes:log.close()
        if any(code!=0 for code in codes):
            state['failed'].append(dict(contest_id=cid,attempt=attempt,reason='CmdStan nonzero exit',codes=codes));save(status_path,state);break
        assert all(estimator.csv_complete(path,samples) for path in paths)
        result_row,diagnostics=original.diagnose(cid,'full',paths,data)
        result=dict(summary=result_row,diagnostics=diagnostics,chain_paths=[str(x) for x in paths],chain_sha256=[digest(x) for x in paths],attempt=attempt,configuration=config)
        save(folder/'diagnostics.json',result)
        print(f"DIAGNOSTICS {cid} attempt={attempt} passed={diagnostics['passed']}",flush=True)
        if diagnostics['passed']:
            save(case/'accepted.json',result);state['finished'].append(dict(contest_id=cid,attempt=attempt,accepted=str(case/'accepted.json')));save(status_path,state);break
        previous=paths
    else:
        state['failed'].append(dict(contest_id=cid,reason='All frozen diagnostic schedules exhausted'))
state['current']=None;state['status']='completed' if not state['failed'] else 'completed_with_failures'
state['posterior_reconstruction_complete']=len(state['finished'])==len(cases);save(status_path,state)
print(json.dumps(dict(status=state['status'],accepted_cases=len(state['finished']),failed=state['failed'])),flush=True)
```
