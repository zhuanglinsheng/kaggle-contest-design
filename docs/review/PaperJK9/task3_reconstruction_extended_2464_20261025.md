# Task 3：2464 的延长抽样

> 诊断口径更正（2026-10-25）：以下历史批次使用了较早 runner 的额外均值 MCSE<=3% 门槛。原最终冻结流程实际使用 `unified_prior_medians.py`，长尾均值误差单列，不作为最终接受门槛。本批已经通过的链仍有效；延长抽样不是最终协议的必需条件。后续接续见[最终协议方法](task4_final_protocol_training_method_20261025.md)。保留以下当时的执行源和记录供追溯。


日期：2026-10-25。原 6 案例重建批次结束，5 例通过，2464 在第三轮 4x8,000 样本后仍因 lambda 均值相对 MCSE 约 5.27% 超过本次重建额外采用的 3% 门槛而未接受；其余混合、ESS、BFMI、发散及树深检查通过。

保留原诊断门槛，增加至每链 32,000 样本，必要时 64,000；不以改变门槛接受该例。模型、数据、先验不变；记录独立随机种子与命令。原批次状态保留为历史记录，不改成全体通过。延长批次另存[状态](reconstruction_20261025/extended_2464_status.json)，通过后才生成该例 accepted.json。来源及逐字节模型快照见[原重建方法](task3_reconstruction_method_20261025.md)。

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
cases=[2464]
status_path=out/'extended_2464_status.json'
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
    if (case/'configuration.json').exists():
        assert json.loads((case/'configuration.json').read_text())==config
    else:
        save(case/'configuration.json',config)
    previous=None
    for attempt,(warmup,samples,delta) in enumerate(((2000,32000,.995),(4000,64000,.9975)),4):
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
            command=[str(model),'sample',f'num_warmup={warmup}',f'num_samples={samples}','adapt',f'delta={delta}','algorithm=hmc','engine=nuts','max_depth=13','metric=dense_e','random',f'seed={20261026+cid}',f'id={j}','data',f"file={folder/'data.json'}",'output',f'file={path}','refresh=100','sig_figs=18',f'init={init_path}']
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
