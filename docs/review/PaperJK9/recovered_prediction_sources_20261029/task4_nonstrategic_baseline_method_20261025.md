# Task 4：两个非战略活动率基准的实际概率预测

日期：2026-10-25。源提交回放和初始化时序已对 73 个竞赛核实后，再开始拟合。范围限定为冻结的事后焦点对与构造小时窗口，不宣称恢复真实小时时间戳。

A：每位参赛者常数率 lambda_i/day，独立 Gamma(.5,.5天) 先验。B：率为 lambda_i*((T-t)/T)^beta，同一 Gamma 先验，双方共享 beta~N(0,1)。B 的率参数条件后验为 Gamma，可解析积分；beta 是严格对数凹的一维后验，以模式附近 Gaussian 求积积分，64/128/256/512 阶加密。B 固定 beta=0 时退回 A。

每小时双方的联合计数概率先在共同 beta 后验下积分；不把两个边际积分分数误作联合分数。模型参数不使用预测期计数更新。预测均值及零计数概率也保存。原小时 75% 切分与上游 UTC+8 完整日期边界切分均报告，保留训练零提交参赛者。

数值核验：预测 PMF 归一化、求积加倍收敛、最小概率若干小时及一个中间小时用独立自适应积分复核。后者在给定被评分计数下优化积分被积函数仅为准确计算概率，不是用该计数重新拟合预测模型。未通过的竞赛不得用于未限定的模型排名。

所有模型参数、时间单位和门槛在评分执行前固定；同时报告全部 73 个竞赛，不选择改善案例。每竞赛平均联合小时对数分数为单位，竞赛层 bootstrap 只比较两个基准，不检验均衡模型增量。

[比较记录](task4_baseline_comparison_20261025.json)、[竞赛分数](task4_baseline_contests_20261025.csv)、[逐时概率预测](task4_baseline_predictions_20261025.csv)。

```python
import csv,json,math,time,hashlib
from pathlib import Path
import numpy as np
from scipy.special import gammaln,logsumexp,roots_hermitenorm
from scipy.optimize import brentq
from scipy.integrate import quad
from pipeline.empirical.estimate import early_cutoff
report=Path('docs/review/PaperJK9');audit=json.loads((report/'task4_source_information_audit_20261025.json').read_text());assert audit['status']=='completed' and audit['replay_passed_contests']==73 and audit['initial_sources_before_cutoff_contests']==73
out=report/'task4_baseline_comparison_20261025.json'
if out.exists():raise RuntimeError('Existing baseline results: inspect before resume')
a0=.5;b0=.5

def log_den_and_moments(beta,z,dt,extra_z=None):
    xx=beta*z+math.log(dt)
    if extra_z is not None:xx=np.r_[xx,beta*extra_z+math.log(dt)];zz=np.r_[z,extra_z]
    else:zz=z
    den=float(np.logaddexp(math.log(b0),logsumexp(xx)));w=np.exp(xx-den);m=float(w@zz);v=float(w@(zz*zz)-m*m)
    return den,m,max(0.,v)
def mode_shape(A,Z,z,dt,extra_z=None):
    derivative=lambda beta:-beta+Z-A*log_den_and_moments(beta,z,dt,extra_z)[1]
    bound=16.
    while derivative(-bound)*derivative(bound)>0:
        bound*=2
        if bound>128:raise RuntimeError('Could not bracket posterior mode')
    mode=float(brentq(derivative,-bound,bound,xtol=1e-11));den,m,var=log_den_and_moments(mode,z,dt,extra_z)
    return mode,1/math.sqrt(1+A*var)
def log_integral(A,Z,z,dt,extra_z=None):
    mode,sd=mode_shape(A,Z,z,dt,extra_z)
    logp=lambda beta:-.5*beta*beta+beta*Z-A*log_den_and_moments(beta,z,dt,extra_z)[0]
    peak=logp(mode)
    def integrand(u):
        v=logp(mode+sd*u)-peak
        return math.exp(v) if v>-745 else 0.
    left,el=quad(integrand,-np.inf,0,epsabs=1e-11,epsrel=1e-10,limit=120)
    right,er=quad(integrand,0,np.inf,epsabs=1e-11,epsrel=1e-10,limit=120)
    return peak+math.log(sd*(left+right)),float((el+er)/(left+right))
def quadrature_posterior(A,Z,z,dt,order):
    mode,sd=mode_shape(A,Z,z,dt);proposal_sd=2*sd;nodes,weights=roots_hermitenorm(order);ok=weights>0;nodes=nodes[ok];weights=weights[ok];beta=mode+proposal_sd*nodes
    den=np.logaddexp(math.log(b0),logsumexp(math.log(dt)+beta[:,None]*z[None,:],axis=1))
    lw=np.log(weights)-.5*beta*beta+beta*Z-A*den+.5*nodes*nodes
    normal=float(logsumexp(lw));lw-=normal
    return beta,den,lw,dict(mode=mode,proposal_sd=proposal_sd,mean=float(np.exp(lw)@beta),log_integral=normal+math.log(proposal_sd),order=order)
def forecast(beta,den,lw,ai,aj,zfuture,counts,dt):
    scores=[];means=[];zeros=[]
    for start in range(0,len(zfuture),128):
        z=zfuture[start:start+128];k=counts[:,start:start+128];v=math.log(dt)+beta[:,None]*z[None,:]-den[:,None];C=gammaln(ai+k[0])-gammaln(ai)-gammaln(k[0]+1)+gammaln(aj+k[1])-gammaln(aj)-gammaln(k[1]+1)
        ll=C[None,:]-(ai+aj)*np.logaddexp(0,v)-(k[0]+k[1])[None,:]*np.logaddexp(0,-v)
        scores.extend(logsumexp(lw[:,None]+ll,axis=0).tolist());meanrate=np.exp(logsumexp(lw[:,None]+v,axis=0));means.extend(np.column_stack((ai*meanrate,aj*meanrate)).tolist());zeros.extend(np.column_stack((np.exp(logsumexp(lw[:,None]-ai*np.logaddexp(0,v),axis=0)),np.exp(logsumexp(lw[:,None]-aj*np.logaddexp(0,v),axis=0)))).tolist())
    return np.array(scores),np.asarray(means),np.asarray(zeros)
# The predictive distribution must normalize, including a zero-training-event case.
k=np.arange(301);aa=.5;den=2.;h=.041666666666666664
lp=gammaln(aa+k)-gammaln(aa)-gammaln(k+1)-aa*np.logaddexp(0,math.log(h/den))-k*np.logaddexp(0,math.log(den/h))
assert abs(float(np.exp(lp).sum())-1)<1e-12
state=dict(status='running',scope='Two nonstrategic baselines on fixed reconstructed windows; conditional on ex-post focal pairs and imputed hourly timeline, not equilibrium-model comparison',gamma_prior_shape=a0,gamma_prior_rate_days=b0,deadline_slope_prior='Normal(0,1), common across the two participants',predictive_unit='Joint pair-hour count probability, nats',source_audit_sha256=hashlib.sha256((report/'task4_source_information_audit_20261025.json').read_bytes()).hexdigest(),predictive_mass_smoke_error=float(abs(np.exp(lp).sum()-1)),records=[],equilibrium_increment_tested=False)
rows=[];hourrows=[]
def save():
 tmp=out.with_suffix('.json.tmp');tmp.write_text(json.dumps(state,indent=2,allow_nan=False));tmp.replace(out)
save();clock=time.monotonic()
for source_record in audit['records']:
    cid=source_record['contest_id'];rawpath=next(Path('results/reestimation/lambda100_sd225/input_snapshots/main').glob(f'contest_{cid}_*.json'));raw=json.loads(rawpath.read_text());N=raw['N_Delta'];dt=raw['Delta2f'];start=raw['count_start_hour'];counts=[]
    for player in ('i','j'):
        ts=np.array([float(t) for t in raw['hat_t_'+player] if 0<float(t)<=N]);idx=np.maximum(1,np.ceil(ts).astype(int))-1;counts.append(np.bincount(idx,minlength=N)[:N])
    counts=np.asarray(counts)
    for split,cut in [('primary_hour',early_cutoff(raw)),('complete_day',source_record['complete_source_day_cutoff_hour'])]:
        assert start<cut<N;train=counts[:,start:cut];test=counts[:,cut:];nt=train.sum(axis=1);ai,aj=nt+a0;A=ai+aj;zt=np.log((N-np.arange(start,cut))/N);zf=np.log((N-np.arange(cut,N))/N);Z=float((train[0]+train[1])@zt)
        A_score,A_mean,A_zero=forecast(np.array([0.]),np.array([math.log(b0+(cut-start)*dt)]),np.array([0.]),float(ai),float(aj),zf,test,dt)
        previous=None;converged=False
        for order in (64,128,256,512):
            beta,den,lw,diag=quadrature_posterior(A,Z,zt,dt,order);B_score,B_mean,B_zero=forecast(beta,den,lw,float(ai),float(aj),zf,test,dt)
            if previous is not None:
                maxdelta=float(np.max(np.abs(B_score-previous[0])));normaldelta=abs(diag['log_integral']-previous[1]);meandelta=abs(diag['mean']-previous[2])
                if maxdelta<1e-5 and normaldelta<1e-8 and meandelta<1e-7:converged=True;break
            previous=(B_score.copy(),diag['log_integral'],diag['mean'])
        # Independent adaptive integration checks the rarest scored hours and a zero hour.
        logZ,quaderr=log_integral(A,Z,zt,dt);selected=set(np.argsort(B_score)[:3].tolist());selected.add(len(zf)//2);check=[]
        for j in sorted(selected):
            ki,kj=test[:,j];K=int(ki+kj);C=float(gammaln(ai+ki)-gammaln(ai)-gammaln(ki+1)+gammaln(aj+kj)-gammaln(aj)-gammaln(kj+1));num,err=log_integral(A+K,Z+K*zf[j],zt,dt,extra_z=float(zf[j]));reference=C+K*math.log(dt)+num-logZ
            check.append(dict(holdout_hour_index=j,quadrature_score=float(B_score[j]),adaptive_score=reference,difference=float(B_score[j]-reference),relative_integration_error_estimate=err))
        independent_error=max(abs(r['difference']) for r in check);ready=bool(converged and independent_error<1e-4)
        row=dict(contest_id=cid,split=split,cutoff_hour=cut,forecast_hours=len(zf),training_count_i=int(nt[0]),training_count_j=int(nt[1]),holdout_count_i=int(test[0].sum()),holdout_count_j=int(test[1].sum()),A_mean_joint_logscore=float(A_score.mean()),B_mean_joint_logscore=float(B_score.mean()),B_minus_A_mean_joint_logscore=float((B_score-A_score).mean()),beta_posterior_mean=diag['mean'],quadrature_order=order,quadrature_converged=converged,independent_max_score_error=independent_error,numerical_score_check_passed=ready)
        rows.append(row);state['records'].append(dict(**row,adaptive_checks=check,training_log_integral_gh=diag['log_integral'],training_log_integral_adaptive=logZ,training_integral_relative_error_estimate=quaderr,input_sha256=source_record['input_sha256']))
        for j in range(len(zf)):
            hourrows.append(dict(contest_id=cid,split=split,hour_index=cut+j+1,count_i=int(test[0,j]),count_j=int(test[1,j]),A_joint_logscore=float(A_score[j]),B_joint_logscore=float(B_score[j]),A_mean_i=float(A_mean[j,0]),A_mean_j=float(A_mean[j,1]),B_mean_i=float(B_mean[j,0]),B_mean_j=float(B_mean[j,1]),A_zero_i=float(A_zero[j,0]),A_zero_j=float(A_zero[j,1]),B_zero_i=float(B_zero[j,0]),B_zero_j=float(B_zero[j,1])))
    state['elapsed_seconds']=time.monotonic()-clock;save()
    if len(rows)%20==0:print('Baseline forecasts',len(rows)//2,'/73',flush=True)
for name,data in [('task4_baseline_contests_20261025.csv',rows),('task4_baseline_predictions_20261025.csv',hourrows)]:
    with (report/name).open('w',newline='') as h:
        w=csv.DictWriter(h,fieldnames=data[0]);w.writeheader();w.writerows(data)
state['status']='completed';state['all_numerical_checks_passed']=all(r['numerical_score_check_passed'] for r in rows);state['all_four_models_compared']=False
rng=np.random.default_rng(2026102554);state['summaries']={}
for split in ['primary_hour','complete_day']:
    rr=[r for r in rows if r['split']==split];diff=np.array([r['B_minus_A_mean_joint_logscore'] for r in rr]);boot=diff[rng.integers(0,len(diff),size=(2000,len(diff)))].mean(axis=1)
    state['summaries'][split]=dict(contests=len(rr),forecast_hours=sum(r['forecast_hours'] for r in rr),mean_B_minus_A_joint_logscore=float(diff.mean()),median_B_minus_A_joint_logscore=float(np.median(diff)),positive_difference_contests=int((diff>0).sum()),contest_bootstrap_95_interval=np.quantile(boot,[.025,.975]).tolist(),ranking_reportable=all(r['numerical_score_check_passed'] for r in rr))
save();print(json.dumps(dict(status=state['status'],all_numerical_checks_passed=state['all_numerical_checks_passed'],summaries=state['summaries']),indent=2),flush=True)
```
