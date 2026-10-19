"""Fresh NUTS recovery under the broad external-scale sensitivity prior."""
import json
import numpy as np
from cmdstanpy import CmdStanModel,from_csv
from pipeline.empirical import nonlinear_reestimate as n
from pipeline.empirical import estimation_detail_audit as a
from pipeline.empirical import paired_prior_improvement as p
from pipeline.empirical import paired_activity_sensitivity as s
from pipeline.empirical import paired_prior_report as report


def run():
    fixture=a.OUT/'feedback_timing/recovery_4657/leading_control'
    previous=fixture/'interval_start_r_weak'
    config=json.loads((previous/'run.json').read_text())
    data=dict(config['inputs'],**s.target(config['inputs']))
    folder=p.OUT/'recovery_4657/leading_control/activity_scale/leading';folder.mkdir(parents=True,exist_ok=True)
    model=CmdStanModel(stan_file=str(a.OUT/'feedback_timing/feedback_model.stan'),exe_file=str(a.OUT/'feedback_timing/feedback_model'))
    old=np.load(previous/'posterior.npz')['values'].reshape(-1,6)
    rng=np.random.default_rng(2017657)
    starts=[dict(zip(('c_i','c_j','sigma','lambda','mu_0','log_r'),old[k].tolist())) for k in rng.choice(len(old),3,replace=False)]
    truth=json.loads((fixture/'truth.json').read_text())['truth']
    starts.append({key:truth[key] for key in ('c_i','c_j','sigma','lambda','mu_0','log_r')})
    run_config=dict(inputs=data,starts=starts,seed=2017657,warmup=1500,samples=2500,adapt_delta=.99,
                    model_sha256=n.digest(a.OUT/'feedback_timing/feedback_model.stan'),
                    fixture_sha256=n.digest(previous/'run.json'),prior_design_sha256=n.digest(p.OUT/'activity_prior_design.json'))
    signature=folder/'run.json'
    if signature.exists() and json.loads(signature.read_text())!=run_config:raise RuntimeError('Control configuration changed')
    n.save_json(signature,run_config)
    files=sorted(folder.glob('feedback_model-*.csv'))
    if len(files)==4:fit=from_csv([str(f) for f in files])
    elif files:raise RuntimeError('Partial control run')
    else:fit=model.sample(data=data,inits=starts,chains=4,parallel_chains=4,iter_warmup=1500,iter_sampling=2500,
         seed=2017657,metric='dense_e',adapt_delta=.99,max_treedepth=12,sig_figs=18,
         output_dir=str(folder),show_progress=False,refresh=500)
    if fit.num_draws_sampling!=2500:raise RuntimeError('Incomplete control')
    draws=fit.draws();values=np.stack([draws[:,:,fit.column_names.index(k)] for k in ('c_i','c_j','sigma','lambda','mu_0','log_r')],axis=2)
    np.savez_compressed(folder/'posterior.npz',values=values)
    method=fit.method_variables();energy=method['energy__']
    n.save_json(folder/'sampler.json',dict(divergences=int(method['divergent__'].sum()),
        treedepth_hits=int((method['treedepth__']>=12).sum()),minimum_bfmi=float(np.min(np.mean(np.diff(energy,axis=0)**2,axis=0)/np.var(energy,axis=0)))))
    summaries,diagnostics,_=report.leading_result(folder,data)
    result=dict(diagnostics=diagnostics,truth=truth,summaries=summaries,
                truth_in_95_interval={key:summaries[key]['q025']<=truth[key]<=summaries[key]['q975'] for key in n.NAMES},
                median_relative_error={key:abs(summaries[key]['median']/truth[key]-1) for key in ('c_i','c_j','sigma','lambda','r')},
                scope='One retained high-activity realization; this is not repeated-sample coverage or nonlinear-approximation validation.')
    n.save_json(folder/'summary.json',result)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':run()
