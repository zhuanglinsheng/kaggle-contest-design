"""Verify the complete isolated empirical revision against inputs and chains."""
import csv
import json
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
from pipeline.empirical import unified_prior_reestimate as u,unified_prior_medians as med,unified_prior_auxiliary as aux
from pipeline.empirical import unified_prior_adopt_global as global_export


def main():
    ids=sorted({int(r['contest_id']) for r in csv.DictReader((u.MAIN_ESTIMATION_DIR/'accepted_chain_sources.csv').open())})
    primary=pd.read_csv(u.OUT/'estimation/posterior_summary_final.csv').set_index(['contest_id','mode'])
    auxiliary=pd.read_csv(u.OUT/'auxiliary/posterior_summary.csv').set_index(['scenario','contest_id','mode'])
    if len(primary)!=146 or not primary.index.is_unique or len(auxiliary)!=251 or not auxiliary.index.is_unique:raise RuntimeError('Published posterior summaries incomplete or duplicated')
    required=[('main',cid,mode) for cid in ids for mode in ('full','early')]
    required.extend((scenario,cid,'full') for scenario in ('pair','L1','L2') for cid in ids)
    required.extend((scenario,cid,mode) for scenario in ('seed_20261008','seed_20261009') for cid in aux.TIME_IDS for mode in ('full','early'))
    rows=[]
    for scenario,cid,mode in required:
        case=(u.OUT if scenario=='main' else u.OUT/'auxiliary'/scenario)/'estimation'/mode/f'contest_{cid}'
        accepted=case/'accepted.json';a=json.loads(accepted.read_text());d=a['diagnostics'];cfg=a['configuration']
        raw=(u.MAIN_DATA_DIR if scenario=='main' else aux.input_directory(scenario))/f'contest_{cid}.json'
        rawdata=json.loads(raw.read_text());data=u.e.stan_data(rawdata,mode);expected=dict(data,**u.prior(data))
        if cfg['inputs']!=expected or cfg['original_input_sha256']!=u.digest(raw):raise RuntimeError(f'Input or prior mismatch: {scenario} {cid} {mode}')
        if cfg['model_sha256']!=u.digest(u.MODEL.with_suffix('.stan')) or cfg['executable_sha256']!=u.digest(u.MODEL):raise RuntimeError('Changed posterior implementation')
        if d.get('median_protocol_sha256')!=u.digest(med.__file__):raise RuntimeError('Point-convention protocol mismatch')
        if not d['passed']:raise RuntimeError('Rejected fit')
        if d.get('hmc_applicable',True):
            if a['summary']['divergences'] or d['saturated_transitions'] or min(d['energy_bfmi'])<.3:raise RuntimeError('Rejected HMC fit')
        else:
            if d.get('sampler_family')!='fixed_mixture_metropolis_hastings' or a['summary']['divergences'] is not None or d['saturated_transitions'] is not None or d['energy_bfmi']!=[]:raise RuntimeError('Non-HMC diagnostics mislabeled')
            reference=Path(d['global_reference'])
            parity=json.loads((reference/'target_parity.json').read_text())
            if not parity['passed'] or u.digest(reference/'target_parity.json')!=d['global_target_parity_sha256'] or u.digest(global_export.__file__)!=d['mh_export_source_sha256']:raise RuntimeError('Unverified global posterior refinement')
        if not all(v['rank_rhat']<=1.01 and min(v['bulk_ess'],v['tail_ess'])>=400 for v in d['quantities'].values()):raise RuntimeError('Rejected rank/ESS diagnostics')
        chains=[]
        for p,h in zip(a['chain_paths'],a['chain_sha256']):
            path=Path(p)
            if u.digest(path)!=h:raise RuntimeError('Changed accepted chain')
            chains.append(u.e.read_columns(path))
        lengths={len(c['lambda']) for c in chains}
        if len(chains)!=4 or len(lengths)!=1:raise RuntimeError('Invalid chain lengths')
        for name in u.NAMES:
            values=np.concatenate([np.exp(c['log_r']) if name=='r' else c[name] for c in chains])
            if not np.isclose(np.median(values),a['summary'][name+'_median'],rtol=1e-12,atol=1e-12):raise RuntimeError('Mismatched primary median')
            if not np.isclose(np.mean(values),a['summary'][name+'_mean'],rtol=1e-12,atol=1e-12):raise RuntimeError('Raw-mean sensitivity is mislabeled')
            published=primary.loc[(cid,mode)] if scenario=='main' else auxiliary.loc[(scenario,cid,mode)]
            if not np.isclose(published[name+'_median'],a['summary'][name+'_median'],rtol=1e-12,atol=1e-12):raise RuntimeError('Published median differs from accepted posterior')
        snapshot=u.OUT/'input_snapshots'/scenario/f'contest_{cid}_{u.digest(raw)}.json'
        snapshot.parent.mkdir(parents=True,exist_ok=True)
        if snapshot.exists() and snapshot.read_bytes()!=raw.read_bytes():raise RuntimeError('Conflicting frozen input')
        if not snapshot.exists():shutil.copy2(raw,snapshot)
        rows.append(dict(scenario=scenario,contest_id=cid,mode=mode,sampler_family=d.get('sampler_family','nuts'),draws_per_chain=lengths.pop(),accepted_sha256=u.digest(accepted),
                         input_snapshot=str(snapshot.relative_to(u.OUT)),input_sha256=u.digest(raw),
                         maximum_rank_rhat=max(v['rank_rhat'] for v in d['quantities'].values()),
                         minimum_bulk_ess=min(v['bulk_ess'] for v in d['quantities'].values()),
                         minimum_tail_ess=min(v['tail_ess'] for v in d['quantities'].values()),chain_sha256=a['chain_sha256']))
        print('VERIFIED',scenario,cid,mode,flush=True)
    frame=pd.read_csv(u.OUT/'validation/chapter5_contest_level.csv')
    if len(frame)!=73 or set(frame.contest_id)!=set(ids):raise RuntimeError('Published main diagnostics use the wrong sample')
    for _,row in frame.iterrows():
        for name in u.NAMES:
            if not np.isclose(row[name],primary.loc[(int(row['contest_id']),'full'),name+'_median'],rtol=1e-12,atol=1e-12):raise RuntimeError('Regression inputs are not the declared posterior medians')
    regressions=pd.read_csv(u.OUT/'validation/chapter5_regressions.csv')
    policies=pd.read_csv(u.OUT/'counterfactual/precision_policy_contests.csv')
    if len(regressions)!=6 or not (regressions.n==73).all() or len(policies)!=73 or set(policies.contest_id)!=set(ids):raise RuntimeError('Derived main outputs incomplete')
    sampler=json.loads((u.OUT/'validation/same_prior_sampler_report.json').read_text())
    if not sampler['complete'] or sampler['flagged_over_five_approximate_median_mcse']:raise RuntimeError('Independent sampler comparisons require review')
    parity=json.loads((u.OUT/'target_parity_all146.json').read_text())
    if not parity['passed'] or parity['fits']!=146:raise RuntimeError('Target parity unresolved')
    joint_parity=json.loads((u.OUT/'target_parity_joint_all146.json').read_text())
    if not joint_parity['passed'] or joint_parity['fits']!=146:raise RuntimeError('Joint target parity unresolved')
    u.save(u.OUT/'verified_fits.json',dict(complete=True,fits=len(rows),main_fits=146,auxiliary_fits=251,checks=rows,
          source_sha256=u.digest(__file__),sampler_report_sha256=u.digest(u.OUT/'validation/same_prior_sampler_report.json'),
          source_target_parity_sha256=u.digest(u.OUT/'target_parity_all146.json'),joint_target_parity_sha256=u.digest(u.OUT/'target_parity_joint_all146.json')))
    print('ALL 397 FITS VERIFIED',flush=True)


if __name__=='__main__':main()
