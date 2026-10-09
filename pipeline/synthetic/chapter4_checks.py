"""Audit numerical integration using retained posterior draws, not only truth."""
import json
import numpy as np
from cmdstanpy import from_csv
from pipeline.synthetic.chapter4 import OUT,NAMES,PARAMS,write_json
from pipeline.synthetic.nonlinear import likelihood_data,log_likelihood


def main():
    manifest=json.loads((OUT/'manifest.json').read_text())
    results={}
    for name in NAMES:
        run_path=OUT/f'{name}_run.json'
        if not run_path.exists(): continue
        run=json.loads(run_path.read_text())
        fit=from_csv(run['csv_files'])
        contests=manifest['experiments'][name]['contests']
        for c in contests: c['events']=np.asarray(c['events'])
        coarse=json.loads((OUT/f'{name}.json').read_text())
        fine=likelihood_data(contests,estimate_r=bool(coarse['estimate_r']),dt=1/48)
        values={p:fit.stan_variable(p).reshape(-1) for p in PARAMS}
        if coarse['estimate_r']: values['r_free']=fit.stan_variable('r_free').reshape(-1)
        indices=np.linspace(0,len(values['S0'])-1,24,dtype=int)
        corrections=[]
        for i in indices:
            p={k:v[i] for k,v in values.items()}
            corrections.append(log_likelihood(fine,p)-log_likelihood(coarse,p))
        centered=np.asarray(corrections)-np.mean(corrections)
        weights=np.exp(centered-centered.max())
        ess=weights.sum()**2/(weights@weights)
        results[name]=dict(draws_checked=len(indices),mean_log_density_correction=float(np.mean(corrections)),
                           sd_log_density_correction=float(np.std(corrections)),
                           max_centered_correction=float(abs(centered).max()),
                           importance_ess_fraction=float(ess/len(indices)))
        print(name,results[name],flush=True)
    write_json(OUT/'integration_checks.json',results)


if __name__=='__main__': main()
