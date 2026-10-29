"""Independent CmdStan checks for the recovered C/D prediction targets."""
import json
import math
import numpy as np
from pipeline.publication import ROOT,digest
from pipeline.jk9_fit import training_data
from pipeline.empirical.unified_prior_reestimate import prior
from pipeline.prediction.importance import Target
from pipeline.storage import compile_stan


def check(raw,cutoff):
    from cmdstanpy import CmdStanModel
    data=training_data(raw,'early',cutoff);data.update(prior(data));records=[]
    for free in (False,True):
        source=ROOT/'pipeline/models'/('free_kernel.stan' if free else 'leading_order.stan')
        model=compile_stan(source)
        target=Target(raw,cutoff,free);offsets=[]
        for j in range(3):
            params=dict(c_i=10+j,c_j=12-j,sigma=1+.1*j,**{'lambda':4.+j},mu_0=.1+.03*j,log_r=math.log(100)+.1*j)
            if free:params.update(deadline_power=.4+.1*j,gap_log_scale=-.2+.1*j)
            z=np.array([[math.log(params[k]) if i<4 else params[k] for i,k in enumerate(target.names)]])
            logtarget,_=target.evaluate(z)
            reference=float(model.log_prob(params,data=data,jacobian=False,sig_figs=18).iloc[0,0])
            offsets.append(float(logtarget[0]-z[0,:4].sum()-reference))
        error=max(offsets)-min(offsets)
        if error>=1e-7:raise ValueError(f'Target density mismatch: {error}')
        records.append(dict(model='C' if free else 'D',model_sha256=digest(source),parameter_points=3,
                            max_density_offset_variation=error,passed=True))
    return records


def main():
    from pipeline.jk9_prediction import raw_input,splits
    row=splits(2445,'primary_hour')[0]
    print(json.dumps(check(raw_input(row),row['cutoff_hour']),indent=2))


if __name__=='__main__':main()
