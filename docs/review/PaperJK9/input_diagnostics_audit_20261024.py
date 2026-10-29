from pathlib import Path
import json, math, hashlib
import numpy as np
import pandas as pd
PRIZE_DIVISOR=100
ROOT=Path(__file__).resolve().parents[3]
EMP=ROOT/'results/reestimation/lambda100_sd225'
def early_cutoff(raw):
    total=int(raw['N_Delta']);prefix=min(int(raw.get('initialization_hours',24)),total)
    return max(2,prefix+int(math.floor(.75*(total-prefix))))
def event_indices(data: dict) -> list[int]:
    n = int(data['N_Delta'])
    stan_indices = {max(1, min(n, int(math.ceil(float(value))))) for value in list(data['hat_t_i']) + list(data['hat_t_j'])}
    return sorted((index - 1 for index in stan_indices))

def implied_paths(data: dict, params: pd.Series) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = int(data['N_Delta'])
    dt = float(data['Delta2f'])
    horizon = n * dt
    theta = float(data['theta']) / PRIZE_DIVISOR
    sigma = float(params['sigma'])
    precision = float(params['lambda'])
    c_i = float(params['c_i'])
    c_j = float(params['c_j'])
    hat_y = np.asarray(data['hat_y'], dtype=float)
    tilde = np.empty(n + 1)
    effort_i = np.empty(n)
    effort_j = np.empty(n)
    tilde[0] = float(params['mu_0'])
    bar_s = sigma / math.sqrt(precision)
    weight = -math.expm1(-math.sqrt(precision) * sigma * dt)
    for index in range(n):
        remaining = horizon - index * dt
        q = sigma * sigma * remaining + bar_s
        kernel = math.exp(-tilde[index] ** 2 / (2.0 * q)) / math.sqrt(2.0 * math.pi * q)
        effort_i[index] = theta * kernel / c_i
        effort_j[index] = theta * kernel / c_j
        tilde[index + 1] = tilde[index] + (effort_i[index] - effort_j[index]) * dt + weight * (hat_y[index] - tilde[index])
    return (tilde, effort_i, effort_j)

def benchmark_row(data: dict, params: pd.Series) -> dict[str, float]:
    indices = event_indices(data)
    dt = float(data['Delta2f'])
    public = np.asarray(data['hat_y'], dtype=float)
    private = np.asarray(data['real_y'], dtype=float)
    previous_stan_index = 0
    weighted_sse = 0.0
    for index in indices:
        stan_index = index + 1
        elapsed = (stan_index - previous_stan_index) * dt
        weighted_sse += elapsed * (private[index] - public[index]) ** 2
        previous_stan_index = stan_index
    lambda_bench = len(indices) / weighted_sse if weighted_sse > 0 else math.nan
    _, effort_i, effort_j = implied_paths(data, params)
    sigma_terms = []
    previous = indices[0] if indices else 0
    for current in indices[1:]:
        if current <= previous:
            continue
        elapsed = (current - previous) * dt
        drift = float(np.sum(effort_i[previous + 1:current + 1] - effort_j[previous + 1:current + 1]) * dt)
        residual = private[current] - private[previous] - drift
        sigma_terms.append(residual * residual / elapsed)
        previous = current
    sigma_bench = math.sqrt(float(np.mean(sigma_terms))) if sigma_terms else math.nan
    diff = private - public
    increments = np.diff(private)
    prefix = min(int(data.get('path_start_index', 24)), len(private) - 2)
    cutoff = early_cutoff(data)
    late_diff = diff[cutoff:]
    late_increments = increments[cutoff - 1:]
    return {'lambda_bench': lambda_bench, 'sigma_bench': sigma_bench, 'D_path': float(np.mean(np.abs(diff[prefix:]))), 'D_late': float(np.mean(np.abs(late_diff))), 'V_path': float(np.std(increments[prefix:], ddof=1)), 'V_late': float(np.std(late_increments, ddof=1))}
frame=pd.read_csv(EMP/'validation/chapter5_contest_level.csv').set_index('contest_id')
summary=pd.read_csv(EMP/'estimation/posterior_summary_final.csv')
full=summary[summary['mode']=='full'].set_index('contest_id')
errors={};hashes=0
for path in sorted((EMP/'input_snapshots/main').glob('*.json')):
    raw=json.loads(path.read_text());cid=int(path.name.split('_')[1]); row=frame.loc[cid]
    assert hashlib.sha256(path.read_bytes()).hexdigest()==full.loc[cid,'input_sha256']
    hashes+=1
    for key,value in benchmark_row(raw,row).items():
        errors[key]=max(errors.get(key,0),abs(value-float(row[key])))
assert hashes==73
assert max(errors.values())<1e-8
out=dict(contests=hashes,verified_input_hashes=hashes,maximum_absolute_errors=errors,
 scope='Rebuild six diagnostics from frozen inputs at published posterior medians; no posterior resampling.')
Path(__file__).with_suffix('.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
