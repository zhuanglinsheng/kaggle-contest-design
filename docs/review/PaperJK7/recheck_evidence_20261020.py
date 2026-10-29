"""Second-pass evidence checks; no scientific inputs or results are modified."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.stats import norm, spearmanr

ROOT = Path(__file__).resolve().parents[3]
EMP = ROOT / "results/reestimation/lambda100_sd225"
out = {}

p = ROOT / "results/theory/numerical_joint_plane_moderate_dense/three_class_comparison"
unresolved = pd.read_csv(p / "unresolved_point_decisions.csv")
grid = pd.read_csv(p / "three_class_regions.csv")
visible = unresolved[unresolved.sigma.between(.03, 3) & unresolved.c.between(.03, .5)]
out["figure3"] = dict(grid_points=len(grid), unresolved=len(unresolved),
    visible_unresolved=len(visible), potentially_competitive_failures=int(visible.competitive_failed_design.sum()),
    interval_ranking_disagreements=int((visible.classification != visible.interval_winner).sum()),
    classifications=visible.classification.value_counts().to_dict(),
    source_matches_paper=hashlib.sha256((p/"numerical_regions_bw.pdf").read_bytes()).digest()
        == hashlib.sha256((ROOT/"paper/numerical_regions_bw.pdf").read_bytes()).digest())
assert out["figure3"]["source_matches_paper"]
assert (len(grid), len(unresolved), len(visible), out["figure3"]["potentially_competitive_failures"]) == (14553,1179,1179,926)

aux = pd.read_csv(EMP / "auxiliary/posterior_summary.csv")
main = pd.read_csv(EMP / "estimation/posterior_summary_final.csv")
full = main[main["mode"] == "full"].set_index("contest_id")
scales = json.loads((EMP/"auxiliary/reports/report.json").read_text())["scales"]
out["table12"] = []
for record in scales:
    f = aux[aux.scenario == record["scenario"]].set_index("contest_id")
    rho = {p:float(spearmanr(f[p+"_median"], full.loc[f.index,p+"_median"]).statistic)
           for p in ["lambda", "sigma"]}
    for p in rho:
        assert abs(rho[p]-record[p+"_rank_correlation"]) < 1e-12
    out["table12"].append(dict(scenario=record["scenario"], n=len(f), **rho))

changed = []
for f in (ROOT/"metakaggle/__jsondata__").glob("contest_*.json"):
    base = json.loads(f.read_text())
    other = json.loads((ROOT/"results/robustness/chapter6/inputs/pair"/f.name).read_text())
    if base["count_start_hour"] != other["count_start_hour"]:
        changed.append(int(f.stem.split("_")[-1]))
assert len(changed) == 43
out["changed_pair_count_windows"] = sorted(changed)

w = np.r_[full.w_i_median, full.w_j_median]
out["incentive_summaries"] = dict(player_contest_pairs_below_one=int((w<1).sum()),
    median=float(np.median(w)), contests_max_w_below_one=int((full.max_w_median<1).sum()))
assert out["incentive_summaries"]["player_contest_pairs_below_one"] == 76
assert out["incentive_summaries"]["contests_max_w_below_one"] == 34

# Independently integrate the Gaussian identity used by the first-order output.
# This tests the printed identity at representative arguments, not nonlinear
# approximation accuracy or the complete equilibrium proof.
identity_errors = []
for mu in [-1., 0., 1.]:
    for Q in [.2, .7, 1.5]:
        q = 2.
        numerical = quad(lambda x:norm.pdf(x,scale=np.sqrt(Q)) *
            (norm.cdf(x/np.sqrt(Q))-.5) * norm.pdf(x,loc=mu,scale=np.sqrt(q-Q)),
            -np.inf,np.inf,epsabs=1e-12)[0]
        formula = norm.pdf(mu,scale=np.sqrt(q)) * (
            norm.cdf(mu*np.sqrt(Q/(q*(2*q-Q))))-.5)
        identity_errors.append(abs(numerical-formula))
assert max(identity_errors) < 1e-10
out["gaussian_identity_max_absolute_error"] = max(identity_errors)

# A's precision term is multiplied by the cost asymmetry. It also vanishes
# at zero initial mean. These checks concern the approximation alone.
def coefficient(ci,cj,mu,precision):
    T, sigma, S0 = 1., .8, 1.2
    q = S0+sigma*sigma*T
    def variance(t):
        if precision == 0:return S0+sigma*sigma*t
        if np.isinf(precision):return 0.
        sbar = sigma/np.sqrt(precision)
        tanh = np.tanh(sigma*np.sqrt(precision)*t)
        return sbar*(S0+sbar*tanh)/(sbar+S0*tanh)
    H = quad(lambda t:norm.cdf(mu*np.sqrt((variance(t)+sigma*sigma*(T-t))/
        (q*(2*q-variance(t)-sigma*sigma*(T-t)))))-.5,0,T,epsabs=1e-12)[0]
    return norm.pdf(mu,scale=np.sqrt(q))*(T*(1/ci+1/cj)/2+(1/ci-1/cj)*H)
precisions = [0.,.1,1.,10.,np.inf]
out["first_order_channels"] = {
    "symmetric_costs": [coefficient(1,1,1,p) for p in precisions],
    "zero_initial_mean": [coefficient(1,2,0,p) for p in precisions],
    "asymmetric_positive_alignment": [coefficient(1,2,1,p) for p in precisions],
}
assert np.ptp(out["first_order_channels"]["symmetric_costs"]) < 1e-12
assert np.ptp(out["first_order_channels"]["zero_initial_mean"]) < 1e-12
assert np.all(np.diff(out["first_order_channels"]["asymmetric_positive_alignment"]) < 0)

out["scope"] = "Read-only saved-evidence checks and Gaussian quadrature; no resampling, no nonlinear equilibrium solves, no complete proof certification."
destination = Path(__file__).with_name("recheck_evidence_20261020.json")
destination.write_text(json.dumps(out,indent=2)+"\n")
print(json.dumps(out,indent=2))
