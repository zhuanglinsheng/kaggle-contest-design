"""Read-only paper audit from saved contest-level results, without resampling.

Run from any directory: python3 /absolute/path/to/this_file.py
Requires numpy, pandas, scipy. Writes only numerical_audit_20261024.json beside
this file. Source-level comparisons use the current PaperJK9 manuscript.
"""
from pathlib import Path
import json
import re
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[3]
EMP = ROOT / "results/reestimation/lambda100_sd225"
PAPER = (ROOT / "paper/PaperJK9.tex").read_text()
checks = []


def check(name, actual, expected, tolerance=1e-9):
    error = float(np.max(np.abs(np.asarray(actual, float) - np.asarray(expected, float))))
    checks.append(dict(check=name, passed=error <= tolerance, maximum_absolute_error=error,
                       tolerance=tolerance))


def table(label):
    start = PAPER.index("\\label{" + label + "}")
    return PAPER[start:PAPER.index("\\end{table}", start)]


def cells(line):
    return [re.sub(r"[{}$]", "", c).strip() for c in line.split("&")]


# Recompute OLS, HC1 covariance, normal-reference p values, and rank correlations.
data = pd.read_csv(EMP / "validation/chapter5_contest_level.csv")
reg = pd.read_csv(EMP / "validation/chapter5_regressions.csv").set_index("specification")
pairs = {
    "lambda_benchmark": ("log1p_lambda", "log1p_lambda_bench"),
    "sigma_benchmark": ("log1p_sigma", "log1p_sigma_bench"),
    "signal_disagreement_full": ("log1p_lambda", "log1p_D_path"),
    "signal_disagreement_time_split": ("log1p_lambda_early", "log1p_D_late"),
    "private_volatility_full": ("log1p_sigma", "log1p_V_path"),
    "private_volatility_time_split": ("log1p_sigma_early", "log1p_V_late"),
}
for name, (xc, yc) in pairs.items():
    x, y = data[xc].to_numpy(), data[yc].to_numpy()
    X = np.column_stack((np.ones(len(x)), x))
    inv = np.linalg.inv(X.T @ X)
    b = inv @ X.T @ y
    e = y - X @ b
    covariance = len(x)/(len(x)-2) * inv @ (X.T @ (e[:, None]**2 * X)) @ inv
    se = np.sqrt(np.diag(covariance))
    rho, rp = stats.spearmanr(x, y)
    expected = reg.loc[name]
    check(name, [b[0], b[1], se[1], 2*stats.norm.sf(abs(b[1]/se[1])),
                 1-e@e/np.sum((y-y.mean())**2), rho, rp],
          expected[["constant", "coefficient", "std_error", "p_value", "r2",
                    "spearman_rho", "spearman_p"]].astype(float).to_numpy())

printed_regressions = {
    "lambda_benchmark": [.386, .112, .167, .606, .121, .364],
    "sigma_benchmark": [.912, .173, .573, 1.250, .320, .597],
    "signal_disagreement_full": [-.013, .005, -.022, -.004, .064, -.346],
    "signal_disagreement_time_split": [-.015, .007, -.028, -.002, .029, -.343],
    "private_volatility_full": [.243, .049, .147, .339, .309, .674],
    "private_volatility_time_split": [.265, .095, .078, .452, .275, .386],
}
for name, values in printed_regressions.items():
    check("printed_main_regression_"+name, values,
          reg.loc[name, ["coefficient", "std_error", "ci_low", "ci_high", "r2", "spearman_rho"]], .00050001)

# Table 4 percentile bootstrap, preserving the producer's RNG order.
rank = pd.read_csv(ROOT / "results/validation/main/chapter5_rank_agreement.csv")
rng = np.random.default_rng(20261008)
indices = rng.integers(0, len(rank), size=(10000,len(rank)))
for col, values, tol in [("rank_correlation", [.937,.984,.902,.963], .00050001),
                          ("top2_overlap", [.651,.500,.568,.733], .00050001),
                          ("top5_overlap", [.666,.800,.597,.732], .00050001)]:
    a = rank[col].to_numpy()
    boot = a[indices].mean(axis=1)
    check("Table4_"+col, [a.mean(), np.median(a), *np.quantile(boot,[.025,.975])], values, tol)

# Table 11: compare all printed robustness coefficients, standard errors, CIs, R2.
aux = pd.read_csv(EMP / "auxiliary/reports/regressions.csv").set_index("specification")
lines = [l for l in table("tab-robustness-benchmarks").splitlines()
         if l.startswith(("Baseline", "Alternative pair", "$L_1$", "$L_2$"))]
names = ["lambda_benchmark", "pair_lambda_benchmark", "L1_lambda_benchmark", "L2_lambda_benchmark",
         "sigma_benchmark", "pair_sigma_benchmark", "L1_sigma_benchmark", "L2_sigma_benchmark"]
for name, line in zip(names, lines):
    values = [float(v) for v in re.findall(r"-?\d+\.\d+", line.split("&",1)[1])]
    source = reg if name in reg.index else aux
    check("Table11_"+name, values,
          source.loc[name,["coefficient","std_error","ci_low","ci_high","r2"]], .00050001)

# All referenced citation keys and TeX labels resolve after a clean build.
combined = PAPER + "\n" + "\n".join((ROOT/"paper"/p).read_text() for p in
    ["appendix_joint_design_jk9.tex","appendix_effort_recovery_jk9.tex","appendix_100k_rows.tex"])
combined = re.sub(r"(?m)(?<!\\)%.*$", "", combined)
bib = (ROOT/"paper/_Literatures.bib").read_text()
keys = re.findall(r"@\w+\s*\{\s*([^,]+),", bib)
cited = {k.strip() for group in re.findall(r"\\cite\w*\*?(?:\[[^\]]*\])*\{([^}]+)\}", combined)
         for k in group.split(",")}
labels = re.findall(r"\\label\{([^}]+)\}", combined)
references = set(re.findall(r"\\(?:ref|eqref)\{([^}]+)\}", combined))
check("missing_citation_keys",len({k.lower() for k in cited}-{k.lower() for k in keys}),0)
check("duplicate_bibliography_keys",len(keys)-len({k.lower() for k in keys}),0)
check("missing_tex_labels",len(references-set(labels)),0)
check("duplicate_tex_labels",len(labels)-len(set(labels)),0)

# Check all 73 printed rows, including counts, prizes, medians, benchmarks.
columns = ["contest_id", "Ni", "Nj", "theta", "public_data_pct", "lambda", "sigma",
           "mu_0", "c_i", "c_j", "r", "lambda_bench", "sigma_bench"]
rows = []
for line in (ROOT / "paper/appendix_100k_rows.tex").read_text().splitlines():
    values = re.findall(r"-?\d+(?:\.\d+)?", line)
    if values:
        rows.append([float(v) for v in values])
printed = pd.DataFrame(rows, columns=columns).set_index("contest_id")
expected = data.set_index("contest_id")[columns[1:]].sort_index()
check("appendix_73_rows_all_13_columns", printed.sort_index().to_numpy(),
      expected.to_numpy(), 0.00050001)
summary = pd.read_csv(EMP / "estimation/posterior_summary_final.csv")
full = summary[summary["mode"] == "full"].set_index("contest_id").sort_index()
for param in ["c_i", "c_j", "sigma", "lambda", "mu_0", "r"]:
    check("contest_level_median_"+param, expected[param], full[param+"_median"], 1e-8)

# Synthetic tables: all reported posterior means, SDs, RMSDs and Table 1 CIs.
params = ["c_i", "c_j", "sigma", "lambda", "mu_0", "r"]
specs = [
    ("tbl-contest90", ["initialization_comparison/steady", "initialization_comparison/nonsteady"]),
    ("tbl-pool-synthetic-data", ["richer_steady/pool10", "richer_steady/pool20"]),
    ("tbl-longterm-synthetic-data", ["richer_steady/days180", "richer_steady/days360"]),
]
for label, paths in specs:
    block = table(label)
    for tag, key in [("Posterior Mean", "mean"), ("Posterior SD", "sd"), ("Post. RMSD", "rmsd")]:
        lines = [l for l in block.splitlines() if "\\celltext{"+tag+"}" in l]
        for line, path in zip(lines, paths):
            values = [float(v) for v in re.findall(r"-?\d+\.\d+", line)]
            saved = json.loads((ROOT / "results/synthetic" / path / "summary.json").read_text())
            check(label+"/"+path+"/"+key, values, [saved[p][key] for p in params], .00050001)
    if label == "tbl-contest90":
        lines = [l for l in block.splitlines() if l.startswith("95\\% Interval")]
        for line, path in zip(lines, paths):
            values = [float(v) for v in re.findall(r"-?\d+\.\d+", line)]
            saved = json.loads((ROOT / "results/synthetic" / path / "summary.json").read_text())
            check(path+"/credible_intervals", values,
                  [saved[p][key] for p in params for key in ("q2_5", "q97_5")], .0050001)

# Reaggregate all joint policies; verify every printed Table 7 row.
policies = pd.read_csv(EMP / "fixed_initial_design/joint_policies.csv")
joint = json.loads((EMP / "fixed_initial_design/summary.json").read_text())["joint_summary"]
block = table("tab-counterfactual-precision")
numeric_lines = [l for l in block.splitlines() if re.match(r"^\d", l)]
printed_A = [[float(v) for v in re.findall(r"-?\d+(?:\.\d+)?", l)] for l in numeric_lines[:9]]
printed_B = [[float(v) for v in re.findall(r"-?\d+(?:\.\d+)?", l)] for l in numeric_lines[9:]]
for index, entry in enumerate(joint):
    g = policies[policies.delta == entry["delta"]]
    counts = g.choice.value_counts()
    actual_A = [entry["delta"], counts.get("full_disclosure", 0),
                counts.get("no_feedback_limit", 0)+counts.get("finite_precision", 0),
                counts.get("zero_prize", 0)]
    reduced = g[g.choice.isin(["no_feedback_limit", "finite_precision"])]
    actual_B = [entry["delta"], 100*(g.time_multiplier.median()-1),
                100*(g.prize_ratio.median()-1), g.cost_saving_pct.median(),
                reduced.additional_saving_pct.median()]
    check("Table7_PanelA_delta_"+str(entry["delta"]), printed_A[index], actual_A)
    check("Table7_PanelB_delta_"+str(entry["delta"]), printed_B[index], actual_B, .0050001)

fixed = pd.read_csv(EMP / "fixed_initial_design/fixed_precision.csv")
check("Table8_regime_counts", [(fixed.precision_kind=="full_disclosure").sum(),
      (fixed.precision_kind=="finite_precision").sum(),
      (fixed.precision_kind=="no_feedback_limit").sum()], [42, 0, 31])
check("Table8_direction_counts", [(fixed.direction==v).sum() for v in
      ["Increase precision", "Decrease precision", "Direction uncertain"]], [24, 16, 33])
for group, values in [(None, [.001,.011,.231]), ("Increase precision", [.001,.007,.038]),
                       ("Decrease precision", [.373,1.049,1.518]),
                       ("Direction uncertain", [.001,.004,.013])]:
    f = fixed if group is None else fixed[fixed.direction==group]
    check("Table8_quantiles_"+str(group), f.output_gain_pct.quantile([.25,.5,.75]), values, .00050001)

recovery = pd.read_csv(EMP / "validation/simulation_recovery_reference.csv")
for param, expected_values in [("lambda", [17.0, 97.5]), ("sigma", [12.1, 82.5])]:
    f = recovery[recovery.parameter==param]
    check("Table10_"+param, [100*f.median_relative_error.abs().mean(),
          100*f.original_credible_interval_covered.mean()], expected_values, .050001)

fits = dict(main_fits=len(summary), contests=len(full),
            maximum_rank_rhat=float(summary.filter(regex="_rank_rhat$").max().max()),
            minimum_bulk_ess=float(summary.filter(regex="_bulk_ess$").min().min()),
            minimum_tail_ess=float(summary.filter(regex="_tail_ess$").min().min()),
            divergences=int(summary.divergences.sum()))
report = dict(scope="Saved summaries and contest-level results; no chain resampling or nonlinear solves.",
              checks=checks, passed=sum(c["passed"] for c in checks),
              failed=sum(not c["passed"] for c in checks), diagnostics=fits)
out = Path(__file__).with_name("numerical_audit_20261024.json")
out.write_text(json.dumps(report, indent=2)+"\n")
print(json.dumps({"passed":report["passed"], "failed":report["failed"], "diagnostics":fits}, indent=2))
for c in checks:
    if not c["passed"]:
        print("FAILED", c)
