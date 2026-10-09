"""Print a paper patch from the accepted Chapter 5 numerical report.

Never write the manuscript here: inspect and apply the printed patch separately.
The estimates must first pass finalization and regenerate the diagnostic report.
"""
from __future__ import annotations

import difflib
import hashlib
import json
import re
from pathlib import Path

from pipeline.config import MAIN_ESTIMATION_DIR, MAIN_VALIDATION_DIR, PAPER_DIR


def blue(value):
    return r"\textcolor{blue}{" + str(value) + "}"


def stars(p):
    return r"\textbf{" + ("***" if p < .001 else "**" if p < .01 else "*") + "}" if p < .05 else ""


def number(value, p=None):
    return f"{value:.3f}" + (stars(p) if p is not None else "")


def probability(p):
    return "$p<0.001$" if p < .001 else f"$p={p:.3f}$"


def environment(source, kind, label):
    for match in re.finditer(r"\\begin\{" + kind + r"\}.*?\\end\{" + kind + r"\}", source, re.S):
        if r"\label{" + label + "}" in match.group():
            return match.group()
    raise ValueError(f"Missing {kind}: {label}")


def figure(block, row):
    lines = block.splitlines()
    coefficients = 0
    for i, line in enumerate(lines):
        if line.lstrip().startswith(r"& \bluecell"):
            key = "constant" if coefficients == 0 else "coefficient"
            se = "constant_se" if coefficients == 0 else "std_error"
            lo, hi = ("constant_ci_low", "constant_ci_high") if coefficients == 0 else ("ci_low", "ci_high")
            p = "constant_p" if coefficients == 0 else "p_value"
            lines[i] = "\t\t\t\t& " + " & ".join(blue(x) for x in (
                number(row[key], row[p]), number(row[se]), f"[{row[lo]:.3f}, {row[hi]:.3f}]")) + r" \\"
            coefficients += 1
        elif r"\bluecell{R$^2$}" in line:
            lines[i] = r"\bluecell{R$^2$} & \multicolumn{3}{c}{" + blue(number(row["r2"])) + r"} \\"
        elif r"\bluecell{Adj. R$^2$}" in line:
            lines[i] = r"\bluecell{Adj. R$^2$} & \multicolumn{3}{c}{" + blue(number(row["adj_r2"])) + r"} \\"
        elif "OLS residual diagnostics:" in line:
            lines[i] = blue("(2) OLS residual diagnostics: " + diagnostics(row) + ".")
        elif "Condition Number =" in line:
            # Existing figure note closes its significance-level group here.
            trailing = "}" if line.endswith("}}") else ""
            lines[i] = blue(f"(3) Condition Number = {row['condition_number']:.3f}.") + trailing
    if coefficients != 2:
        raise ValueError("Expected two coefficient rows in diagnostic figure")
    return "\n".join(lines)


def diagnostics(row):
    return ", ".join(f"{label} = {row[key]:.3f}" for label, key in (
        ("Omnibus", "omnibus"), ("Jarque--Bera", "jarque_bera"),
        ("Skewness", "skewness"), ("Kurtosis", "kurtosis")))


def table(block, full, early, signal):
    lines = block.splitlines()
    for i, line in enumerate(lines):
        name = next((name for name in ("Full path", "Time split") if line.startswith(name + " &")), None)
        if name:
            row = full if name == "Full path" else early
            cells = (number(row["coefficient"], row["p_value"]), number(row["std_error"]),
                     f"[{row['ci_low']:.3f}, {row['ci_high']:.3f}]", number(row["r2"]),
                     number(row["spearman_rho"], row["spearman_p"]))
            lines[i] = name + " & " + " & ".join(map(blue, cells)) + r" \\"
        elif "Omnibus =" in line:
            values = ", ".join(f"{label} = {full[key]:.3f}; {early[key]:.3f}" for label, key in (
                ("Omnibus", "omnibus"), ("Jarque--Bera", "jarque_bera"),
                ("Skewness", "skewness"), ("Kurtosis", "kurtosis")))
            if signal:
                lines[i] = blue("(2) Residual-distribution diagnostics: Omnibus and Jarque--Bera test residual normality; skewness and kurtosis describe asymmetry and tail heaviness. Values are reported as Full path; Time split: " + values + ".")
            else:
                lines[i] = blue("(2) OLS residual diagnostics (Full path; Time split): " + values + ".")
        elif "(3)" in line and ("conditioning:" in line or "Condition Number" in line):
            trailing = "}" if not signal and line.endswith("}}") else ""
            lines[i] = blue(f"(3) Condition Number (Full path; Time split): {full['condition_number']:.3f}; {early['condition_number']:.3f}.") + trailing
    return "\n".join(lines)


def patch(path, old, new):
    if old == new:
        return ""
    hunks = list(difflib.unified_diff(old.splitlines(True), new.splitlines(True), n=3))[2:]
    body = re.sub(r"^@@ -\d+(?:,\d+)? \+\d+(?:,\d+)? @@.*$", "@@", "".join(hunks), flags=re.M)
    return "*** Update File: " + str(path) + "\n" + body


def main():
    report = json.loads((MAIN_VALIDATION_DIR / "chapter5_paper_numbers.json").read_text())
    expected = hashlib.sha256((MAIN_ESTIMATION_DIR / "posterior_summary_final.csv").read_bytes()).hexdigest()
    if report.get("estimation_sha256") != expected:
        raise ValueError("Regenerate the report from the current accepted estimates first")
    if report["fits"] != 146 or report["divergences"] or report["maximum_rhat"] > 1.01:
        raise ValueError("Refusing publication of incomplete or unconverged estimates")
    rows = {row["specification"]: row for row in report["regressions"]}
    path = PAPER_DIR / "PaperJK7.tex"
    old = path.read_text()
    new = re.sub(r"\\footnote\{\\textcolor\{blue\}\{The numerical results below are retained.*?in progress\.\}\}", "", old)
    for label, key in (("fig-validate-lambda", "lambda_benchmark"), ("fig-validation-sigma", "sigma_benchmark")):
        block = environment(new, "figure", label)
        new = new.replace(block, figure(block, rows[key]), 1)
    for label, key, signal in (("tab-signal-disagreement-validation", "signal_disagreement", True),
                               ("tab-private-volatility-diagnostics", "private_volatility", False)):
        block = environment(new, "table", label)
        new = new.replace(block, table(block, rows[key + "_full"], rows[key + "_time_split"], signal), 1)
    start = new.index(r"{\color{blue}The 146 full- and early-sample estimates")
    stop = new.index("\n\n", start)
    text = ("The 146 full- and early-sample estimates contain no divergent transitions, and their MCMC convergence diagnostics satisfy $\\widehat R<1.01$. Across the 146 full-sample player--contest estimates, the posterior median of $w_k$ has a median of "
            f"${report['median_player_posterior_median_w']:.3f}$ and is below one in {report['players_with_posterior_median_w_below_one']} cases. At the stricter contest level, the posterior median of $\\max\\{{w_i,w_j\\}}$ is below one in {report['contests_with_posterior_median_max_w_below_one']} of 73 contests. "
            "Because these incentive indices are not uniformly small, Theorem~\\ref{thm-equilibrium-effort} does not provide a uniform approximation-error guarantee across the empirical sample. The counterfactuals below are therefore leading-order design benchmarks, rather than exact nonlinear-equilibrium predictions.")
    new = new[:start] + blue(text) + new[stop:]
    row = rows["lambda_benchmark"]
    start = new.index("{The estimated slope is")
    stop = new.index("\n\n", start)
    association = "positive" if row["coefficient"] > 0 else "negative"
    text = (f"The estimated slope is ${row['coefficient']:.3f}$ with a 95\\% confidence interval of $[{row['ci_low']:.3f},{row['ci_high']:.3f}]$. "
            f"The rank correlation is Spearman $\\rho={row['spearman_rho']:.3f}$ ({probability(row['spearman_p'])}). "
            f"The {association} association is a consistency diagnostic: the benchmark is approximate and the structural estimate also uses submission behavior.")
    new = new[:start] + blue(text) + new[stop:]
    row = rows["sigma_benchmark"]
    start = new.index(r"{\color{blue}The estimated slope is", new.index(r"\label{fig-validation-sigma}"))
    stop = new.index("\n\n", start)
    text = (f"The estimated slope is ${row['coefficient']:.3f}$ with a 95\\% confidence interval of $[{row['ci_low']:.3f},{row['ci_high']:.3f}]$. "
            f"The rank correlation is Spearman $\\rho={row['spearman_rho']:.3f}$ ({probability(row['spearman_p'])}). "
            "The benchmark remains model-assisted, so this relationship is interpreted as consistency evidence rather than independent validation of latent diffusion risk.")
    new = new[:start] + blue(text) + new[stop:]
    start = new.index("Both slopes are negative, consistent")
    stop = new.index("\n\\medskip", start)
    full, early = rows["signal_disagreement_full"], rows["signal_disagreement_time_split"]
    text = (f"The full-path slope is ${full['coefficient']:.3f}$ ({probability(full['p_value'])}), whereas the time-split slope is ${early['coefficient']:.3f}$ ({probability(early['p_value'])}). ")
    text += ("The time-split confidence interval includes zero, so evidence that early precision estimates predict disagreement in the final quarter remains inconclusive."
             if early["ci_low"] <= 0 <= early["ci_high"] else "The time-split interval excludes zero; this diagnostic separates early estimation from later disagreement but remains conditional on focal-pair selection.")
    new = new[:start] + blue(text) + new[stop:]
    start = new.index(r"{\color{blue}Both volatility regressions")
    stop = new.index("\n\n", start)
    full, early = rows["private_volatility_full"], rows["private_volatility_time_split"]
    text = (f"The full-path volatility slope is ${full['coefficient']:.3f}$ ({probability(full['p_value'])}); the time-split slope is ${early['coefficient']:.3f}$ ({probability(early['p_value'])}). "
            "Because structural innovation risk and private-path volatility are expressed in transformed-score units, these results are consistency diagnostics rather than scale-free validation evidence.")
    new = new[:start] + blue(text) + new[stop:]
    appendix = PAPER_DIR / "appendix_100k_rows.tex"
    print("*** Begin Patch\n" + patch(path, old, new) + patch(appendix, appendix.read_text(), report["appendix_rows"]) + "*** End Patch")


if __name__ == "__main__":
    main()
