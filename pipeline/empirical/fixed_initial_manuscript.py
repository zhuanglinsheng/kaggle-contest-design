"""Update only design passages from fixed_initial_design's verified outputs."""
from __future__ import annotations

import csv
import datetime
import json
from pathlib import Path
import re
import shutil

import numpy as np
import pandas as pd

from pipeline.config import PAPER_DIR, ROOT
from pipeline.empirical.fixed_initial_design import DEST, digest


def number(value, digits=2):
    return '---' if value is None or pd.isna(value) else f'{value:.{digits}f}'


def main():
    report=json.loads((DEST/'summary.json').read_text())
    summary=pd.read_csv(DEST/'joint_summary.csv')
    fixed=pd.read_csv(DEST/'fixed_precision.csv')
    assert report['contests']==73 and len(fixed)==73
    assert (summary[['full_disclosure','finite_precision','no_feedback_limit','zero_prize','observed']].sum(axis=1)==73).all()
    backup=DEST/datetime.datetime.now().strftime('backup_%Y%m%d_%H%M%S')
    backup.mkdir()
    for name in ['PaperJK7.tex','PaperJK7.pdf','section_joint_design.tex','appendix_joint_design.tex','README.md']:
        shutil.copy2(PAPER_DIR/name,backup/name)
    section=r'''\subsection{\textcolor{blue}{Joint Design at a Fixed Quality Target}}\label{sect-precision-design}

\textcolor{blue}{We compare joint changes in the prize, deadline, and feedback precision that reduce costs while maintaining the baseline model's expected quality of the best solution to first order in the prize. A quality requirement avoids assigning a monetary value to solution quality. For each joint posterior draw, we retain both contestants' costs and fix the initial gap distribution at its baseline calibration, including $S_0=\sigma/\sqrt{\lambda^{obs}}$. Candidate precision changes subsequent learning, not this initial distribution. Let $\overline{\mathcal B}^{(1)}_\ell$ denote posterior mean first-order output and $\overline{\mathcal B}^{obs}_\ell$ its value under the observed regime, evaluated using the same formula. For alternative delay-cost levels $\delta>0$, we compare}
\begin{equation}\label{eq-empirical-joint-design}
\textcolor{blue}{\inf_{\theta,T,\lambda}\left\{\theta+\delta\theta_\ell^{obs}\left(\frac{T}{T_\ell^{obs}}\right)^2\right\}
\quad\text{subject to}\quad
\overline{\mathcal B}^{(1)}_\ell(\theta,T,\lambda)\geq\overline{\mathcal B}^{obs}_\ell.}
\end{equation}
\textcolor{blue}{Here $\delta$ is delay cost at the observed deadline relative to the observed prize. Prizes are nonnegative, precision ranges over $(0,\infty]$, and deadlines lie on the prespecified grid from 0.1 to 2 times the observed duration. We also evaluate the no-feedback limit $\lambda\downarrow0$; a solution at this boundary is a limiting cost bound, not an attained positive-feedback design. The observed regime remains available. Appendix~\ref{app-precision-counterfactual} gives the calculation and the fixed-prize, fixed-duration benchmark.}

\begin{table}[htbp]
\centering
\caption{\textcolor{blue}{Quality-Constrained First-Order Design Comparisons}}
\label{tab-counterfactual-precision}
{\color{blue}\small
\setlength{\tabcolsep}{4pt}
\begin{tabularx}{\textwidth}{@{}r*{5}{>{\centering\arraybackslash}X}@{}}
\multicolumn{6}{l}{\textit{Panel A: Design choices and limiting comparisons (number of contests)}}\\
\midrule
$\delta$ & \makecell{Positive prize,\\full disclosure} & \makecell{Positive prize,\\interior precision} & \makecell{Positive prize,\\no-feedback limit} & \makecell{Zero\\prize} & \makecell{Observed\\regime retained}\\
\midrule
'''
    for r in summary.itertuples():
        section+=f'{r.delta:g} & {r.full_disclosure} & {r.finite_precision} & {r.no_feedback_limit} & {r.zero_prize} & {r.observed} '+r'\\'+'\n'
    section+=r'''\bottomrule
\end{tabularx}
\vspace{2ex}
\begin{tabularx}{\textwidth}{@{}r*{4}{>{\centering\arraybackslash}X}@{}}
\multicolumn{5}{l}{\textit{Panel B: Design changes and cost reductions (medians)}}\\
\midrule
$\delta$ & \makecell{Duration /\\observed duration} & \makecell{Prize /\\observed prize} & \makecell{Total cost\\reduction (\%)} & \makecell{Additional reduction\\from allowing\\less disclosure (\%)}\\
\midrule
'''
    for r in summary.itertuples():
        section+=f'{r.delta:g} & {number(r.median_time)} & {number(r.median_prize)} & {number(r.median_saving)} & {number(r.median_additional_saving)} '+r'\\'+'\n'
    section+=r'''\bottomrule
\end{tabularx}
\begin{minipage}{\textwidth}
\footnotesize\textit{Note:} Each scenario covers 73 contests; the five categories in Panel A sum to 73. Precision is irrelevant at zero prize. No-feedback entries evaluate the limit as precision tends to zero, which is excluded from the positive-feedback design set. Their costs are limiting lower bounds and reductions limiting upper bounds. Total cost is prize plus calibrated delay cost; reductions are relative to $(1+\delta)\theta_\ell^{obs}$. All Panel B medians use 73 contests except the last column, which uses positive-prize choices with interior precision or the no-feedback limit and compares their costs with reoptimized full disclosure. Quality constraints use posterior mean first-order output, not a guarantee about realized quality or exact nonlinear output. Delay costs are scenarios, not estimated preferences.
\end{minipage}}
\end{table}

'''
    selected=summary.set_index('delta')
    counts=', '.join(fr"{int(selected.loc[d,'no_feedback_limit'])} at $\delta={d:g}$" for d in [1.,10.,25.])
    section+=r'\textcolor{blue}{Higher delay costs generally favor shorter contests supported by larger prizes. Positive-prize no-feedback limiting comparisons occur in '+counts+r'. These entries indicate the value of allowing very low precision within the first-order calculation; they do not establish an implementable zero-feedback optimum. The direction of the precision effect depends on the initial perceived gap and cost asymmetry, rather than on a change in the initial score distribution.}'+'\n\n'
    if int(summary.finite_precision.sum())==0:
        section+=r'\textcolor{blue}{No interior precision is selected in these scenarios. The first-order comparisons therefore do not provide empirical evidence for an interior optimum of the nonlinear model.}'+'\n\n'
    section+=r'''\textcolor{blue}{All three design instruments vary here under a fixed requirement for model-implied expected quality. Unlike the theoretical utility-maximization problem, these comparisons impose a quality target and use a first-order output expansion. Under symmetric costs or a zero initial perceived gap, this expansion is independent of precision; the symmetric finite-precision mechanism in Proposition~\ref{prop-joint-design} therefore requires higher-order or nonlinear analysis. Large prize changes remain extrapolations beyond the parameter-recovery check.}
'''
    (PAPER_DIR/'section_joint_design.tex').write_text(section)
    source=(PAPER_DIR/'PaperJK7.tex').read_text()
    begin=source.index(r'\subsection{Precision Counterfactual Calculations}')
    end=source.index(r'\input{appendix_joint_design.tex}',begin)
    calculation=r'''\subsection{Precision Counterfactual Calculations}\label{app-precision-counterfactual}

\textcolor{blue}{For each accepted joint posterior draw $s$, retain $(c_{i,s},c_{j,s},\sigma_s,\mu_{0,s},\lambda_s^{obs})$ and set $S_{0,s}^{base}=\sigma_s/\sqrt{\lambda_s^{obs}}$. Hold this initial variance and the initial mean fixed across candidate prizes, deadlines, and precisions. In the expressions below we suppress the contest and draw indices. The Riccati equation starts at $S_0=S_0^{base}$; write $q=S_0+\sigma^2T$ and $Q_t=S_t+\sigma^2(T-t)$. With $G_q$ the centered normal density of variance $q$ and $\beta(\mu,q)=\mathbb E|N(\mu,q)|/2$, the first-order expected output is}
\[
\begin{aligned}
\mathcal B^{(1)}(\theta,T,\lambda)
&=\beta(\mu_0,q)+\theta A(T,\lambda),\\
A(T,\lambda)
&=G_q(\mu_0)\left\{\frac T2\left(\frac1{c_i}+\frac1{c_j}\right)
+\left(\frac1{c_i}-\frac1{c_j}\right)H(T,\lambda)\right\},\\
H(T,\lambda)
&=\int_0^T\left[\Phi\!\left(\frac{\mu_0\sqrt{Q_t}}{\sqrt{q(2q-Q_t)}}\right)-\frac12\right]dt.
\end{aligned}
\]
\textcolor{blue}{To derive the coefficient, use the best-output drift identity in equation~\eqref{eq-best-output-drift-identity}, replacing its centered initial baseline by $\beta(\mu_0,q)$. Under zero effort the perceived gap $X_t$ has law $N(\mu_0,q-Q_t)$. Substitution of $m_k=\theta G_{Q_t}/c_k+O(\theta^2)$ gives}
\[
\begin{aligned}
\mathbb E[G_{Q_t}(X_t)]&=G_q(\mu_0),\\
\mathbb E\!\left[G_{Q_t}(X_t)\left\{\Phi(X_t/\sqrt{Q_t})-\tfrac12\right\}\right]
&=G_q(\mu_0)\left[\Phi\!\left(\frac{\mu_0\sqrt{Q_t}}{\sqrt{q(2q-Q_t)}}\right)-\tfrac12\right].
\end{aligned}
\]
\textcolor{blue}{These Gaussian identities yield the displayed coefficient. Changes in the state law due to first-order drift affect the effort contribution only at second order. The remainder is $O(\theta^2)$ for fixed positive finite precision and other primitives; it is not a uniform error bound for the optimized designs. The coefficient $A$ is positive. If $c_i=c_j$ or $\mu_0=0$, its precision-dependent component vanishes. Preserving the total-effort coefficient through a symmetric cost replacement would omit the asymmetric contribution to best output.}

\textcolor{blue}{The no-effort term $\beta(\mu_0,q)$ is the same at every precision. At the no-feedback limit, $S_t=S_0+\sigma^2t$ and $Q_t=q$. At full disclosure, contestants observe the initial gap; the organizer still averages over $N(\mu_0,S_0)$. The corresponding first-order coefficient is obtained by setting $Q_t=\sigma^2(T-t)$ for $t>0$ in the integral. Neither endpoint resets the initial score distribution.}

\textcolor{blue}{For each candidate duration, maximize the posterior average $\overline A(T,\lambda)$ over precision, including both endpoint limits. Time integration uses Gauss--Legendre quadrature after $t=T(1-u^2)$, which removes the square-root behavior at the full-disclosure terminal endpoint. The precision search evaluates a logarithmic grid and refines bracketed local maxima. Both the time quadrature and the search grid are independently doubled at all 584 contest-duration combinations; all accepted posterior draws enter the calculation. A no-feedback boundary maximizer is reported as a limiting comparison, not as an attained positive-precision design.}

\textcolor{blue}{The baseline quality target is recomputed with the same formula as the candidate designs:}
\[
\overline{\mathcal B}^{obs}_\ell
=\frac1{N_s}\sum_s\left[\beta(\mu_{0,\ell s},S_{0,\ell s}^{base}+\sigma_{\ell s}^2T_\ell^{obs})
+\theta_\ell^{obs}A_{\ell s}(T_\ell^{obs},\lambda_{\ell s}^{obs})\right].
\]
\textcolor{blue}{The selected precision is common across posterior draws, while the baseline integrates uncertainty in the observed precision and its associated initial variance. Table~\ref{tab-fixed-precision} holds the prize and duration fixed and compares posterior mean output at this selected precision with the baseline target. Directional support compares draw-specific preferred endpoints with each draw's observed precision: its direction is determined by the sign of $\mu_0(1/c_i-1/c_j)$, with a positive sign favoring less disclosure. These comparisons condition on the empirical parameter posterior.}

'''
    source=source[:begin]+calculation+source[end:]
    source=source.replace('using a steady-state, leading-order calibration.',
        'using a first-order output comparison that holds initial uncertainty fixed within each posterior draw.')
    source=source.replace('Quality-preserving design comparisons under the steady-state, leading-order calibration show how higher delay costs favor shorter contests and larger prizes, making finite precision more common. Full disclosure remains preferred in most contests across the reported scenarios.',
        'Fixed-initial-distribution, first-order design comparisons show how delay costs affect prizes and deadlines. Precision comparisons depend on initial perceived gaps and cost asymmetry; no-feedback boundary results are reported as limiting comparisons.')
    source=source.replace('The quality-constrained design comparisons are therefore calibrated benchmarks rather than exact joint-design predictions.',
        'The quality-constrained comparisons hold initial distributions fixed but remain conditional on the empirical posterior and a first-order output approximation, rather than exact nonlinear joint-design predictions.')
    # The historical score-scale policy counts used the superseded output map.
    # Retain all estimation/ranking evidence, without presenting those counts
    # as robustness of the corrected policy calculation.
    source=source.replace('Rank correlations with the baseline range from 0.945 to 0.999; full-disclosure versus finite-precision classifications agree in 72 and 72 of 73 contests, respectively.',
        'Parameter rank correlations with the baseline range from 0.945 to 0.999. This exercise concerns parameter estimates; it does not establish robustness of the design comparisons.')
    source=source.replace('robustness of the revised design comparisons','robustness of the design comparisons')
    start=source.index(r'\textcolor{blue}{For comparisons in common score units')
    stop=source.index(r'\FloatBarrier',start)
    source=source[:start]+r'''\textcolor{blue}{For comparisons in common score units, signal precision is $a^2\hat\lambda_a$ and innovation risk is $\hat\sigma_a/a$. Table~\ref{tab-score-scale-robustness} compares the resulting parameter rankings with the baseline. These correlations assess score-scale sensitivity of the estimates, rather than robustness of the design recommendations.}

\begin{table}[htbp]
\centering
\caption{Score-Scale Sensitivity of Parameter Rankings}
\label{tab-score-scale-robustness}
\begin{tabular}{lccc}
\toprule
Specification & $a$ & Rank correlation: $\lambda$ & Rank correlation: $\sigma$\\
\midrule
Baseline & 1.000 & --- & ---\\
$L_1$ & 1.444 & 0.992 & 0.945\\
$L_2$ & 0.781 & 0.999 & 0.996\\
\bottomrule
\end{tabular}
\begin{minipage}{\textwidth}
\footnotesize\textit{Note:} Each specification covers the same 73 contests. Correlations are Spearman correlations between common-unit posterior medians and baseline posterior medians. Re-estimation retains the baseline numerical priors in transformed units.
\end{minipage}
\end{table}
'''+source[stop:]
    (PAPER_DIR/'PaperJK7.tex').write_text(source)
    counts=report['fixed_classification'];directions=report['fixed_direction']
    appendix=r'\textcolor{blue}{\textit{Fixed-prize, fixed-duration benchmark.} Table~\ref{tab-fixed-precision} applies the asymmetric first-order calculation with fixed initial distributions at the observed prize and duration. It selects full disclosure in '+str(counts.get('full_disclosure',0))+r' contests and the no-feedback limit in '+str(counts.get('no_feedback_limit',0))+r'. '+('No contest selects interior precision. ' if counts.get('finite_precision',0)==0 else str(counts['finite_precision'])+' contests select interior precision. ')+r'The median output increase or limiting increase is '+number(report['fixed_gain_quantiles'][1],3)+r'\%. At the 75\% support threshold, '+str(directions.get('Increase precision',0))+r' contests favor increased precision, '+str(directions.get('Decrease precision',0))+r' favor reduced precision, and '+str(directions.get('Direction uncertain',0))+r' have no clear direction.}'+'\n\n'
    appendix+=r'''\begin{table}[htbp]
\centering
\caption{First-Order Precision Comparisons at the Observed Prize and Duration}
\label{tab-fixed-precision}
{\small
\begin{tabularx}{\textwidth}{Xrr}
\multicolumn{3}{l}{\textit{Panel A: Selected regime or boundary limit}}\\
\midrule
Regime & Contests & Share (\%)\\
\midrule
'''
    for label,key in [('Full disclosure','full_disclosure'),('Interior precision','finite_precision'),('No-feedback limit','no_feedback_limit')]:
        n=counts.get(key,0);appendix+=f'{label} & {n} & {100*n/73:.1f} '+r'\\'+'\n'
    appendix+=r'''\bottomrule
\end{tabularx}
\vspace{2ex}
\begin{tabularx}{\textwidth}{Xrrr}
\multicolumn{4}{l}{\textit{Panel B: Output increases or limiting increases (\%)}}\\
\midrule
Posterior direction group & 25th percentile & Median & 75th percentile\\
\midrule
'''
    for name in ['All contests','Increase precision','Decrease precision','Direction uncertain']:
        values=fixed.output_gain_pct if name=='All contests' else fixed.loc[fixed.direction==name,'output_gain_pct']
        vals=np.quantile(values,[.25,.5,.75]) if len(values) else [None]*3
        appendix+=name+' & '+' & '.join(number(v,3) for v in vals)+r'\\'+'\n'
    appendix+=r'''\bottomrule
\end{tabularx}}
\begin{minipage}{\textwidth}
\footnotesize\textit{Note:} Each contest uses one common precision selected by posterior mean first-order output. No-feedback entries are endpoint limits; their reported increases are limiting bounds under the positive-feedback restriction. Direction groups require more than 75\% posterior support. Percentiles describe variation across contests, not credible intervals. Baseline and candidate outputs use the same fixed initial distribution within each posterior draw.
\end{minipage}
\end{table}

\textcolor{blue}{\textit{Quality-constrained joint design.} Write $B_{0\ell}(T)=N_s^{-1}\sum_s\beta(\mu_{0,\ell s},S_{0,\ell s}^{base}+\sigma_{\ell s}^2T)$. At each candidate duration, select the precision maximizing $\overline A_\ell(T,\lambda)$, including endpoint limits, and calculate}
\[
\theta_\ell^{req}(T)=\max\left\{0,\frac{\overline{\mathcal B}^{obs}_\ell-B_{0\ell}(T)}{\overline A_\ell(T,\lambda_\ell^*(T))}\right\}.
\]
\textcolor{blue}{A positive prize scales the first-order contribution, so its optimal precision at a fixed duration is independent of the prize within this approximation. At zero prize precision is irrelevant. The duration grid remains $T/T_\ell^{obs}\in\{0.1,0.25,0.5,0.75,1,1.25,1.5,2\}$. Compare $\theta_\ell^{req}(T)+\delta\theta_\ell^{obs}(T/T_\ell^{obs})^2$ across this grid and with the observed regime. Improvements smaller than $10^{-10}\theta_\ell^{obs}$ are treated as retention. Full disclosure is evaluated with the same initial distributions and quality target, reoptimizing prize and duration on the same grid. The additional reduction in Table~\ref{tab-counterfactual-precision} uses positive-prize interior or no-feedback limiting choices, relative to this reoptimized full-disclosure cost.}

\textcolor{blue}{All selected designs and boundary limits meet the first-order posterior mean quality constraint. Boundary costs at zero precision are infima when strictly positive feedback is required. The delay costs are scenarios, grid-edge durations need not be unconstrained optima, and large prizes are extrapolations. This calculation fixes the initial distribution but does not remove approximation error in the empirical posterior, the feedback implementation, or the first-order output expansion.}
'''
    (PAPER_DIR/'appendix_joint_design.tex').write_text(appendix)
    readme=(PAPER_DIR/'README.md').read_text()
    begin=readme.index('Table 7 ')
    end=readme.index('Every image referenced',begin)
    readme=readme[:begin]+'''Table 7 reports fixed-initial-distribution, asymmetric first-order design
comparisons. Initial variance is calibrated from each observed-regime posterior
draw and held fixed across candidate policies. Results, grid/quadrature checks,
and the repeatable producer are in
`results/reestimation/lambda100_sd225/fixed_initial_design/` and
`pipeline/empirical/fixed_initial_design.py`. Run that module, then
`pipeline.empirical.fixed_initial_manuscript` to update the design passages.
Zero-precision results are labeled no-feedback limits (unattained under the
positive-feedback restriction). Original estimation, validation, and historical
design outputs remain available. Score-scale robustness now reports parameter
rankings; historical policy classifications are not evidence for this revision.
The design passages in `release_summary_zh.txt` describe the earlier calibration;
the current design tables are documented by `fixed_initial_design/summary.json`.

'''+readme[end:]
    (PAPER_DIR/'README.md').write_text(readme)
    # Update evidence references; retain the independent scale-ranking evidence.
    path=ROOT/'docs/PAPER_EVIDENCE.csv'
    with path.open() as handle:
        reader=csv.DictReader(handle);fields=reader.fieldnames;rows=list(reader)
    for row in rows:
        if row['label'] in ('tab-counterfactual-precision','tab-fixed-precision'):
            row['evidence']=str((DEST/'summary.json').relative_to(ROOT))
            row['producer']='pipeline/empirical/fixed_initial_design.py'
            row['scope']='Fixed initial distribution; asymmetric first-order output; no-feedback limits explicitly distinguished'
        if row['label']=='tab-score-scale-robustness':
            row['evidence']='results/reestimation/lambda100_sd225/auxiliary/reports/report.json'
            row['scope']='Parameter rank correlations only; historical precision recommendations excluded'
    with path.open('w') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    (DEST/'integration.json').write_text(json.dumps(dict(backup=str(backup.relative_to(ROOT)),
        summary_sha256=digest(DEST/'summary.json'),producer_sha256=digest(__file__),
        updated_files={str(p.relative_to(ROOT)):digest(p) for p in [PAPER_DIR/'PaperJK7.tex',PAPER_DIR/'section_joint_design.tex',PAPER_DIR/'appendix_joint_design.tex']}),indent=2)+'\n')
    print('Updated design passages; backup:',backup)


if __name__=='__main__':main()
