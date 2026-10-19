"""Integrate verified quality-constrained design comparisons into Table 7."""
from pathlib import Path
import datetime
import hashlib
import json
import re
import shutil
import pandas as pd


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    root=Path('results/reestimation/lambda100_sd225/design_sensitivity')
    out=root/'paper_update';out.mkdir(exist_ok=True)
    paper=Path('paper');source=(paper/'PaperJK7.tex').read_text()
    assert r'\input{section_joint_design.tex}' not in source, 'Already integrated'
    completion=json.loads((root/'completion.json').read_text())
    for name,sha in completion['artifacts'].items():
        assert digest(root/name)==sha,name
    summary=pd.read_csv(root/'output_preserving_delay_summary.csv')
    policies=pd.read_csv(root/'output_preserving_delay_policies.csv')
    front=pd.read_csv(root/'contest_frontiers.csv')
    full=pd.read_csv(root/'full_disclosure_joint_benchmark.csv')
    assert len(front)==584 and len(policies)==730 and len(full)==730
    assert (summary[['full_disclosure','finite_precision','zero_prize','observed']].sum(axis=1)==73).all()
    # Proportional savings at the limiting zero-cost benchmark are zero.
    mask=(full.full_cost_in_observed_prize==0)&(full.joint_cost_in_observed_prize==0)
    full.loc[mask,'finite_option_cost_saving_pct']=0
    full.to_csv(root/'full_disclosure_joint_benchmark.csv',index=False)
    assert (full.full_cost_in_observed_prize+1e-8>=full.joint_cost_in_observed_prize).all()
    rows=[]
    for row in summary[summary.delta>0].itertuples():
        work=policies[policies.delta==row.delta]
        finite=full[(full.delta==row.delta)&(full.chosen=='finite_precision')]
        assert len(work)==73
        for chosen in work.itertuples():
            if chosen.choice=='observed':continue
            b=front[(front.contest_id==chosen.contest_id)&(front.time_multiplier==chosen.time_multiplier)].iloc[0]
            quality=b.no_effort_output+chosen.prize_ratio*b.observed_prize*b.marginal_prize_output
            assert quality>=b.target_current_output-1e-9,(chosen.contest_id,chosen.delta)
        rows.append(dict(delta=row.delta,full=row.full_disclosure,finite=row.finite_precision,
            zero=row.zero_prize,retained=row.observed,lower=row.decrease_supported,
            time=row.median_time_multiplier,prize=row.median_prize_ratio,
            saving=100*row.median_cost_saving_in_observed_prize/(1+row.delta),
            finite_saving=float(finite.finite_option_cost_saving_pct.median())))
    pd.DataFrame(rows).to_csv(out/'table7_values.csv',index=False)
    backup=out/datetime.datetime.now().strftime('backup_%Y%m%d_%H%M%S');backup.mkdir()
    for name in ('PaperJK7.tex','PaperJK7.pdf','README.md','appendix_effort_recovery.tex'):
        shutil.copy2(paper/name,backup/name)
    start=source.index(r'\subsection{{Implication for Precision Design}}')
    end=source.index(r'\section{Robustness}',start)
    oldtable=re.search(r'\\begin\{table\}.*?\\end\{table\}',source[start:end],re.S).group()
    oldtable=oldtable.replace(r'\label{tab-counterfactual-precision}',r'\label{tab-fixed-precision}')
    oldtable=oldtable.replace(r'\caption{{Precision Adjustments and Expected Output Gains}}',
        r'\caption{\textcolor{blue}{Precision Adjustments at the Observed Prize and Duration}}')
    maintext=r'''\subsection{\textcolor{blue}{Joint Design at a Fixed Quality Target}}\label{sect-precision-design}

\textcolor{blue}{We compare joint changes in the prize, deadline, and feedback precision that reduce the organizer's costs while maintaining at least the current expected quality of the best solution. A quality requirement makes these comparisons possible without assigning a monetary value to solution quality. Let $\overline{\mathcal B}_\ell$ denote posterior mean calibrated output and $\overline{\mathcal B}_\ell^{obs}$ its value under the observed regime. For alternative delay-cost levels $\delta>0$, we solve}
\begin{equation}\label{eq-empirical-joint-design}
\textcolor{blue}{\min_{\theta,T,\lambda}\left\{\theta+\delta\theta_\ell^{obs}\left(\frac{T}{T_\ell^{obs}}\right)^2\right\}
\quad\text{subject to}\quad
\overline{\mathcal B}_\ell(\theta,T,\lambda)\geq\overline{\mathcal B}_\ell^{obs}.}
\end{equation}
\textcolor{blue}{Here $\delta$ is the delay cost at the observed deadline relative to the observed prize. Prizes are nonnegative, precision ranges over $(0,\infty]$, and deadlines lie on a prespecified grid from 0.1 to 2 times the observed duration. The observed regime remains available as an alternative. Appendix~\ref{app-precision-counterfactual} provides the calculation details and the fixed-prize, fixed-duration benchmark.}

\begin{table}[htbp]
\centering
\caption{\textcolor{blue}{Quality-Preserving Joint Design under Alternative Delay Costs}}
\label{tab-counterfactual-precision}
{\color{blue}\small
\setlength{\tabcolsep}{4pt}
\begin{tabularx}{\textwidth}{@{}r*{5}{>{\centering\arraybackslash}X}@{}}
\multicolumn{6}{l}{\textit{Panel A: Design choices (number of contests)}}\\
\midrule
$\delta$ & \makecell{Positive prize,\\full disclosure} & \makecell{Positive prize,\\finite precision} & \makecell{Zero\\prize} & \makecell{Observed\\regime retained} & \makecell{Support for\\lower precision}\\
\midrule
'''
    for row in rows:
        maintext+=f"{row['delta']:g} & {row['full']} & {row['finite']} & {row['zero']} & {row['retained']} & {row['lower']} "+r'\\'+'\n'
    maintext+=r'''\bottomrule
\end{tabularx}
\vspace{2ex}
\begin{tabularx}{\textwidth}{@{}r*{4}{>{\centering\arraybackslash}X}@{}}
\multicolumn{5}{l}{\textit{Panel B: Design changes and cost savings (medians)}}\\
\midrule
$\delta$ & \makecell{Duration /\\observed duration} & \makecell{Prize /\\observed prize} & \makecell{Total cost\\saving (\%)} & \makecell{Additional saving\\from finite precision\\(\%)}\\
\midrule
'''
    for row in rows:
        maintext+=f"{row['delta']:g} & {row['time']:.2f} & {row['prize']:.2f} & {row['saving']:.2f} & {row['finite_saving']:.2f} "+r'\\'+'\n'
    maintext+=r'''\bottomrule
\end{tabularx}
\begin{minipage}{\textwidth}
\footnotesize\textit{Note:} Each scenario covers 73 contests. The first four design categories in Panel A sum to 73; precision is irrelevant when the prize is zero. The last column counts contests with more than 75\% draw-wise support for reducing precision, conditional on the selected duration; zero-prize and retained regimes are excluded. Total cost is the prize plus calibrated delay cost. Its saving is relative to the observed regime, whose cost is $(1+\delta)\theta_\ell^{obs}$. All Panel B medians use 73 contests except the last column, which uses contests selecting positive prizes and finite precision. That column compares their costs with full disclosure after reoptimizing both prize and duration under the same quality requirement. The quality constraint integrates the joint parameter posterior. The delay-cost levels are scenarios, not estimated preferences.
\end{minipage}}
\end{table}

\textcolor{blue}{Higher delay costs favor shorter contests supported by larger prizes. Positive-prize finite precision is selected in four contests at $\delta=1$, eight at $\delta=10$, and fourteen at $\delta=25$. Shorter deadlines leave less time for innovation to overturn initial performance gaps, increasing the value of preserving uncertainty through noisy feedback. Full disclosure nevertheless remains preferred in most contests in every reported scenario. Among contests selecting finite precision, its additional median cost saving relative to reoptimized full disclosure is 3.81\%, 2.99\%, and 3.98\% at these three delay-cost levels.}

\textcolor{blue}{These comparisons extend the steady-state, leading-order calibration by imposing a quality requirement. They do not estimate organizers' preferences or solve the utility-maximization problem in Proposition~\ref{prop-joint-design}, which holds initial filtering uncertainty fixed.}

'''
    (paper/'section_joint_design.tex').write_text(maintext)
    source=source[:start]+r'\input{section_joint_design.tex}'+'\n\n'+source[end:]
    replacements=[
        (r'\textcolor{blue}{Using the estimated primitives, we derive contest-specific precision recommendations and expected output gains under a steady-state, leading-order calibration at the observed prize and duration.}',
         r'\textcolor{blue}{Using the estimated primitives, we compare joint changes in prizes, deadlines, and feedback precision that maintain current expected solution quality under alternative delay costs, using a steady-state, leading-order calibration.}'),
        ('We then evaluate how feedback precision should be adjusted at the observed prize and duration, and quantify the associated relative output gains.',
         r'\textcolor{blue}{We then compare joint changes in prizes, deadlines, and feedback precision that reduce costs while maintaining expected solution quality.}'),
        ('At the observed prize and duration, the steady-state, leading-order calibration yields contest-specific precision recommendations. Most contests favor increased precision under this calibration, although the median relative output gain is modest.',
         r'\textcolor{blue}{Quality-preserving design comparisons under the steady-state, leading-order calibration show how higher delay costs favor shorter contests and larger prizes, making finite precision more common. Full disclosure remains preferred in most contests across the reported scenarios.}'),
        ('The precision recommendations are therefore calibrated benchmarks rather than exact joint-design predictions.',
         r'\textcolor{blue}{The quality-constrained design comparisons are therefore calibrated benchmarks rather than exact joint-design predictions.}'),
        ('At the observed prize and duration, calibrated output is',
         r'\textcolor{blue}{At a candidate prize and duration, calibrated output is}'),
        ('We maximize the average output over all accepted draws to select one precision',
         r'\textcolor{blue}{For the fixed-prize, fixed-duration benchmark in Table~\ref{tab-fixed-precision}, we maximize the average output over all accepted draws to select one precision}'),
        ('Panel A separately quantifies directional uncertainty',
         r'\textcolor{blue}{In Table~\ref{tab-fixed-precision}, Panel A} separately quantifies directional uncertainty'),
        (r'repeats the posterior-average precision optimization from Section~\ref{sect-precision-design}',
         r'repeats the \textcolor{blue}{fixed-prize, fixed-duration precision optimization in Table~\ref{tab-fixed-precision}}'),
        (r'integrating the joint posterior as in Table~\ref{tab-counterfactual-precision}',
         r'integrating the joint posterior as in \textcolor{blue}{Table~\ref{tab-fixed-precision}}')]
    for old,new in replacements:
        assert source.count(old)==1,old
        source=source.replace(old,new,1)
    needle='{\n\\small\n\\setlength{\\tabcolsep}{1.2pt}'
    assert source.count(needle)==1
    source=source.replace(needle,r'\input{appendix_joint_design.tex}'+'\n\n'+needle,1)
    (paper/'PaperJK7.tex').write_text(source)
    appendix=r'''\textcolor{blue}{\textit{Fixed-prize, fixed-duration benchmark.} Table~\ref{tab-fixed-precision} reports precision choices at the observed prize and duration. Full disclosure is recommended in 70 contests and finite precision in three; the median relative output gain is 0.058\%. This provides a reference for the joint changes in Table~\ref{tab-counterfactual-precision}.}

'''+oldtable+'\n\n'+r'''\textcolor{blue}{\textit{Quality-constrained joint design.} Posterior averaging gives $\overline{\mathcal B}_\ell(\theta,T,\lambda)=B_{0\ell}(T)+\theta A_\ell(T,\lambda)$, where $B_{0\ell}$ is mean output without effort and $A_\ell$ is the posterior mean quality contribution per unit of prize. At each candidate duration, we choose $\lambda_\ell^*(T)\in\arg\max_\lambda A_\ell(T,\lambda)$ and calculate the least prize that meets the observed quality target:}
\[
\textcolor{blue}{\theta_\ell^{req}(T)=\max\left\{0,\frac{\overline{\mathcal B}_\ell^{obs}-B_{0\ell}(T)}{A_\ell(T,\lambda_\ell^*(T))}\right\}.}
\]
\textcolor{blue}{A positive prize scales the quality contribution without changing the optimal precision at a fixed duration. When no prize is needed, precision is payoff-irrelevant. The duration grid is $T/T_\ell^{obs}\in\{0.1,0.25,0.5,0.75,1,1.25,1.5,2\}$. We minimize $\theta_\ell^{req}(T)+\delta\theta_\ell^{obs}(T/T_\ell^{obs})^2$ over this grid and compare it with the observed regime's cost. An improvement smaller than $10^{-10}\theta_\ell^{obs}$ is treated as retention of the observed regime. We repeat the precision search on a doubled grid at all 584 contest-duration combinations; the observed-duration solutions also reproduce the fixed-design benchmark. The joint parameter posterior is held fixed throughout.}

\textcolor{blue}{For the last column of Panel B in Table~\ref{tab-counterfactual-precision}, we impose full disclosure and reoptimize the prize and duration on the same grid at the same quality target. The reported additional saving is the percentage reduction from this full-disclosure cost, among contests whose joint choice uses a positive prize and finite precision. It separates the benefit of allowing finite precision from savings achievable by adjusting only the prize and deadline. Total savings in the preceding column include all design changes.}

\textcolor{blue}{The quality target is met or exceeded in every selected design. The quadratic delay costs are calibrated scenarios; they are not identified from submission behavior. The exercise compares steady-state feedback regimes and differs from the fixed initial uncertainty in Proposition~\ref{prop-joint-design}. Deadline choices at the edges of the grid need not be unconstrained optima, and large proposed prizes extend beyond the incentive scales covered by the parameter-recovery check.}

'''
    (paper/'appendix_joint_design.tex').write_text(appendix)
    for name in ('section_joint_design.tex','appendix_joint_design.tex'):
        shutil.copy2(paper/name,out/name)
    record=dict(backup=str(backup),source_original_sha256=digest(backup/'PaperJK7.tex'),
        source_new_sha256=digest(paper/'PaperJK7.tex'),
        input_hashes={name:digest(root/name) for name in ('completion.json','output_preserving_delay_summary.csv','output_preserving_delay_policies.csv','contest_frontiers.csv','full_disclosure_joint_benchmark.csv')},
        new_files={name:digest(paper/name) for name in ('section_joint_design.tex','appendix_joint_design.tex')},
        table_scenarios=[row['delta'] for row in rows],fixed_precision_results_preserved_in_appendix=True,
        all_quality_targets_verified=True,all_design_category_counts_verified=True,script_sha256=digest(__file__))
    (out/'integration.json').write_text(json.dumps(record,indent=2))
    print('Updated Table 7, Section 5.3, introduction/conclusion, and dependent appendix references.')


if __name__=='__main__':main()
