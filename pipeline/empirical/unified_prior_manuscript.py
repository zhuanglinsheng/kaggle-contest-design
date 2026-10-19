"""Build a blue-marked manuscript copy from the complete empirical revision."""
import json
import re
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
from pipeline.config import PAPER_DIR
from pipeline.empirical.unified_prior_reestimate import OUT,save,digest


def blue(text):return r'\textcolor{blue}{'+str(text)+'}'
def changed_cell(new,old):
    normalize=lambda x:x.replace(r'\textcolor{blue}','').strip().removesuffix(r'\\').strip().strip('{}')
    return old.strip().removesuffix(r'\\').strip() if normalize(new)==normalize(old) else blue(new)
def stars(p):return r'\textbf{'+('***' if p<.001 else '**' if p<.01 else '*')+'}' if p<.05 else ''
def number(v,p=None):return f'{v:.3f}'+(stars(p) if p is not None else '')
def prob(p):return '$p<0.001$' if p<.001 else f'$p={p:.3f}$'


def paragraph(source,needle,replacement):
    start=source.index(needle)
    stop=source.find('\n\n',start)
    if stop<0:raise RuntimeError('Unterminated paragraph')
    return source[:start]+replacement+source[stop:]


def environment(source,kind,label,transform):
    for match in re.finditer(r'\\begin\{'+kind+r'\}.*?\\end\{'+kind+r'\}',source,re.S):
        if r'\label{'+label+'}' in match.group():
            return source[:match.start()]+transform(match.group())+source[match.end():]
    raise RuntimeError('Missing environment '+label)


def residual(row):
    return ', '.join(f'{label} = {row[key]:.3f}' for label,key in (('Omnibus','omnibus'),('Jarque--Bera','jarque_bera'),('Skewness','skewness'),('Kurtosis','kurtosis')))


def update_figure(block,row):
    lines=block.splitlines();irow=0
    for i,line in enumerate(lines):
        if line.lstrip().startswith('&') and 'coef.' not in line:
            key,se,lo,hi,p=('constant','constant_se','constant_ci_low','constant_ci_high','constant_p') if irow==0 else ('coefficient','std_error','ci_low','ci_high','p_value')
            values=(number(row[key],row[p]),number(row[se]),f"[{row[lo]:.3f}, {row[hi]:.3f}]")
            lines[i]=' & '+' & '.join(changed_cell(value,old) for value,old in zip(values,line.split('&')[1:]))+r' \\'
            irow+=1
        elif r'\bluecell{R$^2$}' in line:lines[i]=r'\bluecell{R$^2$} & \multicolumn{3}{c}{'+blue(number(row['r2']))+r'} \\'
        elif r'\bluecell{Adj. R$^2$}' in line:lines[i]=r'\bluecell{Adj. R$^2$} & \multicolumn{3}{c}{'+blue(number(row['adj_r2']))+r'} \\'
        elif '(2) OLS residual diagnostics:' in line:lines[i]=blue('(2) OLS residual diagnostics: '+residual(row)+'.')
        elif '(3) Condition Number =' in line:lines[i]=blue(f"(3) Condition Number = {row['condition_number']:.3f}.")+('}' if line.endswith('}}') else '')
    if irow!=2:raise RuntimeError('Unexpected coefficient rows')
    return '\n'.join(lines)


def update_diagnostic_table(block,full,early):
    lines=block.splitlines()
    for i,line in enumerate(lines):
        for label,row in (('Full path',full),('Time split',early)):
            if line.startswith(label+' &'):
                cells=(number(row['coefficient'],row['p_value']),number(row['std_error']),f"[{row['ci_low']:.3f}, {row['ci_high']:.3f}]",number(row['r2']),number(row['spearman_rho'],row['spearman_p']))
                lines[i]=label+' & '+' & '.join(changed_cell(value,old) for value,old in zip(cells,line.split('&')[1:]))+r' \\'
        if '(2)' in line and 'Omnibus =' in line:
            values=', '.join(f'{label} = {full[key]:.3f}; {early[key]:.3f}' for label,key in (('Omnibus','omnibus'),('Jarque--Bera','jarque_bera'),('Skewness','skewness'),('Kurtosis','kurtosis')))
            lines[i]=blue('(2) OLS residual diagnostics (Full path; Time split): '+values+'.')
        elif '(3) Condition Number (Full path; Time split):' in line:
            lines[i]=blue(f"(3) Condition Number (Full path; Time split): {full['condition_number']:.3f}; {early['condition_number']:.3f}.")+('}' if line.endswith('}}') else '')
    return '\n'.join(lines)


def update_policy_table(block,summary):
    groups={g['group']:g for g in summary['groups']};lines=block.splitlines()
    for i,line in enumerate(lines):
        for name,g in groups.items():
            if line.startswith(name+' &'):
                cells=([str(g['n_contests']),f"{g['share_pct']:.1f}"] if line.count('&')==2 else
                       [f"{g[k]:.3f}" if g[k] is not None else '---' for k in ('gain_q25_pct','gain_median_pct','gain_q75_pct')])
                old=line.split('&')[1:]
                lines[i]=name+' & '+' & '.join(changed_cell(new,previous) for new,previous in zip(cells,old))+r' \\'
    return '\n'.join(lines)


def main():
    from pipeline.empirical.unified_prior_outputs import refresh_completion
    refresh_completion()
    report=json.loads((OUT/'report.json').read_text())
    aux=json.loads((OUT/'auxiliary/reports/report.json').read_text())
    posterior_files=list((OUT/'estimation').glob('*/contest_*/accepted.json'))+list((OUT/'auxiliary').glob('*/estimation/*/contest_*/accepted.json'))
    mh_files=[p for p in posterior_files if not json.loads(p.read_text())['diagnostics'].get('hmc_applicable',True)]
    main_mh=sum('auxiliary' not in p.relative_to(OUT).parts for p in mh_files)
    robust_mh=sum(any(s in p.relative_to(OUT).parts for s in ('pair','L1','L2')) for p in mh_files)
    if report['fits']!=146 or not aux['complete'] or aux['accepted_fits']!=251:raise RuntimeError('Complete matched results required')
    if aux['baseline_sha256']!=digest(OUT/'estimation/posterior_summary_final.csv'):raise RuntimeError('Auxiliary baseline mismatch')
    draft=OUT/'manuscript';draft.mkdir(parents=True,exist_ok=True)
    for path in PAPER_DIR.iterdir():
        if path.is_file() and path.suffix in ('.tex','.bib','.bst','.cls','.sty','.pdf','.png','.jpg','.eps'):
            if path.name!='PaperJK7.pdf':shutil.copy2(path,draft/path.name)
    for path in (OUT/'paper_assets').glob('*.pdf'):shutil.copy2(path,draft/path.name)
    for path in (OUT/'auxiliary/reports').glob('validate_*_pair.pdf'):shutil.copy2(path,draft/path.name)
    source=(PAPER_DIR/'PaperJK7.tex').read_text();oldsource=source
    if mh_files:
        source=source.replace('We sample the resulting approximate posterior using the No-U-Turn Sampler',
                              'We sample the resulting approximate posterior '+blue('primarily')+' using the No-U-Turn Sampler',1)
    needle=r'$\lambda\sim\operatorname{LogNormal}(\log1,0.75^2)$'
    position=source.index('The empirical priors are')
    end=source.index('Centering the prior',position)
    part=source[position:end]
    if part.count(needle)!=1:raise RuntimeError('Empirical prior location ambiguous')
    source=source[:position]+part.replace(needle,blue(r'$\lambda\sim\operatorname{LogNormal}(\log100,2.25^2)$'))+source[end:]
    old='{For each of the 73 contests, we estimate the model using the full observation window'
    text=(r'For each of the 73 contests, we estimate the model using the full observation window and again using only the first 75\% of the contest-phase window for the time-split diagnostics, retaining the full-sample horizon in both estimations. '
          r'All 146 fits have no divergent transitions or tree-depth saturations, rank-normalized $\widehat R<1.01$, and bulk and tail effective sample sizes above 400. Appendix~\ref{appendix-statistics} reports the full-sample parameter estimates.')
    if main_mh:
        text=(r'For each of the 73 contests, we estimate the model using the full observation window and again using only the first 75\% of the contest-phase window for the time-split diagnostics, retaining the full-sample horizon in both estimations. '
              r'All 146 posteriors have rank-normalized $\widehat R<1.01$ and bulk and tail effective sample sizes above 400. The NUTS runs have no divergent transitions or tree-depth saturations; selected multimodal posteriors use the global refinement described in Appendix~\ref{appendix-statistics}.')
    source=paragraph(source,old,blue(text))
    rows={r['specification']:r for r in report['policy'].get('regressions',[])}
    numbers=json.loads((OUT/'validation/chapter5_paper_numbers.json').read_text())
    rows={r['specification']:r for r in numbers['regressions']}
    for label,key in (('fig-validate-lambda','lambda_benchmark'),('fig-validation-sigma','sigma_benchmark')):
        source=environment(source,'figure',label,lambda block,key=key:update_figure(block,rows[key]))
    r=rows['lambda_benchmark']
    source=paragraph(source,r'{Figure~\ref{fig-validate-lambda} shows',blue(rf"Figure~\ref{{fig-validate-lambda}} reports a slope of ${r['coefficient']:.3f}$ and Spearman $\rho={r['spearman_rho']:.3f}$ ({prob(r['spearman_p'])}). Because the two measures share public-data inputs, their association is a directional consistency check rather than an independent holdout validation."))
    r=rows['sigma_benchmark']
    source=paragraph(source,r'{The positive association in Figure~\ref{fig-validation-sigma}',blue(rf"Figure~\ref{{fig-validation-sigma}} reports a slope of ${r['coefficient']:.3f}$ and Spearman $\rho={r['spearman_rho']:.3f}$ ({prob(r['spearman_p'])}). This model-assisted comparison provides a consistency diagnostic for the innovation-risk interpretation of $\hat\sigma_\ell$."))
    for label,key in (('tab-signal-disagreement-validation','signal_disagreement'),('tab-private-volatility-diagnostics','private_volatility')):
        a,b=rows[key+'_full'],rows[key+'_time_split']
        source=environment(source,'table',label,lambda block,a=a,b=b:update_diagnostic_table(block,a,b))
    a,b=rows['signal_disagreement_full'],rows['signal_disagreement_time_split']
    text=rf"Table~\ref{{tab-signal-disagreement-validation}} reports slopes of ${a['coefficient']:.3f}$ ({prob(a['p_value'])}) for the full path and ${b['coefficient']:.3f}$ ({prob(b['p_value'])}) for the time split. "
    text+=(rf"Higher early precision is associated with less disagreement in the final quarter, with $R^2={b['r2']:.3f}$." if b['ci_high']<0 else 'The time-split confidence interval includes zero, so the predictive evidence from early precision remains inconclusive.' if b['ci_low']<=0<=b['ci_high'] else 'The time-split association differs from the predicted negative relationship.')
    source=paragraph(source,r'Table~\ref{tab-signal-disagreement-validation} reports negative slopes',blue(text))
    a,b=rows['private_volatility_full'],rows['private_volatility_time_split']
    text=rf"Table~\ref{{tab-private-volatility-diagnostics}} reports slopes of ${a['coefficient']:.3f}$ ({prob(a['p_value'])}) for the full path and ${b['coefficient']:.3f}$ ({prob(b['p_value'])}) for the time split. "
    text+=('The positive time-split association also appears without the benchmark\'s drift adjustment.' if b['ci_low']>0 else 'The time-split confidence interval includes zero.' if b['ci_low']<=0<=b['ci_high'] else 'The time-split association differs from the predicted positive relationship.')
    source=paragraph(source,r'{Table~\ref{tab-private-volatility-diagnostics} reports positive slopes',blue(text))
    policy=report['policy'];groups={g['group']:g for g in policy['groups']}
    source=environment(source,'table','tab-counterfactual-precision',lambda block:update_policy_table(block,policy))
    text=f"The recommended precision is full disclosure in {policy['full_disclosure_policies']} contests and finite in {policy['finite_policies']}. "
    text+=rf"At the 75\% posterior-support threshold, {groups['Increase precision']['n_contests']} contests support increased precision, {groups['Decrease precision']['n_contests']} support reduced precision, and {groups['Direction uncertain']['n_contests']} have no clear direction. The median relative output gain is {groups['All contests']['gain_median_pct']:.3f}\% across all contests. "
    text+=r'These recommendations describe precision choices across steady-state feedback regimes at the observed prize and duration. Proposition~\ref{prop-joint-design} studies the broader joint-design problem with the initial filtering variance held fixed.'
    source=paragraph(source,'{The recommended precision is full disclosure',blue(text))
    if rows['signal_disagreement_time_split']['ci_high']>=0:
        source=source.replace('Higher estimated signal precision is associated with smaller public-private leaderboard disagreement, including in the time-split diagnostic, and the positive relationship between structural and benchmark precision estimates also appears under an alternative focal-pair specification.',blue('Higher estimated signal precision is associated with smaller full-path public-private leaderboard disagreement; the time-split evidence is less conclusive. We also examine the relationship between structural and benchmark precision under an alternative focal-pair specification.'))
        source=source.replace('Both relationships are supported by time-split tests using later private-leaderboard outcomes.',blue('Time-split tests provide additional evidence for innovation risk, while the evidence for feedback precision is less conclusive.'))
    time={ (r['mode'],r['parameter']):r for r in aux['time_imputation'] }
    a,b=time[('full','lambda')],time[('early','lambda')]
    needle='For eight contests preselected at evenly spaced ranks of submission counts, two alternative seeds give'
    text=rf"For eight contests preselected at evenly spaced ranks of submission counts, two alternative seeds give median absolute changes in the posterior median of $\lambda$ of {100*a['median_absolute_relative_change']:.1f}\% (full sample) and {100*b['median_absolute_relative_change']:.1f}\% (early sample). The corresponding 95\% posterior intervals overlap in {a['intervals_overlap']+b['intervals_overlap']} of {a['comparisons']+b['comparisons']} comparisons."
    source=paragraph(source,needle,blue(text))
    needle=r'{\textit{Regression inference.}}'
    position=source.index(needle)
    source=source[:position]+blue(r'We summarize empirical parameters by posterior medians to limit sensitivity to long right tails. The replication outputs also retain posterior means and their Monte Carlo errors. Equal-tailed credible intervals and precision counterfactuals use the original joint posterior draws.')+'\n\n'+source[position:]
    if mh_files:
        position=source.index(needle)
        detail=(f'Across the 397 full, early, and auxiliary fits, {len(mh_files)} multimodal posterior'+(' is' if len(mh_files)==1 else 's are')+
                r' sampled using fixed-mixture Metropolis--Hastings refinement under the same likelihood and priors. Both sampling methods use four chains and the same rank-normalized convergence and effective-sample-size criteria; HMC-specific diagnostics apply to NUTS runs.')
        source=source[:position]+blue(detail)+'\n\n'+source[position:]
    needle=r'{\textit{Approximation diagnostics.}}'
    text=rf"In the full-sample fits, posterior medians of $w_k=\theta/(\sigma^2c_k)$ are below one for {numbers['players_with_posterior_median_w_below_one']} of the 146 player--contest pairs, with a cross-pair median of ${numbers['median_player_posterior_median_w']:.3f}$. The posterior median of $\max\{{w_i,w_j\}}$ is below one in {numbers['contests_with_posterior_median_max_w_below_one']} of 73 contests. These are descriptive summaries, not theoretical criteria for approximation accuracy."
    source=paragraph(source,needle,needle+' '+blue(text))
    scales=aux['scales'];ranks=[s[k] for s in scales for k in ('lambda_rank_correlation','sigma_rank_correlation')]
    text=rf"As an indirect check, we calibrate the score-scale parameter $a$ in equation~\eqref{{eq-leaderboard-transform}} to a standard normal reference under $L_1$ and $L_2$ losses and re-estimate all 73 contests, retaining the baseline numerical prior specifications in transformed units. Rank correlations with the baseline range from {min(ranks):.3f} to {max(ranks):.3f}; full-disclosure versus finite-precision classifications agree in {scales[0]['classification_agreement']} and {scales[1]['classification_agreement']} of 73 contests, respectively. This checks sensitivity to score calibration, not Gaussianity itself: changing $a$ does not alter standardized distributional shape. Appendix~\ref{{app-robustness-statistics}} provides the details."
    source=paragraph(source,'{As an indirect check, we calibrate',blue(text))
    fits=pd.read_csv(OUT/'auxiliary/posterior_summary.csv');robust=fits[fits.scenario.isin(['pair','L1','L2'])]
    maximum=robust.filter(regex='_rank_rhat$').max().max()
    text=rf"All 219 robustness posteriors (73 alternative-pair fits and 73 fits at each score scale) have maximum rank-normalized $\widehat R={maximum:.4f}$ and bulk and tail effective sample sizes above 400. The NUTS runs have no divergent transitions or tree-depth saturations. "
    if robust_mh:text+=f'{robust_mh} multimodal posterior'+(' uses' if robust_mh==1 else 's use')+' the global refinement described above. '
    text+=r'Re-estimation retains the baseline numerical priors and parameter bounds in the transformed units, so this exercise includes sensitivity to their scale-dependent regularization. Table~\ref{tab-robustness-benchmarks} reports the benchmark regressions.'
    source=paragraph(source,'{All 219 robustness fits',blue(text))
    auxrows={r['specification']:r for r in aux['regressions']}
    pair_lambda=auxrows['pair_lambda_benchmark'];pair_sigma=auxrows['pair_sigma_benchmark']
    pair_text=(rf"Figure~\ref{{fig-robust-pair-validation}} repeats the benchmark comparisons using alternative focal pairs. The slope for innovation risk is ${pair_sigma['coefficient']:.3f}$ ({prob(pair_sigma['p_value'])}); the slope for signal precision is ${pair_lambda['coefficient']:.3f}$ ({prob(pair_lambda['p_value'])}). ")
    pair_text+=(r'The innovation-risk association remains positive and statistically significant, whereas the signal-precision association is less precisely estimated.' if pair_sigma['ci_low']>0 and pair_lambda['ci_low']<=0<=pair_lambda['ci_high'] else r'The confidence intervals indicate the strength of evidence under this alternative pair selection.')
    pair_text+=r' Appendix~\ref{app-robustness-statistics} reports the detailed statistics and changes in effective observation windows.'
    source=paragraph(source,r'{Figure~\ref{fig-robust-pair-validation} repeats',blue(pair_text))
    if pair_lambda['ci_low']<=0<=pair_lambda['ci_high']:
        original='Higher estimated signal precision is associated with smaller public-private leaderboard disagreement, including in the time-split diagnostic, and the positive relationship between structural and benchmark precision estimates also appears under an alternative focal-pair specification.'
        revised='Higher estimated signal precision is associated with smaller public-private leaderboard disagreement, including in the time-split diagnostic. The alternative focal-pair specification provides weaker evidence for the precision benchmark relationship.'
        source=source.replace(original,blue(revised))
    def robust_table(block):
        lines=block.splitlines();p='lambda'
        for i,line in enumerate(lines):
            if r'Innovation risk $\sigma$' in line:p='sigma'
            for name,key in (('Baseline ($a=1$)',p+'_benchmark'),('Alternative pair','pair_'+p+'_benchmark'),('$L_1$ scale','L1_'+p+'_benchmark'),('$L_2$ scale','L2_'+p+'_benchmark')):
                if line.startswith(name+' &'):
                    r=rows[key] if name.startswith('Baseline') else auxrows[key]
                    cells=(number(r['coefficient']),number(r['std_error']),f"[{r['ci_low']:.3f}, {r['ci_high']:.3f}]",number(r['r2']))
                    lines[i]=name+' & '+' & '.join(changed_cell(value,old) for value,old in zip(cells,line.split('&')[1:]))+r'\\'
            if 'All slopes have $p<0.001$.' in line:
                maximum=max(r['p_value'] for r in list(auxrows.values())+[rows['lambda_benchmark'],rows['sigma_benchmark']])
                lines[i]=line.replace('All slopes have $p<0.001$.',blue('All slopes have $p<0.001$.' if maximum<.001 else f'The largest slope $p$-value is {maximum:.3f}.'))
        return '\n'.join(lines)
    source=environment(source,'table','tab-robustness-benchmarks',robust_table)
    counts=lambda g:'('+','.join(str(g[k]['n_contests']) for k in ('Increase precision','Decrease precision','Direction uncertain'))+')'
    text=rf"For comparisons in common score units, signal precision is $a^2\hat\lambda_a$ and innovation risk is $\hat\sigma_a/a$, where the subscript $a$ denotes the re-estimated scale. Table~\ref{{tab-score-scale-robustness}} compares their rankings with the baseline and repeats the posterior-average precision optimization from Section~\ref{{sect-precision-design}}. Classifications agree with the baseline in {scales[0]['classification_agreement']} of 73 contests under $L_1$ and {scales[1]['classification_agreement']} under $L_2$. At the 75\% support threshold, the counts supporting increased precision, reduced precision, or neither direction are ${counts(groups)}$ in the baseline, "
    for j,s in enumerate(scales):
        d=s['direction_counts'];c='('+','.join(str(d[k]) for k in ('Increase precision','Decrease precision','Direction uncertain'))+')'
        text+=f'${c}$ under $L_{j+1}$'+('.' if j==1 else ', and ')
    source=paragraph(source,'For comparisons in common score units, signal precision is',blue(text))
    def scale_table(block):
        lines=block.splitlines()
        for i,line in enumerate(lines):
            if line.startswith('Baseline &'):
                cells=('1.000','---',str(policy['full_disclosure_policies']),str(policy['finite_policies']),f"{groups['All contests']['gain_median_pct']:.3f}")
                lines[i]='Baseline & '+' & '.join(changed_cell(value,old) for value,old in zip(cells,line.split('&')[1:]))+r'\\'
            for j,s in enumerate(scales):
                if line.startswith(f'$L_{j+1}$ &'):
                    cells=(f"{s['scale_a']:.3f}",f"{s['lambda_rank_correlation']:.3f}; {s['sigma_rank_correlation']:.3f}",str(s['full_disclosure_policies']),str(73-s['full_disclosure_policies']),f"{s['median_output_gain_pct']:.3f}")
                    lines[i]=f'$L_{j+1}$ & '+' & '.join(changed_cell(value,old) for value,old in zip(cells,line.split('&')[1:]))+r'\\'
        return '\n'.join(lines)
    source=environment(source,'table','tab-score-scale-robustness',scale_table)
    # Only parameter summaries change convention; the design objective remains
    # posterior mean output, and the theoretical filtering mean is unchanged.
    replacements=(('posterior mean structural estimate','posterior median structural estimate'),
                  ('posterior mean parameter estimates','posterior median parameter estimates'),
                  ('posterior mean parameters','posterior median parameters'),
                  ('posterior mean estimates','posterior median estimates'),
                  ('posterior mean estimate','posterior median estimate'),
                  ('posterior mean of $\\lambda$','posterior median of $\\lambda$'),
                  ('Bayesian Posterior Mean','Bayesian Posterior Median'),
                  ('common-unit posterior means and baseline posterior means','common-unit posterior medians and baseline posterior medians'))
    empirical_start=source.index('Following the parsimonious specification')
    for old,new in replacements:
        prefix=source[:empirical_start];tail=source[empirical_start:]
        replacement=old.replace('means',blue('medians')).replace('Mean',blue('Median')).replace('mean',blue('median'))
        tail=tail.replace(old,replacement)
        source=prefix+tail
    source=source.replace(r'\setlength{\tabcolsep}{1.5pt}',r'\setlength{\tabcolsep}{1.2pt}',1)
    (draft/'PaperJK7.tex').write_text(source)
    (draft/'appendix_100k_rows.tex').write_text(numbers['appendix_rows'])
    save(OUT/'manuscript_update.json',dict(source_original_sha256=digest(PAPER_DIR/'PaperJK7.tex'),source_new_sha256=digest(draft/'PaperJK7.tex'),
        main_report_sha256=digest(OUT/'report.json'),auxiliary_report_sha256=digest(OUT/'auxiliary/reports/report.json'),
        draft=str(draft),source_script_sha256=digest(__file__),original_manuscript_preserved=True))
    print('BLUE-MARKED DRAFT',draft,flush=True)


if __name__=='__main__':main()
