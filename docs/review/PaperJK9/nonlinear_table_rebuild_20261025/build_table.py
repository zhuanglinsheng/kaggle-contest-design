"""Aggregate only complete, verified original-game results; preserve table form."""
from pathlib import Path
import argparse,hashlib,json,math,re
import numpy as np
from pipeline.config import MAIN_DATA_DIR
from pipeline.empirical.counterfactual import PRIZE_DIVISOR
ROOT=Path(__file__).resolve().parent
DELTAS=(.1,.25,.5,1.,2.,5.,10.,25.,100.)
def median_bounds(known,unknown_count):
 """Sharp median bounds when the missing observations are unrestricted.

 Keep unknown contests in the denominator. Sending their values to opposite
 infinities gives the extremal order statistics, including even-sample medians.
 """
 if unknown_count<0 or int(unknown_count)!=unknown_count:raise ValueError('Invalid unknown count')
 if not all(math.isfinite(v) for v in known):raise ValueError('Nonfinite known comparison')
 if not known and not unknown_count:return None
 low=float(np.median([-math.inf]*unknown_count+sorted(known)))
 high=float(np.median(sorted(known)+[math.inf]*unknown_count))
 return [low if math.isfinite(low) else None,high if math.isfinite(high) else None]

def delimited_benchmark(c,s):
 record=c.get('unresolved_full_comparator_delimitation')
 if not record:return False
 path=Path(record['path'])
 if hashlib.sha256(path.read_bytes()).hexdigest()!=record['sha256']:raise ValueError('Changed benchmark delimitation')
 evidence=json.loads(path.read_text())
 if not (c['contest_id']==evidence['contest_id']==6322
         and evidence['reporting_rule']=='median_bounds_with_arbitrary_unknown'
         and evidence['status']=='unresolved_full_disclosure_benchmark'
         and not evidence['quality_criterion_relaxed'] and not evidence['draws_dropped']
         and s['delta'] in evidence['affected_deltas'] and s['retained_observed']
         and s['selected']==c['observed'] and s['full_comparator'] is None
         and s['additional_reduction_pct'] is None):raise ValueError('Invalid benchmark delimitation scope')
 for source,expected in evidence['inputs_sha256'].items():
  if hashlib.sha256(Path(source).read_bytes()).hexdigest()!=expected:raise ValueError('Changed delimited-comparison inputs')
 return True
def aggregate(paths,expected=73):
 cases=[json.loads(Path(p).read_text()) for p in paths]
 if len(cases)!=expected or len({c['contest_id'] for c in cases})!=expected:raise ValueError('Complete distinct contest coverage required')
 for c in cases:
  if c['status']!='completed_verified_incumbents':raise ValueError('Incomplete nonlinear case')
  if not c.get('publication_ready',False):raise ValueError('A required prize refinement or full-disclosure comparator is incomplete')
  if not c['provenance']['method'].startswith('multilevel original-game nonlinear HJB'):raise ValueError('First-order or unknown quality method')
  if len(c['selections'])!=len(DELTAS) or set(s['delta'] for s in c['selections'])!=set(DELTAS):raise ValueError('Exactly all nine delay-cost scenarios required')
  for s in c['selections']:
   if not all(math.isfinite(s[k]) for k in ['prize_ratio','time_multiplier','cost_saving_pct']):raise ValueError('Nonfinite table input')
   if s['prize_ratio']<0 or s['time_multiplier']<=0:raise ValueError('Invalid design ratios')
   kind='zero_prize' if s['selected']['prize']==0 else 'full_disclosure' if s['selected']['precision']=='full' else 'less_disclosure'
   if s['choice']!=kind:raise ValueError('Classification inconsistent with selected design')
   cost=s['prize_ratio']+s['delta']*s['time_multiplier']**2
   if abs(s['cost_saving_pct']-100*(1-cost/(1+s['delta'])))>1e-8:raise ValueError('Cost saving inconsistent with design ratios')
   if not s['retained_observed']:
    key=json.dumps(s['selected'],sort_keys=True)
    if not c['validation'].get(key,{}).get('passed'):raise ValueError('Selected policy lacks complete quality validation')
   if s['choice']=='less_disclosure' and s['prize_ratio']>0 and s['additional_reduction_pct'] is None:
    if delimited_benchmark(c,s):continue
    raise ValueError('Required full-disclosure comparison unresolved without an audited delimitation')
   if s['choice']=='less_disclosure' and s['prize_ratio']>0:
    full=s.get('full_comparator')
    if not full or (full['prize']>0 and full['precision']!='full'):raise ValueError('Invalid full-disclosure comparator')
    if not c['validation'].get(json.dumps(full,sort_keys=True),{}).get('passed'):raise ValueError('Full-disclosure comparator lacks quality validation')
    if not math.isfinite(s['additional_reduction_pct']):raise ValueError('Nonfinite full-disclosure comparison')
    original_prize=s['selected']['prize']/s['prize_ratio'];original_time=s['selected']['horizon']/s['time_multiplier']
    full_cost=full['prize']/original_prize+s['delta']*(full['horizon']/original_time)**2
    if abs(s['additional_reduction_pct']-100*(1-cost/full_cost))>1e-8:raise ValueError('Additional reduction inconsistent with full-disclosure cost')
 rows=[]
 for delta in DELTAS:
  selected=[next(s for s in c['selections'] if s['delta']==delta) for c in cases];counts={kind:sum(s['choice']==kind for s in selected) for kind in ['full_disclosure','less_disclosure','zero_prize']}
  if sum(counts.values())!=expected:raise ValueError('Invalid Panel A classification')
  additional=[s['additional_reduction_pct'] for s in selected if s['choice']=='less_disclosure' and s['prize_ratio']>0]
  unknown=sum(v is None for v in additional);known=[v for v in additional if v is not None]
  bounds=median_bounds(known,unknown)
  rows.append(dict(delta=delta,**counts,delta_T_pct=float(np.median([100*(s['time_multiplier']-1) for s in selected])),delta_prize_pct=float(np.median([100*(s['prize_ratio']-1) for s in selected])),cost_reduction_pct=float(np.median([s['cost_saving_pct'] for s in selected])),additional_reduction_pct=float(np.median(known)) if known and not unknown else None,additional_reduction_median_bounds=bounds,additional_count=len(additional),additional_resolved_count=len(known),additional_unknown_count=unknown))
 return rows

def replace_numbers(source,rows):
 label=source.index(r'\label{tab-counterfactual-precision}');begin=source.rfind(r'\begin{table}',0,label);end=source.index(r'\end{table}',label)+len(r'\end{table}');table=source[begin:end];panels=table.split(r'\par\vspace{2ex}\noindent');assert len(panels)==2
 for r in rows:
  delta=f"{r['delta']:g}";pattern=r'^'+re.escape(delta)+r' & .*?\\\\[ \t]*$'
  a=delta+' & '+' & '.join(str(r[k]) for k in ['full_disclosure','less_disclosure','zero_prize'])+r' \\'
  extra='--' if r['additional_reduction_pct'] is None else f"{r['additional_reduction_pct']:.2f}"
  if r.get('additional_unknown_count'):
   lo,hi=r['additional_reduction_median_bounds']
   if lo is not None and hi is not None:
    # Round outwards so the displayed interval retains every possible median.
    extra=f'$[{math.floor(lo*100)/100:.2f}, {math.ceil(hi*100)/100:.2f}]$'
  b=delta+' & '+' & '.join(f"{r[k]:.2f}" for k in ['delta_T_pct','delta_prize_pct','cost_reduction_pct'])+' & '+extra+r' \\'
  panels[0],n=re.subn(pattern,lambda _:a,panels[0],flags=re.M);assert n==1
  panels[1],n=re.subn(pattern,lambda _:b,panels[1],flags=re.M);assert n==1
 updated=r'\par\vspace{2ex}\noindent'.join(panels)
 # Only numerical rows here. Captions, notes and surrounding method prose
 # must be synchronized separately before compiling the new manuscript.
 return source[:begin]+updated+source[end:]

def replace_fixed_numbers(source,report):
 """Preserve the existing appendix table's two panels and column structure."""
 if not report.get('ready') or report.get('contests')!=73:raise ValueError('Complete fixed-precision table required')
 if sum(report['counts'].values())!=73 or sum(report['directions'].values())!=73:raise ValueError('Fixed-precision counts must sum to73')
 label=source.index(r'\label{tab-fixed-precision}');begin=source.rfind(r'\begin{table}',0,label);end=source.index(r'\end{table}',label)+len(r'\end{table}');table=source[begin:end]
 for old,new,key in [('Full disclosure','Full disclosure','full_disclosure'),('Interior precision','Other finite precision','interior_precision'),('No-feedback limit','Lowest searched precision','lowest_searched_precision')]:
  count=report['counts'][key];row=f'{new} & {count} & {100*count/73:.1f} '+r'\\'
  table,n=re.subn(r'^(?:'+re.escape(old)+'|'+re.escape(new)+r') & .*?\\\\[ \t]*$',lambda _:row,table,flags=re.M);assert n==1
 for label,key in [('All contests','all'),('Increase precision','increase'),('Decrease precision','decrease'),('Direction uncertain','uncertain')]:
  values=report['percentiles'][key];cells=['--']*3 if values is None else [f'{v:.3f}' for v in values]
  row=label+' & '+' & '.join(cells)+r'\\'
  table,n=re.subn(r'^'+re.escape(label)+r' & .*?\\\\[ \t]*$',lambda _:row,table,flags=re.M);assert n==1
 return source[:begin]+table+source[end:]

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--candidate-tex',type=Path);p.add_argument('--candidate-appendix',type=Path);args=p.parse_args();paths=sorted(ROOT.glob('contests/*/refined_selections.json'))
 try:rows=aggregate(paths)
 except ValueError as e:
  (ROOT/'table_readiness.json').write_text(json.dumps(dict(ready=False,completed_cases=len(paths),reason=str(e)),indent=2));print('TABLE NOT READY:',e);raise SystemExit(2)
 (ROOT/'table_numbers.json').write_text(json.dumps(rows,indent=2));(ROOT/'table_readiness.json').write_text(json.dumps(dict(ready=True,completed_cases=len(paths),scope='Verified numerical incumbents; unknown benchmark retained in subgroup median bounds; global optimality not certified',additional_unknown_counts={str(r['delta']):r['additional_unknown_count'] for r in rows},requires_notes_on_median_bounds=any(r['additional_unknown_count'] for r in rows)),indent=2))
 if args.candidate_tex:
  source=Path('paper/PaperJK9.tex').read_text();args.candidate_tex.write_text(replace_numbers(source,rows))
 if args.candidate_appendix:
  from fixed_precision_table import aggregate as aggregate_fixed
  report=aggregate_fixed();source=Path('paper/appendix_joint_design_jk9.tex').read_text();args.candidate_appendix.write_text(replace_fixed_numbers(source,report))
 print('All 73 cases ready for format-preserving table replacement.')
