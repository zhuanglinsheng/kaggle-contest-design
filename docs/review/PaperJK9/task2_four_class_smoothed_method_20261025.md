# Task 2：平滑三区域边界＋保留数值困难网格

按用户要求，恢复原图的三区域平滑包络（完全复用fitted_display_boundaries与原三分类数据），再在最上层叠加1179个实际未解决单元的深灰斜纹，不平滑、删减或扩展困难点。斜纹完全遮住其下的推测分类和曲线。网格与CSV分类不变；平滑包络是展示近似，可能省略原始小岛，不能作为新边界估计。另存PNG/PDF和源哈希，原未平滑四分类图、论文及正式图均保留。

```python
from pathlib import Path
import csv,json,hashlib,collections
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap,BoundaryNorm
from matplotlib.patches import Patch,Rectangle
from matplotlib.collections import PatchCollection
from pipeline.theory.joint_design_moderate_plane import fitted_display_boundaries
root=Path.cwd();base=root/'results/theory/numerical_joint_plane_moderate_dense/three_class_comparison';out=root/'docs/review/PaperJK9/task2_four_class_smoothed_20261025';out.mkdir(exist_ok=True)
files=[base/'three_class_regions.csv',base/'unresolved_point_decisions.csv'];sources={str(f.relative_to(root)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
original=list(csv.DictReader(files[0].open()));uncertain=list(csv.DictReader(files[1].open()));key=lambda r:(float(r['sigma']),float(r['c']));u={key(r):r for r in uncertain};assert len(u)==1179 and sum(r['competitive_failed_design']=='True' for r in uncertain)==926
xs=sorted({float(r['sigma']) for r in original});ys=sorted({float(r['c']) for r in original});assert (len(xs),len(ys),len(original))==(77,189,14553);assert set(u)<=set(map(key,original))
rows=[]
for r in original:
 k=key(r);v=u.get(k);rows.append(dict(sigma=k[0],c=k[1],regime='numerical_difficulty' if v else r['regime'],original_three_class_regime=r['regime'],competitive_failed_candidate=v['competitive_failed_design'] if v else 'False'))
counts=dict(collections.Counter(r['regime'] for r in rows));assert counts==dict(finite=4947,numerical_difficulty=1179,zero=6236,full=2191);assert all(r['regime']==r['original_three_class_regime'] for r in rows if key(r) not in u)
with (out/'four_class_regions.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
codes=dict(finite=0,full=1,zero=2,numerical_difficulty=3);lookup={key(r):codes[r['regime']] for r in rows};z=np.array([[lookup[x,y] for x in xs] for y in ys]);assert z.shape==(189,77) and np.count_nonzero(z==3)==1179

def edges(v):
 v=np.array(v);mid=(v[:-1]+v[1:])/2;return np.r_[2*v[0]-mid[0],mid,2*v[-1]-mid[-1]]
xe,ye=edges(xs),edges(ys)
plt.rcParams.update({'font.size':11,'axes.linewidth':.9,'pdf.fonttype':42,'ps.fonttype':42,'savefig.facecolor':'white','hatch.linewidth':.45,'font.family':'DejaVu Sans'})
fig,ax=plt.subplots(figsize=(8.2,5.8));colors=['#c2c2c2','#ffffff','#eeeeee','#777777']
# Same display-only spline fits as the original paper figure.
original_lookup={key(r):codes[r['regime']] for r in original}
z_original=np.array([[original_lookup[x,y] for x in xs] for y in ys])
sx,finite,exit=fitted_display_boundaries(xs,ys,z_original)
ax.fill_between(sx,min(ys),max(ys),color=colors[1],linewidth=0,zorder=1)
ax.fill_between(sx,exit,max(ys),color=colors[2],linewidth=0,zorder=1)
ax.fill_between(sx,min(ys),finite,color=colors[0],linewidth=0,zorder=1)
visible=finite>min(ys)+1e-8
ax.plot(sx[visible],finite[visible],color='black',linewidth=1.05,zorder=2)
separate=exit-finite>1e-8
ax.plot(sx[separate],exit[separate],color='black',linewidth=1.05,linestyle='dashed',zorder=2)
# Hatch complete runs of unresolved cells, retaining exact grid support.
patches=[]
for j in range(len(xs)):
 indices=np.flatnonzero(z[:,j]==3)
 for run in np.split(indices,np.flatnonzero(np.diff(indices)>1)+1):
  if len(run):patches.append(Rectangle((xe[j],ye[run[0]]),xe[j+1]-xe[j],ye[run[-1]+1]-ye[run[0]]))
pc=PatchCollection(patches,facecolor=colors[3],edgecolor='#f7f7f7',linewidth=0,hatch='///',zorder=4);ax.add_collection(pc)
ax.set(xlim=(.03,3.),ylim=(.03,.5),xlabel=r'Innovation risk $\sigma$',ylabel=r'Effort cost $c$');ax.set_xticks([.03,.5,1.,1.5,2.,2.5,3.]);ax.set_yticks([.03,.1,.2,.3,.4,.5]);ax.minorticks_off()
ax.text(1.9,.43,r'$\theta^*=0,\quad T^*>0$',ha='center',va='center',fontsize=12)
ax.text(1.8,.14,r'$\theta^*>0,\quad T^*>0$'+'\n'+r'$\lambda^*=\infty$',ha='center',va='center',fontsize=12,linespacing=1.5)
ax.text(.64,.17,r'$\theta^*>0,\quad T^*>0$'+'\n'+r'$\lambda^*<\infty$',ha='center',va='center',fontsize=12,linespacing=1.5)
labels=['Positive prize: finite precision','Positive prize: full disclosure','Zero prize','Numerical difficulty (unresolved)'];handles=[Patch(facecolor=c,edgecolor='#555555',linewidth=.6,label=t) for c,t in zip(colors[:3],labels[:3])];handles.append(Patch(facecolor=colors[3],edgecolor='#f7f7f7',linewidth=.5,hatch='///',label=labels[3]));ax.legend(handles=handles,loc='lower left',bbox_to_anchor=(0,1.02),ncol=2,frameon=False,fontsize=9.4,columnspacing=1.4,handlelength=2)
fig.subplots_adjust(left=.11,right=.975,bottom=.145,top=.85)
fig.text(.11,.035,'Smoothed regime boundaries; hatching retains the actual unresolved grid cells.',fontsize=9,color='#444444')
for ext in ['png','pdf']:fig.savefig(out/f'numerical_regions_four_class_smoothed_bw.{ext}',dpi=300)
plt.close(fig)
meta=dict(source_sha256=sources,grid_points=len(rows),grid_shape=list(z.shape),counts=counts,unresolved_with_competitive_failed_candidates=926,unresolved_other_reliability_issues=253,unchanged_resolved_labels=True,display='Original three-regime display-only smoothing splines with RMS residual target 0.0035; dark hatched overlay preserves all1179 actual unresolved cells with midpoint edges; smoothing does not change CSV labels or establish regime frontiers',new_equilibrium_solves=0,manuscript_or_original_figure_modified=False)
(out/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n');caption='The figure shows three design regimes and a separate numerical-difficulty category on the evaluated parameter grid. Dark hatched cells denote the 1,179 points whose regime classification remains unresolved, including 926 points with potentially competitive candidates that failed numerical validation. The three design regions use the original display-only smoothed envelopes. Dark hatching preserves the actual unresolved grid support and covers any smoothed boundary passing through those cells. Smoothing does not resolve classification uncertainty or certify continuous-domain global optima.'
(out/'README.md').write_text('# Four-category numerical design map\n\n'+caption+'\n\nSources and unchanged labels are checked in metadata.json. No original figure or manuscript was replaced. Reproduction: Python fence in ../task2_four_class_smoothed_method_20261025.md.\n')
print(json.dumps(meta,indent=2))
```
