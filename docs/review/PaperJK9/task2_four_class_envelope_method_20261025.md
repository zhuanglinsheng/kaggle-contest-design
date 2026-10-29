# Task 2：黑色平滑外轮廓包裹数值困难区域

用户希望平滑黑边包住困难区域，避免黑线穿过零散斜纹。保留原三区域平滑底图，给实际困难单元及有限精度边界的一条窄连接带作平滑外包络，以黑线描外缘、深灰斜纹填充。它覆盖全部1179个原始未解决点及部分原已分类点，图例/图注明确称display envelope；不把扩大后的图面范围当作新的分类或误差界。原始分类CSV保持不变，另存图。

```python
from pathlib import Path
import csv,json,hashlib,collections
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap,BoundaryNorm
from matplotlib.patches import Patch,Rectangle
from scipy.ndimage import gaussian_filter,distance_transform_edt
from scipy.interpolate import RegularGridInterpolator
from pipeline.theory.joint_design_moderate_plane import fitted_display_boundaries
root=Path.cwd();base=root/'results/theory/numerical_joint_plane_moderate_dense/three_class_comparison';out=root/'docs/review/PaperJK9/task2_four_class_envelope_20261025';out.mkdir(exist_ok=True)
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
# Smooth outer envelopes of the unresolved support. Link the small uncertain
# cells along the finite-precision boundary by a narrow presentation ribbon.
# The resulting envelope may include resolved points: it is explicitly labeled
# a display envelope, never a replacement classification.
gx=np.linspace(.03,3.,1801);gy=np.linspace(.03,.5,1201)
xx,yy=np.meshgrid(gx,gy)
ix=np.clip(np.searchsorted(xe,gx,side='right')-1,0,len(xs)-1)
iy=np.clip(np.searchsorted(ye,gy,side='right')-1,0,len(ys)-1)
actual=(z[np.ix_(iy,ix)]==3)
seed=actual.copy()
fcurve=np.interp(gx,sx,finite)
near_finite=actual & (xx>.55) & (np.abs(yy-fcurve[None,:])<.04)
# A uniform narrow ribbon avoids a scalloped outline around individual cells.
ribbon_halfwidth=float(np.max(np.abs(yy-fcurve[None,:])[near_finite]))+.001
seed|=((xx>=.83)&(xx<=2.43)&(np.abs(yy-fcurve[None,:])<=ribbon_halfwidth))
left=actual & (xx<.55)
seed|=((xx<=float(xx[left].max())) & (yy>=float(yy[left].min())))
sampling=(1/(len(gy)-1),1/(len(gx)-1))
signed=distance_transform_edt(~seed,sampling=sampling)-distance_transform_edt(seed,sampling=sampling)
field=gaussian_filter(signed,sigma=(4.,6.),mode='nearest')
level=max(.0018,float(field[actual].max())+.0008)
assert np.all(field[actual]<=level)
envelope=field<=level
cf=ax.contourf(gx,gy,field,levels=[float(field.min())-1.,level],colors=[colors[3]],hatches=['///'],antialiased=True,zorder=4)
cf.set_edgecolor('#f7f7f7');cf.set_linewidth(0)
ax.contour(gx,gy,field,levels=[level],colors='black',linewidths=.85,zorder=5)
# Recheck all source-point centers, not just the raster mask.
interpolator=RegularGridInterpolator((gy,gx),field)
assert all(float(interpolator([[float(r['c']),float(r['sigma'])]])[0])<=level for r in uncertain)
covered_centers=interpolator(np.array([[r['c'],r['sigma']] for r in rows]))<=level
additional_centers=sum(bool(v) and r['regime']!='numerical_difficulty' for v,r in zip(covered_centers,rows))
ax.set(xlim=(.03,3.),ylim=(.03,.5),xlabel=r'Innovation risk $\sigma$',ylabel=r'Effort cost $c$');ax.set_xticks([.03,.5,1.,1.5,2.,2.5,3.]);ax.set_yticks([.03,.1,.2,.3,.4,.5]);ax.minorticks_off()
ax.text(1.9,.43,r'$\theta^*=0,\quad T^*>0$',ha='center',va='center',fontsize=12)
ax.text(1.8,.14,r'$\theta^*>0,\quad T^*>0$'+'\n'+r'$\lambda^*=\infty$',ha='center',va='center',fontsize=12,linespacing=1.5)
ax.text(.64,.17,r'$\theta^*>0,\quad T^*>0$'+'\n'+r'$\lambda^*<\infty$',ha='center',va='center',fontsize=12,linespacing=1.5)
labels=['Positive prize: finite precision','Positive prize: full disclosure','Zero prize','Numerical difficulty (display envelope)'];handles=[Patch(facecolor=c,edgecolor='#555555',linewidth=.6,label=t) for c,t in zip(colors[:3],labels[:3])];handles.append(Patch(facecolor=colors[3],edgecolor='#f7f7f7',linewidth=.5,hatch='///',label=labels[3]));ax.legend(handles=handles,loc='lower left',bbox_to_anchor=(0,1.02),ncol=2,frameon=False,fontsize=9.4,columnspacing=1.4,handlelength=2)
fig.subplots_adjust(left=.11,right=.975,bottom=.145,top=.85)
fig.text(.11,.035,'Hatching encloses unresolved cells; its smoothed envelope is for display only.',fontsize=9,color='#444444')
for ext in ['png','pdf']:fig.savefig(out/f'numerical_regions_four_class_envelope_bw.{ext}',dpi=300)
plt.close(fig)
meta=dict(source_sha256=sources,grid_points=len(rows),grid_shape=list(z.shape),counts=counts,unresolved_with_competitive_failed_candidates=926,unresolved_other_reliability_issues=253,unchanged_resolved_labels=True,display='Original smoothed three-regime background; black smooth contour encloses a hatched conservative presentation envelope around all1179 unresolved cells, linking small finite-boundary uncertainties by a narrow ribbon',all1179_unresolved_centers_enclosed=True,additional_resolved_centers_in_display_envelope=additional_centers,envelope_changes_classification=False,envelope_level_normalized=float(level),finite_boundary_ribbon_halfwidth=ribbon_halfwidth,new_equilibrium_solves=0,manuscript_or_original_figure_modified=False)
(out/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n');caption='The figure shows three design regimes and a separate numerical-difficulty category on the evaluated parameter grid. Dark hatched cells denote the 1,179 points whose regime classification remains unresolved, including 926 points with potentially competitive candidates that failed numerical validation. The three design regions use the original display-only smoothed envelopes. Dark hatching is a smoothed presentation envelope enclosing all unresolved grid cells and a narrow connecting ribbon along the finite-precision boundary. It also covers some resolved points; its area must not be interpreted as the exact number or extent of unresolved classifications. Black outlines follow the outer edges of this envelope. Smoothing does not resolve classification uncertainty or certify continuous-domain global optima.'
(out/'README.md').write_text('# Four-category numerical design map\n\n'+caption+'\n\nSources and unchanged labels are checked in metadata.json. No original figure or manuscript was replaced. Reproduction: Python fence in ../task2_four_class_envelope_method_20261025.md.\n')
print(json.dumps(meta,indent=2))
```
