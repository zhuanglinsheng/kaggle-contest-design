# Task 2：平滑曲线接入困难单元顶点

按用户最新说明：撤销扩大包络，保留原1179个困难单元的方形/多边形外缘；黑线只在无困难单元的区间平滑，在困难连通分量的左上、右下角接入，沿其实际单元外缘包围。大块左上困难区使用实际下缘和右下角接回曲线。相邻困难单元内公共边不画；无新困难点或连接带。修改仅为图面展示，未修改分类CSV和论文。逐分量锚点保存metadata.json。

```python
from pathlib import Path
import csv,json,hashlib,collections
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap,BoundaryNorm
from matplotlib.patches import Patch,Rectangle
from matplotlib.collections import PatchCollection,LineCollection
from scipy.ndimage import label
from pipeline.theory.joint_design_moderate_plane import fitted_display_boundaries
root=Path.cwd();base=root/'results/theory/numerical_joint_plane_moderate_dense/three_class_comparison';out=root/'docs/review/PaperJK9/task2_four_class_corners_20261025';out.mkdir(exist_ok=True)
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
# Keep the exact grid-cell support, but route the smooth lines to component
# corners. The line does not run through an unresolved rectangle.
original_lookup={key(r):codes[r['regime']] for r in original}
z_original=np.array([[original_lookup[x,y] for x in xs] for y in ys])
sx,finite,exit=fitted_display_boundaries(xs,ys,z_original)
components,ncomponents=label(z==3)
boxes={'finite':[],'exit':[]};anchor_records=[]
for k in range(1,ncomponents+1):
 ii,jj=np.where(components==k);j0,j1=int(jj.min()),int(jj.max())
 yl=int(ii[jj==j0].max());yr=int(ii[jj==j1].min())
 L=(float(xe[j0]),float(ye[yl+1]));R=(float(xe[j1+1]),float(ye[yr]))
 cx=np.array(xs)[jj];cy=np.array(ys)[ii]
 group='finite' if np.mean(np.abs(cy-np.interp(cx,sx,finite)))<=np.mean(np.abs(cy-np.interp(cx,sx,exit))) else 'exit'
 block=dict(L=L,R=R,large=len(ii)>100,component=k)
 if block['large']:block['L']=(.03,float(np.interp(.03,sx,finite)))
 boxes[group].append(block);anchor_records.append(dict(component=k,grid_points=len(ii),boundary=group,left_corner=L,right_corner=R,large_left_block=block['large']))

def routed(base,blocks):
 blocks=sorted(blocks,key=lambda q:q['L'][0]);fullx=[];fully=[];segments=[]
 previous=(.03,float(np.interp(.03,sx,base)))
 def gap(L,R):
  x0,y0=L;x1,y1=R
  if x1<x0-1e-12:raise ValueError('Overlapping component intervals')
  if x1-x0<1e-12:return np.array([x0,x1]),np.array([y0,y1])
  xx=np.unique(np.r_[x0,sx[(sx>x0)&(sx<x1)],x1]);yy=np.interp(xx,sx,base)
  width=min(.075,(x1-x0)/3)
  t=np.clip((xx-x0)/width,0.,1.);left_weight=1-t*t*(3-2*t)
  t=np.clip((xx-(x1-width))/width,0.,1.);right_weight=t*t*(3-2*t)
  yy+=(y0-np.interp(x0,sx,base))*left_weight+(y1-np.interp(x1,sx,base))*right_weight
  yy[0]=y0;yy[-1]=y1
  return xx,yy
 for block in blocks:
  L,R=block['L'],block['R'];xx,yy=gap(previous,L);fullx.extend(xx);fully.extend(yy)
  if L[0]>previous[0]+1e-12 or L[1]!=previous[1]:segments.append((xx,yy))
  xx=np.unique(np.r_[L[0],sx[(sx>L[0])&(sx<R[0])],R[0]])
  # Under the opaque cell mask this line is only a fill delimiter, not drawn.
  yy=np.interp(xx,[L[0],R[0]],[L[1],R[1]])
  fullx.extend(xx);fully.extend(yy);previous=R
 xx,yy=gap(previous,(3.,float(np.interp(3.,sx,base))));fullx.extend(xx);fully.extend(yy);segments.append((xx,yy))
 return np.array(fullx),np.array(fully),segments
fx,fy,fsegments=routed(finite,boxes['finite']);ex,ey,esegments=routed(exit,boxes['exit'])
shared=np.interp(ex,sx,exit-finite)<=1e-8;ey[shared]=np.interp(ex[shared],fx,fy)
# Use the routed edges for the fills as well as their outlines.
ax.fill_between(sx,min(ys),max(ys),color=colors[1],linewidth=0,zorder=1)
ax.fill_between(ex,ey,max(ys),color=colors[2],linewidth=0,zorder=1)
ax.fill_between(fx,min(ys),np.minimum(fy,np.interp(fx,ex,ey)),color=colors[0],linewidth=0,zorder=1)
for xx,yy in fsegments:
 visible=yy>min(ys)+1e-8
 ax.plot(xx[visible],yy[visible],color='black',linewidth=1.05,zorder=2)
for xx,yy in esegments:
 separate=np.interp(xx,sx,exit-finite)>1e-8
 ax.plot(xx[separate],yy[separate],color='black',linewidth=1.05,linestyle='dashed',zorder=2)
# Exact unresolved-cell union: no expansion, rounding or connecting ribbon.
patches=[]
for j in range(len(xs)):
 indices=np.flatnonzero(z[:,j]==3)
 for run in np.split(indices,np.flatnonzero(np.diff(indices)>1)+1):
  if len(run):patches.append(Rectangle((xe[j],ye[run[0]]),xe[j+1]-xe[j],ye[run[-1]+1]-ye[run[0]]))
pc=PatchCollection(patches,facecolor=colors[3],edgecolor='#f7f7f7',linewidth=0,hatch='///',zorder=4);ax.add_collection(pc)
outline=[]
for i,j in zip(*np.where(z==3)):
 if i==0 or z[i-1,j]!=3:outline.append([(xe[j],ye[i]),(xe[j+1],ye[i])])
 if i==len(ys)-1 or z[i+1,j]!=3:outline.append([(xe[j],ye[i+1]),(xe[j+1],ye[i+1])])
 if j==0 or z[i,j-1]!=3:outline.append([(xe[j],ye[i]),(xe[j],ye[i+1])])
 if j==len(xs)-1 or z[i,j+1]!=3:outline.append([(xe[j+1],ye[i]),(xe[j+1],ye[i+1])])
ax.add_collection(LineCollection(outline,colors='black',linewidths=.85,zorder=5))
ax.set(xlim=(.03,3.),ylim=(.03,.5),xlabel=r'Innovation risk $\sigma$',ylabel=r'Effort cost $c$');ax.set_xticks([.03,.5,1.,1.5,2.,2.5,3.]);ax.set_yticks([.03,.1,.2,.3,.4,.5]);ax.minorticks_off()
ax.text(1.9,.43,r'$\theta^*=0,\quad T^*>0$',ha='center',va='center',fontsize=12)
ax.text(1.8,.14,r'$\theta^*>0,\quad T^*>0$'+'\n'+r'$\lambda^*=\infty$',ha='center',va='center',fontsize=12,linespacing=1.5)
ax.text(.64,.17,r'$\theta^*>0,\quad T^*>0$'+'\n'+r'$\lambda^*<\infty$',ha='center',va='center',fontsize=12,linespacing=1.5)
labels=['Positive prize: finite precision','Positive prize: full disclosure','Zero prize','Numerical difficulty (unresolved)'];handles=[Patch(facecolor=c,edgecolor='#555555',linewidth=.6,label=t) for c,t in zip(colors[:3],labels[:3])];handles.append(Patch(facecolor=colors[3],edgecolor='#f7f7f7',linewidth=.5,hatch='///',label=labels[3]));ax.legend(handles=handles,loc='lower left',bbox_to_anchor=(0,1.02),ncol=2,frameon=False,fontsize=9.4,columnspacing=1.4,handlelength=2)
fig.subplots_adjust(left=.11,right=.975,bottom=.145,top=.85)
fig.text(.11,.035,'Smooth curves meet unresolved cells at corners; hatching follows the exact grid.',fontsize=9,color='#444444')
for ext in ['png','pdf']:fig.savefig(out/f'numerical_regions_four_class_corners_bw.{ext}',dpi=300)
plt.close(fig)
meta=dict(source_sha256=sources,grid_points=len(rows),grid_shape=list(z.shape),counts=counts,unresolved_with_competitive_failed_candidates=926,unresolved_other_reliability_issues=253,unchanged_resolved_labels=True,display='Exact1179-cell unresolved support with polygonal black outlines; smooth regime curves routed to upper-left/lower-right component corners, never smoothed across unresolved components',additional_resolved_centers_in_hatching=0,unresolved_connected_components=ncomponents,corner_connections=anchor_records,new_equilibrium_solves=0,manuscript_or_original_figure_modified=False)
(out/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n');caption='The figure shows three design regimes and a separate numerical-difficulty category on the evaluated parameter grid. Dark hatched cells denote the 1,179 points whose regime classification remains unresolved, including 926 points with potentially competitive candidates that failed numerical validation. The three design regions use the original display-only smoothed envelopes. Dark hatching preserves exactly the original unresolved grid support, with no expanded envelope or connecting ribbon. Black curves are smooth away from unresolved components and connect to their upper-left and lower-right corners, following their polygonal outer edges at unresolved components. Smoothing does not resolve classification uncertainty or certify continuous-domain global optima.'
(out/'README.md').write_text('# Four-category numerical design map\n\n'+caption+'\n\nSources and unchanged labels are checked in metadata.json. No original figure or manuscript was replaced. Reproduction: Python fence in ../task2_four_class_corners_method_20261025.md.\n')
print(json.dumps(meta,indent=2))
```
