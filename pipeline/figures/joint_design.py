"""JK9 display reconstruction from frozen regime CSVs; no HJB rerun."""
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
from scipy.interpolate import BSpline
from scipy.optimize import minimize,LinearConstraint
from pipeline.theory.joint_design_moderate_plane import fitted_display_boundaries
from pipeline.storage import ROOT, RUNS
root=ROOT;base=root/'results/releases/paperjk9-20261029/evidence/theory_source';out=RUNS/'theory_figure';out.mkdir(parents=True,exist_ok=True)
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
constraints={'finite':[],'exit':[]};fit_diagnostics={};constraint_records=[]
for k in range(1,ncomponents+1):
 ii,jj=np.where(components==k);cx=np.array(xs)[jj];cy=np.array(ys)[ii]
 group='finite' if np.mean(np.abs(cy-np.interp(cx,sx,finite)))<=np.mean(np.abs(cy-np.interp(cx,sx,exit))) else 'exit'
 for j in sorted(set(jj)):
  i=ii[jj==j];lo=float(ye[i.min()]);hi=float(ye[i.max()+1]);x=float(xs[j])
  constraints[group].append((x,lo,hi));constraint_records.append(dict(component=k,boundary=group,sigma=x,lower=lo,upper=hi))

def jointfit(base,items,name):
 # Solve for one global, curvature-penalized spline correction, rather than
 # patching individual gaps. Unknown cell intervals constrain where it passes.
 left,right=.03,3.;t=(sx-left)/(right-left)
 knots=np.r_[np.zeros(4),np.linspace(0,1,45)[1:-1],np.ones(4)]
 nb=len(knots)-4;basis=BSpline(knots,np.eye(nb),3);B=basis(t);D=basis.derivative(2)(t)
 Q=B.T@B/len(t)+1e-5*(D.T@D/len(t))+np.eye(nb)*1e-10
 x=np.array([v[0] for v in items]);lo=np.array([v[1] for v in items]);hi=np.array([v[2] for v in items]);A=basis((x-left)/(right-left));reference=np.interp(x,sx,base);scale=.0025
 constraint=LinearConstraint(A,(lo-reference)/scale,(hi-reference)/scale)
 opt=minimize(lambda v:float(v@Q@v),np.zeros(nb),jac=lambda v:2*Q@v,method='SLSQP',constraints=[constraint],options=dict(ftol=1e-10,maxiter=1500))
 violation=max(float(np.max((lo-reference)/scale-A@opt.x)),float(np.max(A@opt.x-(hi-reference)/scale)),0.)
 assert opt.success and violation<1e-7,(name,opt.message,violation)
 fitted=base+scale*(B@opt.x);centers=reference+scale*(A@opt.x)
 fit_diagnostics[name]=dict(method='Single global B-spline correction with minimum displacement and curvature penalty; unresolved intervals enter joint constraints, no per-cell local reconnects',success=bool(opt.success),message=str(opt.message),curvature_penalty=1e-5,constraints=len(items),maximum_interval_violation=violation*scale,maximum_shift_from_original=float(np.max(np.abs(fitted-base))),rms_shift_from_original=float(np.sqrt(np.mean((fitted-base)**2))))
 for xx,y,l,h in zip(x,centers,lo,hi):assert l-1e-9<=y<=h+1e-9
 return fitted
finite_fit=jointfit(finite,constraints['finite'],'finite');exit_fit=jointfit(exit,constraints['exit'],'exit')
# Keep the original shared boundary before full disclosure first appears.
original_gap=np.maximum(exit-finite,0.);gap_delta=exit_fit-exit-(finite_fit-finite)
full_columns=np.flatnonzero(np.any(z_original==1,axis=0));first=int(full_columns[0]);a,b=xs[first-1],xs[first]
blend=np.clip((sx-a)/(b-a),0.,1.);blend=blend*blend*(3-2*blend)
exit_fit=finite_fit+np.maximum(0.,original_gap+blend*gap_delta)
# Check the displayed curves as well, after the shared-boundary operation.
for group,curve in [('finite',finite_fit),('exit',exit_fit)]:
 for x,lo,hi in constraints[group]:
  value=float(np.interp(x,sx,curve));assert lo-2e-5<=value<=hi+2e-5,(group,x,value,lo,hi)
ax.fill_between(sx,min(ys),max(ys),color=colors[1],linewidth=0,zorder=1)
ax.fill_between(sx,exit_fit,max(ys),color=colors[2],linewidth=0,zorder=1)
ax.fill_between(sx,min(ys),finite_fit,color=colors[0],linewidth=0,zorder=1)
visible=finite_fit>min(ys)+1e-8
ax.plot(sx[visible],finite_fit[visible],color='black',linewidth=1.05,zorder=2)
separate=exit_fit-finite_fit>1e-8
ax.plot(sx[separate],exit_fit[separate],color='black',linewidth=1.05,linestyle='dashed',zorder=2)
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
fig.text(.11,.035,'Display curves are fitted jointly with unresolved-cell intervals; classifications are unchanged.',fontsize=9,color='#444444')
for ext in ['png','pdf']:fig.savefig(out/f'numerical_regions_four_class_jointfit_bw.{ext}',dpi=300)
plt.close(fig)
with (out/'joint_fit_boundary_coordinates.csv').open('w',newline='') as f:
 w=csv.writer(f);w.writerow(['sigma','finite_original','finite_joint_fit','exit_original','exit_joint_fit']);w.writerows(zip(sx,finite,finite_fit,exit,exit_fit))
meta=dict(source_sha256=sources,grid_points=len(rows),grid_shape=list(z.shape),counts=counts,unresolved_with_competitive_failed_candidates=926,unresolved_other_reliability_issues=253,unchanged_resolved_labels=True,display='Exact1179-cell unresolved support with black outlines; global smooth boundary corrections jointly constrained by the unresolved intervals, without local gap patching or prescribed corner/edge anchors',additional_resolved_centers_in_hatching=0,unresolved_connected_components=ncomponents,spline_fits=fit_diagnostics,unresolved_interval_constraints=constraint_records,new_equilibrium_solves=0,manuscript_or_original_figure_modified=False)
(out/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n');caption='The figure shows three design regimes and a separate numerical-difficulty category on the evaluated parameter grid. Dark hatched cells denote the 1,179 points whose regime classification remains unresolved, including 926 points with potentially competitive candidates that failed numerical validation. The three design regions use display-only smooth boundaries fitted jointly with the unresolved-cell intervals. Dark hatching preserves exactly the original unresolved grid support, with no expanded envelope or connecting ribbon. Black outlines enclose the exact unresolved cell union. The unresolved vertical intervals enter a global curvature-penalized spline fit, influencing the overall boundary trajectory rather than individual local connections. No corner or edge-intersection coordinates are prescribed. This fitting convention is for presentation, not an estimate of the unknown economic frontier. Smoothing does not resolve classification uncertainty or certify continuous-domain global optima.'
(out/'README.md').write_text('# Four-category numerical design map\n\n'+caption+'\n\nSources and unchanged labels are checked in metadata.json. No original figure or manuscript was replaced. Reproduction: Python fence in ../task2_four_class_jointfit_method_20261025.md.\n')
print(json.dumps(meta,indent=2))
