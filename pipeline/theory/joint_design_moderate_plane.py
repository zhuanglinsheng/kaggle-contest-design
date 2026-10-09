"""Moderate-curvature joint-design experiment; output solves are preference-free.

Run with python -m pipeline.theory.joint_design_moderate_plane.
Existing strong-curvature *output* samples may be reused, never their optima or
region labels. Every objective is reevaluated under u(B)=3(1-exp(-2B)), D=T²/2.
The classification is a numerical design-grid comparison, not a theorem.
"""
from __future__ import annotations

import csv
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

from pipeline.config import ROOT
from pipeline.theory.joint_design_plane import run_jobs, upper_output
from pipeline.theory.joint_design_refined_plane import load_cache

OUT = ROOT / "results/theory/numerical_joint_plane_moderate"
COSTS = np.geomspace(.003, 3., 181)
SIGMAS = np.unique(np.r_[np.geomspace(.06, 3., 17), .8, 1.])
PRECISIONS = (0., .01, .1, 1., 10., 100., math.inf)
ALPHAS = np.geomspace(.0002, 12., 14)
HORIZONS = np.geomspace(.02, math.sqrt(6.), 11)


def loss_output(b):
    return 3. * np.exp(-2. * np.asarray(b))


def candidate_output_upper(sigma,alpha,t,precision):
    if precision==0:
        # Symmetric no-signal play remains at posterior mean zero. Each player
        # earns theta/2 minus its effort cost, so int m² <= theta/c = alpha.
        # Best output equals the Gaussian baseline plus int m, bounded by
        # baseline + sqrt(alpha*T), not the generic sqrt(2*alpha*T) bound.
        return math.sqrt((1+sigma*sigma*t)/(2*math.pi))+math.sqrt(alpha*t)
    return upper_output(sigma,alpha,t,precision)


def zero_design(s):
    def derivative(t):
        b = math.sqrt((1+s*s*t)/(2*math.pi))
        return 6*math.exp(-2*b)*s*s/(2*math.sqrt(2*math.pi*(1+s*s*t)))-t
    t = brentq(derivative, 0., math.sqrt(6.))
    b = math.sqrt((1+s*s*t)/(2*math.pi))
    return t, float(loss_output(b)+t*t/2), b


def latest(cache, key):
    row=next((cache[key+(float(k),)] for k in (6,5,4,3,2,1,0)
                 if key+(float(k),) in cache), None)
    if (row is not None and key[3]==0 and row.get("status")=="stable_BE_LF_audit"
            and math.isfinite(row.get("error",math.nan))):
        from pipeline.theory.joint_design_stable_audit import NO_SIGNAL_VERIFICATION_TOL
        residual=row["error"]/max(1.,math.sqrt(key[1]*key[2]))
        if residual>NO_SIGNAL_VERIFICATION_TOL:
            # Old cached numbers remain intact for audit, but are not Nash
            # policies merely because an artificial-viscosity solve converged.
            return dict(row,B=math.nan,status=f"failed: no-signal Nash verification residual {residual}")
    return row


def families(s, designs, cache, costs=None):
    costs=COSTS if costs is None else np.asarray(costs)
    data, errors, failed = [], [], []
    for a, t, l in sorted(designs):
        key = (s, a, t, l)
        r = latest(cache, key)
        if r is None or not math.isfinite(r["B"]):
            failed.append((a, float(loss_output(candidate_output_upper(s,a,t,l)))+t*t/2, l))
            continue
        earlier = [cache[key+(float(k),)] for k in range(7)
                   if key+(float(k),) in cache and k<float(r["level"])
                   and math.isfinite(cache[key+(float(k),)]["B"])]
        error_b = 3*abs(r["B"]-earlier[-1]["B"]) if earlier else 0.
        if not earlier and r.get("status") in ("stable_BE_LF_audit","verified_physical_policy_audit"):
            # A stable time integrator is not a resolution error certificate.
            # With only one mesh, retain the analytic output screen but do
            # not let this unrefined numerical value certify a winning family.
            error_b=math.inf
        if math.isfinite(r["error"]):
            error_b = max(error_b, r["error"])
        data.append((a,t,l,r["B"],float(r["level"])))
        errors.append(float(loss_output(max(0.,r["B"]-error_b))-loss_output(r["B"])))
    data = np.asarray(data)
    scores = costs[:,None]*data[:,0]+loss_output(data[:,3])[None,:]+data[:,1][None,:]**2/2
    indices=[]
    for mask in (np.isfinite(data[:,2]), np.isinf(data[:,2])):
        sub = np.flatnonzero(mask)
        indices.append(sub[np.argmin(scores[:,sub],axis=1)])
    return data, scores, indices, np.asarray(errors), failed


def interval_indices(data,scores,errors,floors=None):
    """Minimize error intervals over designs, not only point-estimate optima.

    min_j (L_j-error_j) <= min_j true_L_j <= min_j (L_j+error_j).
    The right envelope uses resolution-checked candidates when available.
    Opposite-family optimistic candidates remain in the left envelope.
    """
    lower=scores-errors[None,:]
    if floors is not None:lower=np.maximum(lower,floors)
    upper=scores+errors[None,:]
    result=[]
    for mask in (np.isfinite(data[:,2]),np.isinf(data[:,2])):
        sub=np.flatnonzero(mask)
        fine=sub[data[sub,4]>=1]
        right=fine if len(fine) else sub
        result.append((sub[np.argmin(lower[:,sub],axis=1)],
                       right[np.argmin(upper[:,right],axis=1)]))
    return result,lower,upper


def winners(s, designs, cache, costs=None):
    costs=COSTS if costs is None else np.asarray(costs)
    data, scores, indices, errors, _ = families(s,designs,cache,costs)
    floors=costs[:,None]*data[:,0]+data[:,1][None,:]**2/2
    floors+=np.array([float(loss_output(candidate_output_upper(s,a,t,l)))
                      for a,t,l,b,level in data])[None,:]
    intervals,_,_=interval_indices(data,scores,errors,floors)
    all_indices=np.r_[indices[0],indices[1],*(idx for pair in intervals for idx in pair)]
    return {(float(data[i,0]),float(data[i,1]),float(data[i,2]))
            for i in np.unique(all_indices)}


def classify(s, designs, cache, costs=None):
    costs=COSTS if costs is None else np.asarray(costs)
    data, scores, (fi,ui), errors, failed = families(s,designs,cache,costs)
    floors=costs[:,None]*data[:,0]+data[:,1][None,:]**2/2
    floors+=np.array([float(loss_output(candidate_output_upper(s,a,t,l)))
                      for a,t,l,b,level in data])[None,:]
    intervals,lower,upper=interval_indices(data,scores,errors,floors)
    (fli,fui),(uli,uui)=intervals
    t0, zloss, b0 = zero_design(s)
    rows=[]
    for k,c in enumerate(costs):
        f,u = int(fi[k]),int(ui[k])
        fl,ul = float(scores[k,f]),float(scores[k,u])
        ef,eu = errors[f],errors[u]
        fl_bound,fu_bound=float(lower[k,fli[k]]),float(upper[k,fui[k]])
        ul_bound,uu_bound=float(lower[k,uli[k]]),float(upper[k,uui[k]])
        # Vanishing prizes approach the zero-prize value arbitrarily closely.
        # A fixed payoff floor would manufacture a broad unresolved exit band.
        # Resolution differences already protect the nonzero family comparison.
        tol=max(1e-9,64*np.finfo(float).eps*min(fl,ul,zloss))
        fail_f=min((v+c*a for a,v,l in failed if math.isfinite(l)),default=math.inf)
        fail_u=min((v+c*a for a,v,l in failed if math.isinf(l)),default=math.inf)
        finite_ready=data[fui[k],4]>=1 and data[uli[k],4]>=1
        full_ready=data[uui[k],4]>=1 and data[fli[k],4]>=1
        zero_ready=data[fli[k],4]>=1 and data[uli[k],4]>=1
        if finite_ready and fu_bound+tol<min(zloss,ul_bound,fail_u):
            regime="finite"
        elif full_ready and uu_bound+tol<min(zloss,fl_bound,fail_f):
            regime="full"
        elif zero_ready and zloss+tol<min(fl_bound,ul_bound,fail_f,fail_u):
            regime="zero"
        else:
            regime="unresolved"
        idx=f if fl<ul else u
        if regime=="finite":idx=int(fui[k])
        elif regime=="full":idx=int(uui[k])
        a,t,l,b,level=data[idx]
        positive=min(fl,ul)<zloss
        # Caps and low-precision failures remain diagnostics, not assumed optima.
        rows.append(dict(sigma=s,c=float(c),regime=regime,
            theta=float(c*a) if positive else 0.,alpha=float(a) if positive else 0.,
            T=float(t) if positive else t0,precision=float(l) if positive else math.nan,
            B=float(b) if positive else b0,loss=min(fl,ul,zloss),
            finite_loss=fl,full_loss=ul,zero_loss=zloss,
            finite_lower=fl_bound,finite_upper=fu_bound,
            full_lower=ul_bound,full_upper=uu_bound,
            comparison_upper=min(zloss,fu_bound,uu_bound),
            finite_error=float(ef),full_error=float(eu),
            failed_finite_lower=fail_f,failed_full_lower=fail_u,
            alpha_cap=bool(positive and a>=12),precision_cap=bool(positive and l==100)))
    return rows


def smooth_class_display(sigmas,costs,z):
    """Display-only smoothing of one-hot labels, in linear physical units.

    Local bandwidths are sigma=0.025 and c=0.00125. No ordering, monotonicity,
    or number of switches is imposed; original classifications are untouched.
    """
    from scipy.interpolate import RegularGridInterpolator
    from scipy.ndimage import gaussian_filter
    sx=np.linspace(min(sigmas),max(sigmas),1401)
    cy=np.linspace(min(costs),max(costs),1001)
    xx,yy=np.meshgrid(sx,cy)
    points=np.column_stack((yy.ravel(),xx.ravel()))
    weights=[]
    for code in range(3):
        values=RegularGridInterpolator((costs,sigmas),(z==code).astype(float))(points)
        weights.append(gaussian_filter(values.reshape(xx.shape),
            sigma=(.00125/(cy[1]-cy[0]),.025/(sx[1]-sx[0])),mode="nearest"))
    weights=np.array(weights)
    return sx,cy,weights,np.argmax(weights,axis=0)


def fitted_display_boundaries(sigmas,costs,z):
    """Smoothing splines of the two region envelopes, for display only.

    Fit RMS residual 0.0035 in effort cost, without a monotonicity constraint.
    This envelope illustration does not preserve small isolated grid islands.
    """
    from scipy.interpolate import UnivariateSpline
    sx=np.linspace(min(sigmas),max(sigmas),1801)
    curves=[]
    for code in (0,2):
        points=[]
        for j,s in enumerate(sigmas):
            indices=np.flatnonzero(z[:,j]==code)
            if not len(indices):continue
            i=indices[-1] if code==0 else indices[0]
            adjacent=min(i+1,len(costs)-1) if code==0 else max(i-1,0)
            points.append((s,(costs[i]+costs[adjacent])/2))
        x,y=np.array(points).T
        fit=UnivariateSpline(x,y,k=min(3,len(x)-1),s=len(x)*.0035**2)
        evaluated=fit(np.clip(sx,x[0],x[-1]))
        if code==0:
            tail=sx>x[-1]
            evaluated[tail]=fit(x[-1])+min(0.,fit.derivative()(x[-1]))*(sx[tail]-x[-1])
        curves.append(np.clip(evaluated,min(costs),max(costs)))
    finite,exit=curves
    gap=np.maximum(exit-finite,0.)
    # A shared finite/zero boundary must not acquire an artificial white sliver
    # merely because its two copies were fitted independently. Preserve the
    # initial shared segment and blend the separation across the first grid
    # interval in which full disclosure appears.
    full_columns=np.flatnonzero(np.any(z==1,axis=0))
    if not len(full_columns):
        gap[:]=0.
    elif full_columns[0]>0:
        first=int(full_columns[0])
        left,right=sigmas[first-1],sigmas[first]
        blend=np.clip((sx-left)/(right-left),0.,1.)
        gap*=blend*blend*(3-2*blend)
    return sx,exit-gap,exit


def draw(rows, destination=None, xlim=None, three_class_estimates=False,
         smooth_boundaries=False, ylim=None):
    destination=OUT if destination is None else destination
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm
    from matplotlib.patches import Patch
    plt.rcParams.update({"font.size":11,"axes.linewidth":.9,"pdf.fonttype":42,
                         "ps.fonttype":42,"savefig.facecolor":"white"})
    # The paper view excludes the very-low-cost exploratory search where the
    # alpha cap often binds; retain all those observations in regions.csv.
    rows=[r for r in rows if r["c"]>=.03-1e-12]
    sigmas=sorted({r["sigma"] for r in rows})
    costs=sorted({r["c"] for r in rows})
    codes={"finite":0,"full":1,"zero":2,"unresolved":3}
    lookup={(r["sigma"],r["c"]):codes[r["regime"]] for r in rows}
    z=np.array([[lookup[s,c] for s in sigmas] for c in costs])
    def edges(xs):
        xs=np.asarray(xs); mids=(xs[:-1]+xs[1:])/2
        return np.r_[2*xs[0]-mids[0],mids,2*xs[-1]-mids[-1]]
    fig,ax=plt.subplots(figsize=(7.6,5.3))
    shades=["#c2c2c2","#ffffff","#eeeeee","#707070"]
    if smooth_boundaries:
        if np.any(z==3):
            raise ValueError("Display smoothing requires a three-class panel")
        sx,finite,exit=fitted_display_boundaries(sigmas,costs,z)
        ax.fill_between(sx,min(costs),max(costs),color=shades[1],linewidth=0)
        ax.fill_between(sx,exit,max(costs),color=shades[2],linewidth=0)
        ax.fill_between(sx,min(costs),finite,color=shades[0],linewidth=0)
        visible=finite>min(costs)+1e-8
        ax.plot(sx[visible],finite[visible],color="black",linewidth=1.05)
        separate=exit-finite>1e-8
        ax.plot(sx[separate],exit[separate],color="black",linewidth=1.05,linestyle="dashed")
    else:
        ax.pcolormesh(edges(sigmas),edges(costs),z,shading="flat",
                  cmap=ListedColormap(shades),norm=BoundaryNorm(np.arange(-.5,4.5),4),
                  rasterized=True)
    # Optional smoothed boundaries are presentation only, not new estimates.
    for value,style in [(0,"solid"),(2,"dashed")]:
        if smooth_boundaries:continue
        mask=(z==value).astype(float)
        if mask.min()!=mask.max():
            ax.contour(sigmas,costs,mask,levels=[.5],colors="black",
                       linewidths=1.05,linestyles=style)
    ax.set_xscale("linear");ax.set_yscale("linear")
    ax.set_xlim(*(xlim or (min(sigmas),max(sigmas))));ax.set_ylim(*(ylim or (0.,.5)))
    if xlim is not None and xlim[0]<min(sigmas):
        ax.axvspan(xlim[0],min(sigmas),facecolor="white",edgecolor="#999999",
                   hatch="///",linewidth=0.,zorder=2)
    # Do not extend the first computed regime into the excluded low-cost strip.
    if (ylim or (0.,.5))[0]<min(costs):
        ax.axhspan((ylim or (0.,.5))[0],min(costs),facecolor="white",edgecolor="#999999",
                   hatch="///",linewidth=0.,zorder=2)
    ax.set_xlabel(r"Innovation risk $\sigma$");ax.set_ylabel(r"Effort cost $c$")
    ax.set_xticks(([xlim[0]] if xlim is not None else [])
                 +[.5,1.,1.5,2.,2.5,3.])
    ax.set_yticks([(ylim or (0.,.5))[0],.1,.2,.3,.4,.5])
    ax.minorticks_off()
    labels=["Positive prize: finite precision", "Positive prize: full disclosure",
            "Zero prize", "Numerically unresolved"]
    legend_items=list(zip(shades,labels))[:3 if three_class_estimates else 4]
    legend_position=(dict(loc="upper right",ncol=1,borderaxespad=.9)
                     if three_class_estimates else
                     dict(loc="lower left",bbox_to_anchor=(0,1.015),ncol=2,borderaxespad=0.))
    if three_class_estimates:
        ax.text(1.9,.43,r"$\theta^*=0,\quad T^*>0$",
                ha="center",va="center",fontsize=12)
        ax.text(1.8,.14,r"$\theta^*>0,\quad T^*>0$"+"\n"+r"$\lambda^*=\infty$",
                ha="center",va="center",fontsize=12,linespacing=1.5)
        ax.text(.64,.17,r"$\theta^*>0,\quad T^*>0$"+"\n"+r"$\lambda^*<\infty$",
                ha="center",va="center",fontsize=12,linespacing=1.5)
    else:
        ax.legend(handles=[Patch(facecolor=color,edgecolor="black",linewidth=.6,label=label)
            for color,label in legend_items],frameon=False,fontsize=9,**legend_position)
    fig.subplots_adjust(left=.12,right=.98,bottom=.14,top=.97 if three_class_estimates else .84)
    for ext in ("pdf","png"):
        fig.savefig(destination/f"numerical_regions_bw.{ext}",dpi=350)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    path=OUT/"output_samples.csv"
    cache=load_cache(ROOT/"results/theory/numerical_joint_plane_strong64/validated_samples.csv")
    cache.update(load_cache(path))
    cache.update(load_cache(OUT/"stable_audit_samples.csv"))
    designs={}
    for ss in SIGMAS:
        s=float(ss);t0=zero_design(s)[0]
        times=np.unique(np.r_[HORIZONS,t0*np.array([.7,1.,1.3])])
        times=times[(times>0)&(times<=math.sqrt(6.))]
        items={(float(a),float(t),l) for a in ALPHAS for t in times for l in PRECISIONS}
        # Old outputs broaden the search at matching risk slices without imposing
        # their old utility, delay function, or optimized designs.
        items.update((k[1],k[2],k[3]) for k in cache if k[0]==s and .0001<=k[1]<=12
                     and .001<=k[2]<=math.sqrt(6.) and k[-1]==0.)
        designs[s]=items
    jobs=[(s,a,t,l,0.) for s,items in designs.items() for a,t,l in sorted(items)]
    print(f"Moderate baseline: {len(SIGMAS)} risk slices, {len(COSTS)} costs; {len(jobs)} candidate outputs",flush=True)
    run_jobs(jobs,cache,4,path)
    for iteration in range(2):
        jobs=[]
        for s,items in designs.items():
            extra=set()
            for a,t,l in winners(s,items,cache):
                for af,tf in [(.7,1),(1.4,1),(1,.85),(1,1.15)]:
                    na,nt=float(a*af),float(t*tf)
                    if .00005<=na<=12 and .001<=nt<=math.sqrt(6.):
                        extra.update((na,nt,p) for p in (l,math.inf))
                if 0<l<100:
                    extra.update((a,t,l*f) for f in (.4,2.5))
            items.update(extra)
            jobs.extend((s,a,t,l,0.) for a,t,l in sorted(extra))
        print(f"Joint design refinement {iteration+1}: {len(jobs)} probes",flush=True)
        run_jobs(jobs,cache,4,path)
    for level in (1.,2.):
        for iteration in range(3):
            jobs={(s,a,t,l,level) for s,items in designs.items()
                  for a,t,l in winners(s,items,cache) if (s,a,t,l,level) not in cache}
            if not jobs:break
            print(f"Resolution {level}, pass {iteration+1}: {len(jobs)} checks",flush=True)
            run_jobs(sorted(jobs),cache,4,path)
    # Refine failed designs only when their conservative upper-output bound
    # can still change an unresolved cell. Never silently discard that failure.
    for iteration in range(2):
        jobs=set()
        for s,items in designs.items():
            unresolved=[r for r in classify(s,items,cache) if r["regime"]=="unresolved"]
            costs=np.asarray([r["c"] for r in unresolved])
            best=np.asarray([r["comparison_upper"] for r in unresolved])
            for a,t,l in items:
                r=latest(cache,(s,a,t,l))
                if not len(costs) or math.isfinite(r["B"]) or r["level"]>=2:
                    continue
                lower=loss_output(candidate_output_upper(s,a,t,l))+t*t/2+costs*a
                if np.any(lower<best):jobs.add((s,a,t,l,2.))
        if not jobs:break
        print(f"Competitive failure checks: {len(jobs)}",flush=True)
        run_jobs(sorted(jobs),cache,4,path)
        # A newly recovered solution can change either family optimum.
        winner_jobs={(s,a,t,l,2.) for s,items in designs.items()
                     for a,t,l in winners(s,items,cache) if (s,a,t,l,2.) not in cache}
        run_jobs(sorted(winner_jobs),cache,4,path)
    rows=[r for s,items in designs.items() for r in classify(s,items,cache)]
    with (OUT/"regions.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    metadata=dict(S0=1.,utility="3*(1-exp(-2*B))",delay="T**2/2",
        sigma=SIGMAS.tolist(),c=COSTS.tolist(),alpha=ALPHAS.tolist(),T=HORIZONS.tolist(),
        precision=["infinity" if math.isinf(l) else l for l in PRECISIONS],
        counts=dict(Counter(r["regime"] for r in rows)),
        qualification="Reoptimized design-grid comparison; not certified continuous-domain boundaries. Failed competitive solves and resolution ties remain unresolved.",
        output_reuse="Preference-independent equilibrium output only; all utility and delay values newly evaluated.",
        horizon_bound="D(T)>=3 at T>=sqrt(6) excludes such designs relative to T=0.",
        alpha_bound="Numerical search cap 12, not an exogenous model restriction.")
    metadata["paper_view_c_range"]=[.03,3.]
    metadata["payoff_floor"]=1e-9
    (OUT/"metadata.json").write_text(json.dumps(metadata,indent=2)+"\n")
    draw(rows)
    print(json.dumps(metadata["counts"],indent=2),flush=True)


if __name__=="__main__":
    main()
