"""Fine-grid precision profiles and checks of the degenerate no-signal boundary."""
import json
import os

from archive.experiments.precision_conditions import OUT, evaluate


def main():
    rows=[]
    for w in [.5,1.,1.5,2.]:
        for ell in [1.,3.,5.,10.,30.,100.,1e3,1e5]:
            rows.extend(evaluate(w,1.,ell,[0.],dx=.025,steps=1000))
    checks=[]
    for w,s0,ell in [(1.,1.,0.),(2.,1.,0.),(1.5,.1,0.),(2.,.5,0.),(2.,1.,5.),(2.,1.,1e5)]:
        try:
            checks.extend(evaluate(w,s0,ell,[0.],dx=.015,steps=1800))
        except RuntimeError as error:
            checks.append(dict(w=w,s0=s0,ell=ell,error=str(error)))
    (OUT/'fine_validation.json').write_text(json.dumps(dict(rows=rows,checks=checks),indent=2)+'\n')
    os.environ.setdefault('MPLCONFIGDIR','/tmp/kaggle-contest-design-matplotlib')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,3.8),constrained_layout=True)
    for w in [.5,1.,1.5,2.]:
        selected=[r for r in rows if r['w']==w]
        high=selected[-1]['B']
        axes[0].plot([r['ell'] for r in selected],[r['B']-high for r in selected],label=f'w = {w}',marker='o',ms=3)
    axes[0].set_ylabel('Output relative to precision $10^5$')
    selected=[r for r in rows if r['w']==2]
    for key,label in [('half_effort','Half total effort'),('gap_contribution','Gap-drift contribution'),('B','Best output')]:
        axes[1].plot([r['ell'] for r in selected],[r[key]-selected[-1][key] for r in selected],label=label,marker='o',ms=3)
    axes[1].set_ylabel('Contribution relative to precision $10^5$')
    for axis in axes:
        axis.axhline(0,color='grey',lw=.7)
        axis.set_xscale('log');axis.set_xlabel(r'Normalized precision $\ell$')
        axis.legend(frameon=False,fontsize=8)
        axis.spines[['top','right']].set_visible(False)
    fig.savefig(OUT/'mechanism_refined.png',dpi=180)
    plt.close(fig)
    print('Saved refined profiles and boundary failures.',flush=True)


if __name__=='__main__': main()
