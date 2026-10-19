"""One-off local completion controller; no recurring automation is created."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from pipeline.empirical.unified_prior_reestimate import OUT,ROOT,save


def module(name):
    logs=OUT/'logs';logs.mkdir(parents=True,exist_ok=True)
    path=logs/(name+'.log')
    with path.open('w') as handle:
        result=subprocess.run([sys.executable,'-m','pipeline.empirical.'+name],cwd=ROOT,
                              stdout=handle,stderr=subprocess.STDOUT)
    if result.returncode:raise RuntimeError(f'{name} failed; inspect {path}')
    print('COMPLETED STAGE',name,flush=True)


def main():
    main_done=False;repair=None;attempted=set()
    while True:
        count_main=len(list((OUT/'estimation').glob('*/contest_*/accepted.json')))
        count_aux=len(list((OUT/'auxiliary').glob('*/estimation/*/contest_*/accepted.json')))
        if repair is not None and repair[0].poll() is not None:
            repair[1].close()
            if repair[0].returncode:raise RuntimeError('Long convergence refinement failed; inspect refinement log')
            repair=None
        if repair is None:
            for name in ('median_failures.json','auxiliary_median_failures.json'):
                path=OUT/name
                if not path.exists():continue
                for item in json.loads(path.read_text()):
                    scenario,cid,mode=item['task'];key=(scenario,cid,mode)
                    root=OUT if scenario=='main' else OUT/'auxiliary'/scenario
                    if (root/'estimation'/mode/f'contest_{cid}/accepted.json').exists() or key in attempted:continue
                    attempted.add(key)
                    logs=OUT/'logs';logs.mkdir(parents=True,exist_ok=True)
                    log=(logs/f'refine_{scenario}_{cid}_{mode}.log').open('w')
                    command=[sys.executable,'-m','pipeline.empirical.unified_prior_refine_medians',str(cid),'--mode',mode]
                    if scenario!='main':command.extend(['--scenario',scenario])
                    repair=(subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT),log)
                    print('REFINING',key,flush=True);break
                if repair is not None:break
        if count_main==146 and not main_done:
            for name in ('unified_prior_outputs','unified_prior_sampler_check','unified_prior_interval_report','unified_prior_policy_mcse'):
                module(name)
            main_done=True
        save(OUT/'finalization_progress.json',dict(accepted_main=count_main,required_main=146,accepted_auxiliary=count_aux,
              required_auxiliary=251,main_outputs_complete=main_done,complete=False,controller_pid=os.getpid()))
        if main_done and count_aux==251 and repair is None:break
        print('WAITING',count_main,'/146',count_aux,'/251',flush=True)
        time.sleep(30)
    module('unified_prior_aux_summary')
    module('unified_prior_verify')
    module('unified_prior_manuscript')
    save(OUT/'finalization_progress.json',dict(accepted_main=146,required_main=146,accepted_auxiliary=251,required_auxiliary=251,
                                           main_outputs_complete=True,numerical_and_draft_complete=True,pdf_compilation_pending=True,complete=False))
    print('ALL NUMERICAL OUTPUTS AND BLUE-MARKED DRAFT COMPLETE',flush=True)


if __name__=='__main__':main()
