"""Serial GPU queue with per-job timeout, persisted state and stop-on-failure."""
from common import *
import argparse,subprocess,time

def main(args):
    manifest=EXP/'execution_manifest.json'
    files={p.name:sha(p) for p in EXP.glob('*.py')};files['protocol.json']=sha(EXP/'protocol.json')
    if manifest.exists():
        old=json.loads(manifest.read_text());assert old['files']==files,'Source changed after launch; audit and version before resuming'
    else:save(manifest,{'utc':utc(),'files':files,'heldout_policy':'sealed until final freeze'})
    jobs=[]
    for dataset in CFG['development']:
        for seed in CFG['seeds']:
            if dataset!='Worm_Neuron':jobs.append(('train_scsamac.py',dataset,seed,'scSAMAC'))
    for variant in ['NB-GCL','NB-GCL-fusion','NB-GCL-context']:
        for dataset in CFG['development']:
            for seed in CFG['seeds']:jobs.append(('train_gcl.py',dataset,seed,variant))
    for variant in ['both','positive','negative','none','far','uniformity','anchored']:
        for dataset in CFG['development']:
            for seed in CFG['seeds']:jobs.append(('train_r6.py',dataset,seed,variant))
    state={'status':'RUNNING','pid':os.getpid(),'started_utc':utc(),'jobs_total':len(jobs),'completed':[]}
    for index,(script,dataset,seed,variant) in enumerate(jobs):
        out=EXP/'training'/dataset/variant/f'seed{seed}'/'main'
        if (out/'run.json').exists():
            result=json.loads((out/'run.json').read_text())
            if result['status']=='COMPLETE':state['completed'].append(str(out));continue
            raise RuntimeError(f'Existing incomplete attempt requires inspection: {out}')
        # NB-H extraction is a stage0 dependency; do not consume an in-progress file.
        if script!='train_scsamac.py':
            assert all((EXP/'cache'/dataset/f'nb_h_seed{s}.npy').exists() for s in CFG['seeds'])
        for name,digest in files.items():assert sha(EXP/name)==digest,'Frozen source modified during execution'
        command=[str(PYTHON),str(EXP/script),'--dataset',dataset,'--seed',str(seed)]
        if script!='train_scsamac.py':command+=['--variant',variant]
        log=EXP/'queue_logs'/f'{index:03}_{dataset}_{variant}_{seed}.log';log.parent.mkdir(exist_ok=True)
        with log.open('w',encoding='utf8') as f:
            proc=subprocess.Popen(command,stdout=f,stderr=subprocess.STDOUT,cwd=EXP)
            state.update(current=dict(index=index,command=command,log=str(log),pid=proc.pid,started_utc=utc()))
            save(EXP/'queue_state.json',state)
            started=time.monotonic()
            while proc.poll() is None:
                state['current']['elapsed_seconds']=time.monotonic()-started
                save(EXP/'queue_state.json',state)
                if time.monotonic()-started>CFG['timeout_seconds_per_job']:
                    f.write('\nHard timeout reached; terminating this job.\n');f.flush();proc.kill();proc.wait();break
                time.sleep(10)
        if proc.returncode:
            state.update(status='FAILED',exit_code=proc.returncode,failed_utc=utc());save(EXP/'queue_state.json',state)
            if (out/'run.json').exists():
                result=json.loads((out/'run.json').read_text());result.update(status='FAILED',exit_code=proc.returncode,log=str(log));save(out/'run.json',result)
            return
        state['completed'].append(str(out));save(EXP/'queue_state.json',state)
    state.update(status='COMPLETE',completed_utc=utc());save(EXP/'queue_state.json',state)
    subprocess.run([str(PYTHON),str(EXP/'evaluate.py')],check=True)
if __name__=='__main__':main(argparse.ArgumentParser().parse_args())
