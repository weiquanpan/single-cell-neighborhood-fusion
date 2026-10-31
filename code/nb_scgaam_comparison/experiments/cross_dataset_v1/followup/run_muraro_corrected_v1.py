"""Serial holdout queue: immutable adapter hash; four smokes before formal runs."""
from pathlib import Path
import sys,json,subprocess,os,time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import *
H=EXP/'heldout_muraro_corrected_v1';AD=EXP/'followup/muraro_train_adapter_v1.py'

def result(d,s,m,smoke):
    if m=='NB-GCL-fusion':return H/('smoke_workspace' if smoke else 'candidate_workspace')/'training'/d/m/f'seed{s}'/('smoke' if smoke else 'main')/'run.json'
    return H/'runs'/d/'full'/m/f"{'smoke' if smoke else 'main'}_seed{s}"/'attempt_001/run.json'

def main():
    lock=H/'training.lock';fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,str(os.getpid()).encode());os.close(fd)
    state={'status':'RUNNING','started_utc':utc(),'pid':os.getpid(),'completed':[],'adapter_sha256':sha(AD),'supervisor_sha256':sha(__file__)}
    frozen=json.loads((EXP/'followup/final_method_v1/manifest.json').read_text())
    datasets=[d for d in frozen['heldout_datasets'] if (H/'prepared'/d/'full/ready.json').exists()]
    state['unavailable_datasets']=[d for d in frozen['heldout_datasets'] if d not in datasets]
    methods=['A','NB','scGAAM','NB-GCL-fusion']
    jobs=[(datasets[0],101,m,True) for m in methods]+[(d,s,m,False) for d in datasets for s in frozen['seeds'] for m in methods]
    state['total_jobs']=len(jobs);save(H/'training_state.json',state)
    try:
        for i,(d,s,m,smoke) in enumerate(jobs):
            assert sha(AD)==state['adapter_sha256']
            meta=result(d,s,m,smoke)
            if meta.exists():
                assert json.loads(meta.read_text())['status'] in ['COMPLETE','COMPLETED'],str(meta)
                state['completed'].append(str(meta));continue
            log=H/'logs'/f'{i:03}_{d}_{s}_{m}_{smoke}.log';log.parent.mkdir(exist_ok=True)
            assert not log.exists(),'Preserve previous attempt before restarting'
            cmd=[str(PYTHON),str(AD),'--dataset',d,'--seed',str(s),'--method',m]+(['--smoke'] if smoke else [])
            with log.open('w',encoding='utf8') as f:
                p=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,cwd=EXP);start=time.monotonic()
                state['current']={'index':i,'dataset':d,'seed':s,'method':m,'smoke':smoke,'pid':p.pid,'log':str(log),'started_utc':utc()}
                while p.poll() is None:
                    state['current']['elapsed_seconds']=time.monotonic()-start;save(H/'training_state.json',state)
                    if time.monotonic()-start>(1800 if smoke else 28800):
                        subprocess.run(['taskkill','/PID',str(p.pid),'/T','/F'],stdout=f,stderr=f);p.wait();raise TimeoutError(str(log))
                    time.sleep(10)
            assert p.returncode==0,('Worker failed',p.returncode,str(log))
            assert json.loads(meta.read_text())['status'] in ['COMPLETE','COMPLETED']
            state['completed'].append(str(meta));save(H/'training_state.json',state)
        state.update(status='AVAILABLE_DATASETS_COMPLETE',ended_utc=utc());save(H/'training_state.json',state)
    except BaseException as exc:
        state.update(status='FAILED',error=repr(exc),ended_utc=utc());save(H/'training_state.json',state);raise
    finally:lock.unlink()

if __name__=='__main__':main()
