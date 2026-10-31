"""Posthoc technical controls. Starts only after original queue exits successfully."""
from pathlib import Path
import sys,os,json,time,subprocess
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from common import EXP,CFG,PYTHON,sha,save,utc
HERE=Path(__file__).resolve().parent

def main():
    original=json.loads((EXP/'queue_state.json').read_text())
    assert original['status']=='COMPLETE','Original GPU queue must complete first'
    # Check live process command lines rather than stale PIDs, which may be reused.
    query="Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'python.exe' -and $_.CommandLine -match 'cross_dataset_v1' -and $_.CommandLine -match '(train_gcl|train_r6|train_scsamac|run_queue)\\.py' } | Select-Object -ExpandProperty ProcessId"
    live=subprocess.check_output(['powershell.exe','-NoProfile','-Command',query],text=True).strip()
    assert not live,'Original training/supervisor still alive: '+live
    manifest=json.loads((HERE/'manifest.json').read_text())
    def verify():
        for name,h in manifest['original_sources'].items():assert sha(EXP/name)==h,name
        for name,h in manifest['control_sources'].items():assert sha(HERE/name)==h,name
    verify()
    # Atomic lock prevents accidental concurrent control supervisors.
    lock=HERE/'queue.lock';fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    os.write(fd,str(os.getpid()).encode());os.close(fd)
    state={'status':'RUNNING','pid':os.getpid(),'started_utc':utc(),'completed':[], 'formal_jobs':24,'smoke_jobs':2,'executor_sha256':sha(__file__)}
    configs=[('train_gcl_control.py','NB-GCL','NB-GCL-expression-union-control'),('train_r6_control.py','both_gentle','both_gentle')]
    jobs=[(script,variant,name,'Worm_Neuron',11,True) for script,variant,name in configs]
    jobs += [(script,variant,name,d,s,False) for script,variant,name in configs for d in CFG['development'] for s in CFG['seeds']]
    try:
        for index,(script,variant,name,d,s,smoke) in enumerate(jobs):
            verify();out=EXP/'control_training'/d/name/f'seed{s}'/('smoke' if smoke else 'main')
            if (out/'run.json').exists():
                prior=json.loads((out/'run.json').read_text())
                assert prior['status']=='COMPLETE','Inspect failed existing attempt before retrying: '+str(out)
                state['completed'].append(str(out));continue
            cmd=[str(PYTHON),str(HERE/script),'--dataset',d,'--seed',str(s),'--variant',variant]+(['--smoke'] if smoke else [])
            log=HERE/f'job_{index:02}.log';start=time.monotonic()
            with log.open('w',encoding='utf8') as f:
                p=subprocess.Popen(cmd,cwd=EXP,stdout=f,stderr=subprocess.STDOUT)
                state['current']={'command':cmd,'pid':p.pid,'out':str(out),'log':str(log),'started_utc':utc(),'smoke':smoke}
                while p.poll() is None:
                    state['current']['elapsed_seconds']=time.monotonic()-start;save(HERE/'queue_state.json',state)
                    if time.monotonic()-start>(1800 if smoke else CFG['timeout_seconds_per_job']):
                        f.write('\nHard timeout reached; terminating job.\n');f.flush();p.kill();p.wait();break
                    time.sleep(10)
            if p.returncode:
                state.update(status='FAILED',exit_code=p.returncode,ended_utc=utc());save(HERE/'queue_state.json',state)
                if (out/'run.json').exists():
                    meta=json.loads((out/'run.json').read_text());meta.update(status='FAILED',exit_code=p.returncode,log=str(log));save(out/'run.json',meta)
                return
            assert json.loads((out/'run.json').read_text())['status']=='COMPLETE'
            state['completed'].append(str(out));save(HERE/'queue_state.json',state)
        state.update(status='COMPLETE',ended_utc=utc());save(HERE/'queue_state.json',state)
    except BaseException as exc:
        state.update(status='FAILED',error=repr(exc),ended_utc=utc());save(HERE/'queue_state.json',state);raise
    finally:
        lock.unlink()

if __name__=='__main__':main()
