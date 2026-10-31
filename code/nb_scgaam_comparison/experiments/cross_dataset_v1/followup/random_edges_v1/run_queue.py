from pathlib import Path
import sys,os,time,subprocess,json,traceback
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from common import *
HERE=Path(__file__).resolve().parent

def main():
    lock=HERE/'queue.lock'
    fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,str(os.getpid()).encode());os.close(fd)
    state=dict(status='RUNNING',pid=os.getpid(),started_utc=utc(),completed=[],planned=12)
    try:
        for d in CFG['development']:
            for seed in CFG['seeds']:
                for p,h in json.loads((HERE/'original_sources.json').read_text()).items(): assert sha(p)==h,p
                for p,h in json.loads((HERE/'execution_sources.json').read_text()).items(): assert sha(HERE/p)==h,p
                out=HERE/'runs'/d/'random_equal_edges'/f'seed{seed}'/'main'
                assert not out.exists(),'Never overwrite prior attempt: '+str(out)
                log=HERE/f'{d}_{seed}.log'
                cmd=[str(PYTHON),'-u',str(HERE/'train_random.py'),'--dataset',d,'--seed',str(seed),'--variant','NB-GCL-fusion']
                with log.open('x',encoding='utf8') as f:
                    p=subprocess.Popen(cmd,cwd=EXP,stdout=f,stderr=subprocess.STDOUT)
                    state['current']=dict(dataset=d,seed=seed,pid=p.pid,out=str(out),log=str(log),started_utc=utc())
                    while p.poll() is None:
                        state['updated_utc']=utc();save(HERE/'state.json',state);time.sleep(5)
                assert p.returncode==0,f'Training failed {d} {seed}, exit {p.returncode}'
                assert json.loads((out/'run.json').read_text())['status']=='COMPLETE'
                state['completed'].append(str(out));save(HERE/'state.json',state)
        state.update(status='TRAINING_COMPLETE',ended_utc=utc());save(HERE/'state.json',state)
        if (HERE/'evaluate_supplement.py').exists():
            with (HERE/'evaluation.log').open('x',encoding='utf8') as f:
                subprocess.run([str(PYTHON),str(HERE/'evaluate_supplement.py')],cwd=EXP,stdout=f,stderr=subprocess.STDOUT,check=True)
            state.update(status='COMPLETE_PENDING_REVIEW',updated_utc=utc());save(HERE/'state.json',state)
    except BaseException as ex:
        state.update(status='FAILED',error=repr(ex),traceback=traceback.format_exc(),updated_utc=utc());save(HERE/'state.json',state);raise
    finally:
        lock.unlink()

if __name__=='__main__':main()
