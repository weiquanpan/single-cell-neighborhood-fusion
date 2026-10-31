"""Serial, durable experiment queue. Stop on failure; no silent retry."""
from pathlib import Path
import argparse,json,os,subprocess,sys,time,traceback
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'cross_dataset_v1'))
import common as c

def main(resume=False):
    p=json.loads((HERE/'protocol.json').read_text());lock=HERE/'queue.lock'
    fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,str(os.getpid()).encode());os.close(fd)
    state=dict(status='RUNNING',pid=os.getpid(),started_utc=c.utc(),completed=[],total_jobs=len(p['jobs']),
               protocol_sha256=c.sha(HERE/'protocol.json'))
    if resume and (HERE/'state.json').exists():state['completed']=json.loads((HERE/'state.json').read_text())['completed']
    try:
        for job in p['jobs']:
            if job['id'] in state['completed']:continue
            for name,h in p['worker_hashes'].items():assert c.sha(HERE/name)==h,('Worker changed',name)
            assert c.sha(HERE/'protocol.json')==state['protocol_sha256']
            log=HERE/'logs'/(job['id']+'.log');log.parent.mkdir(exist_ok=True)
            assert not log.exists(),('Prior attempt preserved; inspect before retry',str(log))
            cmd=[p['python'],'-u',str(HERE/job['script']),*job['args']]
            with log.open('w',encoding='utf8') as f:
                child=subprocess.Popen(cmd,cwd=HERE,stdout=f,stderr=subprocess.STDOUT)
                t=time.monotonic();state['current']=dict(id=job['id'],pid=child.pid,command=cmd,log=str(log),started_utc=c.utc())
                while child.poll() is None:
                    state['current']['elapsed_seconds']=time.monotonic()-t
                    state['current']['process_alive']=True;state['heartbeat_utc']=c.utc();c.save(HERE/'state.json',state)
                    if time.monotonic()-t>job['timeout_seconds']:
                        state['current']['timeout_action']='Terminating owned child process tree after frozen hard timeout'
                        c.save(HERE/'state.json',state)
                        subprocess.run(['taskkill','/PID',str(child.pid),'/T','/F'],stdout=f,stderr=f)
                        child.wait();raise TimeoutError(job['id'])
                    time.sleep(10)
            assert child.returncode==0,('Child failed',job['id'],child.returncode)
            if job.get('result'):
                result=json.loads((HERE/job['result']).read_text());assert result['status']=='COMPLETE',job['id']
            state['completed'].append(job['id']);state['current']['process_alive']=False;c.save(HERE/'state.json',state)
        state.update(status='COMPLETE',finished_utc=c.utc());c.save(HERE/'state.json',state)
    except BaseException:
        if 'current' in state:state['current']['process_alive']=False
        state.update(status='FAILED',error=traceback.format_exc(),finished_utc=c.utc());c.save(HERE/'state.json',state);raise
    finally:lock.unlink(missing_ok=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--resume',action='store_true');main(ap.parse_args().resume)
