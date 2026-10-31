"""Single-instance supervised sequential comparison. Never overwrites an attempt."""
from common import *
import argparse,subprocess,time,traceback,psutil
p=argparse.ArgumentParser();p.add_argument('--mode',choices=['smoke','pilot','formal'],required=True);a=p.parse_args()
lock=R/'logs/queue.lock';lock.parent.mkdir(exist_ok=True)
if lock.exists():
    old=json.loads(lock.read_text());pid=old['pid']
    if psutil.pid_exists(pid):
        try:
            if 'run_queue.py' in ' '.join(psutil.Process(pid).cmdline()):raise RuntimeError('A queue is already alive')
        except psutil.NoSuchProcess:pass
    lock.rename(lock.with_name('queue_stale_'+str(time.time_ns())+'.json'))
with lock.open('x') as f:json.dump({'pid':os.getpid(),'mode':a.mode,'started':utc()},f)
config=json.loads((R/'configs/experiment.json').read_text());state={'status':'RUNNING','mode':a.mode,'pid':os.getpid(),'started_utc':utc(),'completed':[]}
def record(event,**fields):
    with (R/'logs/events.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(dict(utc=utc(),event=event,**fields))+'\n')
def command(cmd,log,out=None,timeout=28800):
    start=time.monotonic();peak=0
    with log.open('w',encoding='utf8') as stream:
        proc=subprocess.Popen(cmd,stdout=stream,stderr=subprocess.STDOUT,cwd=R,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        state.update(child_pid=proc.pid,command=cmd);save(R/'state.json',state)
        while proc.poll() is None:
            try:
                parent=psutil.Process(proc.pid);family=[parent]+parent.children(recursive=True);rss=sum(x.memory_info().rss for x in family if x.is_running());peak=max(peak,rss)
            except (psutil.NoSuchProcess,psutil.AccessDenied):pass
            if out:save(out/'supervisor.json',dict(pid=proc.pid,alive=True,elapsed_seconds=time.monotonic()-start,peak_process_tree_rss=peak,updated_utc=utc()))
            if time.monotonic()-start>timeout:
                record('HARD_TIMEOUT',command=cmd)
                for child in psutil.Process(proc.pid).children(recursive=True):child.terminate()
                proc.terminate();proc.wait(timeout=30)
                if out:
                    save(out/'supervisor.json',dict(pid=proc.pid,alive=False,exit_code=proc.returncode,wall_seconds=time.monotonic()-start,peak_process_tree_rss=peak,ended_utc=utc(),hard_timeout=True))
                    if (out/'run.json').exists():
                        failed=json.loads((out/'run.json').read_text());failed.update(status='FAILED',error='HARD_TIMEOUT',ended_utc=utc());save(out/'run.json',failed)
                raise RuntimeError('Hard timeout, attempt retained')
            time.sleep(3)
    if out:save(out/'supervisor.json',dict(pid=proc.pid,alive=False,exit_code=proc.returncode,wall_seconds=time.monotonic()-start,peak_process_tree_rss=peak,ended_utc=utc()))
    if proc.returncode:
        if out and (out/'run.json').exists():
            failed=json.loads((out/'run.json').read_text())
            if failed['status']=='RUNNING':failed.update(status='FAILED',error='PROCESS_EXIT_'+str(proc.returncode),ended_utc=utc());save(out/'run.json',failed)
        raise RuntimeError('Command failed; log='+str(log))
try:
    if a.mode=='formal':
        assert json.loads((R/'reports/pilot_gate.json').read_text())['status']=='PASSED'
        frozen=json.loads((R/'configs/frozen_manifest.json').read_text())
        for name,h in frozen['files'].items():assert sha(R/name)==h,name
    assert json.loads((R/'reports/adaptation_checks.json').read_text())['status']=='PASSED'
    datasets=['Limb_Muscle'] if a.mode in ['smoke','pilot'] else config['datasets']
    depths=[1.] if a.mode in ['smoke','pilot'] else config['depths']
    for depth in depths:
      for dataset in datasets:
        roles=[a.mode] if a.mode!='formal' else (['pilot','main'] if depth==1 else ['main'])
        for role in roles:
          for seed in [config['pilot_seed']] if role!='main' else config['seeds']:
            while not (R/'data/raw'/DATASETS[dataset][0]).exists():
                state.update(status='WAITING_DATA',dataset=dataset);save(R/'state.json',state)
                ds=json.loads((R/'logs/download_state.json').read_text());download_alive=psutil.pid_exists(ds['pid'])
                if not download_alive:raise RuntimeError('Download process absent before dataset ready; inspect logs')
                time.sleep(30)
            task=taskpath(dataset,depth,seed)
            if not (task/'ready.json').exists():
                log=R/'logs'/f'prepare_{dataset}_{depth}_{seed}_{time.time_ns()}.log';command([str(PYTHON),'-X','utf8','-u',str(R/'scripts/prepare.py'),'--dataset',dataset,'--depth',str(depth),'--seed',str(seed)],log,timeout=1800)
            if a.mode=='formal':
                command([str(PYTHON),'-X','utf8',str(R/'scripts/audit_prepared.py'),'--dataset',dataset,'--depth',str(depth),'--seed',str(seed)],R/'logs'/f'prepared_audit_{dataset}_{depth}_{seed}_{time.time_ns()}.log',timeout=1800)
            for method in config['methods']:
                runroot=R/'runs'/dataset/('full' if depth==1 else 'half')/method/f'{role}_seed{seed}'
                completed=[f.parent for f in runroot.glob('attempt_*/metrics.json') if json.loads((f.parent/'run.json').read_text())['status']=='COMPLETED']
                if completed:
                    assert len(completed)==1,'Multiple successful attempts need review'
                    if a.mode=='formal' and not (completed[0]/'independent_audit.json').exists():
                        command([str(PYTHON),'-X','utf8',str(R/'scripts/audit_run.py'),'--run',str(completed[0])],completed[0]/'audit.log',timeout=600)
                    record('REUSE_COMPLETED',run=str(completed[0]));continue
                attempts=list(runroot.glob('attempt_*'));out=runroot/f'attempt_{len(attempts)+1:03d}';out.mkdir(parents=True,exist_ok=False)
                state.update(status='RUNNING',dataset=dataset,depth=depth,method=method,role=role,seed=seed,active_run=str(out));save(R/'state.json',state);record('START',run=str(out))
                cmd=[str(PYTHON),'-X','utf8','-u',str(R/'scripts/train_one.py'),'--dataset',dataset,'--method',method,'--seed',str(seed),'--depth',str(depth),'--role',role,'--out',str(out)]
                command(cmd,out/'training.log',out,config['smoke_timeout_seconds'] if role=='smoke' else config['formal_timeout_seconds'])
                command([str(PYTHON),'-X','utf8',str(R/'scripts/evaluate.py'),'--run',str(out)],out/'evaluation.log',timeout=600)
                command([str(PYTHON),'-X','utf8',str(R/'scripts/audit_run.py'),'--run',str(out)],out/'audit.log',timeout=600)
                state['completed'].append(str(out));record('COMPLETED',run=str(out))
          if a.mode=='formal':
            command([str(PYTHON),'-X','utf8',str(R/'scripts/build_report.py')],R/'logs'/f'report_{dataset}_{depth}_{time.time_ns()}.log',timeout=600)
    state.update(status='COMPLETED',ended_utc=utc());save(R/'state.json',state)
    save(R/'reports'/f'{a.mode}_queue_completion.json',state)
except BaseException as err:
    state.update(status='FAILED',error=repr(err),traceback=traceback.format_exc(),ended_utc=utc());save(R/'state.json',state);record('FAILED',error=repr(err));raise
finally:
    if lock.exists() and json.loads(lock.read_text())['pid']==os.getpid():lock.unlink()
