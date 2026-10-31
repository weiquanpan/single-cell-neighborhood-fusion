"""Sequential single-GPU queue; fixed protocol, retained failures, no automatic retry."""
from pathlib import Path
import sys,json,subprocess,time,os,datetime
B=Path(__file__).resolve().parent;P=json.loads((B/'protocol.json').read_text());PY=sys.executable
def save(p,x):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2,ensure_ascii=False),encoding='utf8');os.replace(t,p)
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
lock=B/'queue.lock'
try:fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
except FileExistsError:raise SystemExit('Queue lock exists; inspect actual process before audited recovery')
os.write(fd,str(os.getpid()).encode());os.close(fd)
state={'status':'RUNNING','pid':os.getpid(),'started_utc':utc(),'jobs':[]};logs=B/'logs';logs.mkdir(exist_ok=True)
try:
 for m in P['methods']:
  matches=list((B/'runs/Romanov'/m/'seed101').glob('smoke_*/run.json'))
  assert any(json.loads(p.read_text())['status']=='COMPLETE' for p in matches),f'Missing smoke success: {m}'
 jobs=[]
 for m in ['scAGCL','scSAMAC','scCAGN']:
  datasets=([] if m=='scSAMAC' else list(P['development']))+P['primary']+P['supplement']
  for d in datasets:
   for s in (P['development_seeds'] if d in P['development'] else P['seeds']):jobs.append((m,d,s))
 state['total_jobs']=len(jobs);save(B/'training_state.json',state)
 for m,d,s in jobs:
  out=B/'runs'/d/m/f'seed{s}'/'attempt_001'
  if out.exists():
   meta=json.loads((out/'run.json').read_text()) if (out/'run.json').exists() else {'status':'UNKNOWN'}
   if meta['status']=='COMPLETE':state['jobs'].append({'method':m,'dataset':d,'seed':s,'status':'REUSED','seconds':meta['seconds']});continue
   raise RuntimeError(f'Existing incomplete attempt needs inspection: {out}')
  log=logs/f'{m}_{d}_{s}.log';cmd=[PY,'-u',str(B/'run_one.py'),'--method',m,'--dataset',d,'--seed',str(s)]
  job={'method':m,'dataset':d,'seed':s,'status':'RUNNING','started_utc':utc(),'log':str(log),'command':cmd};state['jobs'].append(job)
  start=time.monotonic()
  with log.open('w',encoding='utf8') as f:
   proc=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,cwd=B);job['pid']=proc.pid;save(B/'training_state.json',state)
   while proc.poll() is None:
    job['elapsed_seconds']=time.monotonic()-start;state['heartbeat_utc']=utc();save(B/'training_state.json',state)
    if job['elapsed_seconds']>P['timeout_seconds_per_job']:
     job['status']='TIMED_OUT';proc.kill();proc.wait();break
    time.sleep(20)
  job.update(returncode=proc.returncode,seconds=time.monotonic()-start,finished_utc=utc())
  if job['status']!='TIMED_OUT':job['status']='COMPLETE' if proc.returncode==0 else 'FAILED'
  save(B/'training_state.json',state)
  if job['status']!='COMPLETE':
   # Do not silently retry or spend a full grid on a failing adapter.
   state['status']='NEEDS_INSPECTION';break
  # Refresh only after a full dataset x seed group, without interfering with training.
  expected=P['development_seeds'] if d in P['development'] else P['seeds']
  if s==expected[-1]:
   with (logs/'evaluation.log').open('a',encoding='utf8') as f:
    rc=subprocess.call([PY,str(B/'evaluate_external.py')],stdout=f,stderr=subprocess.STDOUT,cwd=B)
   if rc:state['evaluation_error']=True
 else:
  state['status']='TRAINING_COMPLETE'
  with (logs/'evaluation.log').open('a',encoding='utf8') as f:
   rc=subprocess.call([PY,str(B/'evaluate_external.py'),'--package'],stdout=f,stderr=subprocess.STDOUT,cwd=B)
  state['status']='COMPLETE' if rc==0 else 'EVALUATION_OR_PACKAGE_FAILED'
except BaseException:
 import traceback
 state.update(status='FAILED',error=traceback.format_exc());raise
finally:
 state['updated_utc']=utc();save(B/'training_state.json',state);lock.unlink(missing_ok=True)
