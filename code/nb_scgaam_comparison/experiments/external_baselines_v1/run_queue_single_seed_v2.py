"""Sequential single-GPU queue; fixed protocol, retained failures, no automatic retry."""
from pathlib import Path
import sys,json,subprocess,time,os,datetime
B=Path(__file__).resolve().parent;P=json.loads((B/'protocol_single_seed_v2.json').read_text());PY=sys.executable
def save(p,x):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2,ensure_ascii=False),encoding='utf8');os.replace(t,p)
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()

def reviewed_spleen_failure(m,d,seed,out):
 if m!='scSAMAC' or d!='Quake_10x_Spleen' or seed not in P['seeds']:return False
 try:
  import hashlib,re
  meta=json.loads((out/'run.json').read_text())
  reference=json.loads((B/'runs/Quake_10x_Spleen/scSAMAC/seed101/attempt_001/run.json').read_text())
  log=(B/'logs'/f'{m}_{d}_{seed}.log').read_text(encoding='utf8')
  first=re.search(r'Pretrain epoch\s+(\d+), NB loss: nan',log)
  known=(meta['status']=='FAILED' and meta['phase']=='joint' and meta['optimizer_steps']=={'pretrain:Adam':11400}
   and meta['input_sha256']==reference['input_sha256'] and meta['runner_sha256']==reference['runner_sha256']
   and hashlib.sha256((B/'run_one.py').read_bytes()).hexdigest()==reference['runner_sha256']
   and 'assert torch.isfinite(z).all()' in meta['error'] and meta['error'].rstrip().endswith('AssertionError') and first is not None)
  if known:
   save(B/'monitoring'/f'spleen{seed}_reviewed_failure.json',{'utc':utc(),'first_nan_epoch':int(first.group(1)),'attempt':str(out),'action':'Retained numerical failure; no retry or parameter change; continue independent jobs','rule':'audit/resume_20261018T1033Z/RESUME_NOTE.json'})
  return known
 except (OSError,KeyError,ValueError):return False

# Audited takeover: old scheduler stopped alone; existing training process stays alive.
import ctypes
wait_pid=22368
h=ctypes.windll.kernel32.OpenProcess(0x00100000,False,wait_pid)
if h:
 while ctypes.windll.kernel32.WaitForSingleObject(h,20000)==258:
  save(B/'handoff_single_seed_v2.json',{'status':'WAITING_FOR_EXISTING_TRAINING','pid':os.getpid(),'training_pid':wait_pid,'utc':utc()})
 ctypes.windll.kernel32.CloseHandle(h)
prior=json.loads((B/'runs/Limb_Muscle/scCAGN/seed33/attempt_001/run.json').read_text())
assert prior['status']=='COMPLETE', 'Existing training failed: inspect retained attempt before continuing'
save(B/'handoff_single_seed_v2.json',{'status':'EXISTING_TRAINING_COMPLETE','utc':utc()})
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
 state['total_jobs']=len(jobs);save(B/'training_state_single_seed_v2.json',state)
 for m,d,s in jobs:
  out=B/'runs'/d/m/f'seed{s}'/'attempt_001'
  if out.exists():
   meta=json.loads((out/'run.json').read_text()) if (out/'run.json').exists() else {'status':'UNKNOWN'}
   if meta['status']=='COMPLETE':state['jobs'].append({'method':m,'dataset':d,'seed':s,'status':'REUSED','seconds':meta['seconds']});continue
   if reviewed_spleen_failure(m,d,s,out):
    state['jobs'].append({'method':m,'dataset':d,'seed':s,'status':'FAILED','seconds':meta['seconds'],'review':f'monitoring/spleen{s}_reviewed_failure.json','retained_attempt':str(out),'retried':False});continue
   raise RuntimeError(f'Existing incomplete attempt needs inspection: {out}')
  log=logs/f'{m}_{d}_{s}.log';cmd=[PY,'-u',str(B/'run_one.py'),'--method',m,'--dataset',d,'--seed',str(s)]
  job={'method':m,'dataset':d,'seed':s,'status':'RUNNING','started_utc':utc(),'log':str(log),'command':cmd};state['jobs'].append(job)
  start=time.monotonic()
  with log.open('w',encoding='utf8') as f:
   proc=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,cwd=B);job['pid']=proc.pid;save(B/'training_state_single_seed_v2.json',state)
   while proc.poll() is None:
    job['elapsed_seconds']=time.monotonic()-start;state['heartbeat_utc']=utc();save(B/'training_state_single_seed_v2.json',state)
    if job['elapsed_seconds']>P['timeout_seconds_per_job']:
     job['status']='TIMED_OUT'
     if os.name=='nt':subprocess.run(['taskkill','/PID',str(proc.pid),'/T','/F'],stdout=f,stderr=subprocess.STDOUT)
     else:proc.kill()
     proc.wait();break
    time.sleep(20)
  job.update(returncode=proc.returncode,seconds=time.monotonic()-start,finished_utc=utc())
  if job['status']!='TIMED_OUT':job['status']='COMPLETE' if proc.returncode==0 else 'FAILED'
  save(B/'training_state_single_seed_v2.json',state)
  if job['status']!='COMPLETE':
   if job['status']=='FAILED' and reviewed_spleen_failure(m,d,s,out):
    job['review']=f'monitoring/spleen{s}_reviewed_failure.json';job['retried']=False;save(B/'training_state_single_seed_v2.json',state)
   else:
    state['status']='NEEDS_INSPECTION';break
  # Refresh only after a full dataset x seed group, without interfering with training.
  expected=P['development_seeds'] if d in P['development'] else P['seeds']
  if s==expected[-1]:
   with (logs/'evaluation.log').open('a',encoding='utf8') as f:
    rc=subprocess.call([PY,str(B/'evaluate_external_single_seed_v2.py')],stdout=f,stderr=subprocess.STDOUT,cwd=B)
   if rc:state['evaluation_error']=True
 else:
  state['status']='TRAINING_COMPLETE'
  with (logs/'evaluation.log').open('a',encoding='utf8') as f:
   rc=subprocess.call([PY,str(B/'evaluate_external_single_seed_v2.py')]+[],stdout=f,stderr=subprocess.STDOUT,cwd=B)
  state['status']=('EXHAUSTED_WITH_RETAINED_FAILURES' if any(j['status']=='FAILED' for j in state['jobs']) else 'COMPLETE') if rc==0 else 'EVALUATION_OR_PACKAGE_FAILED'
except BaseException:
 import traceback
 state.update(status='FAILED',error=traceback.format_exc());raise
finally:
 state['updated_utc']=utc();save(B/'training_state_single_seed_v2.json',state);lock.unlink(missing_ok=True)
