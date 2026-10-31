from pathlib import Path
import json,shutil,difflib,datetime,hashlib
B=Path(__file__).resolve().parent;A=B/'audit/resume_20261018T1033Z';A.mkdir(exist_ok=False)
for name in ['training_state.json','run_queue_r2.py','protocol.json']:shutil.copy2(B/name,A/name)
for name in ['queue_r2.stdout.log','queue_r2.stderr.log']:
 if (B/name).exists():shutil.copy2(B/name,A/name)
for seed in [101,202]:
 meta=json.loads((B/f'runs/Quake_10x_Spleen/scSAMAC/seed{seed}/attempt_001/run.json').read_text())
 assert meta['status']=='FAILED' and meta['optimizer_steps']['pretrain:Adam']==11400
 assert meta['runner_sha256']==hashlib.sha256((B/'run_one.py').read_bytes()).hexdigest()
baseline=json.loads((B/'runs/Quake_10x_Spleen/scSAMAC/seed101/attempt_001/run.json').read_text())
other=json.loads((B/'runs/Quake_10x_Spleen/scSAMAC/seed202/attempt_001/run.json').read_text());assert baseline['input_sha256']==other['input_sha256']
s=(B/'run_queue_r2.py').read_text();old=s
helper='''
def reviewed_spleen_failure(m,d,seed,out):
 if m!='scSAMAC' or d!='Quake_10x_Spleen' or seed not in P['seeds']:return False
 try:
  import hashlib,re
  meta=json.loads((out/'run.json').read_text())
  reference=json.loads((B/'runs/Quake_10x_Spleen/scSAMAC/seed101/attempt_001/run.json').read_text())
  log=(B/'logs'/f'{m}_{d}_{seed}.log').read_text(encoding='utf8')
  first=re.search(r'Pretrain epoch\\s+(\\d+), NB loss: nan',log)
  known=(meta['status']=='FAILED' and meta['phase']=='joint' and meta['optimizer_steps']=={'pretrain:Adam':11400}
   and meta['input_sha256']==reference['input_sha256'] and meta['runner_sha256']==reference['runner_sha256']
   and hashlib.sha256((B/'run_one.py').read_bytes()).hexdigest()==reference['runner_sha256']
   and 'assert torch.isfinite(z).all()' in meta['error'] and meta['error'].rstrip().endswith('AssertionError') and first is not None)
  if known:
   save(B/'monitoring'/f'spleen{seed}_reviewed_failure.json',{'utc':utc(),'first_nan_epoch':int(first.group(1)),'attempt':str(out),'action':'Retained numerical failure; no retry or parameter change; continue independent jobs','rule':'audit/resume_20261018T1033Z/RESUME_NOTE.json'})
  return known
 except (OSError,KeyError,ValueError):return False

'''
s=s.replace("lock=B/'queue.lock'",helper+"lock=B/'queue.lock'")
s=s.replace("if (m,d,s)==('scSAMAC','Quake_10x_Spleen',101) and meta['status']=='FAILED':","if reviewed_spleen_failure(m,d,s,out):")
s=s.replace("'review':'monitoring/spleen101_diagnostic.json'","'review':f'monitoring/spleen{s}_reviewed_failure.json'")
s=s.replace("  if job['status']!='COMPLETE':\n   # Do not silently retry or spend a full grid on a failing adapter.\n   state['status']='NEEDS_INSPECTION';break", "  if job['status']!='COMPLETE':\n   if job['status']=='FAILED' and reviewed_spleen_failure(m,d,s,out):\n    job['review']=f'monitoring/spleen{s}_reviewed_failure.json';job['retried']=False;save(B/'training_state.json',state)\n   else:\n    state['status']='NEEDS_INSPECTION';break")
compile(s,'run_queue_r3.py','exec');(B/'run_queue_r3.py').write_text(s,encoding='utf8')
(A/'queue.patch').write_text(''.join(difflib.unified_diff(old.splitlines(True),s.splitlines(True))),encoding='utf8')
note={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'reviewed_failures':{'Spleen101':'NaN starts epoch74','Spleen202':'NaN starts epoch72'},'input_hashes_identical':True,'reason':'Continue independent scheduled jobs following repeated identified numerical failure; do not alter algorithms or retry failed attempts.','strict_continuation_signature':['scSAMAC and Spleen only, original five seeds','same frozen runner and input hashes as inspected seed101','all11400 pretrain steps executed, no joint optimizer steps','NaN in pretrain log, assertion at nonfinite embedding entering joint stage'],'other_errors':'stop for inspection','unchanged':['run_one.py','evaluate_external.py','protocol.json','training budgets','fusion','source algorithms'],'successful_jobs_reused':57,'retained_failures':2,'pending':55}
(A/'RESUME_NOTE.json').write_text(json.dumps(note,indent=2,ensure_ascii=False),encoding='utf8')
print(json.dumps(note))
