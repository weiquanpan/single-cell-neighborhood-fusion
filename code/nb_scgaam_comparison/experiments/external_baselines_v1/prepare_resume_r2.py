from pathlib import Path
import json,shutil,difflib,datetime
B=Path(__file__).resolve().parent;A=B/'audit/resume_20261018T0928Z';A.mkdir(exist_ok=False)
for name in ['training_state.json','run_queue.py','protocol.json'] :shutil.copy2(B/name,A/name)
for name in ['queue.stdout.log','queue.stderr.log']:
 if (B/name).exists():shutil.copy2(B/name,A/name)
s=(B/'run_queue.py').read_text();old=s
needle="   raise RuntimeError(f'Existing incomplete attempt needs inspection: {out}')"
replacement="""   if (m,d,s)==('scSAMAC','Quake_10x_Spleen',101) and meta['status']=='FAILED':
    state['jobs'].append({'method':m,'dataset':d,'seed':s,'status':'FAILED','seconds':meta['seconds'],'review':'monitoring/spleen101_diagnostic.json','retained_attempt':str(out),'retried':False});continue
   raise RuntimeError(f'Existing incomplete attempt needs inspection: {out}')"""
assert needle in s;s=s.replace(needle,replacement)
s=s.replace("rc=subprocess.call([PY,str(B/'evaluate_external.py'),'--package'],stdout=f,stderr=subprocess.STDOUT,cwd=B)","rc=subprocess.call([PY,str(B/'evaluate_external.py')]+([] if any(j['status']=='FAILED' for j in state['jobs']) else ['--package']),stdout=f,stderr=subprocess.STDOUT,cwd=B)")
s=s.replace("state['status']='COMPLETE' if rc==0 else 'EVALUATION_OR_PACKAGE_FAILED'","state['status']=('EXHAUSTED_WITH_RETAINED_FAILURES' if any(j['status']=='FAILED' for j in state['jobs']) else 'COMPLETE') if rc==0 else 'EVALUATION_OR_PACKAGE_FAILED'")
# Terminate the Windows process tree on the existing 48h engineering timeout.
s=s.replace("job['status']='TIMED_OUT';proc.kill();proc.wait();break","job['status']='TIMED_OUT'\n     if os.name=='nt':subprocess.run(['taskkill','/PID',str(proc.pid),'/T','/F'],stdout=f,stderr=subprocess.STDOUT)\n     else:proc.kill()\n     proc.wait();break")
(B/'run_queue_r2.py').write_text(s,encoding='utf8');compile(s,'run_queue_r2.py','exec')
(A/'queue.patch').write_text(''.join(difflib.unified_diff(old.splitlines(True),s.splitlines(True),fromfile='run_queue.py',tofile='run_queue_r2.py')),encoding='utf8')
note={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'reason':'Reviewed scSAMAC Spleen101 numerical failure; all prepared inputs finite/hash verified, no established engineering cause. Continue remaining independent jobs without retrying or overwriting failed attempt.','changes':['queue-only allowlist for exactly one reviewed failure','all future unreviewed failures still stop for inspection','do not declare complete or completion-package with missing results','Windows timeout terminates launcher and worker tree'],'unchanged':['training source','evaluation source','algorithm budgets','data','fusion protocol'],'successful_jobs_reused':57,'retained_failure_count':1,'pending_jobs':56}
(A/'RESUME_NOTE.json').write_text(json.dumps(note,indent=2,ensure_ascii=False),encoding='utf8')
print(json.dumps(note))
