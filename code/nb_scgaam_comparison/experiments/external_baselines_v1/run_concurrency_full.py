from pathlib import Path
import sys,json,time,subprocess,os,traceback
B=Path(__file__).resolve().parent;V=B/'performance_concurrency_v1';sys.path.insert(0,str(B.parent/'cross_dataset_v1'));import common as c
state={'status':'RUNNING','pid':os.getpid(),'started_utc':c.utc()}
def save():c.save(V/'state.json',state)
def launch(name,seed):
 f=(V/(name+'.log')).open('x',encoding='utf8')
 p=subprocess.Popen([sys.executable,'-u',str(B/'validate_concurrency_full.py'),'--case',name,'--seed',str(seed)],stdout=f,stderr=subprocess.STDOUT,cwd=B)
 return p,f
def wait(jobs):
 while any(p.poll() is None for p,f in jobs):
  state['heartbeat_utc']=c.utc();save();time.sleep(20)
 for p,f in jobs:
  f.close()
  if p.returncode:raise RuntimeError('Full concurrency validation failed; logs retained')
try:
 state['phase']='serial202';t=time.perf_counter();jobs=[launch('serial202',202)];state['child_pids']=[jobs[0][0].pid];save();wait(jobs);serial202=time.perf_counter()-t
 state['phase']='parallel101_202';t=time.perf_counter();jobs=[launch('parallel101',101),launch('parallel202',202)];state['child_pids']=[p.pid for p,f in jobs];save();wait(jobs);parallel=time.perf_counter()-t
 results={}
 for seed in [101,202]:
  ref=B/'performance_validation_v1/baseline' if seed==101 else V/'serial202';candidate=V/f'parallel{seed}'
  a=json.loads((ref/'validation_complete.json').read_text());b=json.loads((candidate/'validation_complete.json').read_text());ta=json.loads((ref/'trajectory.json').read_text());tb=json.loads((candidate/'trajectory.json').read_text())
  results[str(seed)]={'full_trajectory_exact':ta==tb,'different_epoch_records':[i for i,(x,y) in enumerate(zip(ta,tb)) if x!=y],'final_equality':{k:a[k]==b[k] for k in ['optimizer_steps','selected_restart','checkpoint_tensor_hashes','output_hashes']}}
 base=json.loads((B/'performance_validation_v1/baseline/validation_complete.json').read_text())
 state.update(status='COMPLETE_PENDING_REVIEW',phase='review',results=results,serial_pair_seconds=base['wall_seconds']+serial202,parallel_pair_seconds=parallel,throughput_speedup=(base['wall_seconds']+serial202)/parallel,deployed=False,finished_utc=c.utc(),caveat='Production task remained running; one serial reference collected earlier. Check contention, full equality and large-data memory before promotion.')
 save()
except BaseException:
 state.update(status='FAILED_NEEDS_INSPECTION',error=traceback.format_exc());save();raise
