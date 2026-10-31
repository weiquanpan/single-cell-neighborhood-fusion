"""Run sequential equivalence gates, then graph and independent-seed throughput tests.
Never deploy automatically: full-grid coverage and unconfounded speed need review.
"""
from pathlib import Path
import sys,json,subprocess,time,os,traceback
B=Path(__file__).resolve().parent;V=B/'performance_validation_v1';V.mkdir(exist_ok=True);sys.path.insert(0,str(B.parent/'cross_dataset_v1'));import common as c
state={'status':'RUNNING','pid':os.getpid(),'started_utc':c.utc(),'stages':[]}
def update():c.save(V/'state.json',state)
def run(cmd,log):
 with log.open('w',encoding='utf8') as f:
  p=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,cwd=B);state['child_pid']=p.pid;update()
  return p.wait()
def equivalent(a,b):
 import torch
 if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a.cpu(),b.cpu())
 if isinstance(a,dict):return a.keys()==b.keys() and all(equivalent(a[k],b[k]) for k in a)
 if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equivalent(x,y) for x,y in zip(a,b))
 return a==b
try:
 for mode in ['baseline','resident']:
  state['phase']='full_'+mode;update()
  rc=run([sys.executable,'-u',str(B/'validate_resident_full.py'),'--mode',mode],V/(mode+'.log'))
  if rc:raise RuntimeError(f'Full {mode} validation failed; inspect retained log, no retry')
 a=json.loads((V/'baseline/validation_complete.json').read_text());b=json.loads((V/'resident/validation_complete.json').read_text())
 ta=json.loads((V/'baseline/trajectory.json').read_text());tb=json.loads((V/'resident/trajectory.json').read_text())
 full_equal=ta==tb and all(a[k]==b[k] for k in ['optimizer_steps','selected_restart','checkpoint_tensor_hashes','output_hashes'])
 state['resident']={'full_run_bitwise_equal':full_equal,'epoch_records':len(ta),'baseline_seconds':a['wall_seconds'],'resident_seconds':b['wall_seconds'],'timing_caveat':'concurrent unchanged production task; one paired full case is not all-data proof'};update()
 if not full_equal:
  c.save(V/'equivalence_failure.json',{'first_mismatch':next((i for i,(x,y) in enumerate(zip(ta,tb)) if x!=y),None),'a':a,'b':b})
 # Graph microtest is a separate candidate, even if resident equivalence fails.
 import torch
 state['phase']='cuda_graph_diagnostic';update()
 for mode in ['eager','graph']:
  rc=run([sys.executable,'-u',str(B/'benchmark_acceleration.py'),'--mode',mode,'--seed','913','--out',str(V/('graph_'+mode))],V/('graph_'+mode+'.log'))
  if rc:
   state['graph']={'status':'FAILED_DIAGNOSTIC','mode':mode,'deployed':False};break
 else:
  a=torch.load(V/'graph_eager/state.pt',weights_only=True,map_location='cpu');b=torch.load(V/'graph_graph/state.pt',weights_only=True,map_location='cpu')
  state['graph']={'model_bitwise_equal':equivalent(a['model'],b['model']),'optimizer_bitwise_equal':equivalent(a['optimizer'],b['optimizer']),'eager':json.loads((V/'graph_eager/result.json').read_text()),'graph':json.loads((V/'graph_graph/result.json').read_text()),'deployed':False}
 update();state['phase']='concurrency_diagnostic';update()
 # Serial and concurrent trials use same two seeds and10000 updates each.
 start=time.perf_counter()
 for seed in [914,915]:
  rc=run([sys.executable,'-u',str(B/'benchmark_acceleration.py'),'--mode','eager','--seed',str(seed),'--out',str(V/f'serial_{seed}')],V/f'serial_{seed}.log')
  if rc:raise RuntimeError('Serial concurrency reference failed')
 serial=time.perf_counter()-start;start=time.perf_counter();procs=[];handles=[]
 for seed in [914,915]:
  f=(V/f'parallel_{seed}.log').open('w',encoding='utf8');handles.append(f);procs.append(subprocess.Popen([sys.executable,'-u',str(B/'benchmark_acceleration.py'),'--mode','eager','--seed',str(seed),'--out',str(V/f'parallel_{seed}')],stdout=f,stderr=subprocess.STDOUT,cwd=B))
 codes=[p.wait() for p in procs];parallel=time.perf_counter()-start
 for f in handles:f.close()
 if any(codes):raise RuntimeError('Concurrent diagnostic failed')
 same=all(equivalent(torch.load(V/f'serial_{s}/state.pt',weights_only=True,map_location='cpu'),torch.load(V/f'parallel_{s}/state.pt',weights_only=True,map_location='cpu')) for s in [914,915])
 state['concurrency']={'states_bitwise_equal':same,'serial_wall_seconds':serial,'parallel_wall_seconds':parallel,'throughput_speedup':serial/parallel,'steps_per_seed':10000,'deployed':False,'caveat':'Full-budget concurrent verification and representative large-data memory needed before promotion.'}
 state['status']='DIAGNOSTICS_COMPLETE_PENDING_REVIEW';state['phase']='review';state['finished_utc']=c.utc();update()
except BaseException:
 state.update(status='FAILED_NEEDS_INSPECTION',error=traceback.format_exc());update();raise
