"""Full-budget isolated equivalence run; never writes formal training outputs."""
from pathlib import Path
import sys,json,argparse,shutil,hashlib,importlib.util,time
B=Path(__file__).resolve().parent;V=B/'performance_concurrency_v1';E=B.parent/'cross_dataset_v1'
sys.path.insert(0,str(E));import common as c
import numpy as np,torch
from torch.utils.data import Dataset

def digest(value):
 h=hashlib.sha256()
 def walk(v):
  if isinstance(v,torch.Tensor):
   a=v.detach().cpu().contiguous();h.update(str(a.dtype).encode());h.update(str(tuple(a.shape)).encode());h.update(a.numpy().tobytes())
  elif isinstance(v,np.ndarray):h.update(str(v.dtype).encode());h.update(str(v.shape).encode());h.update(v.tobytes())
  elif isinstance(v,dict):
   for k in sorted(v,key=str):h.update(str(k).encode());walk(v[k])
  elif isinstance(v,(list,tuple)):
   for x in v:walk(x)
  else:h.update(repr(v).encode())
 walk(value);return h.hexdigest()

def main(case_name,seed):
 mode="baseline"
 case=V/case_name;case.mkdir(parents=True,exist_ok=False)
 shutil.copy2(B/'protocol.json',case/'protocol.json')
 shutil.copytree(B/'adapted/scCAGN',case/'adapted/scCAGN',ignore=shutil.ignore_patterns('__pycache__'))
 shutil.copytree(B/'upstream/scCAGN',case/'upstream/scCAGN',ignore=shutil.ignore_patterns('.git','__pycache__'))
 # Frozen protocol verifies all adapted files; only scCAGN is needed in this case.
 sys.path.insert(0,str(case/'adapted/scCAGN'));import pretrain as pre
 OriginalDataset=pre.LoadDataset;original_train=pre.pretrain_ae;original_step=torch.optim.Adamax.step
 observed={'step':0,'optimizer':None,'order':hashlib.sha256()};trajectory=[];timing={'audit_seconds':0.0}
 class TrackedDataset(Dataset):
  def __init__(self,data):self.x=data;self.original=OriginalDataset(data);self.gpu=torch.from_numpy(data).cuda() if mode=='resident' else None
  def __len__(self):return len(self.x)
  def __getitem__(self,i):
   observed['order'].update(int(i).to_bytes(8,'little'))
   return (self.gpu[i],torch.from_numpy(np.array(i))) if self.gpu is not None else self.original[i]
 pre.LoadDataset=TrackedDataset
 def counted_step(opt,*a,**kw):
  result=original_step(opt,*a,**kw);observed['optimizer']=opt;observed['step']+=1;return result
 torch.optim.Adamax.step=counted_step
 def audited_train(model,dataset,restart,*a,**kw):
  original_print=pre.print;observed['order']=hashlib.sha256();observed['step']=0
  def report(*xs,**kwargs):
   original_print(*xs,**kwargs)
   if xs and str(xs[0]).startswith('Pretrain restart='):
    t=time.perf_counter()
    import random
    trajectory.append({'restart':int(restart),'epoch':observed['step']//len(dataset),'steps':observed['step'],'sample_order_sha256':observed['order'].hexdigest(),'model_sha256':digest(model.state_dict()),'optimizer_sha256':digest(observed['optimizer'].state_dict()),'rng_sha256':digest([torch.get_rng_state(),torch.cuda.get_rng_state_all(),np.random.get_state(),random.getstate()]),'message':str(xs[0]).split(' checkpoint=')[0]})
    c.save(case/'trajectory.json',trajectory);timing['audit_seconds']+=time.perf_counter()-t
  pre.print=report
  try:return original_train(model,dataset,restart,*a,**kw)
  finally:pre.print=original_print
 pre.pretrain_ae=audited_train
 spec=importlib.util.spec_from_file_location('validation_original_runner',B/'run_one.py');runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner);runner.B=case
 from types import SimpleNamespace
 start=time.perf_counter()
 runner.main(SimpleNamespace(dataset='Romanov',seed=seed,method='scCAGN',smoke=False,attempt='001'))
 run=case/'runs/Romanov/scCAGN'/f'seed{seed}'/'attempt_001';meta=json.loads((run/'run.json').read_text())
 assert len(trajectory)==1000 and trajectory[-1]['steps']==len(np.load(E/'heldout_v1/prepared/Romanov/full/cell_positions.npy'))*50
 tensors={p.name:digest(torch.load(p,weights_only=True,map_location='cpu')) for p in run.glob('*.pt')}
 with np.load(run/'outputs.npz') as f:outputs={k:digest(f[k]) for k in f.files}
 c.save(case/'validation_complete.json',{'mode':mode,'utc':c.utc(),'wall_seconds':time.perf_counter()-start,'audit_seconds':timing['audit_seconds'],'formal_budget':{'pretrain_restarts':20,'epochs_each':50,'joint_epochs':30},'epochs_compared':len(trajectory),'optimizer_steps':meta['optimizer_steps'],'selected_restart':meta['selected_pretrain_restart'],'checkpoint_tensor_hashes':tensors,'output_hashes':outputs,'scope':'Full Romanov 101 run only. Runs concurrently with unchanged formal queue; timing affected by shared resources.'})

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--case',required=True);p.add_argument('--seed',type=int,required=True);a=p.parse_args();main(a.case,a.seed)
