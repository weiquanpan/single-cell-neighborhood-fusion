"""Short isolated diagnostic, not a formal experiment or changed training protocol."""
from pathlib import Path
import sys,time,json,copy
B=Path(__file__).resolve().parent;E=B.parent/'cross_dataset_v1';sys.path.insert(0,str(E))
import common as c
import numpy as np,torch
from sklearn.preprocessing import MinMaxScaler,normalize
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from torch.utils.data import DataLoader,Dataset
sys.path.insert(0,str(B/'adapted/scCAGN'));from pretrain import AE,LoadDataset
c.setup(913);torch.set_default_dtype(torch.float32)
out=B/'performance_audit_v1';out.mkdir(exist_ok=True)
x=normalize(MinMaxScaler().fit_transform(np.load(E/'heldout_v1/prepared/Romanov/full/counts.npy').astype('float32'))).astype('float32')
def model():return AE(100,200,200,200,200,100,x.shape[1],8).cuda()
initial=copy.deepcopy(model().state_dict());gx=torch.from_numpy(x).cuda()
class Resident(Dataset):
 def __len__(self):return len(x)
 def __getitem__(self,i):return gx[i],torch.tensor(i)
def trial(resident,profile=False):
 torch.manual_seed(913);m=model();m.load_state_dict(initial);opt=torch.optim.Adamax(m.parameters(),lr=.001)
 dl=DataLoader(Resident() if resident else LoadDataset(x),batch_size=None,shuffle=True);it=iter(dl);order=[]
 torch.cuda.synchronize();start=time.perf_counter()
 for step in range(256):
  row,idx=next(it);order.append(int(idx));row=row.to('cuda');bar,_=m(row);loss=torch.nn.functional.mse_loss(bar,row);opt.zero_grad();loss.backward();opt.step()
 torch.cuda.synchronize();seconds=time.perf_counter()-start
 return seconds,{k:v.detach().cpu().clone() for k,v in m.state_dict().items()},order
# Warm up kernels independently of measured runs.
trial(False)
results=[]
for i in range(3):
 a,wa,ia=trial(False);b,wb,ib=trial(True)
 results.append({'cpu_dataset_seconds':a,'gpu_resident_seconds':b,'same_order':ia==ib,'all_weights_bitwise_equal':all(torch.equal(wa[k],wb[k]) for k in wa),'max_weight_abs_diff':max(float((wa[k]-wb[k]).abs().max()) for k in wa)})
# CPU/operator attribution; profiler overhead makes this unsuitable for wall-time extrapolation.
m=model();m.load_state_dict(initial);opt=torch.optim.Adamax(m.parameters(),lr=.001);it=iter(DataLoader(LoadDataset(x),batch_size=None,shuffle=True))
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
 for step in range(32):
  with torch.profiler.record_function('data_loading'):row,idx=next(it)
  with torch.profiler.record_function('host_to_device'):row=row.cuda()
  with torch.profiler.record_function('forward'):bar,_=m(row);loss=torch.nn.functional.mse_loss(bar,row)
  with torch.profiler.record_function('backward'):opt.zero_grad();loss.backward()
  with torch.profiler.record_function('optimizer'):opt.step()
 torch.cuda.synchronize()
(out/'operator_profile.txt').write_text(prof.key_averages().table(sort_by='self_cpu_time_total',row_limit=25),encoding='utf8')
stages={}
torch.cuda.synchronize();t=time.perf_counter()
with torch.no_grad():bar,z=m(torch.from_numpy(x).cuda());z=z.cpu().numpy()
stages['full_embedding_seconds']=time.perf_counter()-t
t=time.perf_counter();km=KMeans(n_clusters=7,n_init=20).fit(z);stages['kmeans20_seconds']=time.perf_counter()-t
t=time.perf_counter();score=silhouette_score(z,km.labels_);stages['silhouette_seconds']=time.perf_counter()-t
t=time.perf_counter();torch.save(m.state_dict(),out/'diagnostic_checkpoint.pt');stages['checkpoint_seconds']=time.perf_counter()-t
attribution={e.key:{'cpu_total_ms':e.cpu_time_total/1000,'cpu_self_ms':e.self_cpu_time_total/1000,'device_total_ms':e.device_time_total/1000} for e in prof.key_averages() if e.key in ['data_loading','host_to_device','forward','backward','optimizer']}
r={'utc':c.utc(),'scope':'Diagnostic only, Romanov input,256 steps/trial and32 profiler steps; formal GPU process continues concurrently, absolute speed estimates are contention-affected. No labels read; no formal files changed.','trials':results,'stages':stages,'profile_ranges':attribution,'cell_count':len(x),'formal_pretrain_updates_per_Worm_run':4186*1000}
c.save(out/'profile_results.json',r);print(json.dumps(r,indent=2))
