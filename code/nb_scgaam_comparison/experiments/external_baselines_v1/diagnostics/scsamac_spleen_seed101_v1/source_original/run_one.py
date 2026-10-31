"""Label-isolated adapters. Original algorithms and all adaptations are separately audited."""
from pathlib import Path
import sys,os,json,time,argparse,traceback,platform
B=Path(__file__).resolve().parent;E=B.parent/'cross_dataset_v1';sys.path.insert(0,str(E))
import common as c
import numpy as np,torch
from types import SimpleNamespace
from threadpoolctl import threadpool_limits
P=json.loads((B/'protocol.json').read_text())

def prepared(d):
 if d in P['development']:return c.ROOT/'data/prepared'/d/'full'
 return E/('heldout_muraro_corrected_v1' if d=='Muraro' else 'heldout_v1')/'prepared'/d/'full'

def main(a):
 out=B/'runs'/a.dataset/a.method/f'seed{a.seed}'/(('smoke_' if a.smoke else 'attempt_')+a.attempt);out.mkdir(parents=True,exist_ok=False)
 start=time.perf_counter();task=prepared(a.dataset);ids=np.load(task/'cell_positions.npy');k=P['development'].get(a.dataset)
 if k is None:k=json.loads((task/'ready.json').read_text())['known_k']
 meta={'status':'RUNNING','utc':c.utc(),'pid':os.getpid(),'method':a.method,'dataset':a.dataset,'seed':a.seed,'smoke':a.smoke,'protocol_sha256':c.sha(B/'protocol.json'),'runner_sha256':c.sha(__file__),'labels_read':False,'phase':'initializing','optimizer_steps':{},'phase_seconds':{},'input_sha256':{p.name:c.sha(p) for p in task.glob('*.npy') if p.name in ['counts.npy','encoder_input.npy','log_input.npy','cell_positions.npy']},'environment':{'python':sys.version,'torch':torch.__version__,'numpy':np.__version__,'gpu':torch.cuda.get_device_name(),'platform':platform.platform()}}
 def update(**kw):meta.update(kw,seconds=time.perf_counter()-start);c.save(out/'run.json',meta)
 def progress(*xs,**kw):
  kw['flush']=True;print(*xs,**kw);c.save(out/'progress.json',{'utc':c.utc(),'phase':meta['phase'],'seconds':time.perf_counter()-start,'message':' '.join(map(str,xs)),'optimizer_steps':meta['optimizer_steps']})
 def phase(name):meta['phase']=name;update()
 # Count optimizer steps without modifying optimizer math. Per-phase and class counts.
 for cls in [torch.optim.Adam,torch.optim.Adamax,torch.optim.Adadelta]:
  old=cls.step
  def step(self,*args,_old=old,_name=cls.__name__,**kw):
   result=_old(self,*args,**kw);key=meta['phase']+':'+_name;meta['optimizer_steps'][key]=meta['optimizer_steps'].get(key,0)+1;return result
  cls.step=step
 update()
 try:
  assert c.sha(E/'followup/final_method_v1/manifest.json')==P['frozen_fusion_manifest_sha256']
  for name,h in P['adapted_sha256'].items():assert c.sha(B/name)==h,name
  source=(c.ROOT/'experiments/scsamac_author_worm_v1/author_source' if a.method=='scSAMAC' else B/'upstream'/a.method)
  for name,h in P['source'][a.method]['files'].items():assert c.sha(source/name)==h,name
  sys.addaudithook(c.guard);c.setup(a.seed)
  if a.smoke:ids=ids[:384]
  n=len(ids)
  if a.method=='scSAMAC':
   sys.path.insert(0,str(source));import scSAMAC as mod
   torch.set_default_dtype(torch.float64);mod.print=progress
   original=mod.KMeans
   class StableKMeans(original):
    def fit_predict(self,*args,**kw):
     with threadpool_limits(limits=1):return super().fit_predict(*args,**kw)
   mod.KMeans=StableKMeans
   class Observed(mod.scSAMAC):
    def encodeBatch(self,X,batch_size=256):
     z=super().encodeBatch(X,batch_size);assert torch.isfinite(z).all();self.last_z=z.detach().cpu().numpy().copy();self.last_state={k:v.detach().cpu().clone() for k,v in self.state_dict().items()};return z
   x=np.load(task/'encoder_input.npy')[:n].astype('float64');raw=np.load(task/'counts.npy')[:n].astype('float64')
   with np.load(task/'statistics.npz') as f:sf=f['size_factors'][:n].astype('float64')
   model=Observed(input_dim=x.shape[1],z_dim=32,encodeLayer=[256,64],decodeLayer=[64,256],sigma=2.5,alpha=1.,gamma=1.,device='cuda')
   phase('pretrain');t=time.perf_counter();model.pretrain_autoencoder(x,raw,sf,batch_size=256,lr=.0001,epochs=2 if a.smoke else 300,ae_weights=str(out/'pretrained.pt'));meta['phase_seconds']['pretrain']=time.perf_counter()-t
   phase('joint');t=time.perf_counter();pred=model.fit(x,raw,sf,n_clusters=k,y=None,lr=1.,batch_size=256,num_epochs=2 if a.smoke else 2000,update_interval=1,tol=.001,save_dir=str(out))[0];meta['phase_seconds']['joint']=time.perf_counter()-t
   z=model.last_z.copy();state=model.last_state;torch.save(state,out/'prediction_checkpoint.pt');model.load_state_dict(state)
   with torch.no_grad():again=model.encodeBatch(torch.tensor(x,device='cuda'));q=model.soft_assign(again)
   np.testing.assert_array_equal(again.cpu().numpy(),z);np.testing.assert_array_equal(q.argmax(1).cpu().numpy(),pred);meta['checkpoint_reload_exact']=True
  elif a.method=='scAGCL':
   # Upstream only uses scatter_add(sum, dim=0); PyG provides the same operation
   # via native torch scatter_add, avoiding an unavailable compiled extension.
   import types
   from torch_geometric.utils import scatter
   shim=types.ModuleType('torch_scatter');shim.scatter_add=lambda src,index,dim=0,dim_size=None:scatter(src,index,dim=dim,dim_size=dim_size,reduce='sum');sys.modules['torch_scatter']=shim
   meta['compatibility_adapter']='torch_scatter.scatter_add -> torch_geometric.utils.scatter(reduce=sum); no graph or objective change'
   sys.path.insert(0,str(source));import utils as u;import train as tr;from model import Encoder,AGCLModel;from torch_geometric.data import Data
   torch.set_default_dtype(torch.float32);tr.print=progress
   x=np.load(task/'log_input.npy')[:n].astype('float32');data=Data(x=torch.from_numpy(x))
   phase('graph');t=time.perf_counter();g=u.GraphConstruction(data,x,k);meta['phase_seconds']['graph']=time.perf_counter()-t
   model=AGCLModel(Encoder(data.num_features,256,num_layers=2),256,256,.5).cuda();opt=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=.00001)
   phase('joint');t=time.perf_counter();emb=tr.Train(model,data,g,'cuda',2 if a.smoke else 500,1.,100,.01,10,opt,.4,.3,.3,.4,400);meta['phase_seconds']['joint']=time.perf_counter()-t
   z=emb.detach().cpu().numpy();torch.save(model.state_dict(),out/'terminal_checkpoint.pt')
   from sklearn.cluster import KMeans
   from sklearn.preprocessing import normalize
   with threadpool_limits(limits=8):pred=KMeans(n_clusters=k,n_init='auto',random_state=0).fit_predict(normalize(z))
   meta['checkpoint_note']='Author selects embedding at minimum training loss; terminal weights are NOT that embedding checkpoint. Exact embedding retained.'
  else:
   sys.path.insert(0,str(B/'adapted/scCAGN'));import scCAGN as mod;import pretrain as pre;import utils as u;import KNN
   from sklearn.preprocessing import MinMaxScaler,normalize
   import scipy.sparse as sp
   torch.set_default_dtype(torch.float32);mod.print=progress;pre.print=progress;pre.OUTPUT_DIR=out
   x=normalize(MinMaxScaler().fit_transform(np.load(task/'counts.npy')[:n].astype('float32'))).astype('float32')
   mod.device=torch.device('cuda');mod.args=SimpleNamespace(Auto=False,pretrain_frequency=1 if a.smoke else 20,pretrain_epoch=2 if a.smoke else 50,name=a.dataset,n_input=x.shape[1],n_z=8,lr=.001,Train_epoch=2 if a.smoke else 30)
   phase('graph');t=time.perf_counter();edges=KNN.calculateKNNgraphDistanceMatrixStatsSingleThread(x);edges=np.asarray([e[:2] for e in edges if e[2]==1],dtype=np.int64)
   adj=sp.coo_matrix((np.ones(len(edges)),(edges[:,0],edges[:,1])),shape=(n,n),dtype=np.float32);adj=adj+adj.T.multiply(adj.T>adj)-adj.multiply(adj.T>adj);adj=u.sparse_mx_to_torch_sparse_tensor(u.normalize(adj+sp.eye(n))).coalesce();mod.load_graph=lambda name:adj;meta['phase_seconds']['graph']=time.perf_counter()-t
   phase('pretrain');t=time.perf_counter();best=mod.pretarin_cluster(k,x,mod.device);meta['phase_seconds']['pretrain']=time.perf_counter()-t;meta['selected_pretrain_restart']=int(best);mod.args.pretrain_path=str(out/f'pretrain_{best}.pt')
   phase('joint');t=time.perf_counter();model,z,pred=mod.train_cluster(SimpleNamespace(x=x),k,mod.device);meta['phase_seconds']['joint']=time.perf_counter()-t;torch.save(model.state_dict(),out/'terminal_checkpoint.pt')
  phase('readout');t=time.perf_counter()
  from sklearn.cluster import KMeans
  from sklearn.preprocessing import normalize
  assert z.shape[0]==n and np.isfinite(z).all() and len(pred)==n
  with threadpool_limits(limits=8):matched=KMeans(n_clusters=k,n_init=20,random_state=a.seed).fit_predict(normalize(z))
  np.save(out/'cell_positions.npy',ids);np.savez_compressed(out/'outputs.npz',embedding=z,prediction=pred,prediction_kmeans20=matched)
  meta['phase_seconds']['readout']=time.perf_counter()-t;meta['peak_cuda_allocated_bytes']=torch.cuda.max_memory_allocated();update(status='COMPLETE',phase='done',finished_utc=c.utc());progress('COMPLETE',meta['seconds'])
 except BaseException:
  update(status='FAILED',error=traceback.format_exc(),finished_utc=c.utc());raise

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--method',choices=P['methods'],required=True);p.add_argument('--dataset',required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--smoke',action='store_true');p.add_argument('--attempt',default='001');main(p.parse_args())
