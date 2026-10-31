"""Isolated fixed-step diagnostic; NEVER a substitute for a formal full-budget run."""
from pathlib import Path
import sys,json,time,argparse,copy,hashlib
B=Path(__file__).resolve().parent;E=B.parent/'cross_dataset_v1';sys.path.insert(0,str(E));import common as c
import numpy as np,torch
from sklearn.preprocessing import MinMaxScaler,normalize
sys.path.insert(0,str(B/'adapted/scCAGN'));from pretrain import AE
def main(a):
 c.setup(a.seed);torch.set_default_dtype(torch.float32)
 out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
 x=normalize(MinMaxScaler().fit_transform(np.load(E/'heldout_v1/prepared/Romanov/full/counts.npy').astype('float32'))).astype('float32');gx=torch.from_numpy(x).cuda()
 m=AE(100,200,200,200,200,100,2000,8).cuda();initial=copy.deepcopy(m.state_dict());opt=torch.optim.Adamax(m.parameters(),lr=.001,capturable=a.mode=='graph')
 order=np.random.RandomState(a.seed).randint(0,len(x),size=a.steps)
 if a.mode=='graph':
  # Warm up on a separate stream. Reset every parameter and optimizer tensor
  # in place after capture so captured addresses stay valid.
  static=gx[0].clone();stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
  with torch.cuda.stream(stream):
   for _ in range(3):
    opt.zero_grad(set_to_none=True);bar,_=m(static);torch.nn.functional.mse_loss(bar,static).backward();opt.step()
  torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
  graph=torch.cuda.CUDAGraph()
  opt.zero_grad(set_to_none=True)
  with torch.cuda.graph(graph):
   opt.zero_grad(set_to_none=True);bar,_=m(static);torch.nn.functional.mse_loss(bar,static).backward();opt.step()
  with torch.no_grad():
   for key,value in m.state_dict().items():value.copy_(initial[key])
   for st in opt.state.values():
    for v in st.values():
     if torch.is_tensor(v):v.zero_()
  torch.cuda.synchronize()
 torch.cuda.synchronize();start=time.perf_counter()
 for i in order:
  if a.mode=='graph':static.copy_(gx[int(i)]);graph.replay()
  else:
   row=gx[int(i)];opt.zero_grad();bar,_=m(row);torch.nn.functional.mse_loss(bar,row).backward();opt.step()
 torch.cuda.synchronize();seconds=time.perf_counter()-start
 torch.save({'model':m.state_dict(),'optimizer':opt.state_dict()},out/'state.pt')
 c.save(out/'result.json',{'mode':a.mode,'seed':a.seed,'steps':a.steps,'seconds':seconds,'order_sha256':hashlib.sha256(order.tobytes()).hexdigest(),'pid':__import__('os').getpid(),'scope':'diagnostic only; not counted toward formal budget'})
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--mode',choices=['eager','graph'],required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--steps',type=int,default=10000);p.add_argument('--out',required=True);main(p.parse_args())
