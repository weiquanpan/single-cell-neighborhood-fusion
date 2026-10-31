from pathlib import Path
import sys,json
B=Path(__file__).resolve().parent;V=B/'performance_validation_v1';sys.path.insert(0,str(B.parent/'cross_dataset_v1'));import common as c
import numpy as np,torch
from sklearn.metrics import adjusted_rand_score
def delta(a,b):
 pairs=[]
 def visit(x,y,path=''):
  if isinstance(x,torch.Tensor):
   x=x.cpu();y=y.cpu();pairs.append({'path':path,'equal':torch.equal(x,y),'max_abs':float((x-y).abs().max()) if x.numel() else 0})
  elif isinstance(x,dict):
   for k in x:visit(x[k],y[k],path+'/'+str(k))
  elif isinstance(x,(list,tuple)):
   for i,(xx,yy) in enumerate(zip(x,y)):visit(xx,yy,path+'/'+str(i))
 visit(a,b);return {'all_equal':all(p['equal'] for p in pairs),'max_abs':max((p['max_abs'] for p in pairs),default=0),'different_tensors':sum(not p['equal'] for p in pairs)}
a=V/'baseline/runs/Romanov/scCAGN/seed101/attempt_001';b=V/'resident/runs/Romanov/scCAGN/seed101/attempt_001'
res={'utc':c.utc(),'resident':{},'graph':{}}
res['resident']['checkpoints']={p.name:delta(torch.load(p,weights_only=True,map_location='cpu'),torch.load(b/p.name,weights_only=True,map_location='cpu')) for p in a.glob('*.pt')}
with np.load(a/'outputs.npz') as x,np.load(b/'outputs.npz') as y:
 res['resident']['outputs']={k:{'exact':bool(np.array_equal(x[k],y[k])),'max_abs':float(np.max(np.abs(x[k]-y[k]))),**({'partition_ari_between_outputs':float(adjusted_rand_score(x[k],y[k])),'different_labels':int(np.sum(x[k]!=y[k]))} if k!='embedding' else {})} for k in x.files}
res['graph']=delta(torch.load(V/'graph_eager/state.pt',weights_only=True,map_location='cpu'),torch.load(V/'graph_graph/state.pt',weights_only=True,map_location='cpu'))
res['decision']='Do not deploy resident or CUDA Graph: strict full-output/state equality not met. Concurrency short test passed; full-budget test next. No frozen source change.'
c.save(V/'review_v1.json',res);print(json.dumps(res,indent=2))
