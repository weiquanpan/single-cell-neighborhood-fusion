from common import *
import numpy as np, torch
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops
sys.path.insert(0,str(R/'source/scgaam'))
from model import HybridGATModel
from GATEncoder import GATEncoderHybrid
from utils import GraphConstruction
torch.set_num_threads(8);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False
ns={};text=(R/'upstream/nqm314-scGAAM-4b3c345/GATEncoder.py').read_text(encoding='utf8')
text='\n'.join(line for line in text.split('\n') if not line.startswith('from torch_scatter import'))
exec(compile(text,'original_GATEncoder','exec'),ns)
x=np.load(taskpath('Limb_Muscle',1,2211)/'log_input.npy');d=Data(x=torch.tensor(x));GraphConstruction(d,x,6)
edge=d.edge_index.cuda();edge,_=add_self_loops(torch.cat([edge,edge.flip(0)],dim=1),num_nodes=len(x));xx=d.x.cuda()
models=[]
state=torch.load(R/'runs/Limb_Muscle/full/scGAAM/pilot_seed2211/attempt_002/final.pt',map_location='cuda',weights_only=True)
for cls in [ns['GATEncoderHybrid'],GATEncoderHybrid]:
    m=HybridGATModel(cls(x.shape[1],256,256,4,.3),256,256,.5).cuda();m.load_state_dict(state);m.eval();models.append(m)
with torch.no_grad():
    old=models[0](xx,edge,None);new=models[1](xx,edge,None)
    torch.testing.assert_close(new,old,rtol=1e-5,atol=1e-5)
    for _ in range(8):assert torch.equal(new,models[1](xx,edge,None))
# The dense training path, gradients and optimizer updates are unchanged exactly.
adj=torch.eye(40,device='cuda');outs=[]
for m in models:
    m.train();torch.manual_seed(14);z=m(xx[:40],adj);loss=z.square().mean();loss.backward();outs.append(z.detach())
assert torch.equal(outs[0],outs[1])
assert all(torch.equal(p.grad,q.grad) for p,q in zip(models[0].parameters(),models[1].parameters()) if p.grad is not None)
save(R/'reports/sparse_adaptation_checks.json',dict(status='PASSED',utc=utc(),full_shape=list(x.shape),dense_training_output_and_gradients_exact=True,full_inference_repeat_exact=True,repeat_count=8,original_vs_adapted_max_abs_error=float((old-new).abs().max()),original_vs_adapted_rtol=1e-5,original_vs_adapted_atol=1e-5,checkpoint_requirement='exact; not relaxed'))
print('SPARSE_ADAPTATION_PASSED')
