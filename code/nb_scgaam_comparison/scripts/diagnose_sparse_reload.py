from common import *
import numpy as np,torch
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops
sys.path.insert(0,str(R/'source/scgaam'))
from model import HybridGATModel
from GATEncoder import GATEncoderHybrid
from utils import GraphConstruction
torch.set_num_threads(8);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False
x=np.load(taskpath('Limb_Muscle',1,2211)/'log_input.npy');data=Data(x=torch.tensor(x));GraphConstruction(data,x,6)
edge=data.edge_index.cuda();edge,_=add_self_loops(torch.cat([edge,edge.flip(0)],dim=1),num_nodes=len(x));xx=data.x.cuda()
m=HybridGATModel(GATEncoderHybrid(x.shape[1],256,256,4,.3),256,256,.5).cuda();m.load_state_dict(torch.load(R/'runs/Limb_Muscle/full/scGAAM/pilot_seed2211/attempt_002/final.pt',weights_only=True));m.eval()
origmm=torch.sparse.mm;rows={};bases={}
for mode in ['original_coo','csr_mm','deterministic_index_add']:
    if mode=='csr_mm':torch.sparse.mm=lambda a,b:origmm(a.to_sparse_csr(),b)
    if mode=='deterministic_index_add':
        def stable_mm(a,b):
            a=a.coalesce();idx=a.indices();v=a.values();out=b.new_zeros((a.shape[0],b.shape[1]))
            return out.index_add_(0,idx[0],v[:,None]*b.index_select(0,idx[1]))
        torch.sparse.mm=stable_mm
    arrays=[]
    with torch.no_grad():
        for i in range(8):arrays.append(m(xx,edge,None).cpu().numpy())
    bases[mode]=arrays[0];rows[mode]=dict(repeated_exact=all(np.array_equal(arrays[0],z) for z in arrays),max_abs_repeat_error=max(float(np.max(np.abs(z-arrays[0]))) for z in arrays))
torch.sparse.mm=origmm
rows['between_methods_max_abs_error']=float(np.max(np.abs(bases['original_coo']-bases['csr_mm'])))
save(R/'reports/sparse_reload_diagnosis.json',rows);print(json.dumps(rows,indent=2))
