"""Compare the original training loop against the adaptation under fixed RNG."""
from common import *
import random, copy, numpy as np, torch
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops
sys.path.insert(0,str(R/'source/scgaam'))
from utils import GraphConstruction
from model import HybridGATModel
from GATEncoder import GATEncoderHybrid
from train import Train
torch.set_num_threads(8);torch.use_deterministic_algorithms(True)
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
namespace={'principal_components':None}
exec(compile((R/'upstream/nqm314-scGAAM-4b3c345/train.py').read_text(encoding='utf8'),'original_train','exec'),namespace)
def seed():
    random.seed(81);np.random.seed(81);torch.manual_seed(81)
x=np.random.RandomState(73).normal(size=(81,29)).astype(np.float32)
d=Data(x=torch.tensor(x),y=torch.arange(81)%4);g=GraphConstruction(d,x,4)
seed();base=HybridGATModel(GATEncoderHybrid(29,8,8,2,.3),8,8,.5).cuda();state=copy.deepcopy(base.state_dict())
models=[];outputs=[]
for fn in [namespace['Train'],Train]:
    model=HybridGATModel(GATEncoderHybrid(29,8,8,2,.3),8,8,.5).cuda();model.load_state_dict(state)
    seed();opt=torch.optim.Adam(model.parameters(),lr=.0005,weight_decay=.00001)
    outputs.append(fn(model,d,g,'cuda',4,1.,4,100,.01,2,opt,.4,.3,.3,.4,40));models.append(model)
assert torch.equal(outputs[0][0],outputs[1][0])
np.testing.assert_array_equal(outputs[0][1],outputs[1][1])
assert all(torch.equal(v,models[1].state_dict()[k]) for k,v in models[0].state_dict().items())
model=models[1];model.load_state_dict(model.best_state);model.eval();edge=d.edge_index.cuda();edge,_=add_self_loops(torch.cat([edge,edge.flip(0)],dim=1),num_nodes=len(x))
with torch.no_grad():again=model(d.x.cuda(),edge,None)
assert torch.equal(again,outputs[1][0])
save(R/'reports/training_equivalence.json',dict(status='PASSED',time_utc=utc(),original_loop_rounds=5,embedding_exact=True,predictions_exact=True,final_weights_exact=True,checkpoint_reload_exact=True,original_undefined_global_supplied_as_None=True,deterministic_algorithms=True))
print('TRAINING_EQUIVALENCE_PASSED')
