"""Numerical equivalence and label-independent execution checks before any main run."""
from common import *
import ast, types, numpy as np, torch, networkx as nx
from scipy.spatial.distance import pdist,squareform
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Data
sys.path.insert(0,str(R/'source/scgaam'))
from utils import GraphConstruction,GraphAdversarialAttack
from model import HybridGATModel
from GATEncoder import GATEncoderHybrid
torch.set_num_threads(8)
original=(R/'upstream/nqm314-scGAAM-4b3c345/utils.py').read_text(encoding='utf8');tree=ast.parse(original)
selected=ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['GraphConstruction','GraphAdversarialAttack']],type_ignores=[])
ns=dict(np=np,torch=torch,nx=nx,pdist=pdist,squareform=squareform,NearestNeighbors=NearestNeighbors)
exec(compile(selected,'author_original_selected_functions','exec'),ns)
x=np.random.RandomState(19).normal(size=(53,29)).astype(np.float32);d1=Data(x=torch.tensor(x));d2=Data(x=torch.tensor(x))
g1=ns['GraphConstruction'](d1,x,6);g2=GraphConstruction(d2,x,6)
assert torch.equal(d1.edge_index,d2.edge_index)
assert list(g1.edges())==list(g2.edges()),'Graph insertion ordering must remain identical'
nodes=np.random.RandomState(9).permutation(53)[:25];assert list(g1.subgraph(nodes).edges())==list(g2.subgraph(nodes).edges())
torch.manual_seed(17);model=HybridGATModel(GATEncoderHybrid(29,8,8,2,.3),8,8,.5).cuda()
features=torch.tensor(x[:12]).cuda();adj=torch.eye(12,device='cuda');aug=features.clone()
torch.manual_seed(88);a1,y1=ns['GraphAdversarialAttack'](model,adj,adj,features,aug,2,.2,100,.01,None)
torch.manual_seed(88);a2,y2=GraphAdversarialAttack(model,adj,adj,features,aug,2,.2,100,.01)
assert torch.equal(a1,a2) and torch.equal(y1,y2)
assert torch.max(abs(y2-aug))<=.040001
train_tree=ast.parse((R/'source/scgaam/train.py').read_text(encoding='utf8'))
assert not any(isinstance(n,ast.Attribute) and n.attr=='y' for n in ast.walk(train_tree))
save(R/'reports/adaptation_checks.json',dict(status='PASSED',graph_edges_exact=True,graph_node_and_edge_order_exact=True,unused_parameter_attack_output_exact=True,reference_labels_removed_from_train=True,test_device=torch.cuda.get_device_name(),time_utc=utc()))
print('ADAPTATION_CHECKS_PASSED')
