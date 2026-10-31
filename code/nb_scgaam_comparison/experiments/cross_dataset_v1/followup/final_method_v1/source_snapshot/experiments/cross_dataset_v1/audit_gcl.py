"""Verify zero-NB control preserves upstream training and NB likelihood algebra."""
from common import *
import random
import numpy as np
import torch
import networkx as nx
from train_gcl import HybridGATModel,GATEncoderHybrid,NBModel
from torch_geometric.data import Data
from train import Train as Original
from gcl_train import Train as Modified
from threadpoolctl import threadpool_limits

setup(77)
def reset():random.seed(77);np.random.seed(77);torch.manual_seed(77)
def run(fn):
    reset();x=torch.rand(48,20);edge=torch.tensor([[i,i+1] for i in range(47)]).T
    data=Data(x=x,edge_index=edge);g=nx.Graph();g.add_nodes_from(range(48));g.add_edges_from(edge.T.tolist())
    m=HybridGATModel(GATEncoderHybrid(20,256,256,4,dropout_prob=.3).cuda(),256,256,.5).cuda()
    opt=torch.optim.Adam(m.parameters(),lr=.0005,weight_decay=.00001)
    with threadpool_limits(limits=8):z,p=fn(m,data,g,'cuda',2,1.,3,100,.01,2,opt,.4,.3,.3,.4,48)
    return z.detach(),p,m.best_state
a,ap,aw=run(Original);b,bp,bw=run(Modified)
torch.testing.assert_close(a,b,rtol=0,atol=0);np.testing.assert_array_equal(ap,bp)
assert all(torch.equal(aw[k],bw[k]) for k in aw)
z=torch.randn(8,256,device='cuda');counts=torch.poisson(torch.full((8,20),2.));sf=torch.linspace(.5,1.5,8)
m=NBModel(GATEncoderHybrid(20,256,256,4).cuda(),256,256,.5).cuda();m.add_nb(counts,sf,.1)
loss=m.nb_loss(z,np.arange(8));mu=torch.nn.functional.softplus(m.nb_decoder(z)).clamp_min(1e-5)*sf.cuda()[:,None];theta=torch.nn.functional.softplus(m.nb_theta).clamp_min(1e-4)
reference=-torch.distributions.NegativeBinomial(total_count=theta,logits=(mu/theta).log()).log_prob(counts.cuda()).mean()*.1
torch.testing.assert_close(loss,reference,rtol=1e-5,atol=1e-6)
save(EXP/'gcl_audit.json',{'status':'PASS','zero_nb_training_exact':True,'nb_likelihood_distribution_check':True,'labels_read':False,'scope':'48-node deterministic toy equivalence; does not establish full-dataset outcome'})
print('PASS: original scGAAM == zero-NB adapted loop; NB likelihood verified')
