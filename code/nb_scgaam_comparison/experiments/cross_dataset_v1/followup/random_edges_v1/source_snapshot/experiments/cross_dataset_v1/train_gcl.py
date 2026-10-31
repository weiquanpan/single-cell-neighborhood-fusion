"""Sequential additive scGAAM ablations, preserving upstream optimizer/PGD loop."""
from common import *
import argparse,time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from threadpoolctl import threadpool_limits
sys.path.insert(0,str(ROOT/'source/scgaam'))
from model import HybridGATModel
from GATEncoder import GATEncoderHybrid
from utils import GraphConstruction
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops
from gcl_train import Train

class TwoHop(nn.Module):
    """R6-inspired cell-context tokens. No balancing or occupancy loss."""
    def __init__(self,encoder):
        super().__init__();self.encoder=encoder
        self.hop=nn.Parameter(torch.randn(3,256)*.02)
        self.block=nn.TransformerEncoderLayer(256,4,512,dropout=0,batch_first=True,activation='gelu')
        self.norm=nn.LayerNorm(256)
    def forward(self,x,edge,edge_weight=None):
        z=self.encoder(x,edge,edge_weight)
        if edge.shape[0]==2 and edge.dtype==torch.long:
            value=torch.ones(edge.shape[1],device=z.device) if edge_weight is None else edge_weight
            a=torch.sparse_coo_tensor(edge,value,(len(z),len(z))).coalesce()
            degree=torch.sparse.sum(a,dim=1).to_dense().clamp_min(1)[:,None]
            one=torch.sparse.mm(a,z)/degree;two=torch.sparse.mm(a,one)/degree
        else:
            degree=edge.sum(1).clamp_min(1)[:,None];one=edge@z/degree;two=edge@one/degree
        token=torch.stack([z,one,two],1)+self.hop
        # Chunk attention to keep full-graph inference memory bounded.
        context=torch.cat([self.block(token[i:i+256])[:,0] for i in range(0,len(z),256)])
        return self.norm(z+context)

class NBModel(HybridGATModel):
    def add_nb(self,counts,sf,weight):
        self.nb_decoder=nn.Sequential(nn.Linear(256,256),nn.ReLU(),nn.Linear(256,counts.shape[1])).cuda()
        self.nb_theta=nn.Parameter(torch.zeros(counts.shape[1],device='cuda'))
        self.counts=counts;self.sf=sf;self.nb_weight=weight
    def nb_loss(self,z,indices):
        x=self.counts[indices].to(z.device);sf=self.sf[indices].to(z.device)
        mu=F.softplus(self.nb_decoder(z)).clamp_min(1e-5)*sf[:,None]
        theta=F.softplus(self.nb_theta).clamp_min(1e-4)
        nll=torch.lgamma(theta)+torch.lgamma(x+1)-torch.lgamma(x+theta)+(theta+x)*torch.log1p(mu/theta)+x*(torch.log(theta)-torch.log(mu.clamp_min(1e-8)))
        return self.nb_weight*nll.mean()

def main(args):
    sys.addaudithook(guard);setup(args.seed);start=time.monotonic();task=prepared(args.dataset)
    out=EXP/'training'/args.dataset/args.variant/f'seed{args.seed}'/('smoke' if args.smoke else 'main');out.mkdir(parents=True,exist_ok=False)
    meta=dict(status='RUNNING',labels_read=False,seed=args.seed,dataset=args.dataset,variant=args.variant,pid=os.getpid(),started_utc=utc(),protocol_sha256=sha(EXP/'protocol.json'),source_sha256={p.name:sha(p) for p in [Path(__file__),EXP/'gcl_train.py']})
    save(out/'run.json',meta)
    gx=np.load(task/'log_input.npy');counts=np.load(task/'counts.npy');ids=np.load(task/'cell_positions.npy')
    with np.load(task/'statistics.npz') as f:sf=f['size_factors']
    if args.smoke:gx=gx[:384];counts=counts[:384];ids=ids[:384];sf=sf[:384]
    k=CFG['development'][args.dataset];data=Data(x=torch.tensor(gx,dtype=torch.float32))
    graph=GraphConstruction(data,gx,k)
    if args.variant in ['NB-GCL-fusion','NB-GCL-context']:
        from train_r6 import context_graph
        h=np.load(EXP/'cache'/args.dataset/f'nb_h_seed{args.seed}.npy')[:len(gx)]
        a,_,_,_=context_graph((h-h.mean(0))/np.maximum(h.std(0),1e-6))
        co=sparse.triu(a,k=1).tocoo();graph.add_edges_from(zip(co.row.tolist(),co.col.tolist()))
        # A single union graph is used for both augmented training and full inference.
        edges=np.array(sorted({tuple(sorted(e)) for e in graph.edges() if e[0]!=e[1]}),dtype=np.int64)
        data.edge_index=torch.tensor(edges.T.copy(),dtype=torch.long)
    enc=GATEncoderHybrid(gx.shape[1],256,256,4,dropout_prob=.3).cuda()
    if args.variant=='NB-GCL-context':enc=TwoHop(enc).cuda()
    cls=HybridGATModel if args.variant=='scGAAM-control' else NBModel
    model=cls(enc,256,256,.5).cuda()
    if isinstance(model,NBModel):model.add_nb(torch.tensor(counts,dtype=torch.float32),torch.tensor(sf,dtype=torch.float32),.1)
    cfg=json.loads((ROOT/'configs/experiment.json').read_text())['scGAAM']
    opt=torch.optim.Adam(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    def progress(epoch,loss,l1,l2):
        row=dict(epoch=epoch,loss=loss,contrastive=l1,adversarial=l2,seconds=time.monotonic()-start)
        save(out/'progress.json',row)
        with (out/'history.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    model.event_callback=progress
    with threadpool_limits(limits=8):
        z,pred=Train(model,data,graph,'cuda',2 if args.smoke else 500,1.,k,cfg['alpha'],cfg['beta'],2 if args.smoke else 10,opt,cfg['edge_r1'],cfg['edge_r2'],cfg['feature_r1'],cfg['feature_r2'],min(400,len(gx)))
    model.load_state_dict(model.best_state);model.eval();edge=data.edge_index.cuda();edge,_=add_self_loops(torch.cat([edge,edge.flip(0)],1),num_nodes=len(gx))
    with torch.no_grad():again=model(data.x.cuda(),edge,None)
    reload_max_abs=float((z-again).abs().max())
    torch.testing.assert_close(z,again,rtol=1e-5,atol=1e-6)
    with threadpool_limits(limits=8):km=KMeans(n_clusters=k,n_init=20,random_state=args.seed).fit_predict(normalize(z.cpu().numpy()))
    np.save(out/'cell_positions.npy',ids);np.savez_compressed(out/'outputs.npz',embedding=z.cpu().numpy(),prediction=km,author_prediction=pred)
    torch.save(model.best_state,out/'final.pt')
    meta.update(status='COMPLETE',seconds=time.monotonic()-start,selected_epoch=model.best_epoch,checkpoint_reload_exact=reload_max_abs==0,checkpoint_reload_max_abs=reload_max_abs,checkpoint_reload_tolerance={'rtol':1e-5,'atol':1e-6},readout='provisional L2 KMeans20; all embeddings retained for global frozen readout',graph_edges=graph.number_of_edges())
    save(out/'run.json',meta)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',choices=CFG['development'],required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--variant',choices=['scGAAM-control','NB-GCL','NB-GCL-fusion','NB-GCL-context'],required=True);p.add_argument('--smoke',action='store_true');main(p.parse_args())
