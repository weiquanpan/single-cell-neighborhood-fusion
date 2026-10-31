"""Float32 R6 with checkpointed context batches and sampled relations."""
from common import *
import argparse, time, math
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from scipy import sparse
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits
sys.path.insert(0,str(ROOT/'experiments/nb_structural_context_v1'))
from context_model import StructuralGraph, student_q

def context_graph(h,quota=15):
    n=len(h);v=h-h.mean(1,keepdims=True);v=v/np.maximum(np.linalg.norm(v,axis=1,keepdims=True),1e-8)
    nn=NearestNeighbors(n_neighbors=min(quota+1,n),metric='euclidean',n_jobs=8).fit(v)
    d,j=nn.kneighbors(v);i=np.repeat(np.arange(n),j.shape[1]);j=j.ravel();w=1-d.ravel()**2/2
    keep=(i!=j)&(w>0)
    a=sparse.csr_matrix((w[keep],(i[keep],j[keep])),shape=(n,n));a=a.maximum(a.T)
    p=sparse.diags(1/np.maximum(np.asarray(a.sum(1)).ravel(),1e-8))@a
    two=(p@p).tocsr();reach=two.astype(bool)
    context=np.full((n,1+quota*2),-1,np.int64);context[:,0]=np.arange(n)
    for row in range(n):
        js=a[row].indices;vs=a[row].data;direct=js[np.lexsort((js,-vs))[:quota]]
        js=two[row].indices;vs=two[row].data;keep=(js!=row)&~np.isin(js,a[row].indices)
        js=js[keep];vs=vs[keep];indirect=js[np.lexsort((js,-vs))[:quota]]
        context[row,1:1+len(direct)]=direct;context[row,1+quota:1+quota+len(indirect)]=indirect
    dist=np.full((n,1+quota*2,1+quota*2),3,np.int64)
    for row,c in enumerate(context):
        valid=c>=0;s=c.clip(0);sub=a[s][:,s].toarray()>0;sub2=reach[s][:,s].toarray()
        dist[row]=np.where(s[:,None]==s[None,:],0,np.where(sub,1,np.where(sub2,2,3)))
        dist[row][~(valid[:,None]&valid[None,:])]=3
    return a,reach,context,dist

def encode_batch(model,h,c,d):
    valid=c>=0;x=model.input(h[c.clamp_min(0)])+model.rank(model.rank_ids)[None]+model.hop(model.hop_ids)[None]
    x=model.input_norm(x)*valid[:,:,None]
    for b in model.blocks:x=b(x,d,valid)
    return model.norm(h[c[:,0]]+model.output(x[:,0]))

def main(args):
    sys.addaudithook(guard);setup(args.seed);start=time.monotonic();task=prepared(args.dataset)
    out=EXP/'training'/args.dataset/args.variant/f'seed{args.seed}'/('smoke' if args.smoke else 'main')
    out.mkdir(parents=True,exist_ok=False)
    meta=dict(status='RUNNING',labels_read=False,seed=args.seed,dataset=args.dataset,variant=args.variant,pid=os.getpid(),protocol_sha256=sha(EXP/'protocol.json'),source_sha256=sha(__file__),started_utc=utc())
    save(out/'run.json',meta)
    h0=np.load(EXP/'cache'/args.dataset/f'nb_h_seed{args.seed}.npy').astype(np.float32)
    ids=np.load(task/'cell_positions.npy')
    if args.smoke:h0=h0[:384];ids=ids[:384]
    h0=(h0-h0.mean(0))/np.maximum(h0.std(0),1e-6)
    a,two,c,d=context_graph(h0);upper=sparse.triu(a,k=1).tocoo();n=len(h0)
    h=torch.tensor(h0,device='cuda');ct=torch.tensor(c,device='cuda');dt=torch.tensor(d,device='cuda')
    model=StructuralGraph().float().cuda();k=CFG['development'][args.dataset]
    pos=torch.tensor(np.stack([upper.row,upper.col]),device='cuda')
    rng=np.random.default_rng(args.seed);batch=CFG['r6']['batch_size']
    def encode():
        return torch.cat([checkpoint(lambda cc,dd:encode_batch(model,h,cc,dd),ct[i:i+batch],dt[i:i+batch],use_reentrant=False) for i in range(0,n,batch)])
    # Cell confidence uses agreement of local pair memberships across three seed H partitions.
    confidence=np.zeros(n,np.float32);neighbors=c[:,1:16];valid=neighbors>=0
    if args.variant=='anchored':
        memberships=[]
        for s in CFG['seeds']:
            hs=np.load(EXP/'cache'/args.dataset/f'nb_h_seed{s}.npy')[:n]
            with threadpool_limits(limits=8):y=KMeans(n_clusters=k,n_init=20,random_state=s).fit_predict(hs)
            memberships.append(y[:,None]==y[neighbors.clip(0)])
        freq=np.mean(memberships,axis=0)
        confidence=(np.sum(np.maximum(freq,1-freq)*valid,axis=1)/np.maximum(valid.sum(1),1)).astype(np.float32)
    conf=torch.tensor(confidence,device='cuda')
    hn=F.normalize(h,dim=1);anchor_index=torch.tensor(neighbors.clip(0),device='cuda');anchor_mask=torch.tensor(valid,device='cuda')
    ph=torch.softmax((hn[:,None]*hn[anchor_index]).sum(-1).masked_fill(~anchor_mask,-1e4),dim=1).detach()
    ni=rng.integers(n,size=10000);nj=rng.integers(n,size=10000)
    median=float(np.median((hn[ni]*hn[nj]).sum(-1).cpu().numpy()))
    def relations(z):
        total=z.sum()*0
        if args.variant not in ['negative','none']:
            ij=pos[:,torch.randint(pos.shape[1],(min(8192,pos.shape[1]),),device='cuda')]
            total=total+.5*F.softplus(-(z[ij[0]]*z[ij[1]]).sum(-1)/math.sqrt(32)-model.edge_bias).mean()
        if args.variant=='uniformity':
            ij=torch.randint(n,(2,8192),device='cuda');zn=F.normalize(z,dim=1)
            total=total+.5*torch.log(torch.exp(-2*(zn[ij[0]]-zn[ij[1]]).square().sum(-1)).mean()+1e-8)
        elif args.variant not in ['positive','none']:
            i=rng.integers(n,size=32768);j=rng.integers(n,size=32768);keep=(i!=j)&(np.asarray(a[i,j]).ravel()==0)
            if args.variant in ['far','anchored']:
                keep&=~np.asarray(two[i,j]).ravel()
                keep&=((hn[i]*hn[j]).sum(-1).cpu().numpy()<=median)
            i=i[keep][:8192];j=j[keep][:8192]
            if len(i):total=total+.5*F.softplus((z[i]*z[j]).sum(-1)/math.sqrt(32)+model.edge_bias).mean()
        return total
    steps=2 if args.smoke else CFG['r6']['pretrain_updates'];history=[];previous=None
    for phase in ['pretrain','joint']:
        if phase=='joint':
            with torch.no_grad():z=encode()
            with threadpool_limits(limits=8):km=KMeans(n_clusters=k,n_init=20,random_state=args.seed).fit(z.cpu().numpy())
            np.savez_compressed(out/'pretrain_outputs.npz',embedding=z.cpu().numpy(),prediction=km.labels_)
            model.mu=torch.nn.Parameter(torch.tensor(km.cluster_centers_,device='cuda'))
        opt=torch.optim.Adam(model.parameters(),lr=.001,amsgrad=True)
        for step in range(steps):
            opt.zero_grad(set_to_none=True);z=encode();feature=(model.decoder(z)-h).square().mean();edge=relations(z);total=feature+.2*edge;dec=total.new_zeros(());anchor=dec
            if args.variant=='anchored':
                zn=F.normalize(z,dim=1);logpz=F.log_softmax((zn[:,None]*zn[anchor_index]).sum(-1).masked_fill(~anchor_mask,-1e4),dim=1)
                anchor=(conf*(ph*(ph.clamp_min(1e-8).log()-logpz)).sum(1)).mean();total=total+.1*anchor
            if phase=='joint':
                q=student_q(z,model.mu)
                if step%10==0:
                    # 2x2 uses original DEC. Corrected variants use gentle, non-frequency-normalized DEC.
                    p=q.detach().pow(2 if args.variant in ['both','positive','negative','none'] else 1.5)
                    if args.variant in ['both','positive','negative','none']:p=p/q.detach().sum(0)
                    p=p/p.sum(1,keepdim=True)
                    pred=q.argmax(1).detach();change=None if previous is None else float((pred!=previous).float().mean());previous=pred
                dec=(p*(p.clamp_min(1e-8).log()-(q+1e-6).log())).sum(1).mean();total=total+dec
            if not torch.isfinite(total):raise FloatingPointError('nonfinite R6 loss')
            total.backward();opt.step()
            if step%10==0:
                row=dict(phase=phase,step=step,loss=float(total.detach()),feature=float(feature.detach()),edge=float(edge.detach()),anchor=float(anchor.detach()),dec=float(dec.detach()),seconds=time.monotonic()-start,label_change=change if phase=='joint' else None)
                history.append(row);save(out/'progress.json',row);print(row,flush=True)
    with torch.no_grad():z=encode();q=student_q(z,model.mu)
    with threadpool_limits(limits=8):km=KMeans(n_clusters=k,n_init=20,random_state=args.seed).fit(z.cpu().numpy())
    np.save(out/'cell_positions.npy',ids);np.savez_compressed(out/'outputs.npz',embedding=z.cpu().numpy(),prediction=q.argmax(1).cpu().numpy(),kmeans_prediction=km.labels_)
    torch.save(model.state_dict(),out/'final.pt');save(out/'history.json',history)
    meta.update(status='COMPLETE',seconds=time.monotonic()-start,embedding_variance=float(z.var(0).sum()),stop_reason='fixed_budget_with_label_change_diagnostics',engineering_changes='float32, sampled edges, checkpointed context minibatches; not bitwise original R6')
    save(out/'run.json',meta)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',choices=CFG['development'],required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--variant',choices=['both','positive','negative','none','far','uniformity','anchored'],required=True);p.add_argument('--smoke',action='store_true');main(p.parse_args())
