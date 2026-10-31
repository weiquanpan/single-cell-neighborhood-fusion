"""Label-isolated reviewer experiments; frozen Fusion training loop is reused."""
from pathlib import Path
import argparse, hashlib, json, os, random, sys, time, traceback

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXP = ROOT / 'experiments/cross_dataset_v1'
HOLD = EXP / 'heldout_v1'
sys.path.insert(0, str(EXP))
import common as c
import numpy as np
import torch
from scipy import sparse
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from threadpoolctl import threadpool_limits


def check_frozen():
    frozen = json.loads((EXP/'followup/final_method_v1/manifest.json').read_text())
    for rel, expected in frozen['source_sha256'].items():
        assert c.sha(ROOT/rel) == expected, ('Frozen source changed', rel)
    return frozen


def latent_edges(h):
    h = (h-h.mean(0))/np.maximum(h.std(0), 1e-6)
    v = h-h.mean(1, keepdims=True)
    v = v/np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-8)
    nn = NearestNeighbors(n_neighbors=min(16,len(v)), metric='euclidean', n_jobs=8).fit(v)
    d,j = nn.kneighbors(v)
    i = np.repeat(np.arange(len(v)),j.shape[1]); j=j.ravel(); w=1-d.ravel()**2/2
    keep=(i!=j)&(w>0)
    a=sparse.csr_matrix((w[keep],(i[keep],j[keep])),shape=(len(v),len(v)))
    co=sparse.triu(a.maximum(a.T),k=1).tocoo()
    return set(zip(co.row.tolist(),co.col.tolist()))


def pair_array(edges):
    return np.array(sorted(edges), dtype=np.int64).reshape(-1,2)


def dense_expression(gx, expression, count):
    """Smallest expanded kNN shell reaching target, trim final shell by distance."""
    from scipy.spatial.distance import pdist, squareform
    dist=squareform(pdist(gx,metric='correlation'))
    n=len(gx); q=min(32,n)
    while True:
        nn=NearestNeighbors(n_neighbors=q,metric='precomputed').fit(dist)
        dd,jj=nn.kneighbors(dist)
        # Rank rounds preserve all previous kNN shells. Pair orientation is irrelevant.
        added=set(); boundary=None
        for rank in range(q):
            js=jj[:,rank]
            shell={(min(i,int(j)),max(i,int(j))) for i,j in enumerate(js) if i!=j}
            shell-=expression|added
            if len(added)+len(shell)>=count:
                ordered=sorted(shell,key=lambda e:(float(dist[e[0],e[1]]),e[0],e[1]))
                added.update(ordered[:count-len(added)]); boundary=rank+1; break
            added.update(shell)
        if boundary is not None:
            assert len(added)==count and not added&expression
            return added,dict(query_neighbors=boundary,exact_edge_count=True,
                              final_shell_rule='Pearson distance, then ascending cell pair')
        if q==n: raise RuntimeError('Cannot fill expression edge budget')
        q=min(n,q*2)


def readouts(z,k,seed):
    with threadpool_limits(limits=8):
        raw=KMeans(n_clusters=k,n_init=20,random_state=seed).fit_predict(z)
        l2=KMeans(n_clusters=k,n_init=20,random_state=seed).fit_predict(normalize(z))
    return raw,l2


def mse_representation(dataset,seed,smoke,progress):
    out=HERE/('smoke_representations' if smoke else 'representations')/dataset/f'seed{seed}'
    if (out/'run.json').exists():
        assert json.loads((out/'run.json').read_text())['status']=='COMPLETE'
        return np.load(out/'H.npy')
    out.mkdir(parents=True,exist_ok=False)
    task=HOLD/'prepared'/dataset/'full'
    x=np.load(task/'encoder_input.npy').astype('float64')
    if smoke:x=x[:384]
    sys.path.insert(0,str(ROOT/'source/scdeepcluster'))
    from methods import Model
    model=Model(x.shape[1],'NB',seed)
    old=HOLD/'runs'/dataset/'full/NB'/f'main_seed{seed}'/'attempt_001'
    model.load_state_dict(torch.load(old/'initial.pt',map_location='cuda',weights_only=True))
    # Same encoder/decoder hidden weights and initial output linear map. Gaussian
    # reconstruction needs an unconstrained output for standardized log inputs.
    model._dec_mean[1]=torch.nn.Identity()
    class MSELoss(torch.nn.Module):
        def forward(self,x,mean,**kw):return torch.nn.functional.mse_loss(mean,x)
    model.zinb_loss=MSELoss()
    rng=torch.load(old/'initial_rng.pt',map_location='cpu',weights_only=False)
    random.setstate(rng['python']);np.random.set_state(rng['numpy'])
    torch.set_rng_state(rng['torch']);torch.cuda.set_rng_state_all(rng['cuda'])
    model.batch_generator.set_state(rng['batch']);model.pretrain_patience=401
    matched_epochs=json.loads((old/'result.json').read_text())['pretrain_epochs']
    meta=dict(status='RUNNING',dataset=dataset,seed=seed,labels_read=False,
              initial_weights_source=str(old/'initial.pt'),initial_rng_source=str(old/'initial_rng.pt'),
              target='gene-standardized library-normalized log1p input',loss='MSE',
              output_activation='linear',sigma=2.5,batch_size=256,lr=.001,amsgrad=True,
              epochs=2 if smoke else matched_epochs,unused_dispersion_head_retained=True,
              budget_rule='Match corresponding frozen NB actual pretraining epochs, including its early stopping')
    c.save(out/'run.json',meta);t=time.monotonic()
    def callback(phase,row):
        c.save(out/'progress.json',dict(phase=phase,**row))
        if row['epoch']%10==0:progress('mse_pretrain',**row)
    model.event_callback=callback
    model.pretrain_autoencoder(x,x,np.ones(len(x)),batch_size=256,lr=.001,
                               epochs=2 if smoke else matched_epochs,ae_weights=str(out/'pretrained.pt'))
    h=model.encodeBatch(torch.tensor(x,dtype=torch.float64)).cpu().numpy()
    assert np.isfinite(h).all();np.save(out/'H.npy',h)
    meta.update(status='COMPLETE',seconds=time.monotonic()-t,epochs_done=model.pretrain_epochs_done)
    c.save(out/'run.json',meta);del model;torch.cuda.empty_cache()
    return h


def main(a):
    frozen=check_frozen()
    protocol=json.loads((HERE/'protocol.json').read_text())
    assert a.dataset in protocol['datasets'] and a.seed in protocol['seeds']
    for name,expected in protocol['worker_hashes'].items():
        assert c.sha(HERE/name)==expected,('Worker changed',name)
    sys.addaudithook(c.guard)
    c.setup(a.seed)
    task=HOLD/'prepared'/a.dataset/'full'
    out=HERE/('smoke' if a.smoke else 'runs')/a.dataset/a.variant/f'seed{a.seed}'
    out.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    meta=dict(status='RUNNING',dataset=a.dataset,seed=a.seed,variant=a.variant,
              pid=os.getpid(),started_utc=c.utc(),labels_read=False,smoke=a.smoke,
              protocol_sha256=c.sha(HERE/'protocol.json'),worker_sha256=c.sha(__file__))
    c.save(out/'run.json',meta)
    def progress(phase,**row):
        row=dict(phase=phase,seconds=time.monotonic()-started,utc=c.utc(),**row)
        c.save(out/'progress.json',row)
        with (out/'history.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(row)+'\n')
    try:
        ready=json.loads((task/'ready.json').read_text())
        gx=np.load(task/'log_input.npy');raw=np.load(task/'counts.npy')
        ids=np.load(task/'cell_positions.npy');k=ready['known_k']
        with np.load(task/'statistics.npz') as f:sf=f['size_factors']
        hp=HOLD/'candidate_workspace/cache'/a.dataset/f'nb_h_seed{a.seed}.npy'
        assert c.sha(hp)==protocol['inputs'][a.dataset]['nb_h'][str(a.seed)]
        h=np.load(hp)
        if a.smoke:gx=gx[:384];raw=raw[:384];ids=ids[:384];sf=sf[:384];h=h[:384]
        n=len(gx);assert h.shape==(n,32)
        torch.set_default_dtype(torch.float32)
        from train_gcl import NBModel,GATEncoderHybrid,GraphConstruction
        from gcl_train import Train
        from torch_geometric.data import Data
        from torch_geometric.utils import add_self_loops
        data=Data(x=torch.tensor(gx,dtype=torch.float32))
        graph=GraphConstruction(data,gx,k)
        expression={tuple(sorted(e)) for e in graph.edges() if e[0]!=e[1]}
        nb_edges=latent_edges(h);nb_added=nb_edges-expression
        added=set();extra={}
        if a.variant in ['fusion_reproduce','fusion_noaux']:added=nb_added
        elif a.variant=='random':
            sys.path.insert(0,str(EXP/'followup/random_edges_v1'))
            from random_graph import sample_edges
            added=sample_edges(n,expression,len(nb_added),a.seed)
        elif a.variant=='expression_dense':
            added,extra=dense_expression(gx,expression,len(nb_added))
        elif a.variant in ['pca_union','mse_union']:
            if a.variant=='pca_union':
                x=np.load(task/'encoder_input.npy')[:n]
                pc=PCA(n_components=32,svd_solver='randomized',random_state=a.seed)
                with threadpool_limits(limits=8):rep=np.ascontiguousarray(pc.fit_transform(x))
            else:rep=mse_representation(a.dataset,a.seed,a.smoke,progress)
            added=latent_edges(rep)-expression
            p_raw,p_l2=readouts(rep,k,a.seed)
            np.savez_compressed(out/'representation.npz',embedding=rep,prediction_raw=p_raw,prediction_l2=p_l2)
        elif a.variant!='expression':raise ValueError(a.variant)
        graph.add_edges_from(sorted(added))
        edges=pair_array(expression|added)
        data.edge_index=torch.tensor(edges.T.copy(),dtype=torch.long)
        np.savez_compressed(out/'graphs.npz',expression=pair_array(expression),added=pair_array(added),
                            nb_added=pair_array(nb_added),latent=pair_array(nb_edges))
        np.save(out/'cell_positions.npy',ids)
        # Graph algorithms use only private RNGs. Reset all training RNGs to the
        # exact frozen seed after representation pretraining and graph diagnostics.
        random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
        torch.set_default_dtype(torch.float32)
        model=NBModel(GATEncoderHybrid(gx.shape[1],256,256,4,dropout_prob=.3).cuda(),256,256,.5).cuda()
        model.add_nb(torch.tensor(raw,dtype=torch.float32),torch.tensor(sf,dtype=torch.float32),
                     0. if a.variant=='fusion_noaux' else .1)
        # Keeping the zero-weight decoder preserves paired backbone initialization.
        digest=hashlib.sha256()
        for name,v in model.state_dict().items():digest.update(name.encode());digest.update(v.detach().cpu().numpy().tobytes())
        meta.update(initial_state_sha256=digest.hexdigest(),expression_edges=len(expression),
                    added_edges=len(added),target_nb_added_edges=len(nb_added),graph_edges=len(edges),
                    optimizer_updates=3 if a.smoke else 501,nb_aux_weight=model.nb_weight,
                    actual_adversary='features only; frozen code; 10 steps, beta=.01, Linf=.04',**extra)
        c.save(out/'run.json',meta);progress('graph_ready',edges=len(edges))
        cfg=json.loads((ROOT/'configs/experiment.json').read_text())['scGAAM']
        opt=torch.optim.Adam(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
        model.event_callback=lambda epoch,loss,l1,l2:progress('graph_train',epoch=epoch,loss=loss,base_loss=l1,adversarial_loss=l2)
        with threadpool_limits(limits=8):
            z,native=Train(model,data,graph,'cuda',2 if a.smoke else 500,1.,k,cfg['alpha'],cfg['beta'],
                           2 if a.smoke else 10,opt,cfg['edge_r1'],cfg['edge_r2'],cfg['feature_r1'],cfg['feature_r2'],min(400,n))
        model.load_state_dict(model.best_state);model.eval()
        edge=data.edge_index.cuda();edge,_=add_self_loops(torch.cat([edge,edge.flip(0)],1),num_nodes=n)
        with torch.no_grad():again=model(data.x.cuda(),edge,None)
        torch.testing.assert_close(z,again,rtol=1e-5,atol=1e-6)
        emb=z.detach().cpu().numpy();assert np.isfinite(emb).all()
        with threadpool_limits(limits=8):pred=KMeans(n_clusters=k,n_init=20,random_state=a.seed).fit_predict(normalize(emb))
        np.savez_compressed(out/'outputs.npz',embedding=emb,prediction=pred,author_prediction=native)
        torch.save(model.best_state,out/'selected.pt')
        if a.variant=='fusion_reproduce' and not a.smoke:
            original=HOLD/'candidate_workspace/training'/a.dataset/'NB-GCL-fusion'/f'seed{a.seed}'/'main'
            with np.load(original/'outputs.npz') as f:
                maxdiff=float(np.max(np.abs(emb-f['embedding'])))
                np.testing.assert_allclose(emb,f['embedding'],rtol=1e-5,atol=1e-6)
                np.testing.assert_array_equal(pred,f['prediction'])
            meta.update(reproduction_max_abs=maxdiff,reproduction_predictions_exact=True,
                        original_source=str(original))
        meta.update(status='COMPLETE',seconds=time.monotonic()-started,finished_utc=c.utc(),
                    selected_epoch=model.best_epoch,checkpoint_reload_max_abs=float((z-again).abs().max()),
                    peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated())
        c.save(out/'run.json',meta);progress('COMPLETE')
    except BaseException:
        meta.update(status='FAILED',error=traceback.format_exc(),finished_utc=c.utc())
        c.save(out/'run.json',meta);raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--seed',type=int,required=True)
    p.add_argument('--variant',required=True);p.add_argument('--smoke',action='store_true');main(p.parse_args())
