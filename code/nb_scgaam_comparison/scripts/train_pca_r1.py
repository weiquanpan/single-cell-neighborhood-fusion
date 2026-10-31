"""One label-isolated model attempt. Evaluation is a separate process."""
from common import *
import argparse, math, random, time, traceback
def guard(event,args):
    if event=='open' and args and isinstance(args[0],(str,bytes,os.PathLike)):
        s=os.fsdecode(args[0]).replace('\\','/').lower()
        if '/evaluation_only/' in s or 'reference_labels' in s or '/data/raw/' in s:raise PermissionError('Training label isolation guard: '+s)
sys.addaudithook(guard)
import numpy as np, torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
p=argparse.ArgumentParser();p.add_argument('--dataset',choices=DATASETS,required=True);p.add_argument('--method',choices=['A','NB','scGAAM','PCA_KMeans'],required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--depth',type=float,default=1);p.add_argument('--role',choices=['smoke','pilot','main'],required=True);p.add_argument('--out',required=True);a=p.parse_args()
assert a.method=='PCA_KMeans','Revision driver is restricted to PCA_KMeans'
out=Path(a.out);out.mkdir(parents=True,exist_ok=True);assert not (out/'run.json').exists()
config=json.loads((R/'configs/experiment.json').read_text());task=taskpath(a.dataset,a.depth,a.seed);data_meta=json.loads((task/'ready.json').read_text())
start=time.monotonic();meta=dict(status='RUNNING',**vars(a),started_utc=utc(),pid=os.getpid(),labels_read=False,label_guard=True,config_sha256=sha(R/'configs/experiment.json'),data_identity_sha256=sha(task/'ready.json'),source_sha256={str(f.relative_to(R)):sha(f) for f in (R/'source').rglob('*.py')},driver_sha256=sha(__file__),torch=torch.__version__,protocol_id=config['protocol_id'])
meta['method_protocol_id']='pca_fit_then_transform_contiguous_r1'
save(out/'run.json',meta)
(out/'worker_source.py').write_text(Path(__file__).read_text(encoding='utf8'),encoding='utf8')
def progress(phase,**fields):
    row=dict(phase=phase,elapsed_seconds=time.monotonic()-start,**fields);save(out/'progress.json',row)
    with (out/'history.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(row)+'\n')
try:
    torch.set_num_threads(8);torch.set_num_interop_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    assert torch.cuda.is_available();torch.cuda.reset_peak_memory_stats();random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
    x=np.load(task/'encoder_input.npy');raw=np.load(task/'counts.npy');gx=np.load(task/'log_input.npy');ids=np.load(task/'cell_positions.npy');genes=np.load(task/'gene_positions.npy')
    with np.load(task/'statistics.npz') as f:stats={k:f[k] for k in f.files}
    if a.role=='smoke':
        chosen=np.sort(np.random.RandomState(92211).choice(len(x),min(384,len(x)),replace=False))
        x=x[chosen];raw=raw[chosen];gx=gx[chosen];ids=ids[chosen]
        for key in ['size_factors','library_sizes']:stats[key]=stats[key][chosen]
    np.save(out/'cell_positions.npy',ids);np.save(out/'gene_positions.npy',genes)
    k=DATASETS[a.dataset][3];progress('loaded',shape=list(x.shape));graph_seconds=0;prediction_seconds=0;train_start=time.monotonic()
    if a.method in ['A','NB']:
        sys.path.insert(0,str(R/'source/scdeepcluster'));from methods import Model
        torch.use_deterministic_algorithms(True);m=Model(x.shape[1],a.method,a.seed);m.configure_data(raw,stats)
        if a.method=='NB':
            parentroot=out.parents[2]/'A'/out.parent.name
            candidates=sorted(parentroot.glob('attempt_*/result.json'))
            candidates=[f.parent for f in candidates if json.loads(f.read_text()).get('status')=='COMPLETED']
            assert len(candidates)==1,('Need one completed paired A',str(parentroot))
            parent=candidates[0];pm=json.loads((parent/'run.json').read_text());assert pm['data_identity_sha256']==meta['data_identity_sha256'] and pm['config_sha256']==meta['config_sha256']
            initial=torch.load(parent/'initial.pt',map_location='cuda',weights_only=True);own=m.state_dict();common={key:v for key,v in initial.items() if key in own and own[key].shape==v.shape};own.update(common);m.load_state_dict(own)
            assert all(torch.equal(m.state_dict()[key],v) for key,v in common.items())
            rng=torch.load(parent/'initial_rng.pt',map_location='cpu',weights_only=False);random.setstate(rng['python']);np.random.set_state(rng['numpy']);torch.set_rng_state(rng['torch']);torch.cuda.set_rng_state_all(rng['cuda']);m.batch_generator.set_state(rng['batch'])
            meta.update(initialization_parent=str(parent),initial_common_keys=list(common),paired_initialization_exact=True)
        torch.save(m.state_dict(),out/'initial.pt');torch.save(dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all(),batch=m.batch_generator.get_state()),out/'initial_rng.pt')
        def callback(phase,row):
            progress(phase,**row)
            if (phase=='pretraining' and row['epoch']%50==0) or (phase=='joint' and row['update']%500==0):torch.save(m.state_dict(),out/'rolling.pt')
        m.event_callback=callback;m.max_updates=4 if a.role=='smoke' else config['A_NB']['max_updates']
        m.pretrain_autoencoder(x.astype('float64'),raw.astype('float64'),stats['size_factors'].astype('float64'),epochs=2 if a.role=='smoke' else config['A_NB']['pretrain_epochs'],batch_size=256,ae_weights=str(out/'pretrained.pt'))
        m.fit(x.astype('float64'),raw.astype('float64'),stats['size_factors'].astype('float64'),n_clusters=k,y=None,batch_size=256,num_epochs=math.ceil(m.max_updates/math.ceil(len(x)/256))+1,update_interval=1,tol=0 if a.role=='smoke' else config['A_NB']['tol'],save_dir=str(out))
        torch.cuda.synchronize();train_seconds=time.monotonic()-train_start;infer_start=time.monotonic()
        with torch.no_grad():z=m.encodeBatch(torch.tensor(x,dtype=torch.float64));q=m.soft_assign(z)
        pred=q.argmax(1).cpu().numpy();emb=z.cpu().numpy();torch.cuda.synchronize();prediction_seconds=time.monotonic()-infer_start;torch.save(m.state_dict(),out/'final.pt')
        m2=Model(x.shape[1],a.method,a.seed);m2.mu=torch.nn.Parameter(torch.zeros(k,32,device='cuda',dtype=torch.float64));m2.load_state_dict(torch.load(out/'final.pt',map_location='cuda',weights_only=True))
        with torch.no_grad():z2=m2.encodeBatch(torch.tensor(x,dtype=torch.float64));q2=m2.soft_assign(z2)
        assert torch.equal(z,z2) and torch.equal(q,q2)
        details=dict(pretrain_epochs=m.pretrain_epochs_done,joint_updates=m.joint_updates,stop_reason=m.stop_reason,paired_initialization_exact=meta.get('paired_initialization_exact'),checkpoint_reload_exact=True)
    elif a.method=='scGAAM':
        sys.path.insert(0,str(R/'source/scgaam'));from model import HybridGATModel
        from GATEncoder import GATEncoderHybrid
        from utils import GraphConstruction
        from train import Train
        from torch_geometric.data import Data
        from torch_geometric.utils import add_self_loops
        torch.set_default_dtype(torch.float32);cfg=config['scGAAM'];data=Data(x=torch.tensor(gx,dtype=torch.float32))
        gt=time.monotonic();graph=GraphConstruction(data,gx,k);graph_seconds=time.monotonic()-gt;progress('graph_ready',seconds=graph_seconds,edges=graph.number_of_edges())
        enc=GATEncoderHybrid(gx.shape[1],256,256,4,dropout_prob=.3).cuda();m=HybridGATModel(enc,256,256,.5).cuda()
        torch.save(m.state_dict(),out/'initial.pt');opt=torch.optim.Adam(m.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
        m.event_callback=lambda epoch,loss,l1,l2:progress('scgaam_train',epoch=epoch,loss=loss,base_loss=l1,adversarial_loss=l2)
        train_start=time.monotonic()
        embt,pred=Train(m,data,graph,'cuda',2 if a.role=='smoke' else cfg['num_epochs'],cfg['lambda'],k,cfg['alpha'],cfg['beta'],2 if a.role=='smoke' else cfg['num_itersAdv'],opt,cfg['edge_r1'],cfg['edge_r2'],cfg['feature_r1'],cfg['feature_r2'],min(cfg['subgraph_size'],len(gx)))
        torch.cuda.synchronize();train_seconds=time.monotonic()-train_start;infer_start=time.monotonic();emb=embt.detach().cpu().numpy()
        torch.save(m.best_state,out/'final.pt');m.load_state_dict(torch.load(out/'final.pt',map_location='cuda',weights_only=True));m.eval()
        edge=data.edge_index.cuda();edge,_=add_self_loops(torch.cat([edge,edge.flip(0)],dim=1),num_nodes=len(gx))
        with torch.no_grad():again=m(data.x.cuda(),edge,None).cpu().numpy()
        np.testing.assert_array_equal(emb,again)
        details=dict(selected_epoch=m.best_epoch,loop_iterations=(3 if a.role=='smoke' else 501),stop_reason='fixed_author_budget_minimum_loss',checkpoint_reload_exact=True,final_clustering_included_in_training_time=True)
    else:
        import pickle
        pc=PCA(n_components=min(50,gx.shape[0]-1,gx.shape[1]),random_state=a.seed,svd_solver='randomized')
        pc.fit(gx);pc.components_=np.ascontiguousarray(pc.components_);emb=pc.transform(gx)
        km=KMeans(n_clusters=k,n_init=20,random_state=a.seed);pred=km.fit_predict(emb)
        with (out/'final.pkl').open('wb') as f:pickle.dump((pc,km),f)
        train_seconds=time.monotonic()-train_start;infer_start=time.monotonic()
        with (out/'final.pkl').open('rb') as f:pc2,km2=pickle.load(f)
        again=pc2.transform(gx)
        np.testing.assert_array_equal(emb,again)
        np.testing.assert_array_equal(pred,km2.predict(again));m=None
        details=dict(checkpoint_reload_exact=True,stop_reason='pca_kmeans_complete',pca_projection_protocol='fit_then_transform_contiguous_r1')
    assert np.isfinite(emb).all() and len(pred)==len(ids)
    np.savez_compressed(out/'outputs.npz',embedding=emb,prediction=pred)
    result=dict(status='COMPLETED',dataset=a.dataset,method=a.method,seed=a.seed,depth=a.depth,role=a.role,shape=list(x.shape),parameters=sum(p.numel() for p in m.parameters()) if m else 0,graph_seconds=graph_seconds,train_seconds=train_seconds,prediction_seconds=prediction_seconds,compute_seconds=graph_seconds+train_seconds+prediction_seconds,inference_and_reload_seconds=time.monotonic()-infer_start,run_wall_seconds=time.monotonic()-start,shared_preparation_seconds=data_meta['shared_preparation_seconds'],peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),device=torch.cuda.get_device_name(),labels_read=False,deterministic_algorithms=True,**details)
    save(out/'result.json',result);meta.update(status='COMPLETED',completed_utc=utc());save(out/'run.json',meta);progress('COMPLETED');print(json.dumps(result,indent=2))
except BaseException as error:
    meta.update(status='FAILED',error=repr(error),traceback=traceback.format_exc(),ended_utc=utc());save(out/'run.json',meta);progress('FAILED',error=repr(error));raise
