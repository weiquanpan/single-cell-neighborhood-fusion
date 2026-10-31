"""Reanalyse archived outputs; never opens evaluation labels. Resume per result."""
from common import *
import argparse, time, traceback
import numpy as np
from threadpoolctl import threadpool_limits
from readouts import readout, coassociation_knn, graph_readout, merge_connectivity

R6_DIRS={
 'Worm_Neuron':['nb_structural_context_v1','nb_structural_context_v1_expand'],
 'Limb_Muscle':['nb_structural_context_limb_v1','nb_r6_diag_limb_full_s22','nb_r6_diag_limb_full_s33'],
 'Tosches':['nb_structural_context_tosches_r2_retry_002','nb_r6_diag_tosches_full_s22'],
 'Bach':[]}

def embeddings(dataset,seed):
    task=prepared(dataset);ids=np.load(task/'cell_positions.npy');out={};sources={}
    for method,name in [('NB','NB-DEC'),('scGAAM','scGAAM'),('PCA_KMeans','PCA')]:
        p=original(dataset,method,seed)
        np.testing.assert_array_equal(ids,np.load(p/'cell_positions.npy'))
        with np.load(p/'outputs.npz') as f:out[name]=f['embedding']
        sources[name]={'path':str(p/'outputs.npz'),'sha256':sha(p/'outputs.npz')}
    cache=EXP/'cache'/dataset/f'nb_h_seed{seed}.npy';cache.parent.mkdir(parents=True,exist_ok=True)
    p=original(dataset,'NB',seed)/'pretrained.pt'
    if not cache.exists():
        import torch
        sys.path.insert(0,str(ROOT/'source/scdeepcluster'))
        from methods import Model
        m=Model(2000,'NB',seed)
        state=torch.load(p,map_location='cuda',weights_only=True)['ae_state_dict']
        m.load_state_dict(state,strict=True);m.eval()
        with torch.no_grad():h=m.encodeBatch(torch.as_tensor(np.load(task/'encoder_input.npy'),dtype=torch.float64)).cpu().numpy()
        np.save(cache,h);del m;torch.cuda.empty_cache()
    out['NB-H']=np.load(cache);sources['NB-H']={'path':str(p),'sha256':sha(p)}
    candidates=[]
    for d in R6_DIRS[dataset]:
        for p in (ROOT/'experiments'/d/'runs').glob(f'seed{seed}/attempt_*/outputs.npz'):
            meta=json.loads((p.parent/'run.json').read_text())
            if meta['status'] in ['COMPLETE','COMPLETED']:candidates.append(p)
    if len(candidates)==1:
        p=candidates[0];np.testing.assert_array_equal(ids,np.load(p.parent/'cell_positions.npy'))
        with np.load(p) as f:out['R6']=f['embedding']
        sources['R6']={'path':str(p),'sha256':sha(p)}
    return out,sources,ids

def main(datasets,methods):
    sys.addaudithook(guard);setup(11)
    save(EXP/'stage0_manifest.json',{'utc':utc(),'protocol_sha256':sha(EXP/'protocol.json'),'sources':{p.name:sha(p) for p in EXP.glob('*.py')},'labels_read':False})
    for dataset in datasets:
        k=CFG['development'][dataset];parts=[];four_parts=[];complete_r6=True
        for seed in CFG['seeds']:
            emb,sources,ids=embeddings(dataset,seed)
            folder=EXP/'stage0'/dataset/f'seed{seed}';folder.mkdir(parents=True,exist_ok=True)
            np.save(folder/'cell_positions.npy',ids);save(folder/'sources.json',sources)
            for view,z in emb.items():
                assert len(z)==len(ids) and np.isfinite(z).all()
                for method in methods:
                    path=folder/f'{view}__{method}.npz'
                    if path.exists():continue
                    start=time.monotonic();print(dataset,seed,view,method,flush=True)
                    try:
                        with threadpool_limits(limits=8):y=readout(z,k,seed,method)
                        np.savez_compressed(path,prediction=y)
                        save(path.with_suffix('.json'),{'status':'COMPLETE','seconds':time.monotonic()-start,'k':int(len(np.unique(y))),'labels_read':False})
                    except Exception as e:
                        save(path.with_suffix('.json'),{'status':'FAILED','error':repr(e),'traceback':traceback.format_exc()})
            for view in ['NB-H','scGAAM']:
                with threadpool_limits(limits=8):y=readout(emb[view],k,seed,'kmeans')
                parts.append(y);four_parts.append(y)
            for factor in CFG['consensus_pca_k_factors']:
                with threadpool_limits(limits=8):y=readout(emb['PCA'],k*factor,seed,'kmeans')
                parts.append(y);four_parts.append(y)
            if 'R6' in emb:
                with threadpool_limits(limits=8):four_parts.append(readout(emb['R6'],k,seed,'kmeans'))
            else:complete_r6=False
        for name,part in [('consensus3',parts)]+([('consensus4',four_parts)] if complete_r6 else []):
            folder=EXP/'stage0'/dataset/name;folder.mkdir(parents=True,exist_ok=True)
            np.save(folder/'cell_positions.npy',ids)
            print(dataset,name,'coassociation',flush=True)
            a=coassociation_knn(part)
            for seed in CFG['seeds']:
                for method in ['spectral','leiden','overmerge2','overmerge3']:
                    path=folder/f'seed{seed}__{method}.npz'
                    if path.exists():continue
                    try:
                        with threadpool_limits(limits=8):
                            if method.startswith('overmerge'):
                                y=merge_connectivity(a,graph_readout(a,int(method[-1])*k,seed,'leiden'),k)
                            else:y=graph_readout(a,k,seed,method)
                        np.savez_compressed(path,prediction=y)
                        save(path.with_suffix('.json'),{'status':'COMPLETE','n_partitions':len(part),'labels_read':False,'ensemble_seeds':CFG['seeds'],'note':'Three readout seeds on one shared ensemble, not independent training replicates'})
                    except Exception as e:save(path.with_suffix('.json'),{'status':'FAILED','error':repr(e)})
        save(EXP/'stage0'/dataset/'status.json',{'complete_r6_views':complete_r6,'labels_read':False})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--datasets',nargs='+',default=list(CFG['development']));p.add_argument('--readouts',nargs='+',default=CFG['readouts']);a=p.parse_args();main(a.datasets,a.readouts)
