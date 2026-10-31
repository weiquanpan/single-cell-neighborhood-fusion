"""Post-training development readouts, with a preregistered manifest and no labels."""
from pathlib import Path
import sys,time,traceback
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import *
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from threadpoolctl import threadpool_limits
from readouts import coassociation_knn,graph_readout,merge_connectivity

OUT=EXP/'followup/new_embedding_readouts_v1'
VARIANTS=['both','positive','negative','none','far','uniformity','anchored','both_gentle']

def km(z,k,seed):
    # Matches train_gcl.py frozen readout, preserving embedding dtype.
    with threadpool_limits(limits=8):return KMeans(n_clusters=k,n_init=20,random_state=seed).fit_predict(normalize(z))

def main():
    sys.addaudithook(guard)
    assert json.loads((EXP/'queue_state.json').read_text())['status']=='COMPLETE'
    assert json.loads((EXP/'followup/controls_v1/queue_state.json').read_text())['status']=='COMPLETE'
    OUT.mkdir(exist_ok=True)
    manifest={'created_utc':utc(),'script_sha256':sha(__file__),'readouts_sha256':sha(EXP/'readouts.py'),'datasets':CFG['development'],'seeds':CFG['seeds'],'r6_variants':VARIANTS,'r6_readout':'L2 KMeans20, random_state=training seed, final 800-step embedding','consensus_views':['NB-H','PCA at K/2K/3K','scGAAM','new R6 both'],'extra_consensus_view':'NB-GCL-fusion','graph_readouts':['spectral','leiden','overmerge2','overmerge3'],'limitations':['PCA has three partitions per seed versus one for each other view','coassociation top-k ties broken by index','three readout seeds share one ensemble, not independent trainings'],'labels_read':False,'heldout_content_read':False}
    mp=OUT/'manifest.json'
    if mp.exists():
        old=json.loads(mp.read_text());assert old['script_sha256']==manifest['script_sha256'] and old['readouts_sha256']==manifest['readouts_sha256']
    else:save(mp,manifest)
    state={'status':'RUNNING','started_utc':utc(),'completed':[],'failures':[]}
    save(OUT/'queue_state.json',state)
    try:
        for d,k in CFG['development'].items():
            ids=np.load(prepared(d)/'cell_positions.npy');parts=[];extra=[]
            dest=OUT/d;dest.mkdir(exist_ok=True);np.save(dest/'cell_positions.npy',ids)
            for seed in CFG['seeds']:
                for v in VARIANTS:
                    src=EXP/('control_training' if v=='both_gentle' else 'training')/d/v/f'seed{seed}'/'main'
                    assert json.loads((src/'run.json').read_text())['status']=='COMPLETE'
                    np.testing.assert_array_equal(ids,np.load(src/'cell_positions.npy'))
                    path=dest/f'{v}__L2KMeans20__seed{seed}.npz'
                    if not path.exists():
                        with np.load(src/'outputs.npz') as f:z=f['embedding'];assert np.isfinite(z).all()
                        np.savez_compressed(path,prediction=km(z,k,seed))
                        save(path.with_suffix('.json'),{'status':'COMPLETE','source_sha256':sha(src/'outputs.npz'),'labels_read':False})
                    if v=='both':
                        with np.load(path) as f:parts.append(f['prediction'])
                for view in ['NB-H','scGAAM']:
                    src=EXP/'stage0'/d/f'seed{seed}'
                    np.testing.assert_array_equal(ids,np.load(src/'cell_positions.npy'))
                    with np.load(src/f'{view}__kmeans.npz') as f:parts.append(f['prediction'])
                src=original(d,'PCA_KMeans',seed)
                np.testing.assert_array_equal(ids,np.load(src/'cell_positions.npy'))
                with np.load(src/'outputs.npz') as f:z=f['embedding'].astype(np.float64)
                for factor in CFG['consensus_pca_k_factors']:parts.append(km(z,k*factor,seed))
                src=EXP/'training'/d/'NB-GCL-fusion'/f'seed{seed}'/'main'
                np.testing.assert_array_equal(ids,np.load(src/'cell_positions.npy'))
                with np.load(src/'outputs.npz') as f:extra.append(f['prediction'])
            for name,p in [('consensus4_newR6',parts),('consensus5_plusfusion',parts+extra)]:
                state['current']={'dataset':d,'method':name,'started_utc':utc()};save(OUT/'queue_state.json',state)
                print(d,name,flush=True)
                with threadpool_limits(limits=8):a=coassociation_knn(p)
                for seed in CFG['seeds']:
                    for method in manifest['graph_readouts']:
                        path=dest/f'{name}__{method}__seed{seed}.npz'
                        if path.exists():continue
                        if path.with_suffix('.json').exists():continue # Preserve failed attempt; no silent retry.
                        try:
                            with threadpool_limits(limits=8):
                                if method.startswith('overmerge'):y=merge_connectivity(a,graph_readout(a,int(method[-1])*k,seed,'leiden'),k)
                                else:y=graph_readout(a,k,seed,method)
                            np.savez_compressed(path,prediction=y)
                            save(path.with_suffix('.json'),{'status':'COMPLETE','partitions':len(p),'labels_read':False})
                        except Exception as exc:
                            fail={'status':'FAILED','path':str(path),'error':repr(exc),'traceback':traceback.format_exc()};save(path.with_suffix('.json'),fail);state['failures'].append(fail)
                state['completed'].append(d+'/'+name);save(OUT/'queue_state.json',state)
        state.update(status='COMPLETE',ended_utc=utc());save(OUT/'queue_state.json',state)
    except BaseException as exc:
        state.update(status='FAILED',error=repr(exc),traceback=traceback.format_exc());save(OUT/'queue_state.json',state);raise

if __name__=='__main__':main()
