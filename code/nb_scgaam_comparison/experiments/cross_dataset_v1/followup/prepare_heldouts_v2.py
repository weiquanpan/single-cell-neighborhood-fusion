"""Prepare all sealed holdouts only after final-method freeze; never select a model."""
from pathlib import Path
import sys,time,traceback
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import *
import numpy as np
import scipy.sparse as sp
import h5py
import scanpy as sc

def matrix(q):
    if isinstance(q,h5py.Group):return sp.csr_matrix((q['data'][...],q['indices'][...],q['indptr'][...]),shape=tuple(q['shape'][...]))
    return sp.csr_matrix(q[...])

def main():
    frozen=EXP/'followup/final_method_v1/manifest.json';m=json.loads(frozen.read_text())
    assert m['status']=='FROZEN_FOR_HELDOUT_VALIDATION'
    for name,h in m['source_sha256'].items():assert sha(ROOT/name)==h,name
    out=EXP/'heldout_v1';out.mkdir(exist_ok=True)
    state={'status':'RUNNING','started_utc':utc(),'freeze_sha256':sha(frozen),'driver_sha256':sha(__file__),'completed':[],'input_failures':[]}
    save(out/'preparation_state.json',state)
    hashes=json.loads((EXP/'heldout_manifest.json').read_text())['files']
    try:
        for d in m['heldout_datasets']:
            state['current']=d;save(out/'preparation_state.json',state)
            dest=out/'prepared'/d/'full'
            if (dest/'ready.json').exists():state['completed'].append(d);continue
            start=time.monotonic();p=EXP/'heldout_sealed'/f'{d}.h5';assert sha(p)==hashes[d]['sha256']
            with h5py.File(p,'r') as f:
                if 'obs' in f:x=matrix(f['exprs']);y=f['obs/cell_type1'][...]
                else:x=matrix(f['X']);y=f['Y'][...]
            y=np.asarray(y).reshape(-1);assert x.shape[0]==len(y)
            if not (np.isfinite(x.data).all() and (x.data>=0).all() and np.equal(x.data,np.floor(x.data)).all()):
                state['input_failures'].append({'dataset':d,'error':'Noninteger counts; original source recovery required, no rounding or replacement'});save(out/'preparation_state.json',state);continue
            assert x.data.max()<2**24
            shape=x.shape;x=x.astype(np.float32);ids=np.flatnonzero(np.asarray(x.sum(1)).ravel()>0);x=x[ids]
            genes=np.flatnonzero(np.asarray(x.sum(0)).ravel()>0);x=x[:,genes]
            ad=sc.AnnData(x.copy());sc.pp.normalize_total(ad,exclude_highly_expressed=True);sc.pp.log1p(ad)
            sc.pp.highly_variable_genes(ad,min_mean=.0125,max_mean=3,min_disp=.5,n_top_genes=2000,subset=False)
            sel=np.flatnonzero(ad.var['highly_variable'].to_numpy());counts=x[:,sel].toarray();gx=ad.X[:,sel].toarray().astype(np.float32)
            before=ids.copy();keep=counts.sum(1)>0;excluded=ids[~keep];counts=counts[keep];gx=gx[keep];ids=ids[keep]
            lib=counts.sum(1);assert len(lib)>1 and (lib>0).all();ref=np.float32(np.median(lib));sf=lib/ref
            enc=np.log1p(counts/sf[:,None]);mean=enc.mean(0,dtype=np.float64).astype(np.float32);scale=enc.std(0,ddof=1,dtype=np.float64).astype(np.float32);scale[scale==0]=1;enc=(enc-mean)/scale
            assert np.isfinite(enc).all() and np.isfinite(gx).all()
            dest.mkdir(parents=True,exist_ok=False)
            for name,v in [('counts',counts),('log_input',gx),('encoder_input',enc),('cell_positions',ids),('gene_positions',genes[sel]),('pre_hvg_cell_positions',before),('excluded_hvg_zero_cell_positions',excluded)]:np.save(dest/(name+'.npy'),v)
            np.savez(dest/'statistics.npz',size_factors=sf,library_sizes=lib,reference_total=ref,gene_mean=mean,gene_scale=scale)
            groups,labels=np.unique(y.astype(str),return_inverse=True);ev=out/'evaluation_only'/d;ev.mkdir(parents=True,exist_ok=False);np.save(ev/'reference_labels.npy',labels);save(ev/'groups.json',groups.tolist())
            save(dest/'ready.json',{'status':'READY','dataset':d,'known_k':len(groups),'raw_shape':shape,'shape':counts.shape,'raw_sha256':hashes[d]['sha256'],'freeze_sha256':sha(frozen),'shared_preparation_seconds':time.monotonic()-start,'excluded_hvg_cells':len(excluded),'preprocessing':'common_hvg_positive_library_r1','files':{p.name:sha(p) for p in dest.iterdir() if p.is_file()}})
            state['completed'].append(d);save(out/'preparation_state.json',state)
        state.update(status='COMPLETE_WITH_INPUT_FAILURES' if state['input_failures'] else 'COMPLETE',ended_utc=utc());save(out/'preparation_state.json',state)
    except BaseException as exc:
        state.update(status='FAILED',error=repr(exc),traceback=traceback.format_exc());save(out/'preparation_state.json',state);raise

if __name__=='__main__':main()
