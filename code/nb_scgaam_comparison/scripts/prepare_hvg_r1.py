from common import *
import argparse, time, h5py, numpy as np, scipy.sparse as sp
import scanpy as sc
p=argparse.ArgumentParser();p.add_argument('--dataset',choices=DATASETS,required=True);p.add_argument('--depth',type=float,default=1);p.add_argument('--seed',type=int,default=2211);a=p.parse_args()
out=taskpath(a.dataset,a.depth,a.seed)
if (out/'ready.json').exists():print('Already prepared',out);sys.exit(0)
start=time.monotonic();fname,n,g,k=DATASETS[a.dataset];rawpath=R/'data/raw'/fname
download=json.loads((R/'logs/download_state.json').read_text())['files'][fname];assert download['status']=='VERIFIED'
with h5py.File(rawpath,'r') as f:
    if 'obs' in f:
        q=f['exprs'];x=sp.csr_matrix((q['data'][...],q['indices'][...],q['indptr'][...]),shape=tuple(q['shape'][...])) if isinstance(q,h5py.Group) else sp.csr_matrix(q[...])
        y=f['obs/cell_type1'][...]
        technology=[str(z) for z in np.unique(f['obs/platform'][...])] if 'platform' in f['obs'] else []
    else:x=sp.csr_matrix(f['X'][...]);y=f['Y'][...];technology=[]
assert x.shape==(n,g),(x.shape,n,g)
assert len(y)==n and len(np.unique(y))==k
assert np.isfinite(x.data).all() and (x.data>=0).all() and np.equal(x.data,np.floor(x.data)).all(),'Not raw integer counts; no rounding permitted'
assert x.data.max()<2**24
if a.depth==.5:
    x.data=np.random.RandomState(a.seed+300000).binomial(x.data.astype(np.int64),.5).astype(np.float32);x.eliminate_zeros()
else:assert a.depth==1
x=x.astype(np.float32);ids=np.flatnonzero(np.asarray(x.sum(1)).ravel()>0);x=x[ids]
genes=np.flatnonzero(np.asarray(x.sum(0)).ravel()>0);x=x[:,genes]
ad=sc.AnnData(x.copy())
sc.pp.normalize_total(ad,exclude_highly_expressed=True);sc.pp.log1p(ad)
sc.pp.highly_variable_genes(ad,min_mean=.0125,max_mean=3,min_disp=.5,n_top_genes=2000,subset=False)
sel=np.flatnonzero(ad.var['highly_variable'].to_numpy());counts=x[:,sel].toarray();logx=ad.X[:,sel].toarray().astype(np.float32)
# r1: retain common HVG-positive cells for every method, using observed counts only.
# HVG selection and graph normalization stay fitted to pre-HVG positive observed cells.
pre_hvg_ids=ids.copy();keep=counts.sum(1)>0;excluded_hvg_ids=ids[~keep].copy()
assert keep.any(), 'No HVG-positive cells; requires review'
counts=counts[keep];logx=logx[keep];ids=ids[keep]
# Parent A/NB preprocessing on the common raw features; graph method retains author log input.
lib=counts.sum(1);assert (lib>0).all()
ref=np.float32(np.median(lib));sf=lib/ref
enc=np.log1p(counts/sf[:,None]);mean=enc.mean(0,dtype=np.float64).astype(np.float32);scale=enc.std(0,ddof=1,dtype=np.float64).astype(np.float32);scale[scale==0]=1
enc=(enc-mean)/scale
out.mkdir(parents=True,exist_ok=True)
np.save(out/'pre_hvg_cell_positions.npy',pre_hvg_ids)
np.save(out/'excluded_hvg_zero_cell_positions.npy',excluded_hvg_ids)
for name,v in [('counts',counts),('log_input',logx),('encoder_input',enc),('cell_positions',ids),('gene_positions',genes[sel])]:np.save(out/(name+'.npy'),v)
np.savez(out/'statistics.npz',size_factors=sf,library_sizes=lib,reference_total=ref,gene_mean=mean,gene_scale=scale)
ev=R/'data/evaluation_only'/a.dataset;ev.mkdir(parents=True,exist_ok=True)
if not (ev/'reference_labels.npy').exists():
    groups,encoded=np.unique(y.astype(str),return_inverse=True);np.save(ev/'reference_labels.npy',encoded);save(ev/'groups.json',groups.tolist())
meta={'preprocessing_protocol_id':'common_hvg_positive_library_r1','hvg_zero_filter':'counts_on_selected_observed_HVG.sum(1)>0; common cells for all methods','pre_hvg_cell_count':len(pre_hvg_ids),'excluded_hvg_zero_cell_count':len(excluded_hvg_ids),'normalization_population':'observed-positive cells before post-HVG filtering; no HVG refit','preparation_driver_sha256':sha(__file__),'status':'READY','dataset':a.dataset,'depth':a.depth,'observation_seed':None if a.depth==1 else a.seed+300000,'raw_sha256':download['sha256'],'raw_shape':[n,g],'shape':list(counts.shape),'known_k':k,'nonnegative_integer_verified':True,'technology_metadata':technology,'same_reported_shape':True,'exact_paper_version_proven':False,'scanpy_version':sc.__version__,'shared_preparation_seconds':time.monotonic()-start,'normalization_scgaam':'normalize_total exclude_highly_expressed=True; log1p; seurat HVG2000','normalization_A_NB':'common raw HVG counts; median library size-factor; log1p; ddof1 scale','files':{f.name:sha(f) for f in out.iterdir() if f.is_file()},'completed_utc':utc()}
save(out/'ready.json',meta);save(R/'reports/data_checks'/f'{a.dataset}_{out.name}.json',meta);print(json.dumps(meta,indent=2))
