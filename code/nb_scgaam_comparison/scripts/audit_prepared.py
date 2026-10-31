"""Independent count/normalization checks before exposing a task to training."""
from common import *
import argparse, h5py, numpy as np, scipy.sparse as sp
p=argparse.ArgumentParser();p.add_argument('--dataset',choices=DATASETS,required=True);p.add_argument('--depth',type=float,required=True);p.add_argument('--seed',type=int,required=True);a=p.parse_args()
task=taskpath(a.dataset,a.depth,a.seed);dest=R/'reports/data_checks'/f'{a.dataset}_{task.name}_independent.json'
meta=json.loads((task/'ready.json').read_text())
if dest.exists():
    old=json.loads(dest.read_text());assert old['status']=='PASSED' and old['ready_sha256']==sha(task/'ready.json');print('REUSE_PREPARATION_AUDIT');sys.exit(0)
for name,h in meta['files'].items():assert sha(task/name)==h,name
with h5py.File(R/'data/raw'/DATASETS[a.dataset][0],'r') as f:
    if 'obs' in f:
        q=f['exprs'];raw=sp.csr_matrix((q['data'][...],q['indices'][...],q['indptr'][...]),shape=tuple(q['shape'][...])) if isinstance(q,h5py.Group) else sp.csr_matrix(q[...]);truth=f['obs/cell_type1'][...]
    else:raw=sp.csr_matrix(f['X'][...]);truth=f['Y'][...]
if a.depth==.5:
    observed=np.random.RandomState(a.seed+300000).binomial(raw.data.astype(np.int64),.5)
    assert (observed<=raw.data).all();raw.data=observed.astype(np.float32);raw.eliminate_zeros()
counts=np.load(task/'counts.npy');ids=np.load(task/'cell_positions.npy');genes=np.load(task/'gene_positions.npy')
np.testing.assert_array_equal(counts,raw[ids][:,genes].toarray())
np.testing.assert_array_equal(ids,np.flatnonzero(np.asarray(raw.sum(1)).ravel()>0))
np.testing.assert_array_equal(np.unique(truth.astype(str),return_inverse=True)[1],np.load(R/'data/evaluation_only'/a.dataset/'reference_labels.npy'))
assert counts.shape[1]>=2000 and counts.shape[1]<=2020,'Unexpected HVG ties/count, requires explicit review'
assert np.isfinite(counts).all() and (counts>=0).all() and np.equal(counts,np.floor(counts)).all()
with np.load(task/'statistics.npz') as f:stats={k:f[k] for k in f.files}
lib=counts.sum(axis=1);sf=lib/np.float32(np.median(lib));enc=np.log1p(counts/sf[:,None])
mean=np.mean(enc,axis=0,dtype=np.float64).astype(np.float32);scale=np.std(enc,axis=0,ddof=1,dtype=np.float64).astype(np.float32);scale[scale==0]=1
np.testing.assert_array_equal(stats['size_factors'],sf);np.testing.assert_array_equal(stats['gene_mean'],mean);np.testing.assert_array_equal(stats['gene_scale'],scale)
np.testing.assert_array_equal(np.load(task/'encoder_input.npy'),(enc-mean)/scale)
# Reconstruct Scanpy's excluded-high-count library normalization from the observation.
raw=raw.astype(np.float32)[ids];tot=np.asarray(raw.sum(1)).ravel();coo=raw.tocoo()
# Scanpy 1.12's CSR implementation uses float64 max_fraction * row sum,
# and the median over all excluded-gene library sums (including zero sums).
high=coo.data.astype(np.float64)>.05*tot[coo.row].astype(np.float64)
large=np.bincount(coo.col[high],minlength=raw.shape[1])>0
normlib=np.asarray(raw[:,~large].sum(1)).ravel();target=np.median(normlib)
denom=normlib/target;denom[denom==0]=1
expected=np.log1p(raw[:,genes].toarray()/denom[:,None]);logged=np.load(task/'log_input.npy')
# Scanpy sparse in-place arithmetic versus dense arithmetic: explicit float32 precision bound.
maxerr=float(np.max(np.abs(expected-logged)));np.testing.assert_allclose(logged,expected,rtol=2e-6,atol=2e-6)
save(dest,dict(status='PASSED',utc=utc(),ready_sha256=sha(task/'ready.json'),raw_selected_counts_exact=True,observation_recomputed=True,reference_labels_exact=True,encoder_transform_exact=True,log_transform_max_abs_error=maxerr,log_transform_rtol=2e-6,log_transform_atol=2e-6,shape=list(counts.shape),labels_used_for_features=False))
print('PREPARATION_AUDIT_PASSED',a.dataset,task.name)
