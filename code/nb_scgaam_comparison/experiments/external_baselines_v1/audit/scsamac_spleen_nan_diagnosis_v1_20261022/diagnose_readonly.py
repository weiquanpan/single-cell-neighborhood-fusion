"""Read-only NaN diagnosis: arrays/checkpoints/source audit and scalar loss probes.
No trainer is imported, no model is fitted, and optimizer.step is blocked.
All generated files are confined to this NEW audit directory.
"""
from pathlib import Path
import os,sys,json,hashlib,re,subprocess,importlib.util,warnings,time
HERE=Path(__file__).resolve().parent
B=HERE.parents[1];BASE=B.parents[1];E=BASE/'experiments/cross_dataset_v1'
AUTHOR=BASE/'experiments/scsamac_author_worm_v1/author_source'
UPSTREAM=BASE/'upstream/scSAMAC-cluster'
os.environ.update(CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',MPLBACKEND='Agg',MPLCONFIGDIR=str(HERE/'matplotlib_cache'),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
sys.dont_write_bytecode=True
sys.path.extend(['C:/Users/Omega/AppData/Roaming/Python/Python313/site-packages','D:/Anaconda/Lib/site-packages'])
import numpy as np
import scipy.sparse as sp
import h5py,torch
torch.set_num_threads(4)
def forbidden_step(*args,**kw):raise RuntimeError('Optimizer updates are forbidden in this read-only diagnostic')
for cls in [torch.optim.Adam,torch.optim.Adamax,torch.optim.Adadelta,torch.optim.SGD]:cls.step=forbidden_step
def load(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for x in iter(lambda:f.read(8*1024**2),b''):h.update(x)
    return h.hexdigest()
def save(name,v):
    (HERE/name).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
def module(name,p):
    spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def stats(x):
    x=np.asarray(x);good=x[np.isfinite(x)]
    return dict(shape=list(x.shape),dtype=str(x.dtype),nan=int(np.isnan(x).sum()),posinf=int(np.isposinf(x).sum()),neginf=int(np.isneginf(x).sum()),minimum=float(good.min()) if good.size else None,maximum=float(good.max()) if good.size else None,quantiles={str(q):float(np.quantile(good,q)) for q in [0,.01,.5,.99,1]} if good.size else {},nonzero=int(np.count_nonzero(x)))

start=time.monotonic();P=load(B/'protocol.json');protected={}
def protect(p):
    p=Path(p);protected[str(p)]=sha(p);return protected[str(p)]
for p in [B/'run_one.py',B/'protocol.json',E/'followup/prepare_heldouts_v2.py',E/'common.py']+list(AUTHOR.glob('*.py')):protect(p)
source=[]
for name,expected in P['source']['scSAMAC']['files'].items():
    p=AUTHOR/name;h=protect(p);up=sha(UPSTREAM/name)
    assert h==expected==up,name
    source.append(dict(name=name,sha256=h,matches_frozen_protocol=True,matches_original_checkout=True))
head=subprocess.check_output(['git','-C',str(UPSTREAM),'rev-parse','HEAD'],text=True).strip()
assert head==P['source']['scSAMAC']['commit']
status=subprocess.check_output(['git','-C',str(UPSTREAM),'status','--porcelain'],text=True).strip();assert not status
save('source_audit.json',dict(commit=head,checkout_clean=True,files=source,runner_sha256=protect(B/'run_one.py'),protocol_sha256=protect(B/'protocol.json')))

runs=[];checkpoint_details=[]
for seed in [101,202,303,404,505]:
    folder=B/'runs/Quake_10x_Spleen/scSAMAC'/f'seed{seed}'/'attempt_001'
    log=B/'logs'/f'scSAMAC_Quake_10x_Spleen_{seed}.log'
    meta=load(folder/'run.json');protect(folder/'run.json');protect(log)
    lines=log.read_text(encoding='utf8');pairs=re.findall(r'Pretrain epoch\s+(\d+), NB loss:\s*([^\s]+)',lines)
    finite=[(int(e),float(v)) for e,v in pairs if np.isfinite(float(v))]
    bad=[int(e) for e,v in pairs if not np.isfinite(float(v))]
    assert meta['status']=='FAILED' and bad and len(pairs)==300
    assert meta['runner_sha256']==protected[str(B/'run_one.py')]
    assert meta['protocol_sha256']==protected[str(B/'protocol.json')]
    cp=folder/'pretrained.pt';protect(cp);checkpoint=torch.load(cp,map_location='cpu',weights_only=True)
    tensors=[]
    for name,t in checkpoint['ae_state_dict'].items():
        tensors.append(dict(name=name,shape=list(t.shape),dtype=str(t.dtype),numel=t.numel(),nan=int(torch.isnan(t).sum()),inf=int(torch.isinf(t).sum())))
    opt=checkpoint['optimizer_state_dict'];steps=sorted({float(s['step']) for s in opt['state'].values()})
    detail=dict(seed=seed,files=[p.name for p in folder.iterdir() if p.is_file()],state_tensors=tensors,optimizer_param_group={k:v for k,v in opt['param_groups'][0].items() if k!='params'},optimizer_steps=steps,optimizer_nonfinite={k:sum(int((~torch.isfinite(s[k])).sum()) for s in opt['state'].values() if k in s) for k in ['exp_avg','exp_avg_sq','max_exp_avg_sq']})
    checkpoint_details.append(detail)
    runs.append(dict(seed=seed,first_nonfinite_epoch=min(bad),last_finite_epoch=finite[-1][0],last_finite_logged_total_loss=finite[-1][1],pretrain_epochs=300,optimizer_steps=meta['optimizer_steps'],seconds=meta['seconds'],checkpoint_sha256=protected[str(cp)],all_model_values_nan=all(t['nan']==t['numel'] for t in tensors),has_before_failure_checkpoint=False,has_batch_loss_terms=False,source_and_input_identical_to_seed101=meta['input_sha256']==load(B/'runs/Quake_10x_Spleen/scSAMAC/seed101/attempt_001/run.json')['input_sha256']))
    del checkpoint
save('run_timeline.json',runs);save('checkpoint_audit.json',checkpoint_details)
print('RUN_AND_SOURCE_AUDIT_COMPLETE',flush=True)

# All actual saved inputs are inspected, not their metadata alone.
summaries=[]
for d in P['primary']:
    task=E/'heldout_v1/prepared'/d/'full'
    counts=np.load(task/'counts.npy');x=np.load(task/'encoder_input.npy');gx=np.load(task/'log_input.npy');ids=np.load(task/'cell_positions.npy')
    with np.load(task/'statistics.npz') as f:sf=f['size_factors'];ref=float(f['reference_total'])
    ready=load(task/'ready.json');hash_checks={n:protect(task/n)==h for n,h in ready['files'].items()}
    assert all(hash_checks.values())
    norms=np.linalg.norm(x.astype('float64'),axis=1)
    r=dict(dataset=d,counts=stats(counts),encoder=stats(x),log_graph=stats(gx),size_factors=stats(sf),counts_noninteger=int(np.count_nonzero(counts!=np.floor(counts))),counts_negative=int((counts<0).sum()),zero_libraries=int((counts.sum(1)==0).sum()),zero_variance_genes=int((counts.var(0)==0).sum()),unique_cell_positions=len(np.unique(ids)),cell_position_count=len(ids),reference_total=ref,encoder_row_norm=stats(norms),cell_gene_shape_consistent=counts.shape==x.shape==gx.shape,hash_checks=hash_checks)
    summaries.append(r)
save('input_summaries.json',summaries)

d='Quake_10x_Spleen';task=E/'heldout_v1/prepared'/d/'full'
rawfile=E/'heldout_sealed'/f'{d}.h5';raw_hash=protect(rawfile);ready=load(task/'ready.json');assert raw_hash==ready['raw_sha256']
counts=np.load(task/'counts.npy');saved_x=np.load(task/'encoder_input.npy');saved_gx=np.load(task/'log_input.npy');ids=np.load(task/'cell_positions.npy');genes=np.load(task/'gene_positions.npy')
with h5py.File(rawfile,'r') as f:
    q=f['exprs'] if 'exprs' in f else f['X']
    if isinstance(q,h5py.Group):raw=sp.csr_matrix((q['data'][...],q['indices'][...],q['indptr'][...]),shape=tuple(q['shape'][...]))
    else:raw=sp.csr_matrix(q[...])
assert np.isfinite(raw.data).all() and (raw.data>=0).all() and np.equal(raw.data,np.floor(raw.data)).all()
subset=raw[ids][:,genes].toarray();assert np.array_equal(subset,counts)
with np.load(task/'statistics.npz') as f:stored={k:f[k] for k in f.files}
lib=counts.sum(1);sf=lib/np.float32(np.median(lib));enc=np.log1p(counts/sf[:,None]);mean=enc.mean(0,dtype=np.float64).astype(np.float32);scale=enc.std(0,ddof=1,dtype=np.float64).astype(np.float32);scale[scale==0]=1;enc=(enc-mean)/scale
checks=dict(raw_sha256_matches=True,raw_shape=list(raw.shape),raw_nonnegative_finite_integer=True,raw_subset_exact=True,unique_genes=len(np.unique(genes)),size_factor_exact=bool(np.array_equal(sf,stored['size_factors'])),library_size_exact=bool(np.array_equal(lib,stored['library_sizes'])),gene_mean_exact=bool(np.array_equal(mean,stored['gene_mean'])),gene_scale_exact=bool(np.array_equal(scale,stored['gene_scale'])),encoder_formula_exact=bool(np.array_equal(enc,saved_x)),last_minibatch_size=len(ids)%256,raw_cells_retained=len(ids))
assert all(checks[k] for k in ['size_factor_exact','library_size_exact','gene_mean_exact','gene_scale_exact','encoder_formula_exact'])

# Recompute PREPROCESSING ONLY, in memory. Never write prepared inputs.
import scanpy as sc
raw32=raw.astype(np.float32);rowids=np.flatnonzero(np.asarray(raw32.sum(1)).ravel()>0);raw32=raw32[rowids];gids=np.flatnonzero(np.asarray(raw32.sum(0)).ravel()>0);raw32=raw32[:,gids]
ad=sc.AnnData(raw32.copy());sc.pp.normalize_total(ad,exclude_highly_expressed=True);sc.pp.log1p(ad)
sc.pp.highly_variable_genes(ad,min_mean=.0125,max_mean=3,min_disp=.5,n_top_genes=2000,subset=False)
sel=np.flatnonzero(ad.var['highly_variable'].to_numpy());selected=raw32[:,sel].toarray();keep=selected.sum(1)>0
checks.update(hvg_gene_positions_exact=bool(np.array_equal(gids[sel],genes)),hvg_cell_positions_exact=bool(np.array_equal(rowids[keep],ids)),graph_log_input_exact=bool(np.array_equal(ad.X[:,sel].toarray().astype('float32')[keep],saved_gx)))
assert all(checks[k] for k in ['hvg_gene_positions_exact','hvg_cell_positions_exact','graph_log_input_exact'])
save('preprocessing_reconstruction.json',checks)
del ad,raw32,raw,subset,enc,selected

# Run the exact author's normalize() on the SAME selected count matrix, not all genes.
pre=module('scsamac_audit_preprocess',AUTHOR/'preprocess.py')
author_ad=sc.AnnData(counts.astype('float64'))
with warnings.catch_warnings(record=True) as notices:
    warnings.simplefilter('always')
    author_ad=pre.normalize(author_ad,size_factors=True,normalize_input=True,logtrans_input=True)
author_x=np.asarray(author_ad.X);delta=np.abs(author_x-saved_x)
comparison=dict(scope='Exact author normalize() on same 9552 cells / 2000 selected genes; no fitting or training',same_shape=author_x.shape==saved_x.shape,raw_counts_equal=bool(np.array_equal(author_ad.raw.X,counts)),size_factor_max_abs_difference=float(np.max(np.abs(np.asarray(author_ad.obs['size_factors'])-sf))),encoder_max_abs_difference=float(delta.max()),encoder_rms_difference=float(np.sqrt(np.mean(delta**2))),encoder_relative_l2_difference=float(np.linalg.norm(delta)/np.linalg.norm(author_x)),stored_dtype=str(saved_x.dtype),author_dtype=str(author_x.dtype),warnings=sorted(set(str(n.message) for n in notices)),remaining_protocol_differences=['Uniform 2000 HVGs selected by shared protocol; author CLI default select_genes=0 uses all genes supplied','Stored standardized inputs are float32 then cast to float64 by adapter; author CLI normalizes float64 directly','Current PyTorch/Scanpy environment, deterministic flags and fixed evaluation seeds differ from a literal author default launch'])
save('author_preprocessing_comparison.json',comparison)
del author_ad,author_x,delta
print('INPUT_AND_PREPROCESSING_AUDIT_COMPLETE',flush=True)

# Pure arithmetic diagnostics. They prove reachable numerical hazards, NOT the
# historical first failing tensor (which was not saved).
lossmod=module('scsamac_audit_loss',AUTHOR/'st_loss.py')
layers=module('scsamac_audit_layers',AUTHOR/'layers.py')
def contrast_probe(amplitude):
    features=(torch.eye(4,dtype=torch.float64)*amplitude).reshape(2,2,4)
    original=lossmod.SupConLoss(temperature=.07)(features)
    z=torch.cat(torch.unbind(features,dim=1),dim=0);scores=z@z.T/.07
    logits=scores-scores.max(1,keepdim=True).values
    eye=torch.eye(4,dtype=torch.bool);denom=(logits.exp()*(~eye)).sum(1)
    positive=torch.eye(2,dtype=torch.float64).repeat(2,2)*(~eye)
    # Mathematically identical log denominator evaluated without exp-underflow.
    logden=torch.logsumexp(scores.masked_fill(eye,-torch.inf),dim=1,keepdim=True)
    exact=-(positive*(scores-logden)).sum(1).mean()
    return dict(amplitude=amplitude,feature_norms=torch.linalg.vector_norm(z,dim=1).tolist(),original_loss=str(float(original)),original_finite=bool(torch.isfinite(original)),zero_denominators=int((denom==0).sum()),max_logit_gap=float((-logits).max()),same_formula_logsumexp=float(exact),positive_counts=positive.sum(1).tolist(),parameters_changed=False,optimizer_steps=0)
probes=[contrast_probe(1),contrast_probe(8)]
assert probes[0]['original_finite'] and not probes[1]['original_finite'] and probes[1]['zero_denominators']==4
meanprobes=[]
for value in [700.,710.]:
    x=torch.tensor([value],dtype=torch.float64,requires_grad=True);out=layers.MeanAct()(x);grad=torch.autograd.grad(out.sum(),x)[0]
    meanprobes.append(dict(input=value,output=float(out),gradient=str(float(grad)),output_finite=bool(torch.isfinite(out)),gradient_finite=bool(torch.isfinite(grad)),optimizer_steps=0))
nbprobes=[]
for count in [0.,1.,float(counts.max())]:
    for mu in [1e-5,1e6]:
        for disp in [1e-4,1e4]:
            for sizefactor in [float(sf.min()),float(sf.max())]:
                x=torch.tensor([[count]],dtype=torch.float64);m=torch.tensor([[mu]],dtype=torch.float64,requires_grad=True);r=torch.tensor([[disp]],dtype=torch.float64,requires_grad=True)
                v=layers.NBLoss()(x,m,r,torch.tensor([sizefactor],dtype=torch.float64))
                g=torch.autograd.grad(v,(m,r))
                nbprobes.append(dict(count=count,mean=mu,dispersion=disp,size_factor=sizefactor,finite_loss=bool(torch.isfinite(v)),finite_gradients=all(bool(torch.isfinite(a).all()) for a in g)))
save('arithmetic_probes.json',dict(scope='Deterministic finite-input CPU operator probes; not a replay of the failed training; no model/optimizer construction',contrastive=probes,mean_activation=meanprobes,nb_boundary_checks=nbprobes,all_nb_boundary_checks_finite=all(x['finite_loss'] and x['finite_gradients'] for x in nbprobes)))

after={p:sha(p) for p in protected};assert after==protected
save('protected_files_sha256.json',dict(before=protected,after_identical=True))
save('diagnostic_result.json',dict(status='ANALYZED_NOT_CAUSALLY_REPLAYED',optimizer_updates=0,training_started=False,model_parameters_modified=False,existing_files_modified=False,protected_files_verified=len(protected),seed_failures=runs,source_byte_identical=True,preprocessing_reconstruction=checks,author_normalization=comparison,numerical_hazards_reproduced=['SupConLoss exp underflow after diagonal maximum then masking -> zero denominator -> log(0) -> NaN','MeanAct exp overflows before clamp: finite forward output can have NaN gradient'],historical_trigger_proven=False,reason='Only epoch-aggregate total loss and terminal contaminated model/optimizer states exist; no last-finite checkpoint/RNG/batch/component loss',elapsed_seconds=time.monotonic()-start))
print(json.dumps(dict(status='ANALYZED_NOT_CAUSALLY_REPLAYED',seconds=time.monotonic()-start,output=str(HERE)),ensure_ascii=False),flush=True)
