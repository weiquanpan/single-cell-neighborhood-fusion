"""One authorized seed101 diagnostic rerun, fixed inputs/parameters/300 epochs."""
from pathlib import Path
import os,sys,json,time,re,hashlib,traceback,platform
HERE=Path(__file__).resolve().parent;B=HERE.parents[1];E=B.parent/'cross_dataset_v1'
sys.dont_write_bytecode=True
sys.path.insert(0,str(E))
import common as c
import numpy as np,torch
from threadpoolctl import threadpool_limits
import diagnostic_observer as diag

def sha(p):return c.sha(p)
def save(p,v):c.save(p,v)

lock=HERE/'diagnostic.lock'
fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
os.write(fd,str(os.getpid()).encode());os.close(fd)
start=time.perf_counter();out=HERE/'checkpoints';out.mkdir(exist_ok=False)
P=json.loads((B/'protocol.json').read_text())
source=HERE/'source_instrumented'
manifest=json.loads((HERE/'source_manifest.json').read_text())
task=E/'heldout_v1/prepared/Quake_10x_Spleen/full'
old=B/'runs/Quake_10x_Spleen/scSAMAC/seed101/attempt_001/run.json'
old_meta=json.loads(old.read_text())
protected={str(p):sha(p) for p in [old,B/'protocol.json',B/'run_one.py',E/'common.py']+list((B/'runs/Quake_10x_Spleen/scSAMAC/seed101/attempt_001').glob('*'))+list(task.glob('*.npy'))+[task/'statistics.npz']+list((c.ROOT/'experiments/scsamac_author_worm_v1/author_source').glob('*.py'))}
save(HERE/'original_files_before.json',protected)
cfg={'dataset':'Quake_10x_Spleen','seed':101,'epochs':300,'batch_size':256,'lr':.0001,'amsgrad':True,'sigma':2.5,'alpha':1.,'gamma':1.,'z_dim':32,'encodeLayer':[256,64],'decodeLayer':[64,256],'contrast_temperature':.07,'contrast_weight':.01,'joint_max_epochs':2000,'joint_lr':1.,'joint_tol':.001,'purpose':'Diagnostic only; not an official result replacement','observations':'Original arithmetic with named intermediate tensors and read-only hooks; extra sample indices only for traceability','original_attempt':str(old.parent),'no_algorithm_or_parameter_changes':True,'pid':os.getpid(),'started_utc':c.utc()}
save(HERE/'config.json',cfg)
result={'status':'RUNNING','pid':os.getpid(),'seed':101,'started_utc':c.utc()}
save(HERE/'state.json',result)
try:
    assert sha(B/'protocol.json')==old_meta['protocol_sha256']
    assert sha(B/'run_one.py')==old_meta['runner_sha256']
    for name,h in old_meta['input_sha256'].items():assert sha(task/name)==h,name
    for name,m in manifest.items():
        assert sha(source/name)==m['instrumented_sha256'],name
        assert sha(HERE/'source_original'/name)==P['source']['scSAMAC']['files'][name],name
    sys.addaudithook(c.guard)
    c.setup(101)
    sys.path.insert(0,str(source))
    import scSAMAC as mod
    torch.set_default_dtype(torch.float64)
    original_print=print
    def progress(*xs,**kw):kw['flush']=True;original_print(*xs,**kw)
    mod.print=progress
    original=mod.KMeans
    class StableKMeans(original):
        def fit_predict(self,*args,**kw):
            with threadpool_limits(limits=1):return super().fit_predict(*args,**kw)
    mod.KMeans=StableKMeans
    class Observed(mod.scSAMAC):
        def encodeBatch(self,X,batch_size=256):
            z=super().encodeBatch(X,batch_size);assert torch.isfinite(z).all();self.last_z=z.detach().cpu().numpy().copy();self.last_state={k:v.detach().cpu().clone() for k,v in self.state_dict().items()};return z
    x=np.load(task/'encoder_input.npy').astype('float64');raw=np.load(task/'counts.npy').astype('float64')
    with np.load(task/'statistics.npz') as f:sf=f['size_factors'].astype('float64')
    model=Observed(input_dim=x.shape[1],z_dim=32,encodeLayer=[256,64],decodeLayer=[64,256],sigma=2.5,alpha=1.,gamma=1.,device='cuda')
    hist_text=(B/'logs/scSAMAC_Quake_10x_Spleen_101.log').read_text(encoding='utf8')
    historical={int(e):v for e,v in re.findall(r'Pretrain epoch\s+(\d+), NB loss:\s*([^\s]+)',hist_text)}
    diag.initialize(historical)
    torch.save(diag.snapshot(model,None),out/'initial.pt')
    # Forward hooks observe original module outputs and never return replacements.
    hooks=[m.register_forward_hook(diag.module_hook(n)) for n,m in model.named_modules() if isinstance(m,torch.nn.Linear)]
    save(HERE/'environment.json',{'python':sys.version,'torch':torch.__version__,'numpy':np.__version__,'gpu':torch.cuda.get_device_name(),'platform':platform.platform(),'default_dtype':str(torch.get_default_dtype()),'deterministic':torch.are_deterministic_algorithms_enabled(),'threads':torch.get_num_threads(),'interop_threads':torch.get_num_interop_threads(),'tf32':torch.backends.cuda.matmul.allow_tf32})
    model.pretrain_autoencoder(x,raw,sf,batch_size=256,lr=.0001,epochs=300,ae_weights=str(out/'pretrained_diagnostic.pt'))
    for hook in hooks:hook.remove()
    try:
        # Same original post-pretraining entry; normally fails before any joint update.
        model.fit(x,raw,sf,n_clusters=5,y=None,lr=1.,batch_size=256,num_epochs=2000,update_interval=1,tol=.001,save_dir=str(out))
        result['joint_entry']='completed'
    except AssertionError:
        result['joint_entry']='nonfinite_embedding_assertion_same_as_original'
        (HERE/'joint_assertion.txt').write_text(traceback.format_exc(),encoding='utf8')
    diag.finish()
    assert all(sha(p)==h for p,h in protected.items()),'An original file changed'
    result.update(status='DIAGNOSTIC_COMPLETE_PENDING_REVIEW',finished_utc=c.utc(),elapsed_seconds=time.perf_counter()-start,pretrain_steps=diag.STEP,first_nonfinite=diag.FIRST,all_original_files_unchanged=True,matched_epochs=sum(v['exact_printed_match'] for v in diag.EPOCH_MATCH),compared_epochs=len(diag.EPOCH_MATCH))
    save(HERE/'result.json',result);save(HERE/'state.json',result)
    print(json.dumps(result,ensure_ascii=False),flush=True)
except Exception:
    result.update(status='DIAGNOSTIC_ERROR',error=traceback.format_exc(),elapsed_seconds=time.perf_counter()-start)
    save(HERE/'result.json',result);save(HERE/'state.json',result)
    raise
finally:
    lock.unlink()
