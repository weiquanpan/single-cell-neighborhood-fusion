from pathlib import Path
import os, sys, json, hashlib, datetime
EXP = Path(__file__).resolve().parent
ROOT = EXP.parents[1]
PYTHON = ROOT.parent/'scdeepcluster_author_work/pytorch_gpu/env313/Scripts/python.exe'
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.update(OMP_NUM_THREADS='8', MKL_NUM_THREADS='8', MPLBACKEND='Agg')
sys.path.extend([str(ROOT/'vendor'), 'C:/Users/Omega/AppData/Roaming/Python/Python313/site-packages', 'D:/Anaconda/Lib/site-packages'])
CFG = json.loads((EXP/'protocol.json').read_text())
def save(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf8');os.replace(tmp,path)
def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(8*1024**2),b''):h.update(chunk)
    return h.hexdigest()
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def guard(event,args):
    if event=='open' and args and isinstance(args[0],(str,bytes,os.PathLike)):
        p=os.fsdecode(args[0]).replace('\\','/').lower()
        if '/evaluation_only/' in p or '/data/raw/' in p or 'reference_labels' in p or p.endswith('/metrics.json'):
            raise PermissionError('Label isolation: '+p)
def original(dataset,method,seed):
    base=ROOT/'runs'/dataset/'full'/method/f'main_seed{seed}'
    good=[]
    for p in base.glob('attempt_*/result.json'):
        if json.loads(p.read_text()).get('status')=='COMPLETED':good.append(p.parent)
    if method=='PCA_KMeans' and len(good)>1:
        good=[p for p in good if json.loads((p/'result.json').read_text()).get('pca_projection_protocol')=='fit_then_transform_contiguous_r1']
    if len(good)!=1:raise RuntimeError(f'Expected one completed source: {base}: {good}')
    return good[0]
def prepared(dataset):
    if dataset not in CFG['development']:raise PermissionError('Heldout access requires frozen separate protocol')
    return ROOT/'data/prepared'/dataset/'full'
def setup(seed):
    import random,numpy as np,torch
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    torch.set_num_threads(8);torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
