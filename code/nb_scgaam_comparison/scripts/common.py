from pathlib import Path
import datetime, hashlib, json, os, sys
R=Path(__file__).resolve().parents[1]
OLD=R.parent/'scdeepcluster_author_work/pytorch_gpu'
PYTHON=OLD/'env313/Scripts/python.exe'
os.environ.setdefault('MPLBACKEND','Agg')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
for envkey,folder in [('NUMBA_CACHE_DIR','numba'),('MPLCONFIGDIR','matplotlib'),('TMP','tmp'),('TEMP','tmp')]:
    cachepath=R/'cache'/folder;cachepath.mkdir(parents=True,exist_ok=True);os.environ[envkey]=str(cachepath)
os.environ['OMP_NUM_THREADS']='8';os.environ['MKL_NUM_THREADS']='8'
# Keep env313 NumPy/SciPy/torch ahead of host packages; no old environment edits.
sys.path.insert(0,str(R/'vendor'))
sys.path.append('C:/Users/Omega/AppData/Roaming/Python/Python313/site-packages')
sys.path.append('D:/Anaconda/Lib/site-packages')
DATASETS={'Limb_Muscle':('Quake_10x_Limb_Muscle.h5',3909,23341,6),
          'Worm_Neuron':('worm_neuron_cell.h5',4186,13488,10),
          'Tosches':('Tosches.h5',18664,23500,15),'Bach':('Bach.h5',23184,19965,8)}
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def save(path,item):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp');temp.write_text(json.dumps(item,indent=2,ensure_ascii=False),encoding='utf8');os.replace(temp,path)
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def taskpath(dataset,depth,seed):return R/'data/prepared'/dataset/('full' if depth==1 else f'half_seed{seed}')
