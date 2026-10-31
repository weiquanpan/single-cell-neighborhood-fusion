"""Path-only adaptation of frozen trainers. No training-label access."""
from pathlib import Path
import sys,json,importlib.util,runpy,argparse,shutil
E=Path(__file__).resolve().parents[1];sys.path.insert(0,str(E))
import common as c
H=E/'heldout_muraro_corrected_v1';R=c.ROOT

def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--method',choices=['A','NB','scGAAM','NB-GCL-fusion'],required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args()
    manifest=json.loads((E/'followup/final_method_v1/manifest.json').read_text())
    assert a.dataset in manifest['heldout_datasets'] and a.seed in manifest['seeds']
    for n,h in manifest['source_sha256'].items():assert c.sha(R/n)==h,n
    ready={d:json.loads((H/'prepared'/d/'full/ready.json').read_text()) for d in manifest['heldout_datasets'] if (H/'prepared'/d/'full/ready.json').exists()}
    assert a.dataset in ready
    role='smoke' if a.smoke else 'main';out=H/'runs'/a.dataset/'full'/a.method/f'{role}_seed{a.seed}'/'attempt_001'
    if a.method!='NB-GCL-fusion':
        spec=importlib.util.spec_from_file_location('heldout_legacy_common',R/'scripts/common.py');legacy=importlib.util.module_from_spec(spec);spec.loader.exec_module(legacy)
        legacy.DATASETS={d:(d+'.h5',*m['raw_shape'],m['known_k']) for d,m in ready.items()}
        legacy.taskpath=lambda dataset,depth,seed:H/'prepared'/dataset/'full'
        sys.modules['common']=legacy
        sys.argv=[str(R/'scripts/train_one.py'),'--dataset',a.dataset,'--method',a.method,'--seed',str(a.seed),'--role',role,'--out',str(out)]
        runpy.run_path(str(R/'scripts/train_one.py'),run_name='__main__')
    else:
        # Frozen source executes unchanged; only dataset/output/cache paths differ.
        sys.addaudithook(c.guard)
        import numpy as np,torch
        from types import SimpleNamespace
        work=H/('smoke_workspace' if a.smoke else 'candidate_workspace');work.mkdir(exist_ok=True)
        for name in ['protocol.json','gcl_train.py']:
            if not (work/name).exists():shutil.copy2(E/name,work/name)
        cache=work/'cache'/a.dataset/f'nb_h_seed{a.seed}.npy';cache.parent.mkdir(parents=True,exist_ok=True)
        if not cache.exists():
            c.setup(a.seed)
            sys.path.insert(0,str(R/'source/scdeepcluster'));from methods import Model
            x=np.load(H/'prepared'/a.dataset/'full/encoder_input.npy')
            nb=H/'runs'/a.dataset/'full/NB'/f'{role}_seed{a.seed}'/'attempt_001'
            assert json.loads((nb/'run.json').read_text())['status']=='COMPLETED'
            m=Model(x.shape[1],'NB',a.seed);m.load_state_dict(torch.load(nb/'pretrained.pt',map_location='cuda',weights_only=True)['ae_state_dict'],strict=True);m.eval()
            with torch.no_grad():z=m.encodeBatch(torch.as_tensor(x,dtype=torch.float64)).cpu().numpy()
            assert np.isfinite(z).all();np.save(cache,z);del m;torch.cuda.empty_cache()
            # train_gcl.main calls setup again; setting interop threads twice is invalid.
            original_setup=c.setup
            def reseed(seed):
                import random
                random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
            c.setup=reseed
        torch.set_default_dtype(torch.float32)
        import train_gcl as trainer
        trainer.EXP=work;trainer.CFG={**c.CFG,'development':{d:m['known_k'] for d,m in ready.items()}}
        trainer.prepared=lambda dataset:H/'prepared'/dataset/'full'
        trainer.main(SimpleNamespace(dataset=a.dataset,seed=a.seed,variant='NB-GCL-fusion',smoke=a.smoke))

if __name__=='__main__':main()
