"""Full author's NB scSAMAC on all development datasets, y=None."""
from common import *
import argparse,time
import numpy as np
import torch
from threadpoolctl import threadpool_limits
sys.path.insert(0,str(ROOT/'experiments/scsamac_author_worm_v1/author_source'))
from scSAMAC import scSAMAC
import scSAMAC as module

def main(args):
    sys.addaudithook(guard);setup(args.seed);start=time.monotonic();task=prepared(args.dataset)
    out=EXP/'training'/args.dataset/'scSAMAC'/f'seed{args.seed}'/('smoke' if args.smoke else 'main');out.mkdir(parents=True,exist_ok=False)
    cfg=json.loads((ROOT/'experiments/scsamac_author_worm_v1/config.json').read_text())
    meta=dict(status='RUNNING',dataset=args.dataset,seed=args.seed,labels_read=False,started_utc=utc(),pid=os.getpid(),source_sha256=sha(__file__),author_source_sha256={p.name:sha(p) for p in (ROOT/'experiments/scsamac_author_worm_v1/author_source').glob('*.py')})
    save(out/'run.json',meta)
    x=np.load(task/'encoder_input.npy').astype('float64');raw=np.load(task/'counts.npy').astype('float64');ids=np.load(task/'cell_positions.npy')
    with np.load(task/'statistics.npz') as f:sf=f['size_factors'].astype('float64')
    if args.smoke:x=x[:384];raw=raw[:384];sf=sf[:384];ids=ids[:384]
    class Observed(scSAMAC):
        def encodeBatch(self,X,batch_size=256):
            z=super().encodeBatch(X,batch_size)
            if not torch.isfinite(z).all():raise FloatingPointError('nonfinite author embedding')
            self.last_z=z.detach().cpu().numpy().copy();self.last_state={k:v.detach().cpu().clone() for k,v in self.state_dict().items()}
            return z
    original=module.KMeans
    class StableKMeans(original):
        def fit_predict(self,*args,**kwargs):
            with threadpool_limits(limits=1):return super().fit_predict(*args,**kwargs)
    module.KMeans=StableKMeans
    import builtins
    def progress(*values,**kwargs):
        builtins.print(*values,**kwargs,flush=True)
        save(out/'progress.json',{'seconds':time.monotonic()-start,'message':' '.join(map(str,values))})
    module.print=progress
    m=Observed(input_dim=x.shape[1],z_dim=32,encodeLayer=[256,64],decodeLayer=[64,256],sigma=2.5,alpha=1.,gamma=1.,device='cuda')
    m.pretrain_autoencoder(x,raw,sf,batch_size=256,lr=.0001,epochs=2 if args.smoke else 300,ae_weights=str(out/'pretrained.pt'))
    pred=m.fit(x,raw,sf,n_clusters=CFG['development'][args.dataset],y=None,lr=1.,batch_size=256,num_epochs=2 if args.smoke else 2000,update_interval=1,tol=.001,save_dir=str(out))[0]
    z=m.last_z.copy();state=m.last_state
    torch.save(state,out/'prediction_checkpoint.pt');torch.save(m.state_dict(),out/'final.pt')
    m.load_state_dict(state)
    with torch.no_grad():again=m.encodeBatch(torch.tensor(x,device='cuda'));q=m.soft_assign(again)
    np.testing.assert_array_equal(again.cpu().numpy(),z);np.testing.assert_array_equal(q.argmax(1).cpu().numpy(),pred)
    np.save(out/'cell_positions.npy',ids);np.savez_compressed(out/'outputs.npz',embedding=z,prediction=pred)
    meta.update(status='COMPLETE',seconds=time.monotonic()-start,author_returned_prediction=True,checkpoint_reload_exact=True)
    save(out/'run.json',meta)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',choices=CFG['development'],required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--smoke',action='store_true');main(p.parse_args())
