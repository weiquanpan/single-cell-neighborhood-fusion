"""Independent metric formulae, initialization/RNG equality, and output hashes."""
from common import *
import argparse, numpy as np, torch
from scipy.optimize import linear_sum_assignment
p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args();out=Path(a.run)
dest=out/'independent_audit.json'
if dest.exists():
    cached=json.loads(dest.read_text());assert cached['status']=='PASSED'
    for name,h in cached['artifact_sha256'].items():assert sha(out/name)==h,name
    print('REUSE_RUN_AUDIT');sys.exit(0)
meta=json.loads((out/'run.json').read_text());res=json.loads((out/'result.json').read_text());recorded=json.loads((out/'metrics.json').read_text())
assert meta['status']==res['status']=='COMPLETED' and res['checkpoint_reload_exact']
assert meta['label_guard'] and not meta['labels_read'] and not res['labels_read']
task=taskpath(meta['dataset'],meta['depth'],meta['seed']);assert sha(task/'ready.json')==meta['data_identity_sha256']
with np.load(out/'outputs.npz') as f:pred=f['prediction'];emb=f['embedding']
truth=np.load(R/'data/evaluation_only'/meta['dataset']/'reference_labels.npy')[np.load(out/'cell_positions.npy')]
k=DATASETS[meta['dataset']][3];assert np.isfinite(emb).all() and len(pred)==len(truth) and (pred>=0).all() and (pred<k).all()
cm=np.zeros((k,k),dtype=np.int64);np.add.at(cm,(truth,pred),1);rows=cm.sum(1);cols=cm.sum(0);n=len(pred)
choose=lambda v:np.sum(v*(v-1)/2)
c=choose(cm);r=choose(rows);s=choose(cols);expected=r*s/(n*(n-1)/2);den=.5*(r+s)-expected
ari=(c-expected)/den if den else 1.
probs=cm/n;prod=rows[:,None]*cols[None,:];mask=cm>0;mi=np.sum(probs[mask]*np.log(cm[mask]*n/prod[mask]));entropy=lambda v:-np.sum((v[v>0]/n)*np.log(v[v>0]/n))
hs=entropy(rows)+entropy(cols);nmi=2*mi/hs if hs else 1.
rr,cc=linear_sum_assignment(-cm);mapped=np.zeros(k,dtype=int);mapped[cc]=rr;mp=mapped[pred]
recalls={};f1=[]
for i in range(k):
    tp=np.sum((truth==i)&(mp==i));recalls[str(i)]=float(tp/rows[i]) if rows[i] else None
    d=np.sum(truth==i)+np.sum(mp==i)
    if d:f1.append(2*tp/d)
values=dict(ARI=float(ari),NMI=float(nmi),ACC=float(np.mean(mp==truth)),macro_F1=float(np.mean(f1)))
for key,value in values.items():assert abs(value-recorded[key])<1e-12,(key,value,recorded[key])
assert recalls==recorded['group_recalls'] and sha(out/'outputs.npz')==recorded['prediction_sha256']
paired=None
def equal(a,b):
    if torch.is_tensor(a):return torch.equal(a,b)
    if isinstance(a,np.ndarray):return np.array_equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[key],b[key]) for key in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b
if meta['method']=='NB':
    parent=Path(meta['initialization_parent']);aa=torch.load(parent/'initial.pt',map_location='cpu',weights_only=True);bb=torch.load(out/'initial.pt',map_location='cpu',weights_only=True)
    common=[key for key in bb if key in aa and aa[key].shape==bb[key].shape];assert all(torch.equal(aa[key],bb[key]) for key in common)
    assert all(not key.startswith('_dec_pi') for key in bb)
    assert equal(torch.load(parent/'initial_rng.pt',weights_only=False,map_location='cpu'),torch.load(out/'initial_rng.pt',weights_only=False,map_location='cpu'))
    paired=dict(parent=str(parent),common_keys=len(common),initial_weights_exact=True,training_rng_exact=True)
artifacts={f.name:sha(f) for f in out.iterdir() if f.is_file() and f.name not in ['supervisor.json','progress.json','audit.log']}
save(dest,dict(status='PASSED',utc=utc(),independent_metrics=values,group_recalls=recalls,metric_tolerance=1e-12,paired=paired,checkpoint_reload_verified_in_training=True,label_guard_present=True,artifact_sha256=artifacts))
print('RUN_AUDIT_PASSED',str(out))
