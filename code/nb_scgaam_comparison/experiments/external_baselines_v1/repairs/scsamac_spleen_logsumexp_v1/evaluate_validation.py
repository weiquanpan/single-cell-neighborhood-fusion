"""Post-training-only evaluation. Never imported by the training process."""
from pathlib import Path
import sys,json,hashlib,re,datetime
R=Path(__file__).resolve().parent;B=R.parents[1];E=B.parent/'cross_dataset_v1'
sys.path.insert(0,str(E))
import common as c
from evaluate import metrics
import numpy as np,torch
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from threadpoolctl import threadpool_limits
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
out=R/'validation_run';meta=read(out/'run.json')
assert meta['status']=='COMPLETE' and meta['seed']==101 and not meta['labels_read']
assert meta['checkpoint_reload_exact'] and meta['finite_gradient_and_parameter_check_every_optimizer_step']
assert meta['optimizer_steps']['pretrain:Adam']==11400
assert meta['optimizer_steps'].get('joint:Adadelta',0)<=76000
log=(R/'training.stdout.log').read_text(encoding='utf8')
epochs=re.findall(r'Pretrain epoch\s+(\d+), NB loss: ([^\s]+)',log)
assert [int(e) for e,v in epochs]==list(range(1,301))
assert all(np.isfinite(float(v)) for e,v in epochs)
assert 'nan' not in log.lower() and 'inf' not in log.lower()
task=E/'heldout_v1/prepared/Quake_10x_Spleen/full'
ids=np.load(out/'cell_positions.npy');np.testing.assert_array_equal(ids,np.load(task/'cell_positions.npy'))
assert len(ids)==9552 and len(np.unique(ids))==9552
for name,h in meta['input_sha256'].items():assert c.sha(task/name)==h
old=read(B/'runs/Quake_10x_Spleen/scSAMAC/seed101/attempt_001/run.json')
assert old['input_sha256']==meta['input_sha256']
for file,h in read(R/'protected_before.json').items():assert c.sha(file)==h,file
for file,h in read(B/'audit/scsamac_spleen_nan_diagnosis_v1_20261022/protected_files_sha256.json')['before'].items():assert c.sha(file)==h,file
m=read(R/'source_manifest.json');changed=[]
for name,h in m['original'].items():
 assert c.sha(R/'source_original'/name)==h
 assert c.sha(R/'source_stable'/name)==m['stable'][name]
 if h!=m['stable'][name]:changed.append(name)
assert changed==['st_loss.py']
assert meta['runner_sha256']==m['runner_sha256']==c.sha(R/'run_stable.py')
for name in ['pretrained.pt','prediction_checkpoint.pt']:
 cp=torch.load(out/name,map_location='cpu',weights_only=True)
 def check(x):
  if torch.is_tensor(x):assert torch.isfinite(x).all(),name
  elif isinstance(x,dict):
   for v in x.values():check(v)
  elif isinstance(x,(tuple,list)):
   for v in x:check(v)
 check(cp)
f=np.load(out/'outputs.npz');z=f['embedding'];pred=f['prediction'];matched=f['prediction_kmeans20']
assert z.shape==(9552,32) and np.isfinite(z).all()
with threadpool_limits(limits=8):again=KMeans(n_clusters=5,n_init=20,random_state=101).fit_predict(normalize(z))
np.testing.assert_array_equal(matched,again)
labelsfile=E/'heldout_v1/evaluation_only/Quake_10x_Spleen/reference_labels.npy'
y=np.load(labelsfile)[ids];rows=[]
for suffix,pr in [('native',pred),('KMeans20',matched)]:
 assert len(pr)==9552 and np.isfinite(pr).all()
 rows.append(dict(dataset='Quake_10x_Spleen',method='scSAMAC_'+suffix,implementation='stable_logsumexp_v1',seed=101,source=str(out),**metrics(y,pr)))
result=dict(status='VALIDATED',utc=c.utc(),full_training_runs=1,pretrain_epochs=300,pretrain_steps=11400,joint_optimizer_steps=meta['optimizer_steps'].get('joint:Adadelta',0),joint_stopping='original tolerance' if 'Reach tolerance' in log else 'original maximum epochs',rows=rows,seconds=meta['seconds'],phase_seconds=meta['phase_seconds'],checks=dict(all_training_gradients_and_updated_parameters_finite=True,pretrain_model_and_optimizer_finite=True,selected_checkpoint_finite=True,checkpoint_embedding_and_native_readout_reload_exact=True,canonical_cell_indices_exact=True,kmeans20_prediction_recomputed_exact=True,input_hashes_identical_to_original_failed_run=True,original_failures_and_sources_unchanged=True,only_loss_arithmetic_changed=True),source_hashes={str(p):c.sha(p) for p in [out/'outputs.npz',out/'cell_positions.npy',out/'run.json',labelsfile,R/'repair_protocol.json',R/'operator_test.json']},original_failure_cost_seconds=old.get('seconds'),diagnostic_reproduction_cost_seconds=read(B/'diagnostics/scsamac_spleen_seed101_v1/result.json').get('elapsed_seconds'),claim_boundary='A single-seed numerical repair; not multi-seed stability evidence. Other datasets retain their original numerical implementation, and original failures remain historical evidence.')
c.save(R/'evaluation.json',result)
print(json.dumps(result,indent=2,ensure_ascii=False))
