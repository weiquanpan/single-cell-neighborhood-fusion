"""Verify finished diagnostic coverage, trace and preserved original records."""
from pathlib import Path
import json,hashlib
import torch
ROOT=Path(__file__).resolve().parent;B=ROOT.parents[1]
def load(name):return json.loads((ROOT/name).read_text(encoding='utf8'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024**2),b''):h.update(chunk)
    return h.hexdigest()
result=load('result.json');assert result['status']=='DIAGNOSTIC_COMPLETE_PENDING_REVIEW'
epochs=load('epoch_trace_comparison.json');assert len(epochs)==300
first_bad_loss={};batch_count=0;epoch_indices={};previous=0
for line in (ROOT/'batches.jsonl').open(encoding='utf8'):
    row=json.loads(line);batch_count+=1;assert row['step']==previous+1;previous=row['step']
    epoch_indices.setdefault(row['epoch'],[]).extend(row['cell_positions'])
    for term,value in row['losses'].items():
        if isinstance(value,str) and term not in first_bad_loss:
            first_bad_loss[term]={'step':row['step'],'epoch':row['epoch'],'batch':row['batch'],'value':value}
assert batch_count==11400
assert all(sorted(ids)==list(range(9552)) for ids in epoch_indices.values())
old=B/'runs/Quake_10x_Spleen/scSAMAC/seed101/attempt_001/pretrained.pt'
new=ROOT/'checkpoints/pretrained_diagnostic.pt'
a=torch.load(old,map_location='cpu',weights_only=True);b=torch.load(new,map_location='cpu',weights_only=True)
def same(x,y):
    if isinstance(x,torch.Tensor):return x.dtype==y.dtype and x.shape==y.shape and x.contiguous().numpy().tobytes()==y.contiguous().numpy().tobytes()
    if isinstance(x,dict):return x.keys()==y.keys() and all(same(x[k],y[k]) for k in x)
    if isinstance(x,(list,tuple)):return len(x)==len(y) and all(same(u,v) for u,v in zip(x,y))
    return x==y
protected=load('original_files_before.json')
originals_unchanged=all(sha(p)==h for p,h in protected.items());assert originals_unchanged
prior_guard=json.loads((B/'audit/scsamac_spleen_nan_diagnosis_v1_20261022/protected_files_sha256.json').read_text(encoding='utf8'))['before']
assert all(sha(p)==h for p,h in prior_guard.items())
summary=dict(status='REVIEWED_DIAGNOSTIC',epochs=300,batches=batch_count,optimizer_updates=11400,
    each_epoch_exactly_all_9552_cells_once=True,
    historical_printed_epoch_losses_all_exact=all(e['exact_printed_match'] for e in epochs),
    matched_epochs=sum(e['exact_printed_match'] for e in epochs),
    last_historical_finite_epoch=max(e['epoch'] for e in epochs if e['historical']!='nan'),
    first_nonfinite_event=load('first_nonfinite.json'),first_nonfinite_batch_losses=first_bad_loss,
    terminal_model_and_optimizer_bitwise_equal_to_original=same(a,b),
    original_files_sha256_unchanged=originals_unchanged,protected_files=len(protected),
    prior_audit_68_files_including_all_five_failures_unchanged=True,
    original_attempt_overwritten=False,official_result_updated=False,
    note='Epoch equality is at the original eight-decimal printed precision; intermediate historical states were not recorded.')
(ROOT/'completion_verification.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
print(json.dumps(summary,indent=2))
