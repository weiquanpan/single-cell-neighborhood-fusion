"""Confirm that the saved pre-trigger Adam state was finite; no training."""
from pathlib import Path
import torch,json
ROOT=Path(__file__).resolve().parent
records={}
for name in ['last_finite_after','first_nonfinite_before','first_nonfinite_after']:
    checkpoint=torch.load(ROOT/f'checkpoints/{name}.pt',map_location='cpu',weights_only=False)
    states=checkpoint['optimizer']['state']
    kinds={}
    for key in ['step','exp_avg','exp_avg_sq','max_exp_avg_sq']:
        tensors=[v[key] for v in states.values() if key in v]
        kinds[key]={'tensors':len(tensors),'nan':sum(int(torch.isnan(t).sum()) for t in tensors),'inf':sum(int(torch.isinf(t).sum()) for t in tensors),'all_finite':all(bool(torch.isfinite(t).all()) for t in tensors)}
    records[name]=kinds
assert all(k['all_finite'] for k in records['last_finite_after'].values())
assert all(k['all_finite'] for k in records['first_nonfinite_before'].values())
(ROOT/'optimizer_integrity.json').write_text(json.dumps(records,indent=2),encoding='utf8')
print(json.dumps(records,indent=2))
