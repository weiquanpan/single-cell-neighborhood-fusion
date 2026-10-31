from pathlib import Path
import json, sys
B = Path(__file__).resolve().parent
sys.path.insert(0, str(B.parent / 'cross_dataset_v1'))
import common as c
import numpy as np
import torch
from sklearn.metrics import adjusted_rand_score

V = B / 'performance_concurrency_v1'
result = {'utc': c.utc(), 'seeds': {}}
for seed in (101, 202):
    ref = B / 'performance_validation_v1/baseline' if seed == 101 else V / 'serial202'
    test = V / f'parallel{seed}'
    a = json.loads((ref / 'trajectory.json').read_text())
    b = json.loads((test / 'trajectory.json').read_text())
    assert len(a) == len(b) == 1000
    fields = sorted(set().union(*(x.keys() for x in a + b)))
    differences = {k: [i for i, (x, y) in enumerate(zip(a,b)) if x.get(k) != y.get(k)] for k in fields}
    r = {'trajectory_count': len(a), 'differences_by_field': differences}
    ap = ref / f'runs/Romanov/scCAGN/seed{seed}/attempt_001'
    bp = test / f'runs/Romanov/scCAGN/seed{seed}/attempt_001'
    checks = {}
    for p in ap.glob('*.pt'):
        x = torch.load(p, weights_only=True, map_location='cpu')
        y = torch.load(bp / p.name, weights_only=True, map_location='cpu')
        assert x.keys() == y.keys()
        checks[p.name] = {'different_tensors': sum(not torch.equal(x[k], y[k]) for k in x),
                          'max_abs': max(float((x[k]-y[k]).abs().max()) for k in x)}
    r['checkpoints'] = checks
    with np.load(ap / 'outputs.npz') as x, np.load(bp / 'outputs.npz') as y:
        assert x.files == y.files
        r['outputs'] = {k: {'exact': bool(np.array_equal(x[k], y[k])),
                           'max_abs': float(np.max(np.abs(x[k]-y[k]))),
                           **({'different_labels': int(np.sum(x[k]!=y[k])), 'partition_ari': float(adjusted_rand_score(x[k],y[k]))} if k != 'embedding' else {})} for k in x.files}
    r['budget_and_selection'] = {k: json.loads((ref/'validation_complete.json').read_text())[k] == json.loads((test/'validation_complete.json').read_text())[k] for k in ('formal_budget','optimizer_steps','selected_restart')}
    result['seeds'][str(seed)] = r
state = json.loads((V/'state.json').read_text())
result['timing'] = {k: state[k] for k in ('serial_pair_seconds','parallel_pair_seconds','throughput_speedup','caveat')}
result['decision'] = 'Not deployed. Strict complete-state equality failed. No tolerance relaxed. Production unchanged. Large-data memory validation remains pending and is not sufficient to override equality failure.'
out = V/'review_v1.json'
assert not out.exists()
c.save(out, result)
print(json.dumps(result, indent=2))
