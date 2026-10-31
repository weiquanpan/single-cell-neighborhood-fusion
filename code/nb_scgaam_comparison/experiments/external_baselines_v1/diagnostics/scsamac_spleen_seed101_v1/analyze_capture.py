"""Analyze the saved diagnostic event only. No model construction or optimizer."""
from pathlib import Path
import json,hashlib,math
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent
torch.set_num_threads(2)
def load(name):return json.loads((ROOT/name).read_text(encoding='utf8'))
def numeric(x):
    x=float(x);return x if math.isfinite(x) else str(x)
def ts(t):
    a=t.detach().cpu().numpy();f=a[np.isfinite(a)]
    return dict(shape=list(a.shape),nan=int(np.isnan(a).sum()),inf=int(np.isinf(a).sum()),finite=bool(np.isfinite(a).all()),min=float(f.min()) if f.size else None,max=float(f.max()) if f.size else None)
def equal(a,b):
    if isinstance(a,torch.Tensor):return torch.equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b
rec=load('first_nonfinite_batch.json')
t=torch.load(ROOT/'checkpoints/first_nonfinite_tensors.pt',map_location='cpu',weights_only=True)
before=torch.load(ROOT/'checkpoints/first_nonfinite_before.pt',map_location='cpu',weights_only=False)
after=torch.load(ROOT/'checkpoints/first_nonfinite_after.pt',map_location='cpu',weights_only=False)
last=torch.load(ROOT/'checkpoints/last_finite_after.pt',map_location='cpu',weights_only=False)
f=t['forward'];g=t['intermediate_gradients']
summary=dict(first_event=load('first_nonfinite.json'),losses=rec['losses'],
    all_observed_forward_tensors_finite=all(s['finite'] for s in rec['forward'].values()),
    nonfinite_forward=[{'name':n,**s} for n,s in rec['forward'].items() if not s['finite']],
    first_ten_nonfinite_backward=[s for s in rec['backward_order'] if not s['finite']][:10],
    nonfinite_parameter_gradients=[n for n,s in rec['parameter_gradients'].items() if not s['finite']],
    before_model_all_finite=all(torch.isfinite(x).all().item() for x in before['model'].values()),
    first_before_model_equals_last_finite_after=equal(before['model'],last['model']),
    first_before_optimizer_equals_last_finite_after=equal(before['optimizer'],last['optimizer']),
    after_model_nonfinite={n:ts(v) for n,v in after['model'].items() if not torch.isfinite(v).all()},
    input_batch_indices=before['batch']['indices'].tolist(),
    before_optimizer_step_values=sorted({int(s['step']) for s in before['optimizer']['state'].values()}),
    after_optimizer_step_values=sorted({int(s['step']) for s in after['optimizer']['state'].values()}),
    nb_backward=[s for s in rec['backward_order'] if s['tensor'].startswith('nb.')],
    mean_forward={k:v for k,v in rec['forward'].items() if k.startswith('mean.')},
    mean_backward=[s for s in rec['backward_order'] if s['tensor'].startswith('mean.')],
    analysis_optimizer_updates=0)
if 'contrast.denominator' in f:
    d=f['contrast.denominator'];ld=f['contrast.log_denominator'];scores=f['contrast.dot_over_temperature'];shifted=f['contrast.shifted_logits'];features=f['contrast.features'];n=features.shape[0]
    bad=(~torch.isfinite(g.get('contrast.denominator',torch.zeros_like(d)))).flatten()
    rows=torch.nonzero(bad).flatten().tolist()
    if not rows:rows=[int(d.argmin())]
    records=[]
    for r in rows:
        pos=(r+n)%(2*n);other=scores[r].clone();other[r]=-torch.inf
        view=r//n;cell=r%n
        row=dict(contrast_row=r,batch_cell_index=cell,prepared_cell_position=int(before['batch']['indices'][cell]),view=view,
            denominator=numeric(d[r,0]),log_denominator=numeric(ld[r,0]),
            self_score=numeric(scores[r,r]),max_nonself_score=numeric(other.max()),
            self_minus_max_nonself=numeric(scores[r,r]-other.max()),
            positive_score=numeric(scores[r,pos]),positive_shifted_logit=numeric(shifted[r,pos]),
            feature_norm=numeric(torch.linalg.vector_norm(features[cell,view])),
            positive_feature_norm=numeric(torch.linalg.vector_norm(features[cell,1-view])),
            nonself_exp_zero_count=int((f['contrast.masked_exp'][r]==0).sum())-1,
            original_input_row_l2=numeric(torch.linalg.vector_norm(before['batch']['x'][cell])))
        if 'contrast.log_denominator' in g and 'contrast.denominator' in g:
            gl=g['contrast.log_denominator'][r,0];gd=g['contrast.denominator'][r,0]
            row.update(log_denominator_gradient=numeric(gl),observed_denominator_gradient=numeric(gd),log_backward_gl_div_d=numeric(gl/d[r,0]))
        records.append(row)
    summary['contrast_path']={'denominator':ts(d),'zero_denominators':int((d==0).sum()),'rows':records,'log_denominator_gradient':ts(g['contrast.log_denominator']) if 'contrast.log_denominator' in g else None,'denominator_gradient':ts(g['contrast.denominator']) if 'contrast.denominator' in g else None}
summary['after_trigger_batches']=[{'offset':i,'losses':load(f'after_trigger_{i}_batch.json')['losses'],'first_nonfinite_event':load(f'after_trigger_{i}_batch.json')['nonfinite_events'][0]} for i in [1,2]]
(ROOT/'capture_analysis.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf8')
print(json.dumps({k:v for k,v in summary.items() if k not in ['input_batch_indices','after_model_nonfinite','nb_backward','mean_backward']},indent=2,ensure_ascii=False))
