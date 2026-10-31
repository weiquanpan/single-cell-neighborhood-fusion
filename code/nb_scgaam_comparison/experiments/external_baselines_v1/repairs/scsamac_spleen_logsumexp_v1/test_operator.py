"""Loss/gradient tests only: zero optimizer updates, no model training."""
from pathlib import Path
import os,sys,json,importlib.util,math
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import torch
R=Path(__file__).resolve().parent
torch.set_default_dtype(torch.float64);torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32=False
torch.use_deterministic_algorithms(True)
def module(p,name):
 spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
orig=module(R/'source_original/st_loss.py','original_loss').SupConLoss
stable=module(R/'source_stable/st_loss.py','stable_loss').SupConLoss
def calc(cls,x,**kw):
 a=x.detach().clone().requires_grad_(True);v=cls(**kw)(a);g,=torch.autograd.grad(v,a);return v.detach(),g.detach()
torch.manual_seed(761)
rows=[]
for device in ['cpu','cuda']:
 for mode in ['all','one']:
  for scale in [.01,.1,.3]:
   x=torch.randn(17,2,32,device=device)*scale
   a,g=calc(orig,x,contrast_mode=mode);b,h=calc(stable,x,contrast_mode=mode)
   torch.testing.assert_close(a,b,atol=1e-12,rtol=1e-12);torch.testing.assert_close(g,h,atol=1e-12,rtol=1e-12)
   rows.append(dict(device=device,mode=mode,scale=scale,loss_abs_diff=float(abs(a-b)),gradient_max_abs_diff=float((g-h).abs().max())))
capture=torch.load(R.parents[1]/'diagnostics/scsamac_spleen_seed101_v1/checkpoints/first_nonfinite_tensors.pt',map_location='cpu',weights_only=True)
x=capture['forward']['contrast.features'].cuda()
a,g=calc(orig,x);b,h=calc(stable,x)
assert torch.isfinite(a) and not torch.isfinite(g).all()
assert torch.isfinite(b) and torch.isfinite(h).all()
# Independent float64 cross-entropy representation, same self-excluded classes.
y=x.detach().clone().requires_grad_(True);n=y.shape[0];z=torch.cat(torch.unbind(y,dim=1),0)
scores=z@z.T/.07; scores=scores.masked_fill(torch.eye(2*n,device='cuda',dtype=torch.bool),-torch.inf)
target=(torch.arange(2*n,device='cuda')+n)%(2*n)
ref=torch.nn.functional.cross_entropy(scores,target);hg,=torch.autograd.grad(ref,y)
torch.testing.assert_close(b,ref,atol=1e-12,rtol=1e-12);torch.testing.assert_close(h,hg,atol=1e-12,rtol=1e-12)
result=dict(all_passed=True,optimizer_updates=0,benign_tests=rows,captured_failure=dict(original_loss=float(a),stable_loss=float(b),cross_entropy_reference=float(ref),original_gradient_finite=bool(torch.isfinite(g).all()),stable_gradient_finite=bool(torch.isfinite(h).all()),reference_gradient_max_abs_diff=float((h-hg).abs().max()),stable_gradient_max_abs=float(h.abs().max()),note='Original underflow quantizes the denominator, so even its finite loss is already inaccurate; the reference verifies the unchanged mathematical objective.'),torch=torch.__version__)
(R/'operator_test.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
print(json.dumps(result,indent=2))
