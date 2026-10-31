"""Create auditable observation-only copies. Never modify frozen source."""
from pathlib import Path
import json, hashlib, shutil, difflib, ast
HERE=Path(__file__).resolve().parent
B=HERE.parents[1]
BASE=B.parents[1]
AUTHOR=BASE/'experiments/scsamac_author_worm_v1/author_source'
P=json.loads((B/'protocol.json').read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
original=HERE/'source_original'; instrumented=HERE/'source_instrumented'
original.mkdir(exist_ok=False); instrumented.mkdir(exist_ok=False)
manifest={}
for name,expected in P['source']['scSAMAC']['files'].items():
    assert sha(AUTHOR/name)==expected,name
    shutil.copyfile(AUTHOR/name,original/name)
    shutil.copyfile(AUTHOR/name,instrumented/name)
    manifest[name]={'original_sha256':expected}
def edit(name,changes):
    p=instrumented/name;before=p.read_text(encoding='utf8');s=before
    s='import diagnostic_observer as diag\n'+s
    for a,b in changes:
        assert s.count(a)==1,(name,a,s.count(a))
        s=s.replace(a,b,1)
    ast.parse(s)
    p.write_text(s,encoding='utf8')
    (HERE/(name+'.diff')).write_text(''.join(difflib.unified_diff(before.splitlines(True),s.splitlines(True),fromfile='original/'+name,tofile='instrumented/'+name)),encoding='utf8')

edit('scSAMAC.py',[
('dataset = TensorDataset(torch.Tensor(X), torch.Tensor(X_raw), torch.Tensor(size_factor))',
 'dataset = TensorDataset(torch.Tensor(X), torch.Tensor(X_raw), torch.Tensor(size_factor), torch.arange(X.shape[0]))'),
('for batch_idx, (x_batch, x_raw_batch, sf_batch) in enumerate(dataloader):',
 'for batch_idx, (x_batch, x_raw_batch, sf_batch, diagnostic_indices) in enumerate(dataloader):\n                diag.begin_batch(epoch, batch_idx, diagnostic_indices, x_batch, x_raw_batch, sf_batch, self, optimizer)'),
('loss = self.nb_loss(x=x_raw_tensor, mean=mean_tensor, disp=disp_tensor, scale_factor=sf_tensor) + contrast_loss2 * 0.01',
 'nb_term = self.nb_loss(x=x_raw_tensor, mean=mean_tensor, disp=disp_tensor, scale_factor=sf_tensor)\n                loss = nb_term + contrast_loss2 * 0.01\n                diag.losses(nb_term, contrast_loss2, loss)'),
('                loss.backward()\n                optimizer.step()\n                loss_val += loss.item() * len(x_batch)',
 '                loss.backward()\n                diag.before_step(self, optimizer)\n                optimizer.step()\n                diag.after_step(self, optimizer)\n                loss_val += loss.item() * len(x_batch)'),
('            print(\'Pretrain epoch %3d, NB loss: %.8f\' % (epoch+1, loss_val/X.shape[0]))',
 '            print(\'Pretrain epoch %3d, NB loss: %.8f\' % (epoch+1, loss_val/X.shape[0]))\n            diag.epoch_end(epoch, loss_val/X.shape[0])'),
('        attn_weights = nn.Softmax(dim=-1)(scores)',
 '        diag.observe("attention.scores", scores)\n        attn_weights = nn.Softmax(dim=-1)(scores)\n        diag.observe("attention.weights", attn_weights)'),
])
edit('st_loss.py',[
('        contrast_count = features.shape[1]',
 '        diag.observe("contrast.features", features)\n        contrast_count = features.shape[1]'),
('        # for numerical stability',
 '        diag.observe("contrast.dot_over_temperature", anchor_dot_contrast)\n        # for numerical stability'),
('        logits = anchor_dot_contrast - logits_max.detach()',
 '        logits = anchor_dot_contrast - logits_max.detach()\n        diag.observe("contrast.shifted_logits", logits)'),
('        exp_logits = torch.exp(logits) * logits_mask\n        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))',
 '        exp_values = torch.exp(logits)\n        diag.observe("contrast.exp_values", exp_values)\n        exp_logits = exp_values * logits_mask\n        diag.observe("contrast.masked_exp", exp_logits)\n        denominator = exp_logits.sum(1, keepdim=True)\n        diag.observe("contrast.denominator", denominator)\n        log_denominator = torch.log(denominator)\n        diag.observe("contrast.log_denominator", log_denominator)\n        log_prob = logits - log_denominator\n        diag.observe("contrast.log_prob", log_prob)\n        diag.observe("contrast.positive_mask", mask, False)'),
('        loss = loss.view(anchor_count, batch_size).mean()',
 '        loss = loss.view(anchor_count, batch_size).mean()\n        diag.observe("contrast.loss", loss)'),
])
edit('layers.py',[
('        return torch.clamp(torch.exp(x), min=1e-5, max=1e6)',
 '        diag.observe("mean.pre_exp", x)\n        exponentiated = torch.exp(x)\n        diag.observe("mean.exp", exponentiated)\n        clamped = torch.clamp(exponentiated, min=1e-5, max=1e6)\n        diag.observe("mean.clamped", clamped)\n        return clamped'),
('        nb_case = nb_final\n\n        return torch.mean(nb_case)',
 '        nb_case = nb_final\n        diag.observe("nb.scaled_mean", mean)\n        diag.observe("nb.dispersion", disp)\n        diag.observe("nb.t1", t1)\n        diag.observe("nb.t2", t2)\n        diag.observe("nb.per_entry", nb_case)\n        return torch.mean(nb_case)'),
])
for name in manifest:manifest[name]['instrumented_sha256']=sha(instrumented/name)
for p in [B/'run_one.py',B/'protocol.json',B.parent/'cross_dataset_v1/common.py']:
    shutil.copyfile(p,original/p.name)
(HERE/'source_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
print('Prepared observation-only source copies; no training launched.')
