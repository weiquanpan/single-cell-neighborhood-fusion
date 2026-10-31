from pathlib import Path
import hashlib,json,shutil,difflib,datetime
B=Path(__file__).resolve().parent
R=B/'repairs/scsamac_spleen_logsumexp_v1'
A=B.parent/'scsamac_author_worm_v1/author_source'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x): p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
R.mkdir(parents=True,exist_ok=False)
shutil.copytree(A,R/'source_original')
shutil.copytree(A,R/'source_stable')
for name in ['run_one.py','protocol.json','protocol_single_seed_v2.json']:
 shutil.copy2(B/name,R/name)
p=R/'source_stable/st_loss.py'; s=p.read_text(encoding='utf8')
old='        exp_logits = torch.exp(logits) * logits_mask\n        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))'
new='''        # Same denominator over non-self contrasts, evaluated in the log domain.
        # Keep logits finite for the positive-mask product below (avoid 0 * -inf).
        log_denominator = torch.logsumexp(
            logits.masked_fill(logits_mask == 0, -torch.inf), dim=1, keepdim=True)
        log_prob = logits - log_denominator'''
assert s.count(old)==1
p.write_text(s.replace(old,new),encoding='utf8')
(R/'stable_loss.diff').write_text(''.join(difflib.unified_diff(s.splitlines(True),p.read_text(encoding='utf8').splitlines(True),fromfile='original/st_loss.py',tofile='stable/st_loss.py')),encoding='utf8')
s=(B/'run_one.py').read_text(encoding='utf8')
s=s.replace("B=Path(__file__).resolve().parent;E=", "R=Path(__file__).resolve().parent;B=R.parents[1];E=")
s=s.replace("out=B/'runs'/a.dataset/a.method/f'seed{a.seed}'/(('smoke_' if a.smoke else 'attempt_')+a.attempt);out.mkdir(parents=True,exist_ok=False)","assert a.method=='scSAMAC' and a.dataset=='Quake_10x_Spleen' and a.seed==101 and not a.smoke\n assert json.loads((R/'operator_test.json').read_text())['all_passed']\n out=R/'validation_run';out.mkdir(parents=True,exist_ok=False)")
s=s.replace("source=(c.ROOT/'experiments/scsamac_author_worm_v1/author_source' if a.method=='scSAMAC' else B/'upstream'/a.method)","source=R/'source_stable'")
s=s.replace("for name,h in P['source'][a.method]['files'].items():assert c.sha(source/name)==h,name", "for name,h in P['source'][a.method]['files'].items():assert c.sha(R/'source_original'/name)==h,name\n  for name,h in json.loads((R/'source_manifest.json').read_text())['stable'].items():assert c.sha(source/name)==h,name")
s=s.replace("result=_old(self,*args,**kw);key=", "params=[p for g in self.param_groups for p in g['params']]\n   grads=[p.grad for p in params if p.grad is not None]\n   assert grads and torch.stack([torch.isfinite(x).all() for x in grads]).all().item(), 'nonfinite gradient; no update applied'\n   result=_old(self,*args,**kw)\n   assert torch.stack([torch.isfinite(x).all() for x in params]).all().item(), 'nonfinite parameter after update'\n   key=")
s=s.replace("torch.set_default_dtype(torch.float64);mod.print=progress", "torch.set_default_dtype(torch.float64);mod.print=progress\n   meta['numerical_implementation']='masked logsumexp; unchanged objective, parameters, seed and native stopping rule'\n   meta['source_manifest_sha256']=c.sha(R/'source_manifest.json')\n   meta['repair_protocol_sha256']=c.sha(R/'repair_protocol.json')\n   meta['finite_gradient_and_parameter_check_every_optimizer_step']=True")
(R/'run_stable.py').write_text(s,encoding='utf8')
(R/'runner.diff').write_text(''.join(difflib.unified_diff((B/'run_one.py').read_text(encoding='utf8').splitlines(True),s.splitlines(True),fromfile='run_one.py',tofile='run_stable.py')),encoding='utf8')
save(R/'source_manifest.json',{'original':{p.name:sha(p) for p in (R/'source_original').glob('*.py')},'stable':{p.name:sha(p) for p in (R/'source_stable').glob('*.py')},'runner_sha256':sha(R/'run_stable.py')})
protected={}
for seed in [101,202,303,404,505]:
 for p in (B/f'runs/Quake_10x_Spleen/scSAMAC/seed{seed}/attempt_001').rglob('*'):
  if p.is_file(): protected[str(p)]=sha(p)
for p in A.glob('*.py'):protected[str(p)]=sha(p)
save(R/'protected_before.json',protected)
save(R/'repair_protocol.json',{'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'purpose':'One user-authorized numerical repair validation; all original failed attempts retained','dataset':'Quake_10x_Spleen','seed':101,'training_runs_authorized':1,'source_change':'st_loss.py explicit masked exp/sum/log to masked logsumexp only','objective_equivalent_in_exact_arithmetic':True,'bitwise_training_trajectory_equivalence_claimed':False,'no_epsilon_no_clipping_no_normalization_no_hyperparameter_change':True,'pretrain':{'epochs':300,'batch_size':256,'lr':.0001,'expected_optimizer_steps':11400,'amsgrad':True},'joint':{'max_epochs':2000,'batch_size':256,'lr':1.,'update_interval':1,'tol':.001,'stopping':'Original unlabeled label-change stopping retained'},'contrast_temperature':.07,'contrast_weight':.01,'sigma':2.5,'z_dim':32,'encodeLayer':[256,64],'decodeLayer':[64,256],'alpha':1.,'gamma':1.,'dtype':'float64','labels_read_in_training':False,'evaluation':'Original native and row-L2 KMeans20 seed101 separately; exact canonical cell IDs','manuscript_update_gate':'Only after finite complete training, exact checkpoint reload, index and readout verification; preserve original failure disclosure'})
print(R)
