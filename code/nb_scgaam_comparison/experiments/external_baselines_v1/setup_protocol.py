from pathlib import Path
import json,hashlib,subprocess,shutil,difflib,datetime
B=Path(__file__).resolve().parent;R=B.parents[1];E=R/'experiments/cross_dataset_v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def put(p,obj):p.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf8')
assert not (B/'protocol.json').exists(),'Never overwrite frozen protocol'
sources={}
for method in ['scAGCL','scCAGN']:
 p=B/'upstream'/method
 sources[method]={'commit':subprocess.check_output(['git','-C',str(p),'rev-parse','HEAD'],text=True).strip(),'files':{q.name:sha(q) for q in p.glob('*.py')}}
sources['scSAMAC']={'commit':'4ee0d2f4ca392802d43ba43a9eacf0a31cbae384','files':{p.name:sha(p) for p in (R/'experiments/scsamac_author_worm_v1/author_source').glob('*.py')}}
# Archive upstream unchanged and make an explicitly different, label-free scCAGN adapter.
src=B/'upstream/scCAGN';dst=B/'adapted/scCAGN';dst.mkdir(parents=True)
for p in src.glob('*.py'):shutil.copy2(p,dst/p.name)
p=dst/'scCAGN.py';s=p.read_text(encoding='utf8');original=s
s=s.replace('from preprocess import *','')
s=s.replace('    y = dataset.y','    # No labels are supplied to training.')
a=s.index('    best_nmi = -1');b=s.index('    # Training loop',a);s=s[:a]+s[b:]
a=s.index('        res1 = tmp_q.cpu()');b=s.index('        x_bar, q, pred, _ = model(data, adj)',a);s=s[:a]+s[b:]
a=s.index('    # Load the best model based on NMI score');b=s.index('if __name__ == "__main__":',a)
s=s[:a]+'''    # Fixed terminal checkpoint, no label-based epoch selection.
    model.eval()
    with torch.no_grad():
        _, q, _, z_final = model(data, adj)
    return model, z_final.cpu().numpy(), q.cpu().numpy().argmax(1)

'''+s[b:]
s=s.replace('        optimizer.step()','        optimizer.step()\n        print("Joint epoch", epoch + 1, "loss", float(total_loss.detach()), flush=True)')
# Disable unused main entry: the path-isolated runner invokes functions directly.
s=s[:s.index('if __name__ == "__main__":')]
p.write_text(s,encoding='utf8')
(B/'scCAGN_label_free.patch').write_text(''.join(difflib.unified_diff(original.splitlines(True),s.splitlines(True),fromfile='upstream/scCAGN.py',tofile='adapted/scCAGN.py')),encoding='utf8')
p=dst/'pretrain.py';s=p.read_text(encoding='utf8');original=s
s=s.replace('model_path = f"D:/360MoveData/Users/mi/Desktop/scCAGN-main/Cluster_model/dataset/{name}/model/{name}{m}.pkl"','model_path = str(OUTPUT_DIR / f"pretrain_{m}.pt")')
s=s.replace('print(f"模型已成功保存到: {model_path}")','print(f"Pretrain restart={m} epoch={epoch+1} silhouette={silhouette} checkpoint={model_path}", flush=True)')
p.write_text(s,encoding='utf8')
(B/'scCAGN_paths.patch').write_text(''.join(difflib.unified_diff(original.splitlines(True),s.splitlines(True))),encoding='utf8')
protocol={'id':'external_baselines_v1','frozen_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source':sources,
'development':{'Worm_Neuron':10,'Limb_Muscle':6,'Tosches':15,'Bach':8},'development_seeds':[11,22,33],
'primary':['Romanov','Adam','Quake_10x_Bladder','Quake_10x_Spleen','Quake_Smart-seq2_Lung'],'supplement':['Muraro'],'seeds':[101,202,303,404,505],
'methods':['scSAMAC','scAGCL','scCAGN'],'selection':'No hyperparameter search. Uniform author default budgets across datasets. Posthoc comparison extension on previously revealed holdouts.',
'input':'Frozen shared cells and 2000 HVGs; scSAMAC encoder/counts/size factors; scAGCL log_input; scCAGN MinMax(counts) then row L2 as author H5 transform. No rounding of Muraro.',
'outputs':'Author native prediction and separate L2+KMeans20(random_state=train seed) readout, never choose per dataset; Fusion unchanged.',
'scSAMAC':{'pretrain_epochs':300,'joint_max_epochs':2000,'tol':0.001,'batch_size':256,'reuse_development':True},
'scAGCL':{'epochs_argument':500,'actual_optimizer_steps':501,'adversarial_iterations_per_view':10,'subgraph_size':400,'selection':'author minimum label-free training loss; returned embedding is before corresponding update','native_readout':'L2 KMeans n_init=auto random_state=0, following author sklearn1.5 default'},
'scCAGN':{'pretrain_restarts':20,'pretrain_epochs_each':50,'pretrain_batch_size':1,'joint_epochs':30,'pretrain_selection':'maximum final-epoch silhouette, author rule','joint_selection':'fixed terminal checkpoint; original label-NMI selection removed and explicitly reported','Auto':False,'known_implementation_limit':'author adversarial loss uses detached pretraining z; preserved, not silently corrected'},
'timeout_seconds_per_job':172800,'timeout_note':'48h engineering cap, not early stopping; timed-out runs remain incomplete. Full author scCAGN cellwise budget may be costly.',
'evaluation':'Reuse frozen evaluate.metrics; exact cell position equality; complete grid only, dataset equal ARI/rank, seed SD, smallest/worst recall and fragmentation. Muraro separate with exclude-unclear sensitivity.',
'frozen_fusion_manifest_sha256':sha(E/'followup/final_method_v1/manifest.json'),
'adapted_sha256':{str(p.relative_to(B)):sha(p) for p in dst.glob('*.py')}}
put(B/'protocol.json',protocol)
print(json.dumps(protocol,indent=2))
