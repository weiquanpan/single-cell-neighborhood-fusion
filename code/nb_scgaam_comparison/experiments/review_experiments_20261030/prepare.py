"""Freeze the follow-up design and validate all inputs before any new outcome."""
from pathlib import Path
import ast,json,sys
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];EXP=ROOT/'experiments/cross_dataset_v1';HOLD=EXP/'heldout_v1'
sys.path.insert(0,str(EXP))
import common as c

def main():
    assert not (HERE/'protocol.json').exists(),'Do not overwrite an existing frozen design'
    datasets=['Romanov','Adam','Quake_10x_Bladder','Quake_10x_Spleen','Quake_Smart-seq2_Lung'];seeds=[101,202,303,404,505]
    variants=['expression','expression_dense','random','pca_union','mse_union','fusion_noaux']
    repair=ROOT/'experiments/external_baselines_v1/repairs/scsamac_spleen_logsumexp_v1'
    src=(repair/'run_stable.py').read_text(encoding='utf8')
    before="R=Path(__file__).resolve().parent;B=R.parents[1];E=B.parent/'cross_dataset_v1';sys.path.insert(0,str(E))"
    after=f"OUTPUT_ROOT=Path(__file__).resolve().parent\nR=Path({str(repair)!r});B=R.parents[1];E=B.parent/'cross_dataset_v1';sys.path.insert(0,str(E))"
    assert src.count(before)==1;src=src.replace(before,after)
    before="assert a.method=='scSAMAC' and a.dataset=='Quake_10x_Spleen' and a.seed==101 and not a.smoke"
    after=f"assert a.method=='scSAMAC' and a.dataset in {datasets!r} and a.seed in {seeds!r} and not a.smoke"
    assert src.count(before)==1;src=src.replace(before,after)
    before="out=R/'validation_run';out.mkdir(parents=True,exist_ok=False)"
    after="out=OUTPUT_ROOT/'scsamac'/a.dataset/f'seed{a.seed}';out.mkdir(parents=True,exist_ok=False)"
    assert src.count(before)==1;src=src.replace(before,after)
    (HERE/'scsamac_worker.py').write_text(src,encoding='utf8')
    inputs={}
    for d in datasets:
        task=HOLD/'prepared'/d/'full';ready=json.loads((task/'ready.json').read_text())
        for name,h in ready['files'].items():assert c.sha(task/name)==h,(d,name)
        inputs[d]=dict(ready_sha256=c.sha(task/'ready.json'),files=ready['files'],nb_h={},nb_pretrain_epochs={})
        for s in seeds:
            hp=HOLD/'candidate_workspace/cache'/d/f'nb_h_seed{s}.npy'
            inputs[d]['nb_h'][str(s)]=c.sha(hp)
            result=HOLD/'runs'/d/'full/NB'/f'main_seed{s}'/'attempt_001/result.json'
            rr=json.loads(result.read_text());assert rr['status']=='COMPLETED'
            inputs[d]['nb_pretrain_epochs'][str(s)]=rr['pretrain_epochs']
            assert 1<=rr['pretrain_epochs']<=400,(d,s)
    jobs=[]
    def worker(d,s,v,smoke=False):
        base='smoke' if smoke else 'runs';jid=f'{base}__{d}__{v}__{s}'
        jobs.append(dict(id=jid,script='worker.py',args=['--dataset',d,'--seed',str(s),'--variant',v]+(['--smoke'] if smoke else []),
                         result=f'{base}/{d}/{v}/seed{s}/run.json',timeout_seconds=1200 if smoke else 10800))
    # All new branches are exercised before numerical reproducibility and the full grid.
    for v in variants:worker('Romanov',101,v,True)
    worker('Romanov',101,'fusion_reproduce')
    jobs.append(dict(id='evaluate_existing_H',script='evaluate.py',args=[],timeout_seconds=7200))
    for v in variants:
        for d in datasets:
            for s in seeds:worker(d,s,v)
    for d in datasets:
        for s in seeds:
            jobs.append(dict(id=f'scsamac__{d}__{s}',script='scsamac_worker.py',
                             args=['--method','scSAMAC','--dataset',d,'--seed',str(s)],
                             result=f'scsamac/{d}/seed{s}/run.json',timeout_seconds=10800))
    jobs.append(dict(id='evaluate_complete_grid',script='evaluate.py',args=[],timeout_seconds=10800))
    names=['worker.py','supervisor.py','evaluate.py','scsamac_worker.py','prepare.py']
    for name in names:ast.parse((HERE/name).read_text(encoding='utf8'))
    protocol=dict(id='review_experiments_20261030_v1',frozen_utc=c.utc(),user_authorization='Start the needed additional experiments now',
                  datasets=datasets,seeds=seeds,variants=variants,python=str(c.PYTHON),inputs=inputs,jobs=jobs,
                  worker_hashes={name:c.sha(HERE/name) for name in names},
                  frozen_method_manifest_sha256=c.sha(EXP/'followup/final_method_v1/manifest.json'),
                  primary_scope='Five previously evaluated primary datasets. Fixed-design reviewer follow-up; not newly unseen validation.',
                  changes=dict(expression='Same NB auxiliary loss and encoder; E_X at training and inference.',
                               expression_dense='E_X plus expression kNN shells to exactly match the corresponding NB union edge count; final shell trimmed by distance then cell indices.',
                               random='Equal-cardinality added edges absent from E_X; original independent random-edge sampler.',
                               pca_union='PCA32 of the same standardized count-encoder input; same 15-neighbor latent graph recipe.',
                               mse_union='Denoising MSE autoencoder with same 2000-256-64-32 encoder, symmetric hidden decoder, paired NB initial weights/RNG, noise=2.5, Adam AMSGrad .001, batch256, fixed epochs matching each original NB run (<=400, original early stopping included); linear output and standardized-input MSE target.',
                               fusion_noaux='Original NB graph union; auxiliary decoder retained with weight zero to preserve paired initialization.',
                               scSAMAC='Stable masked-logsumexp implementation on every dataset and all five seeds; 300 pretraining, <=2000 joint epochs; native and L2KMeans20 outputs.'),
                  graph_training=dict(updates=501,subgraph_nodes=400,sampling='uniform node induced subgraph',
                                      objective='Frozen base-base + adversarial-adversarial + .1 NB auxiliary',
                                      attack='Frozen feature-only PGD. Do not add topology PGD to these experiments.',readout='L2 KMeans20, training seed'),
                  analysis=dict(primary='dataset-equal ARI',secondary=['NMI','median','paired differences','per-class precision/recall/F1','graph homophily and conductance'],
                                intervals='10000 conditional paired seed bootstraps and hierarchical dataset/seed bootstraps; dataset clusters=5',
                                direct_H=['raw KMeans20','L2 KMeans20'],no_result_driven_tuning=True),
                  scsamac_adapter_parent=dict(path=str(repair/'run_stable.py'),sha256=c.sha(repair/'run_stable.py'),
                                              changes=['allow five datasets and five seeds','redirect outputs','pin original source directory']),
                  monitoring=dict(serial_gpu=True,poll_seconds=10,stop_on_failure=True,automatic_retry=False),
                  deferred=['Development-set expansion to 10 seeds','scMCGF and Leiden benchmarks','Full scCAGN multiseed extension','Additional graph-weighting variants'])
    c.save(HERE/'protocol.json',protocol)
    print(json.dumps({'status':'FROZEN','jobs':len(jobs),'formal_graph_runs':150,'scsamac_runs':25,'new_MSE_pretrains':25,'path':str(HERE)},ensure_ascii=False))

if __name__=='__main__':main()
