"""User-authorized interim evaluation on five holdouts, not six-dataset validation."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import *
from evaluate import metrics
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from scipy.stats import rankdata
from threadpoolctl import threadpool_limits

def main():
    H=EXP/'heldout_muraro_corrected_v1';F=json.loads((EXP/'followup/final_method_v1/manifest.json').read_text())
    exclude='--exclude-unclear' in sys.argv
    datasets=['Muraro'];seeds=F['seeds'];rows=[]
    methods=['NB','scGAAM_original','scGAAM_KMeans20','NB-GCL-fusion']
    for d in datasets:
        ids=np.load(H/'prepared'/d/'full/cell_positions.npy');assert len(ids)==len(np.unique(ids))
        labels=np.load(H/'evaluation_only'/d/'reference_labels.npy')[ids]
        mask=np.ones(len(ids),dtype=bool)
        if exclude:
            import h5py
            with h5py.File(EXP/'heldout_sealed/Muraro.h5','r') as hf:cell_names=np.array([x.decode() for x in hf['obs_names'][...]])[ids]
            unclear=json.loads((H/'protocol_amendment.json').read_text())['unclear_cell_ids']
            mask=~np.isin(cell_names,unclear);assert int((~mask).sum())==5
        for seed in seeds:
            for method in ['NB','scGAAM','NB-GCL-fusion']:
                p=(H/'candidate_workspace/training'/d/method/f'seed{seed}'/'main' if method=='NB-GCL-fusion' else H/'runs'/d/'full'/method/f'main_seed{seed}'/'attempt_001')
                meta=json.loads((p/'run.json').read_text());assert meta['status'] in ['COMPLETE','COMPLETED'],str(p)
                np.testing.assert_array_equal(ids,np.load(p/'cell_positions.npy'))
                with np.load(p/'outputs.npz') as f:
                    z=f['embedding'];pred=f['prediction'];assert np.isfinite(z).all() and np.isfinite(pred).all()
                name='scGAAM_original' if method=='scGAAM' else method
                rows.append(dict(dataset=d,seed=seed,method=name,source=str(p),**metrics(labels[mask],pred[mask])))
                if method=='scGAAM':
                    k=json.loads((H/'prepared'/d/'full/ready.json').read_text())['known_k']
                    with threadpool_limits(limits=8):pred=KMeans(n_clusters=k,n_init=20,random_state=seed).fit_predict(normalize(z))
                    rows.append(dict(dataset=d,seed=seed,method='scGAAM_KMeans20',source=str(p),**metrics(labels[mask],pred[mask])))
    per={};group={}
    for m in methods:
        per[m]={};group[m]={}
        for d in datasets:
            rr=[r for r in rows if r['method']==m and r['dataset']==d];assert sorted(r['seed'] for r in rr)==seeds
            per[m][d]=float(np.mean([r['ari'] for r in rr]))
            group[m][d]={k:float(np.mean([r[k] for r in rr])) for k in ['smallest_group_recall','worst_group_recall','mean_fragmentation']}
            group[m][d]['ari_sd']=float(np.std([r['ari'] for r in rr],ddof=1))
    ranks={m:[] for m in methods}
    for d in datasets:
        for m,r in zip(methods,rankdata([-per[m][d] for m in methods])):ranks[m].append(float(r))
    summary=[]
    for m in methods:
        delta={d:per[m][d]-max(per['NB'][d],per['scGAAM_original'][d]) for d in datasets}
        summary.append(dict(method=m,mean_ari=float(np.mean(list(per[m].values()))),mean_rank=float(np.mean(ranks[m])),per_dataset_ari=per[m],delta_vs_stronger_baseline=delta,guardrail_pass=min(delta.values())>=-.05,group_metrics=group[m]))
    out=H/('evaluation_exclude_unclear' if exclude else 'evaluation_primary');out.mkdir(exist_ok=True);save(out/'rows.json',rows);save(out/'summary.json',{'utc':utc(),'scope':'Muraro corrected fractional-expression supplementary evaluation; extended reconstruction objective, not strict integer-count likelihood','datasets':datasets,'exclude_five_unclear':exclude,'evaluated_cells':int(mask.sum()),'seeds':seeds,'summary':summary,'frozen_method_sha256':sha(EXP/'followup/final_method_v1/manifest.json')})
    lines=['# Muraro校正表达补充评价','','独立修订协议；作者校正小数表达不取整，损失为扩展重构目标。五seed全齐；方法不变。敏感性分支仅在评价时排除5个unclear细胞，不重训、不改变K。','', '| 方法 | 数据集等权ARI | 平均排名 | Muraro ARI护栏 |','|---|---:|---:|---|']
    for r in summary:lines.append(f"| {r['method']} | {r['mean_ari']:.6f} | {r['mean_rank']:.2f} | {r['guardrail_pass']} |")
    lines+=['','| 数据 | NB | 原scGAAM | scGAAM KMeans20 | 融合候选 | 候选相对强基线 |','|---|---:|---:|---:|---:|---:|']
    for d in datasets:lines.append('| '+d+' | '+' | '.join(f'{per[m][d]:.6f}' for m in methods)+f" | {per['NB-GCL-fusion'][d]-max(per['NB'][d],per['scGAAM_original'][d]):+.6f} |")
    lines+=['','## 群体指标与seed波动','| 数据 | 方法 | ARI样本SD | 最小群体召回 | 最差群体召回 | 碎裂度 |','|---|---|---:|---:|---:|---:|']
    for d in datasets:
        for m in methods:
            g=group[m][d];lines.append('| '+d+' | '+m+' | '+' | '.join(f'{g[k]:.6f}' for k in ['ari_sd','smallest_group_recall','worst_group_recall','mean_fragmentation'])+' |')
    lines+=['','召回为Hungarian一对一匹配；碎裂度为各真实群体1−最大单预测簇覆盖率的平均，越低越好。完整逐群体数据在rows.json。原scGAAM为作者读出，KMeans20为与候选一致的冻结读出。融合候选有额外NB-H预训练成本，不能宣称等预算优势。已查看五份主评价及本补充结果，任何后续据此调参都须标为新开发，不能再次视作未见验证。','复现：env313 Python执行followup/audit_muraro_supplement.py和followup/evaluate_muraro_supplement.py（敏感性加 --exclude-unclear）。']
    (out/'REPORT.md').write_text('\n'.join(lines),encoding='utf8');print(json.dumps(summary))

if __name__=='__main__':main()
