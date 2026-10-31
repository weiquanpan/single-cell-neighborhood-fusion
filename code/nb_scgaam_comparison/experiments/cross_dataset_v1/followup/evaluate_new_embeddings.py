"""Complete-grid evaluation of the independently frozen development readouts."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import *
from evaluate import metrics
import numpy as np
from scipy.stats import rankdata

def main():
    src=EXP/'followup/new_embedding_readouts_v1'
    manifest=json.loads((src/'manifest.json').read_text())
    assert sha(EXP/'followup/reanalyse_new_embeddings.py')==manifest['script_sha256']
    assert sha(EXP/'readouts.py')==manifest['readouts_sha256']
    assert json.loads((src/'queue_state.json').read_text())['status']=='COMPLETE'
    rows=[]
    for d in CFG['development']:
        ids=np.load(src/d/'cell_positions.npy');np.testing.assert_array_equal(ids,np.load(prepared(d)/'cell_positions.npy'))
        assert len(ids)==len(np.unique(ids))
        labels=np.load(ROOT/'data/evaluation_only'/d/'reference_labels.npy')[ids]
        for path in (src/d).glob('*.npz'):
            method,seed=path.stem.rsplit('__seed',1);seed=int(seed)
            assert seed in CFG['seeds'] and json.loads(path.with_suffix('.json').read_text())['status']=='COMPLETE'
            with np.load(path) as f:p=f['prediction'];assert p.shape==labels.shape and np.isfinite(p).all()
            rows.append(dict(dataset=d,method='new_'+method,seed=seed,**metrics(labels,p)))
    prior=json.loads((EXP/'evaluation/summary.json').read_text())['complete_grid']
    prior+=json.loads((EXP/'followup/control_evaluation/summary.json').read_text())['complete_grid']
    baseline={r['method']:r for r in prior};complete=[];incomplete=[]
    for m in sorted(set(r['method'] for r in rows)):
        rr=[r for r in rows if r['method']==m]
        if not all(sorted(r['seed'] for r in rr if r['dataset']==d)==sorted(CFG['seeds']) for d in CFG['development']):incomplete.append(m);continue
        means={d:float(np.mean([r['ari'] for r in rr if r['dataset']==d])) for d in CFG['development']}
        delta={d:means[d]-max(baseline['baseline_NB']['per_dataset_ari'][d],baseline['baseline_scGAAM']['per_dataset_ari'][d]) for d in means}
        complete.append(dict(method=m,mean_ari=float(np.mean(list(means.values()))),per_dataset_ari=means,delta_vs_stronger_baseline=delta,guardrail_pass=min(delta.values())>=-.05,worst_group_recall=min(r['worst_group_recall'] for r in rr)))
    pool=prior+complete;ranks={r['method']:[] for r in pool}
    for d in CFG['development']:
        for r,k in zip(pool,rankdata([-r['per_dataset_ari'][d] for r in pool])):ranks[r['method']].append(float(k))
    for r in pool:r['mean_rank']=float(np.mean(ranks[r['method']]))
    pool.sort(key=lambda r:(-r['mean_ari'],r['mean_rank']))
    dest=EXP/'followup/new_embedding_evaluation';dest.mkdir(exist_ok=True)
    save(dest/'rows.json',rows);save(dest/'summary.json',{'utc':utc(),'complete_grid':complete,'all_complete_grid':pool,'incomplete':incomplete,'heldout_content_read':False})
    lines=['# 新嵌入统一读出与共识：完整开发集评价','','| 方法 | 平均ARI | 平均排名 | ARI护栏 | 最差群体召回 |','|---|---:|---:|---|---:|']
    for r in sorted(complete,key=lambda r:-r['mean_ari']):lines.append(f"| {r['method']} | {r['mean_ari']:.6f} | {r['mean_rank']:.2f} | {r['guardrail_pass']} | {r['worst_group_recall']:.6f} |")
    lines+=['','不完整、排除排名：'+', '.join(incomplete),'','R6统一L2+KMeans20采用最终800步嵌入；NB-GCL为500轮且另有NB-H预训练成本，不可直接作等预算架构因果结论。共识三seed是同一集成的读出扰动，不是独立训练。共识PCA三尺度权重及按索引打破并列沿用旧实现；不能把失败推广到所有集成。','', '## 每数据群体指标','| 方法 | 数据 | ARI ± SD | 最小召回 | 最差召回 | 碎裂度 |','|---|---|---:|---:|---:|---:|']
    for r in complete:
        for d in CFG['development']:
            rr=[x for x in rows if x['method']==r['method'] and x['dataset']==d]
            vals=[np.mean([x[k] for x in rr]) for k in ['smallest_group_recall','worst_group_recall','mean_fragmentation']]
            lines.append(f"| {r['method']} | {d} | {r['per_dataset_ari'][d]:.6f} ± {np.std([x['ari'] for x in rr],ddof=1):.6f} | "+' | '.join(f'{v:.6f}' for v in vals)+' |')
    lines+=['','复现：env313 Python依次执行 followup/audit.py、followup/evaluate_controls.py、followup/evaluate_new_embeddings.py。留出内容未读取。']
    (dest/'REPORT.md').write_text('\n'.join(lines),encoding='utf8')
    print(json.dumps(complete));print('INCOMPLETE',incomplete)

if __name__=='__main__':main()
