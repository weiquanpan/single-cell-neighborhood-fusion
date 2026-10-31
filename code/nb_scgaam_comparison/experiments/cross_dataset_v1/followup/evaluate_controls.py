"""Independent development-only control audit; never modifies training outputs."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import *
from evaluate import metrics
import numpy as np
from scipy.stats import rankdata

def main():
    manifest=json.loads((EXP/'followup/controls_v1/manifest.json').read_text())
    for n,h in manifest['original_sources'].items(): assert sha(EXP/n)==h,n
    for n,h in manifest['control_sources'].items(): assert sha(EXP/'followup/controls_v1'/n)==h,n
    rows=[]; jobs=[]
    for p in (EXP/'control_training').glob('*/*/seed*/main/run.json'):
        meta=json.loads(p.read_text());jobs.append({'path':str(p),**meta})
        if meta['status']!='COMPLETE':continue
        d=meta['dataset'];assert d in CFG['development']
        ids=np.load(p.parent/'cell_positions.npy')
        np.testing.assert_array_equal(ids,np.load(prepared(d)/'cell_positions.npy'))
        assert len(np.unique(ids))==len(ids)
        y=np.load(ROOT/'data/evaluation_only'/d/'reference_labels.npy')[ids]
        with np.load(p.parent/'outputs.npz') as f:
            assert len(f['embedding'])==len(ids) and np.isfinite(f['embedding']).all()
            pred=f['prediction'];assert pred.ndim==1 and len(pred)==len(y) and np.isfinite(pred).all()
            rows.append(dict(dataset=d,seed=meta['seed'],method='control_'+p.parents[2].name,source=str(p),**metrics(y,pred)))
    summary=json.loads((EXP/'evaluation/summary.json').read_text())['complete_grid']
    old={r['method']:r for r in summary}; complete=[]
    for m in sorted(set(r['method'] for r in rows)):
        rr=[r for r in rows if r['method']==m]
        if not all(sorted(r['seed'] for r in rr if r['dataset']==d)==sorted(CFG['seeds']) for d in CFG['development']):continue
        per={d:float(np.mean([r['ari'] for r in rr if r['dataset']==d])) for d in CFG['development']}
        delta={d:per[d]-max(old['baseline_NB']['per_dataset_ari'][d],old['baseline_scGAAM']['per_dataset_ari'][d]) for d in per}
        complete.append(dict(method=m,mean_ari=float(np.mean(list(per.values()))),per_dataset_ari=per,delta_vs_stronger_baseline=delta,guardrail_pass=min(delta.values())>=-.05))
    pool=summary+complete
    ranks={r['method']:[] for r in pool}
    for d in CFG['development']:
        for r,k in zip(pool,rankdata([-r['per_dataset_ari'][d] for r in pool])):ranks[r['method']].append(float(k))
    for r in complete:r['mean_rank']=float(np.mean(ranks[r['method']]))
    dest=EXP/'followup/control_evaluation';dest.mkdir(exist_ok=True)
    save(dest/'rows.json',rows);save(dest/'summary.json',{'utc':utc(),'complete_grid':complete,'jobs':jobs,'source_hashes_match':True,'heldout_content_read':False,'ranking_pool':[r['method'] for r in pool]})
    lines=['# 补充技术控制：开发集审计','','仅完整四数据×三seed进入判断；排名在原完整候选加完整控制的共同池计算。原训练输出及评价不改写。','']
    for r in complete:
        m=r['method'];lines += [f"## {m}",f"等权平均ARI {r['mean_ari']:.6f}；平均排名 {r['mean_rank']:.2f}；ARI护栏通过：{r['guardrail_pass']}。",'','| 数据 | ARI均值 ± SD | 强基线差 | 最小群体召回 | 最差群体召回 | 碎裂度 |','|---|---:|---:|---:|---:|---:|']
        for d in CFG['development']:
            rr=[x for x in rows if x['method']==m and x['dataset']==d]
            vals=[np.mean([x[k] for x in rr]) for k in ['smallest_group_recall','worst_group_recall','mean_fragmentation']]
            lines.append(f"| {d} | {r['per_dataset_ari'][d]:.6f} ± {np.std([x['ari'] for x in rr],ddof=1):.6f} | {r['delta_vs_stronger_baseline'][d]:+.6f} | "+' | '.join(f'{v:.6f}' for v in vals)+' |')
        if 'expression-union' in m:
            lines += ['',f"统一表达图控制相对原NB-GCL平均ARI差 {r['mean_ari']-old['trained_NB-GCL']['mean_ari']:+.6f}；fusion相对该控制差 {old['trained_NB-GCL-fusion']['mean_ari']-r['mean_ari']:+.6f}。两项均使用冻结L2+KMeans20、同训练seed和500轮预算。",'这一区分修正了此前将图一致性与H边打包的比较，但仅为事后开发集控制；不等于留出验证或群体全面改善。']
        else:
            lines += ['',f"相对原both均值差 {r['mean_ari']-old['trained_both']['mean_ari']:+.6f}；far相对本控制差 {old['trained_far']['mean_ari']-r['mean_ari']:+.6f}。同400+400预算；DEC读出与主干L2读出不混作同一算法。"]
    lines += ['','## 待办','不完整控制不判有效。继续完成补充控制、新嵌入共识检查、完整方法冻结，再做六留出数据×五seed。尚未访问留出内容。','复现：项目env313 Python执行 followup/audit.py，再执行 followup/evaluate_controls.py。']
    (dest/'REPORT.md').write_text('\n'.join(lines),encoding='utf8')
    print(json.dumps(complete))

if __name__=='__main__':main()
