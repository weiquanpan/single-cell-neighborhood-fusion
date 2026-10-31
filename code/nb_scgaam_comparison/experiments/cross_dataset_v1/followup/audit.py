"""Read-only experiment audit and evaluation refresh; no training source edits."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import *
import numpy as np
import evaluate

def main():
    frozen=json.loads((EXP/'execution_manifest.json').read_text())
    mismatches=[n for n,h in frozen['files'].items() if sha(EXP/n)!=h]
    assert not mismatches, mismatches
    evaluate.main()
    failures=[];valid=0
    for path in (EXP/'stage0').rglob('*.npz'):
        with np.load(path) as f:
            y=f['prediction'];assert y.ndim==1 and np.isfinite(y).all()
        ids=np.load(path.parent/'cell_positions.npy')
        assert len(y)==len(ids) and len(np.unique(ids))==len(ids)
        valid+=1
    for path in (EXP/'stage0').rglob('*.json'):
        item=json.loads(path.read_text())
        if item.get('status')=='FAILED':failures.append({'path':str(path.relative_to(EXP)),'error':item.get('error')})
    jobs=[]
    for path in (EXP/'training').glob('*/*/seed*/main/run.json'):
        item=json.loads(path.read_text());jobs.append({'path':str(path.relative_to(EXP)),**item})
        if item['status']=='COMPLETE':
            with np.load(path.parent/'outputs.npz') as f:
                assert np.isfinite(f['embedding']).all() and len(f['prediction'])==len(f['embedding'])
            ids=np.load(path.parent/'cell_positions.npy')
            np.testing.assert_array_equal(ids,np.load(prepared(item['dataset'])/'cell_positions.npy'))
    held=json.loads((EXP/'heldout_manifest.json').read_text())
    assert set(held['files'])==set(CFG['heldout_names'])
    for name,item in held['files'].items():assert sha(EXP/'heldout_sealed'/f'{name}.h5')==item['sha256']
    rows=json.loads((EXP/'evaluation/rows.json').read_text())
    summary=json.loads((EXP/'evaluation/summary.json').read_text())
    selected=['baseline_NB','baseline_scGAAM','scGAAM__kmeans','scGAAM__gmm_full','scGAAM__gmm_diag','scGAAM__spectral','consensus3__overmerge2','consensus3__spectral']
    bymethod={r['method']:r for r in summary['complete_grid']}
    base=bymethod['baseline_scGAAM']
    lines=['# 实验跟进与阶段0判断','',f'核查时间（UTC）：{utc()}','', '## Material Passport','- Mode: run/validate；范围：开发集结果与运行完整性。','- 留出六份文件仅校验哈希，未读取表达或标签。','- 仅作描述性比较，没有显著性或总体泛化结论。','', '## 运行核查',f'- 冻结源码哈希全部一致；核查 {valid} 个读出输出的长度、有限值和细胞ID唯一性。',f"- 正式训练完成 {sum(j['status']=='COMPLETE' for j in jobs)}/129；运行中 {sum(j['status']=='RUNNING' for j in jobs)}；失败 {sum(j['status']=='FAILED' for j in jobs)}。",f'- 阶段0记录 {len(failures)} 个读出失败，均列在审计 JSON；失败组合不进入完整网格排名。','', '## 四数据 × 三seed：ARI均值','| 方法 | Worm | Limb | Tosches | Bach | 跨数据均值 | 平均排名 | 护栏 |','|---|---:|---:|---:|---:|---:|---:|---|']
    for method in selected:
        r=bymethod[method];values=' | '.join(f'{r["per_dataset_ari"][d]:.4f}' for d in CFG['development'])
        lines.append(f'| {method} | {values} | {r["mean_ari"]:.4f} | {r["mean_rank"]:.2f} | {r["guardrail_pass"]} |')
    lines+=['','## 判断','- scGAAM + KMeans20：冻结的统一读出；ARI提升但群体指标受损，不能宣称全面有效。均值提高 '+f"{bymethod['scGAAM__kmeans']['mean_ari']-base['mean_ari']:.4f}"+'，四数据护栏通过；不是每份数据都提高，也不能称为已经验证的跨新数据收益。KMeans初始化次数和随机种子策略同时改变，当前不能将收益单独归因于 n_init。', '- scGAAM + 全协方差GMM：均值较原读出提高、护栏通过，但弱于KMeans20，不作为首选。对角GMM收益更小。','- scGAAM + 谱聚类：Worm下降超过0.05，护栏失败。尽管它在跨四种表示的额外平均中最高，也不能据此绕过主干护栏。','- 三视图共识与过聚类合并：本轮实现均未通过护栏，不判为有效。PCA贡献三个K尺度而另外视图各一个，且共关联top-k并列按细胞索引打破；这些实现选择限制结论，不能否定所有共识方法。','- 历史R6和四视图共识未满足完整网格，不能作跨数据有效性判断。','- 原队列129项已完成；NB-GCL、图融合、两跳token及R6结果详见各完整组报告和R6_ANCHORED_COMPLETE.md。补充控制与完整方法冻结仍待完成。','', '## 群体与稳定性护栏','| 数据 | 原scGAAM ARI样本SD | KMeans20 ARI样本SD | 原scGAAM最差召回均值 | KMeans20最差召回均值 | 原最小群体召回均值 | 新最小群体召回均值 |','|---|---:|---:|---:|---:|---:|---:|']
    for d in CFG['development']:
        old=[r for r in rows if r['dataset']==d and r['method']=='baseline_scGAAM'];new=[r for r in rows if r['dataset']==d and r['method']=='scGAAM__kmeans']
        vals=[base['per_dataset_sample_sd'][d],bymethod['scGAAM__kmeans']['per_dataset_sample_sd'][d]]
        for key in ['worst_group_recall','smallest_group_recall']:vals += [float(np.mean([r[key] for r in group])) for group in [old,new]]
        lines.append('| '+d+' | '+' | '.join(f'{v:.4f}' for v in vals)+' |')
    lines+=['','最差群体召回存在0，ARI通过不等于稀有群体问题已解决。召回采用一对一Hungarian匹配，另见rows.json中的所有群体覆盖率以区分匹配遗漏和真实碎裂。','', '## 解释边界','- 统一L2归一化改变NB/PCA的几何，当前结果不能直接推翻此前未归一化NB-H KMeans的观察。','- 400步无DEC端点与800步DEC端点预算不等，只能作为诊断，不能单独证明DEC净效应。','- 仅4个开发数据、每数据3seed；不计算伪重复显著性。选择偏差、跨数据外推、效应大小、多重比较、指标替换、预算混杂、失败缺失、类不平衡、集成seed非独立、均值遮蔽及因果归因均已标记，尚无统计检验结论。']
    (EXP/'followup/REPORT.md').write_text('\n'.join(lines),encoding='utf8')
    save(EXP/'followup/audit.json',{'utc':utc(),'source_hashes_match':True,'valid_stage0_outputs':valid,'readout_failures':failures,'training_jobs':jobs,'sealed_holdouts':len(held['files']),'heldout_content_read':False})

if __name__=='__main__':main()
