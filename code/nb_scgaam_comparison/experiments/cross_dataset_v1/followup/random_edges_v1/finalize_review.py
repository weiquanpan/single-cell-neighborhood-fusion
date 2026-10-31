"""Add reviewed appendix and a verified, non-overwriting supplement archive."""
from pathlib import Path
import sys,json,hashlib,tarfile,shutil,datetime
import numpy as np
HERE=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False),encoding='utf8')
def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def main():
    D=Path(read(HERE/'evaluation_latest.json')['path'])
    for manifest,relative in [('original_sources.json',False),('input_manifest.json',False),('execution_sources.json',True),('evaluation_source_manifest.json',True)]:
        for p,h in read(HERE/manifest).items():assert digest(HERE/p if relative else Path(p))==h,(manifest,p)
    assert not (HERE/'queue.lock').exists()
    state=read(HERE/'state.json');assert state['status']=='COMPLETE_PENDING_REVIEW' and len(state['completed'])==12
    for p in state['completed']:
        p=Path(p);m=read(p/'run.json');assert m['status']=='COMPLETE' and m['checkpoint_reload_exact'] and not m['labels_read']
        assert m['random_new_edges']==m['fusion_new_edges']
        with np.load(p/'graphs.npz') as f:
            for k in f.files:assert len(np.unique(f[k],axis=0))==len(f[k]),(p,k)
    gs=read(D/'graph_statistics.json');rows=read(D/'development_rows.json');summary=read(D/'summary.json');ds=list(summary['development'][0]['per_dataset'])
    lines=['# 附录：图结构与群体指标（审核版）','','以下数值为三个seed的均值；图不含自环。同标签率仅为训练后的描述性统计。随机控制等边数，但不等节点度。','','| 数据 | 表达/潜在边Jaccard | 融合新增边占比 | 融合新增边同标签率 | 随机新增边同标签率 |','|---|---:|---:|---:|---:|']
    for d in ds:
        rr=[r for r in gs if r['dataset']==d]
        v=[np.mean([r['jaccard'] for r in rr]),np.mean([r['new_fraction_in_fusion'] for r in rr]),np.mean([r['graphs']['fusion_added']['same_label_fraction'] for r in rr]),np.mean([r['graphs']['random_added']['same_label_fraction'] for r in rr])]
        lines.append('| '+d+' | '+' | '.join(f'{x:.4f}' for x in v)+' |')
    lines+=['','| 数据 | 图 | 边数均值 | 度均值 | 度SD均值 | 最大度均值 | 连通分量均值 |','|---|---|---:|---:|---:|---:|---:|']
    for d in ds:
        for g in ['expression','latent','fusion_union','random_union']:
            rr=[r['graphs'][g] for r in gs if r['dataset']==d]
            v=[np.mean([r[k] for r in rr]) for k in ['edges','degree_mean','degree_sd']]+[np.mean([r['degree_quantiles'][-1] for r in rr]),np.mean([r['components'] for r in rr])]
            lines.append('| '+d+' | '+g+' | '+' | '.join(f'{x:.3f}' for x in v)+' |')
    lines+=['','| 数据 | 方法 | 最小群体召回 | 最差群体召回 | 碎裂度（低为好） |','|---|---|---:|---:|---:|']
    for d in ds:
        for m in ['Expression-consistent','Random-equal-edges','Fusion']:
            rr=[r for r in rows if r['dataset']==d and r['method']==m]
            v=[np.mean([r[k] for r in rr]) for k in ['smallest_group_recall','worst_group_recall','mean_fragmentation']]
            lines.append('| '+d+' | '+m+' | '+' | '.join(f'{x:.4f}' for x in v)+' |')
    lines+=['','解释：融合新增边的同标签率明显高于随机边，且四数据平均ARI均高于随机控制；这反对“只要增加同样多的边就能得到同样收益”的解释。但随机边还改变度分布与跨群连接，不能由此单独证明所有语义互补或因果机制。','相对表达图一致性控制，Fusion平均ARI仅增加0.008414，Tosches降低0.030700；不能把相对随机图的0.306503差值当作方法相对合理基线的提升。','Tosches最小/最差群体召回为0，仍不支持普遍改善稀有群体。','统计单位是数据集与seed，不将边数量当作独立样本计算显著性。']
    (D/'APPENDIX_GRAPH_GROUPS_REVIEWED.md').write_text('\n'.join(lines),encoding='utf8')
    # Reviewed addendum retains the automatic report unchanged.
    (D/'PAPER_STORY_REVIEWED.md').write_text('''# 小论文论证与正文安排（审核版）

核心问题：在单细胞图对比学习中，计数潜在空间提供的邻域关系，能否补充表达图并改善跨数据平均聚类质量？

方法贡献：采用计数潜在邻域与表达邻域的静态并集，在训练和推理阶段使用一致图，结合原图对比/对抗学习及辅助计数重建。该工作是具体方法设计和受控经验验证，不以首次图融合、动态融合或全新计数模型为贡献。

正文故事建议：先说明单一邻域图的局限和构造方法，再展示五数据五seed的平均收益，随后用表达一致性控制与等量随机加边控制解释“增加什么边”与“增加多少边”的区别，最后给出失败边界。

关键证据：四开发数据×三seed，表达一致性控制0.8180，等量随机边0.5199，Fusion0.8264。Fusion比随机边高0.3065，但相对合理的表达图控制仅高0.0084，且Tosches下降0.0307。新增潜在边同标签率92.1%–98.8%，随机边13.9%–23.0%，是事后描述，不能替代度匹配机制验证。

主评价：已有完整五seed的五数据等权ARI为Fusion0.6953、scGAAM原生0.6821、scAGCL原生0.6332；统一读出分别0.6953、0.6708、0.6692。统一读出平均排名Fusion与scAGCL同为1.8。不能称所有数据最优或统计显著。

外部补充：scCAGN/scSAMAC按固定单seed列出；scSAMAC多个成功数据集明显更强，Spleen数值失败不应让其成功结果消失。保留seed数量不对称和作者实现适配、训练预算差异。Muraro单列，校正小数输入与排除5 unclear敏感性沿用原归档。

正文最小配置：方法图；五seed主结果表；单seed外部扩展表；三组核心控制表及一张图结构统计图。附录：逐seed、原生/统一读出、群体召回/碎裂度、图度分布/连通性、训练预算及耗时、失败/中断、Muraro敏感性、复现命令。

现有证据足以开始写经验型小论文，无需为了这一定位继续扩大训练。更强的机制论断仍需度匹配随机控制及每个训练seed多个随机图；未在本轮执行，不声称已排除。额外外部多seed仍为未来附件待办。新颖性需要在写作时与最接近工作逐项核对，不用更换叙述替代实质差异。

原独立NB不列作本篇比较方法，但来源、预训练400轮成本和既有失败护栏审计均保留。四开发数据上的补充为事后验证，不是新的未见数据确认；冻结融合方法和读出没有改变。
''',encoding='utf8')
    reused=read(D/'existing_multiseed_rows.json');paired=[]
    for d in ['Romanov','Adam','Quake_10x_Bladder','Quake_10x_Spleen','Quake_Smart-seq2_Lung']:
        for s in [101,202,303,404,505]:
            rr={r['method']:r for r in reused if r['dataset']==d and r['seed']==s}
            paired.append(dict(dataset=d,seed=s,**{m:rr['Fusion']['ari']-rr[m]['ari'] for m in rr if m!='Fusion'}))
    write(D/'existing_multiseed_paired_differences.json',paired)
    now=datetime.datetime.now(datetime.timezone.utc);stamp=now.strftime('%Y%m%dT%H%M%SZ')
    write(HERE/'review_complete.json',dict(utc=now.isoformat(),training_jobs=12,all_complete=True,input_source_hashes_match=True,graph_counts_unique=True,labels_training_blocked=True,readout_evaluator_passed=True,actual_optimizer_updates_per_job=501,training_wall_minutes=sum(r['random_training_seconds'] for r in read(D/'runtime_budget.json'))/60,degree_matched=False,reviewed_reports=['PAPER_STORY_REVIEWED.md','APPENDIX_GRAPH_GROUPS_REVIEWED.md'],historical_run_variant_note='run.json variant retains inherited NB-GCL-fusion CLI switch; separate random_equal_edges path, protocol, graph artifacts and source diff identify random control. Frozen Fusion unchanged.'))
    dest=Path('F:/cluster-noise/paper_packages')/('fusion_minimal_supplement_'+stamp);dest.mkdir(exist_ok=False)
    shutil.copytree(D,dest/'paper')
    references=[Path('F:/cluster-noise/paper_packages/fusion_paper_v1_20261017/research_data_code_results.tar'),Path('F:/cluster-noise/paper_packages/fusion_external_single_seed_partial_20261022T010827Z/external_baselines_partial_data_code_results.tar')]
    write(dest/'BASE_PACKAGE_REFERENCES.json',[dict(path=str(p),bytes=p.stat().st_size,unchanged=True) for p in references])
    manifest={}
    archive=dest/'supplement_code_results_audit.tar'
    files=sorted(p for p in HERE.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    with tarfile.open(archive,'w') as tar:
        for p in files:
            name=p.relative_to(HERE).as_posix();manifest[name]=digest(p);tar.add(p,arcname=name,recursive=False)
    with tarfile.open(archive,'r') as tar:
        assert len(tar.getmembers())==len(manifest)
        for item in tar:
            h=hashlib.sha256();f=tar.extractfile(item)
            for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
            assert h.hexdigest()==manifest[item.name],item.name
    write(dest/'SHA256_MANIFEST.json',manifest)
    write(dest/'PACKAGE_COMPLETE.json',dict(archive=str(archive),sha256=digest(archive),bytes=archive.stat().st_size,files=len(files),verified=True,new_training_jobs=12,all_new_training_successful=True,external_prior_failure_retained=True))
    (dest/'README.md').write_text('本增量包含12次等量随机加边控制、源码审计、图统计与新版小论文材料。paper/PAPER_STORY_REVIEWED.md及APPENDIX_GRAPH_GROUPS_REVIEWED.md为审核结论；RESULTS_TABLES.md为完整结果表。共享数据/原Fusion及已有外部训练保存在BASE_PACKAGE_REFERENCES.json指向的两个既有包，均未覆盖。补充为事后开发集检验，非度匹配；全部12次成功不代表历史外部方法全成功。',encoding='utf8')
    write(HERE/'package_complete.json',dict(path=str(dest),verified=True,utc=now.isoformat()))
    print(json.dumps(dict(package=str(dest),files=len(files),bytes=archive.stat().st_size)))
if __name__=='__main__':main()
