"""Offline-only audit and manuscript tables. Never imported by training."""
from pathlib import Path
import sys,json,csv,shutil
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from common import *
from evaluate import metrics
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.stats import rankdata
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits
HERE=Path(__file__).resolve().parent
B=ROOT/'experiments/external_baselines_v1'
MAIN=['Romanov','Adam','Quake_10x_Bladder','Quake_10x_Spleen','Quake_Smart-seq2_Lung']

def load(p):return json.loads(Path(p).read_text(encoding='utf8'))

def graph_stats(edges,n,y):
    edges=np.asarray(edges,dtype=np.int64).reshape(-1,2)
    degree=np.bincount(edges.ravel(),minlength=n)
    a=sparse.coo_matrix((np.ones(len(edges)),(edges[:,0],edges[:,1])),shape=(n,n)).tocsr()
    return dict(edges=len(edges),degree_mean=float(degree.mean()),degree_sd=float(degree.std()),degree_quantiles=np.quantile(degree,[0,.25,.5,.75,.95,1]).tolist(),components=int(connected_components(a,directed=False,return_labels=False)),same_label_fraction=float(np.mean(y[edges[:,0]]==y[edges[:,1]])) if len(edges) else None)

def aggregate(rows,methods,ds,seeds):
    result=[]
    for m in methods:
        per={}
        for d in ds:
            rr=[r for r in rows if r['method']==m and r['dataset']==d]
            assert sorted(r['seed'] for r in rr)==seeds,(m,d,'incomplete or duplicate seeds')
            per[d]={k:float(np.mean([r[k] for r in rr])) for k in ['ari','nmi','smallest_group_recall','worst_group_recall','mean_fragmentation']}
            per[d]['ari_sd']=float(np.std([r['ari'] for r in rr],ddof=1)) if len(rr)>1 else None
        result.append(dict(method=m,per_dataset=per,mean_ari=float(np.mean([v['ari'] for v in per.values()]))))
    ranks={m:[] for m in methods}
    for d in ds:
        for r,k in zip(result,rankdata([-r['per_dataset'][d]['ari'] for r in result])):ranks[r['method']].append(float(k))
    for r in result:r['mean_rank']=float(np.mean(ranks[r['method']]))
    return result

def table(summary,ds):
    lines=['| 数据 | '+' | '.join(r['method'] for r in summary)+' |','|---|'+'---:|'*len(summary)]
    for d in ds:
        vals=[]
        for r in summary:
            v=r['per_dataset'][d];vals.append(f"{v['ari']:.4f}"+(f" ± {v['ari_sd']:.4f}" if v['ari_sd'] is not None else ' (SD NA)'))
        lines.append('| '+d+' | '+' | '.join(vals)+' |')
    lines.extend(['| 等权平均 ARI | '+' | '.join(f"{r['mean_ari']:.4f}" for r in summary)+' |','| 平均排名（本表共同池） | '+' | '.join(f"{r['mean_rank']:.2f}" for r in summary)+' |'])
    return lines

def main():
    for p,h in load(HERE/'original_sources.json').items():assert sha(p)==h,p
    for p,h in load(HERE/'execution_sources.json').items():assert sha(HERE/p)==h,p
    stamp=utc().replace(':','').replace('-','').replace('+0000','Z')
    dest=HERE/'evaluation'/stamp;dest.mkdir(parents=True,exist_ok=False)
    rows=[];graphs=[];timing=[];source_hashes={}
    for d in CFG['development']:
        for seed in CFG['seeds']:
            p=HERE/'runs'/d/'random_equal_edges'/f'seed{seed}'/'main';meta=load(p/'run.json')
            assert meta['status']=='COMPLETE' and meta['labels_read']==False
            assert meta['optimizer_updates']==501
            history=[json.loads(x) for x in (p/'history.jsonl').read_text().splitlines()]
            assert [x['epoch'] for x in history]==list(range(1,502))
            ids=np.load(p/'cell_positions.npy');np.testing.assert_array_equal(ids,np.load(prepared(d)/'cell_positions.npy'));assert len(np.unique(ids))==len(ids)
            y=np.load(ROOT/'data/evaluation_only'/d/'reference_labels.npy')[ids]
            for method,folder in [('Random-equal-edges',p),('Fusion',EXP/'training'/d/'NB-GCL-fusion'/f'seed{seed}'/'main'),('Expression-consistent',EXP/'control_training'/d/'NB-GCL-expression-union-control'/f'seed{seed}'/'main')]:
                np.testing.assert_array_equal(ids,np.load(folder/'cell_positions.npy'))
                with np.load(folder/'outputs.npz') as f:
                    z=f['embedding'];pred=f['prediction'];assert len(z)==len(ids) and np.isfinite(z).all()
                    assert pred.shape==(len(ids),) and np.isfinite(pred).all()
                    with threadpool_limits(limits=8):check=KMeans(n_clusters=CFG['development'][d],n_init=20,random_state=seed).fit_predict(normalize(z))
                    np.testing.assert_array_equal(pred,check)
                rows.append(dict(dataset=d,seed=seed,method=method,source=str(folder),**metrics(y,pred)))
                source_hashes[str(folder/'outputs.npz')]=sha(folder/'outputs.npz')
            with np.load(p/'graphs.npz') as f:
                sets={k:set(map(tuple,f[k])) for k in f.files}
            e,h,add,rand=(sets[k] for k in ['expression','latent','fusion_added','random_added'])
            assert add==h-e and len(add)==len(rand) and not(e&rand)
            assert all(0<=u<v<len(ids) for pairs in sets.values() for u,v in pairs)
            graphs.append(dict(dataset=d,seed=seed,overlap=len(e&h),jaccard=len(e&h)/len(e|h),new_fraction_in_fusion=len(add)/len(e|h),random_overlap_fusion_added=len(rand&add),degree_matched=False,graphs={k:graph_stats(sorted(v),len(ids),y) for k,v in dict(expression=e,latent=h,fusion_added=add,random_added=rand,fusion_union=e|h,random_union=e|rand).items()}))
            timing.append(dict(dataset=d,seed=seed,random_training_seconds=meta['seconds'],fusion_training_seconds=load(EXP/'training'/d/'NB-GCL-fusion'/f'seed{seed}'/'main/run.json')['seconds'],optimizer_updates=501,pretraining_cached_reused=True,pretraining_budget_epochs=400))
    dev=aggregate(rows,['Expression-consistent','Random-equal-edges','Fusion'],list(CFG['development']),CFG['seeds'])
    paired=[]
    for d in CFG['development']:
        for seed in CFG['seeds']:
            rr={r['method']:r for r in rows if r['dataset']==d and r['seed']==seed}
            paired.append(dict(dataset=d,seed=seed,**{f'fusion_minus_random_{k}':rr['Fusion'][k]-rr['Random-equal-edges'][k] for k in ['ari','smallest_group_recall','worst_group_recall','mean_fragmentation']}))
    # Preserve original guardrail audit separately; independent NB is not a manuscript comparator.
    old=load(EXP/'evaluation/summary.json')['complete_grid'];old={r['method']:r for r in old}
    guards={r['method']:{d:r['per_dataset'][d]['ari']-max(old['baseline_NB']['per_dataset_ari'][d],old['baseline_scGAAM']['per_dataset_ari'][d]) for d in CFG['development']} for r in dev}
    save(dest/'guardrail_audit.json',guards)
    # Reuse all available complete five-seed main evaluation rows; no seed selection.
    source=B/'evaluation/20261018T192153290667Z/rows.json'
    reused=[r for r in load(source) if r['dataset'] in MAIN and r['method'] in ['Fusion','scGAAM_native','scGAAM_KMeans20','scAGCL_native','scAGCL_KMeans20']]
    # Validate cell alignment, output hash and native predictions against stored row metrics.
    for r in reused:
        folder=Path(r['source']);folder=folder if folder.is_dir() else folder.parent
        ids=np.load(folder/'cell_positions.npy');assert len(np.unique(ids))==len(ids)
        canonical=next(x for x in reused if x['dataset']==r['dataset'] and x['method']=='Fusion' and x['seed']==r['seed'])
        np.testing.assert_array_equal(ids,np.load(Path(canonical['source'])/'cell_positions.npy'))
        source_hashes[str(folder/'outputs.npz')]=sha(folder/'outputs.npz')
        with np.load(folder/'outputs.npz') as f:assert len(f['embedding'])==len(ids) and np.isfinite(f['embedding']).all()
    panels={panel:aggregate(reused,['Fusion','scGAAM_'+panel,'scAGCL_'+panel],MAIN,[101,202,303,404,505]) for panel in ['native','KMeans20']}
    single=load(B/'evaluation_single_seed_v2/20261022T005327589183Z/rows.json')
    for name,value in [('development_rows',rows),('graph_statistics',graphs),('runtime_budget',timing),('paired_seed_differences',paired),('existing_multiseed_rows',reused),('external_single_seed_rows',single),('source_output_hashes',source_hashes)]:save(dest/(name+'.json'),value)
    save(dest/'summary.json',dict(development=dev,existing_multiseed=panels,complete_development_grid=True,new_training_jobs=12,posthoc=True,degree_matched=False))
    lines=['# 小论文结果表：最小补充实验','','开发集为事后机制控制，不重新选择融合方法；所有新控制使用冻结 L2+KMeans20、同训练 seed。','', '## 开发集四数据×三seed','']+table(dev,list(CFG['development']))
    for panel,s in panels.items():lines+=['',f'## 已有完整五seed主评价：{panel}','','Fusion固定读出；其它方法按本面板统一呈现，不逐数据挑读出。','']+table(s,MAIN)
    lines+=['','## 外部固定单seed对比（全部可用单元）','','Fusion为五seed均值；基线每数据固定seed101，SD不可估计。scSAMAC缺Spleen，不计算其五数据平均/排名。','| 数据 | Fusion | scGAAM native / unified | scAGCL native / unified | scSAMAC native / unified | scCAGN native / unified |','|---|---:|---:|---:|---:|---:|']
    for d in MAIN+['Muraro']:
        vals=[]
        for m in ['Fusion','scGAAM','scAGCL','scSAMAC','scCAGN']:
            if m=='Fusion':
                rr=[r['ari'] for r in single if r['dataset']==d and r['method']==m];vals.append(f'{np.mean(rr):.4f}')
            else:
                v=[]
                for panel in ['native','KMeans20']:
                    rr=[r['ari'] for r in single if r['dataset']==d and r['method']==m+'_'+panel];v.append(f'{np.mean(rr):.4f}' if rr else '失败')
                vals.append(' / '.join(v))
        lines.append('| '+d+' | '+' | '.join(vals)+' |')
    (dest/'RESULTS_TABLES.md').write_text('\n'.join(lines),encoding='utf8')
    delta=dev[2]['mean_ari']-dev[1]['mean_ari'];per={d:dev[2]['per_dataset'][d]['ari']-dev[1]['per_dataset'][d]['ari'] for d in CFG['development']}
    claim=('融合在该控制下平均优于等量随机加边，支持边的选择有作用；仍不能排除节点度分布等结构差异。' if delta>0 else '融合未在该控制下取得平均ARI优势，不能宣称潜在邻域边优于随机增密；应将方法定位为经验性融合设计并明确机制证据不足。')
    text=['# 核心贡献与实验故事更新','','研究问题：计数潜在邻域能否在统一图对比学习框架中补充表达邻域，改善跨数据平均聚类质量？','','1. 方法贡献是表达图与计数潜在图的静态并集，以及训练/推理使用一致图；不是动态权重、多组学或首次图融合。','2. 核心消融区分辅助计数损失、图一致性和潜在新增边；本轮加入等边数随机控制，检查纯增密解释。','3. 经验贡献是跨数据平均表现及边界分析，同时公开强基线、读出敏感性、稀有群体失败和预算差异。','',f'随机控制与Fusion等权平均ARI差（Fusion−随机）：{delta:+.6f}。',str(per),claim,'','已有五seed主评价中Fusion平均ARI最高；统一读出平均排名与scAGCL并列，不能声称唯一排名第一。单seed扩展中scSAMAC在多个已成功数据集更强，Spleen失败不能隐藏其它成功结果。','不能支持：所有数据集最优、稀有群体普遍改善、显著优越、无额外预训练成本或该设计首次出现。','图同标签比例仅为训练后描述；标签没有参与图构建、随机边选择、训练或选轮。四开发数据已用于方法开发，本轮不是独立验证。','', '建议正文保留方法图、五seed主表、单seed扩展表、核心消融；附录放逐seed、双读出、群体指标、图结构、预算、失败与Muraro敏感性。','若随机控制不支持机制，不继续用留出调参追求正结果；小论文仍可如实报告经验结果。']
    (dest/'CONTRIBUTIONS_ZH.md').write_text('\n'.join(text),encoding='utf8')
    appendix=['# 附录与复现','','随机控制：独立default_rng(seed)，在表达图补集中均匀无放回采样，保留相同新增边数；不匹配节点度。各seed仅一个随机图，不能推断完整随机图分布。','预算：12次新增训练，每次501次更新、每视图10步PGD、辅助计数权重0.1；复用400轮预训练缓存，预训练预算归属保留，随机边构建成本纳入新训练墙钟时间。','图统计不含自环，实际训练原实现自环处理保持不变；度统计与同标签率见graph_statistics.json。','群体召回为Hungarian一对一匹配；碎裂度为1−每真实群体最大簇覆盖率的平均。逐seed与配对差值均保留，无显著性推断。','既有多seed细胞索引/输出有限性已复查，旧评价数值直接复用；源输出SHA256见source_output_hashes.json。','scSAMAC Spleen数值失败原样保留；Tosches中断及额外计算成本见原外部对比归档。Muraro小数输入、排除5 unclear敏感性单列，不能冒充同质六数据确认。','额外外部多seed留待未来附件，不在本轮自动补跑；等度随机控制也是未来更强机制检验。','', '## 命令','```powershell',f'& "{PYTHON}" "{HERE / "run_queue.py"}"',f'& "{PYTHON}" "{HERE / "evaluate_supplement.py"}"','```','队列拒绝覆盖任何已有正式任务；以上训练命令仅用于干净复制目录，不能直接对已完成目录重复执行。']
    (dest/'APPENDIX_REPRODUCTION.md').write_text('\n'.join(appendix),encoding='utf8')
    save(HERE/'evaluation_latest.json',dict(path=str(dest),utc=utc(),status='COMPLETE_PENDING_REVIEW'))
    print(json.dumps(dict(path=str(dest),fusion_minus_random=delta,per_dataset_delta=per)))

if __name__=='__main__':main()
