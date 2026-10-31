"""Report only audited observations; keep partial cohorts and pilot results separate."""
from common import *
import numpy as np, pandas as pd
from scipy.stats import t
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
config=json.loads((R/'configs/experiment.json').read_text());rows=[];attempts=[];groups=[]
for f in sorted((R/'runs').rglob('run.json')):
    out=f.parent;meta=json.loads(f.read_text());base={key:meta[key] for key in ['dataset','method','depth','seed','role','status']};base['run']=str(out.relative_to(R));base['error']=meta.get('error','')
    sup=json.loads((out/'supervisor.json').read_text()) if (out/'supervisor.json').exists() else {}
    base['attempt_wall_seconds']=sup.get('wall_seconds',sup.get('elapsed_seconds'));base['peak_cpu_rss_bytes']=sup.get('peak_process_tree_rss');attempts.append(base.copy())
    if not (out/'independent_audit.json').exists():continue
    audit=json.loads((out/'independent_audit.json').read_text());assert audit['status']=='PASSED'
    result=json.loads((out/'result.json').read_text());metrics=json.loads((out/'metrics.json').read_text());base.update({k:metrics[k] for k in ['ARI','NMI','ACC','macro_F1']})
    for key in ['graph_seconds','train_seconds','prediction_seconds','compute_seconds','run_wall_seconds','shared_preparation_seconds','parameters','peak_gpu_allocated_bytes']:base[key]=result.get(key)
    if base['compute_seconds'] is not None:base['compute_plus_preparation_seconds']=base['compute_seconds']+base['shared_preparation_seconds']
    rows.append(base)
    for group,value in metrics['group_recalls'].items():groups.append(dict(dataset=meta['dataset'],depth=meta['depth'],method=meta['method'],seed=meta['seed'],role=meta['role'],group=group,recall=value,size=metrics['group_sizes'][group]))
dest=R/'reports';dest.mkdir(exist_ok=True);frame=pd.DataFrame(rows);af=pd.DataFrame(attempts);af.to_csv(dest/'attempts.csv',index=False)
pd.DataFrame(groups).to_csv(dest/'group_recalls.csv',index=False)
if frame.empty:
    save(dest/'report_status.json',dict(status='NO_AUDITED_RUNS',updated_utc=utc()));sys.exit(0)
main=frame[frame.role=='main'].copy();pilot=frame[frame.role=='pilot'].copy();smoke=frame[frame.role=='smoke'].copy()
main.to_csv(dest/'main_metrics.csv',index=False);pilot.to_csv(dest/'pilot_metrics.csv',index=False);smoke.to_csv(dest/'smoke_metrics.csv',index=False)
assert not main.duplicated(['dataset','depth','method','seed']).any(),'Duplicate successful main attempt requires review'
paired=[]
for (dataset,depth),block in main.groupby(['dataset','depth']):
    nb=block[block.method=='NB'].set_index('seed')
    for comparator in ['A','scGAAM','PCA_KMeans']:
        comp=block[block.method==comparator].set_index('seed')
        for seed in sorted(set(nb.index)&set(comp.index)):
            row=dict(dataset=dataset,depth=depth,contrast='NB-'+comparator,seed=seed)
            for metric in ['ARI','NMI','ACC','macro_F1']:row['delta_'+metric]=nb.loc[seed,metric]-comp.loc[seed,metric]
            if nb.loc[seed,'compute_seconds']>0:row['comparator_to_NB_time_ratio']=comp.loc[seed,'compute_seconds']/nb.loc[seed,'compute_seconds']
            paired.append(row)
paired=pd.DataFrame(paired);paired.to_csv(dest/'paired_differences.csv',index=False)
summary=[]
if len(paired):
    for keys,block in paired.groupby(['dataset','depth','contrast']):
        row=dict(zip(['dataset','depth','contrast'],keys));v=block.delta_ARI.to_numpy();n=len(v);mean=float(v.mean());spread=float(v.std(ddof=1)) if n>1 else None
        radius=float(t.ppf(.975,n-1)*spread/np.sqrt(n)) if n>1 else None
        row.update(n=n,mean_delta_ARI=mean,positive_seeds=int((v>0).sum()),sd_delta_ARI=spread,descriptive_t_lower=mean-radius if radius is not None else None,descriptive_t_upper=mean+radius if radius is not None else None,mean_time_ratio=float(block.comparator_to_NB_time_ratio.mean()))
        summary.append(row)
summary=pd.DataFrame(summary);summary.to_csv(dest/'paired_summary.csv',index=False)
expected=len(config['datasets'])*len(config['methods'])*len(config['seeds'])*len(config['depths']);complete=len(main)==expected
lines=['# NB 与 scGAAM 比较报告','',f'更新时间：{utc()}。状态：'+('全部 160 个正式运行已完成独立审计。' if complete else f'进行中；已核验 {len(main)}/{expected} 个正式运行。'),'', 'Material Passport: academic-research-suite / experiment-agent；仅基于保存产物的事实材料，尚未人工审稿。', '', '## 实验口径', '', '四份预先指定的公开归档数据；已知 K；共同细胞和由当前观察计数选择的 2000 HVG。A/NB 为同精度、同骨架、配对初始化的机制对照；scGAAM 保留原生 float32，不能将跨架构速度差全部归因于 NB 似然。详见 PLAN.md、DATA_SOURCES.md、IMPLEMENTATION_AUDIT.md。', '', '主种子 11/22/33/44/55；pilot 2211 与 smoke 分开。半深度使用独立二项观察并重新拟合预处理，不是对冻结全深度模型做输入扰动。', '', '## 正式结果（只列已核验运行）','', '|数据|深度|方法|完成 seeds|平均 ARI|平均 NMI|平均 ACC|平均计算秒数|','|---|---:|---|---:|---:|---:|---:|---:|']
for keys,block in main.groupby(['dataset','depth','method'],sort=False):
    dataset,depth,method=keys;lines.append(f'|{dataset}|{depth}|{method}|{len(block)}/5|{block.ARI.mean():.6f}|{block.NMI.mean():.6f}|{block.ACC.mean():.6f}|{block.compute_seconds.mean():.2f}|')
lines+=['','## 配对比较','', '|数据|深度|对照|配对数|平均 ΔARI|正差值次数|描述性 95% t 区间|','|---|---:|---|---:|---:|---:|---|']
for row in summary.to_dict('records'):
    ci=f"[{row['descriptive_t_lower']:.5f}, {row['descriptive_t_upper']:.5f}]" if row['n']>1 else '尚不足两次'
    lines.append(f"|{row['dataset']}|{row['depth']}|{row['contrast']}|{row['n']}|{row['mean_delta_ARI']:+.6f}|{row['positive_seeds']}/{row['n']}|{ci}|")
lines+=['','同一数据的五种子重复用于描述优化随机性，不能当五个独立数据集。区间是小样本、正态近似的描述性区间，不是预注册的显著性或成功判据。未完成配对不用于最终结论。跨数据需以数据集为单位；四数据不能支持广泛普适性结论。', '', '## 成本和失败', '', 'compute_seconds = 建图 + 训练 + 最终预测，包含算法所需聚类及训练日志/检查点开销。scGAAM 最终 KMeans 包含在训练项中。共享准备一次成本、含审计的进程时间、CPU/GPU 内存单独保存在 CSV；每次的独立评价成本不属于训练内核。', '', '全部尝试和失败成本见 attempts.csv。scGAAM 完整 pilot 曾因 CUDA sparse.mm 推断不逐位稳定导致两次重载核验失败；日志和权重均保留。确定性邻居求和适配通过原式数值等价、稠密训练梯度和重复推断检查后才进入正式组。数据下载和预处理失败另见日志，不混记成模型失败。','', '## 逐群体代价与写作边界','','group_recalls.csv 保存每类召回和样本量，应与整体 ARI 一并解释；不能只报告平均获益而略去小群体退化。NB 去零膨胀头并非新的概率分布发明。只有完成所有指定数据、配对检查和失败审计后，才能据结果形成论文主张。','', '## 文件入口','','main_metrics.csv、pilot_metrics.csv、paired_differences.csv、paired_summary.csv、group_recalls.csv、attempts.csv；各运行目录保存日志、初始化、检查点、独立评价和 independent_audit.json。']
(dest/'COMPARISON_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
if len(main):
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    colors={'A':'#536dfe','NB':'#d64b39','scGAAM':'#138c72','PCA_KMeans':'#777777'}
    for col,depth in enumerate(config['depths']):
        selected=main[main.depth==depth]
        for j,method in enumerate(config['methods']):
            means=[];times=[]
            for dataset in config['datasets']:
                block=selected[(selected.method==method)&(selected.dataset==dataset)]
                means.append(block.ARI.mean());times.append(block.compute_seconds.mean())
            xx=np.arange(len(config['datasets']))+(j-1.5)*.18
            axes[0,col].plot(xx,means,'o-',label=method,color=colors[method],markersize=4)
            axes[1,col].plot(xx,times,'o-',label=method,color=colors[method],markersize=4)
        axes[0,col].set_title(f'Observed depth {depth:g} — '+('complete' if complete else 'partial'))
        axes[0,col].set_ylabel('Mean ARI across completed seeds');axes[1,col].set_ylabel('Compute time (seconds, log scale)');axes[1,col].set_yscale('log')
        for ax in axes[:,col]:ax.set_xticks(range(4),config['datasets'],rotation=15);ax.grid(alpha=.2)
    axes[0,0].legend();folder=R/'figures';folder.mkdir(exist_ok=True)
    fig.savefig(folder/'quality_cost.png',dpi=180);fig.savefig(folder/'quality_cost.pdf');plt.close(fig)
save(dest/'report_status.json',dict(status='COMPLETE' if complete else 'PARTIAL',audited_main=len(main),expected_main=expected,audited_pilot=len(pilot),failed_attempts=int(sum(af.status=='FAILED')),updated_utc=utc()))
print('REPORT_UPDATED',len(main),expected)
