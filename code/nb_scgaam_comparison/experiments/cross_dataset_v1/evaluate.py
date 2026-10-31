"""Evaluation-only process; no training imports, complete-grid selection only."""
from common import *
import csv
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import rankdata
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics.cluster import contingency_matrix

def metrics(y,p):
    c=contingency_matrix(y,p);sizes=c.sum(1);r,j=linear_sum_assignment(-c)
    recall=np.zeros(len(c));recall[r]=c[r,j]/sizes[r]
    coverage=c.max(1)/sizes
    return dict(ari=float(adjusted_rand_score(y,p)),nmi=float(normalized_mutual_info_score(y,p)),worst_group_recall=float(recall.min()),smallest_group_recall=float(recall[np.argmin(sizes)]),worst_max_cluster_coverage=float(coverage.min()),mean_fragmentation=float((1-coverage).mean()),group_recalls=recall.tolist(),group_coverage=coverage.tolist(),group_sizes=sizes.tolist())

def main():
    rows=[]
    for dataset in CFG['development']:
        labels=np.load(ROOT/'data/evaluation_only'/dataset/'reference_labels.npy')
        def add(method,seed,path,ids,key='prediction'):
            with np.load(path) as f:p=f[key]
            y=labels[ids];assert len(p)==len(y)
            rows.append(dict(dataset=dataset,method=method,seed=seed,source=str(path),**metrics(y,p)))
        for seed in CFG['seeds']:
            for m in ['NB','scGAAM']:
                folder=original(dataset,m,seed);add('baseline_'+m,seed,folder/'outputs.npz',np.load(folder/'cell_positions.npy'))
            folder=EXP/'stage0'/dataset/f'seed{seed}'
            for path in folder.glob('*.npz'):add(path.stem,seed,path,np.load(folder/'cell_positions.npy'))
        for folder in (EXP/'stage0'/dataset).glob('consensus*'):
            for path in folder.glob('*.npz'):
                seed,method=path.stem.split('__');add(folder.name+'__'+method,int(seed[4:]),path,np.load(folder/'cell_positions.npy'))
        for folder in (EXP/'training'/dataset).glob('*/seed*/main'):
            if not (folder/'run.json').exists() or json.loads((folder/'run.json').read_text())['status']!='COMPLETE':continue
            method=folder.parents[1].name;seed=int(folder.parent.name[4:]);ids=np.load(folder/'cell_positions.npy')
            add('trained_'+method,seed,folder/'outputs.npz',ids)
            if (folder/'pretrain_outputs.npz').exists():
                add('trained_'+method+'__noDEC',seed,folder/'pretrain_outputs.npz',ids)
                add('trained_'+method+'__postDEC_KMeans',seed,folder/'outputs.npz',ids,'kmeans_prediction')
        if dataset=='Worm_Neuron':
            for seed in CFG['seeds']:
                folder=ROOT/'experiments/scsamac_author_worm_v1/runs'/f'seed{seed}'/'attempt_001'
                if json.loads((folder/'run.json').read_text())['status']=='COMPLETE':add('trained_scSAMAC',seed,folder/'outputs.npz',np.load(folder/'cell_positions.npy'))
    save(EXP/'evaluation/rows.json',rows)
    columns=['dataset','method','seed','ari','nmi','worst_group_recall','smallest_group_recall','worst_max_cluster_coverage','mean_fragmentation']
    with (EXP/'evaluation/rows.csv').open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore');w.writeheader();w.writerows(rows)
    grid={};complete=[]
    for row in rows:grid.setdefault(row['method'],{}).setdefault(row['dataset'],{})[row['seed']]=row
    for method,data in grid.items():
        if all(set(data.get(d,{}))==set(CFG['seeds']) for d in CFG['development']):complete.append(method)
    means={m:{d:float(np.mean([r['ari'] for r in grid[m][d].values()])) for d in CFG['development']} for m in complete}
    ranks={m:[] for m in complete}
    for d in CFG['development']:
        for m,r in zip(complete,rankdata([-means[m][d] for m in complete])):ranks[m].append(float(r))
    summary=[]
    for m in complete:
        delta={d:means[m][d]-max(means['baseline_NB'][d],means['baseline_scGAAM'][d]) for d in CFG['development']}
        summary.append({'method':m,'mean_ari':float(np.mean(list(means[m].values()))),'mean_rank':float(np.mean(ranks[m])),'per_dataset_ari':means[m],'per_dataset_sample_sd':{d:float(np.std([r['ari'] for r in grid[m][d].values()],ddof=1)) for d in CFG['development']},'delta_vs_stronger_baseline':delta,'guardrail_pass':min(delta.values())>=-CFG['guardrail_margin'],'worst_group_recall':min(r['worst_group_recall'] for data in grid[m].values() for r in data.values())})
    summary.sort(key=lambda r:(-r['mean_ari'],r['mean_rank']))
    eligible=[r for r in summary if r['guardrail_pass'] and not r['method'].startswith('baseline_')]
    save(EXP/'evaluation/summary.json',{'complete_grid':summary,'incomplete_methods':sorted(set(grid)-set(complete)),'global_candidate':eligible[0]['method'] if eligible else None,'frozen':False,'heldout_evaluated':False})
    readout_scores={}
    for r in CFG['readouts']:
        members=[v+'__'+r for v in ['NB-H','NB-DEC','PCA','scGAAM']]
        if all(m in means for m in members):readout_scores[r]=float(np.mean([means[m][d] for m in members for d in CFG['development']]))
    save(EXP/'evaluation/readout_selection.json',{'scores_macro_over_four_datasets_and_four_representations':readout_scores,'provisional_global_readout':max(readout_scores,key=readout_scores.get) if readout_scores else None,'frozen':False,'rule':'one readout across all datasets; complete-grid methods only; freeze only after stage0 finishes'})
    lines=['# 跨数据开发集重分析','', '## Material Passport','- Mode: run; verification: ANALYZED','- 原始数据和已有输出：本地归档；本报告为开发集选择结果。','- 留出集未评估，方法尚未冻结。','', '| 方法 | 平均 ARI | 平均排名 | 护栏通过 | 最差群体召回 |','|---|---:|---:|---|---:|']
    for r in summary:lines.append(f"| {r['method']} | {r['mean_ari']:.4f} | {r['mean_rank']:.2f} | {r['guardrail_pass']} | {r['worst_group_recall']:.4f} |")
    lines+=['','只将完整四数据 × 三 seed 方法放入主表；缺失方法不参与排名。共识的三 seed 是同一多 seed 集成的读出扰动，不能解释成三次独立训练。','碎裂度定义为 1 − 各真实群体最大单簇覆盖率；召回采用 Hungarian 一对一匹配。','不完整方法：'+', '.join(sorted(set(grid)-set(complete)))]
    (EXP/'evaluation/REPORT.md').write_text('\n'.join(lines),encoding='utf8')
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
