"""Separate evaluator; never imported by trainers. Immutable timestamped reports."""
from pathlib import Path
import sys,json,datetime,shutil,tarfile,hashlib,csv
B=Path(__file__).resolve().parent;E=B.parent/'cross_dataset_v1';R=B.parents[1];sys.path.insert(0,str(E))
import common as c
from evaluate import metrics
import numpy as np
from scipy.stats import rankdata
P=json.loads((B/'protocol.json').read_text());stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');OUT=B/'evaluation'/stamp;OUT.mkdir(parents=True)
def save(name,obj):c.save(OUT/name,obj)
def prepared(d):return (R/'data' if d in P['development'] else E/('heldout_muraro_corrected_v1' if d=='Muraro' else 'heldout_v1'))/'prepared'/d/'full'
def labelpath(d):return (R/'data' if d in P['development'] else E/('heldout_muraro_corrected_v1' if d=='Muraro' else 'heldout_v1'))/'evaluation_only'/d/'reference_labels.npy'
rows=[];timing=[];missing=[]
for m in P['methods']:
 for d in list(P['development'])+P['primary']+P['supplement']:
  seeds=P['development_seeds'] if d in P['development'] else P['seeds'];ids=np.load(prepared(d)/'cell_positions.npy');labels=np.load(labelpath(d))[ids];assert len(ids)==len(np.unique(ids))
  for seed in seeds:
   reuse=m=='scSAMAC' and d in P['development']
   if reuse:p=(R/'experiments/scsamac_author_worm_v1/runs'/f'seed{seed}'/'attempt_001' if d=='Worm_Neuron' else E/'training'/d/m/f'seed{seed}'/'main')
   else:p=B/'runs'/d/m/f'seed{seed}'/'attempt_001'
   meta=json.loads((p/'run.json').read_text()) if (p/'run.json').exists() else {}
   if meta.get('status')!='COMPLETE':missing.append({'dataset':d,'method':m,'seed':seed,'status':meta.get('status','PENDING')});continue
   np.testing.assert_array_equal(ids,np.load(p/'cell_positions.npy'))
   with np.load(p/'outputs.npz') as f:
    z=f['embedding'];pred=f['prediction'];assert len(z)==len(ids) and np.isfinite(z).all()
    if 'prediction_kmeans20' in f:matched=f['prediction_kmeans20']
    else:
     from sklearn.cluster import KMeans
     from sklearn.preprocessing import normalize
     from threadpoolctl import threadpool_limits
     with threadpool_limits(limits=8):matched=KMeans(n_clusters=P['development'][d],n_init=20,random_state=seed).fit_predict(normalize(z))
   for suffix,y in [('native',pred),('KMeans20',matched)]:
    assert len(y)==len(ids) and np.isfinite(y).all()
    rows.append(dict(dataset=d,method=m+'_'+suffix,seed=seed,source=str(p),**metrics(labels,y)))
    if d=='Muraro':
     import h5py
     with h5py.File(E/'heldout_sealed/Muraro.h5','r') as f:names=np.array([x.decode() for x in f['obs_names'][...]])[ids]
     unclear=json.loads((E/'heldout_muraro_corrected_v1/protocol_amendment.json').read_text())['unclear_cell_ids'];mask=~np.isin(names,unclear);assert sum(~mask)==5
     rows.append(dict(dataset='Muraro_exclude_unclear',method=m+'_'+suffix,seed=seed,source=str(p),**metrics(labels[mask],y[mask])))
   timing.append({'dataset':d,'method':m,'seed':seed,'seconds':meta.get('seconds'),'phase_seconds':meta.get('phase_seconds'),'optimizer_steps':meta.get('optimizer_steps'),'peak_cuda_allocated_bytes':meta.get('peak_cuda_allocated_bytes'),'reused':reuse,'budget':P[m],'source':str(p)})
# Existing frozen comparisons are loaded unchanged; old independent NB stays outside manuscript rows.
for path,rename,override in [(E/'evaluation/rows.json',{'baseline_scGAAM':'scGAAM_native','scGAAM__kmeans':'scGAAM_KMeans20','trained_NB-GCL-fusion':'Fusion'},None),(E/'heldout_v1/evaluation_five/rows.json',{'scGAAM_original':'scGAAM_native','scGAAM_KMeans20':'scGAAM_KMeans20','NB-GCL-fusion':'Fusion'},None),(E/'heldout_muraro_corrected_v1/evaluation_primary/rows.json',{'scGAAM_original':'scGAAM_native','scGAAM_KMeans20':'scGAAM_KMeans20','NB-GCL-fusion':'Fusion'},None),(E/'heldout_muraro_corrected_v1/evaluation_exclude_unclear/rows.json',{'scGAAM_original':'scGAAM_native','scGAAM_KMeans20':'scGAAM_KMeans20','NB-GCL-fusion':'Fusion'},'Muraro_exclude_unclear')]:
 for r in json.loads(path.read_text()):
  if r['method'] in rename:rows.append({**r,'method':rename[r['method']],'dataset':override or r['dataset']})
scopes={'development':list(P['development']),'primary_five':P['primary'],'muraro_supplement':['Muraro'],'muraro_sensitivity':['Muraro_exclude_unclear']};summary={};lines=['# 外部方法扩展比较','','## Material Passport','- 本地实际运行结果；标签仅由独立评价器读取。','- 融合和原评价协议不变；这是揭盲后的外部对比扩充。','- scCAGN为无标签固定终轮适配；共有HVG及细胞，不声称完全复现作者独立预处理。','- native与统一读出分开，禁止按数据选择读出；未完成网格不排名。','- 独立旧NB不进入本论文表格；其历史护栏失败仍见原审计，不改写。','']
for scope,ds in scopes.items():
 seeds=P['development_seeds'] if scope=='development' else P['seeds'];complete=[];per={}
 for m in sorted({r['method'] for r in rows}):
  groups={d:[r for r in rows if r['method']==m and r['dataset']==d] for d in ds}
  if not all(sorted(r['seed'] for r in rr)==seeds for rr in groups.values()):continue
  complete.append(m);per[m]={d:{'ari':float(np.mean([r['ari'] for r in rr])),'ari_sd':float(np.std([r['ari'] for r in rr],ddof=1)),**{k:float(np.mean([r[k] for r in rr])) for k in ['smallest_group_recall','worst_group_recall','mean_fragmentation']}} for d,rr in groups.items()}
 summary[scope]={'datasets':ds,'complete_methods':complete,'per_dataset':per,'ranking_panels':{}}
 lines+=['## '+scope,'','| 方法 | 数据集等权ARI |','|---|---:|']
 for m in complete:lines.append(f"| {m} | {np.mean([per[m][d]['ari'] for d in ds]):.6f} |")
 for panel in ['native','KMeans20']:
  methods=[m for m in complete if m=='Fusion' or m.endswith('_'+panel)];ranks={m:[] for m in methods}
  for d in ds:
   for m,r in zip(methods,rankdata([-per[m][d]['ari'] for m in methods])):ranks[m].append(float(r))
  summary[scope]['ranking_panels'][panel]={m:{'mean_ari':float(np.mean([per[m][d]['ari'] for d in ds])),'mean_rank':float(np.mean(ranks[m]))} for m in methods}
 lines+=['','| 数据 | 方法 | ARI | seed SD | 最小群体召回 | 最差群体召回 | 碎裂度 |','|---|---|---:|---:|---:|---:|---:|']
 for d in ds:
  for m in complete:lines.append('| '+d+' | '+m+' | '+' | '.join(f"{per[m][d][k]:.6f}" for k in ['ari','ari_sd','smallest_group_recall','worst_group_recall','mean_fragmentation'])+' |')
 lines+=['']
lines+=['## 耗时与预算','','含输入加载、构图、全部预训练重启、联合训练和读出；共享预处理时间见prepared/ready.json，另计。旧结果没有记录的细分步骤标记未知，不能假造。比较采用作者预算，不声称等预算。','',f'新增/复用完成运行数：{len(timing)}；仍缺：{len(missing)}。','scCAGN：20×50轮、batch=1预训练，末轮silhouette选择重启；30轮联合训练固定终轮。原代码以标签NMI选轮次，已移除并留patch。原代码中对抗损失使用固定预训练z的行为保留。','scAGCL：参数500轮，作者循环实际501次更新，10步×2视图攻击；按无标签训练损失返回嵌入。','scSAMAC：300轮预训练、最多2000轮联合训练、tol=0.001，记录实际优化器步数。']
save('rows.json',rows);save('summary.json',summary);save('runtime_budget.json',timing);save('missing.json',missing);(OUT/'REPORT.md').write_text('\n'.join(lines),encoding='utf8');c.save(B/'evaluation/latest.json',{'path':str(OUT),'missing':len(missing),'utc':c.utc()})
with (OUT/'runtime_budget.csv').open('w',newline='',encoding='utf8') as f:
 w=csv.DictWriter(f,fieldnames=['method','dataset','seed','seconds','reused'],extrasaction='ignore');w.writeheader();w.writerows(timing)
if '--package' in sys.argv:
 assert not missing,'No completion package for incomplete grid'
 dest=R.parent/'paper_packages'/('fusion_external_comparison_'+stamp);dest.mkdir();shutil.copytree(R.parent/'paper_packages/fusion_paper_v1_20261017/paper',dest/'paper')
 for n in ['REPORT.md','rows.json','summary.json','runtime_budget.json','runtime_budget.csv']:shutil.copy2(OUT/n,dest/'paper'/('external_'+n))
 with (dest/'paper/RESULTS_TABLES.md').open('a',encoding='utf8') as f:f.write('\n\n# 新增外部对比\n\n详见 external_REPORT.md；逐seed、群体指标与预算见 external_rows.json 和 external_runtime_budget.json。\n')
 shutil.make_archive(str(dest/'paper_working_materials'),'zip',dest/'paper')
 manifest=[]
 with tarfile.open(dest/'external_baselines_data_code_results.tar','w',copybufsize=4*1024*1024) as tar:
  for p in sorted(B.rglob('*')):
   if p.is_file() and '.git' not in p.parts and '__pycache__' not in p.parts and p.name not in ['queue.lock','training_state.json']:
    name=str(p.relative_to(R.parent));tar.add(p,arcname=name);manifest.append({'path':name,'bytes':p.stat().st_size,'sha256':c.sha(p)})
 c.save(dest/'SHA256_MANIFEST.json',manifest)
 (dest/'README.md').write_text('新增外部比较的完整增量包。共享数据及此前冻结融合结果位于 ../fusion_paper_v1_20261017/research_data_code_results.tar；本增量不重复54GB共享内容。保留原包和本包即可复核整轮。\n',encoding='utf8');c.save(B/'package_complete.json',{'path':str(dest),'utc':c.utc(),'files':len(manifest)})
print(json.dumps({'evaluation':str(OUT),'missing':len(missing),'completed_runs':len(timing)}))
