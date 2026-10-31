"""Explicit partial-result archive; never declares all experiments successful."""
from pathlib import Path
import json, hashlib, tarfile, datetime, shutil, statistics
B=Path(__file__).resolve().parent
ROOT=B.parents[2]
stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
D=ROOT/'paper_packages'/('fusion_external_single_seed_partial_'+stamp)
D.mkdir()
E=Path(json.loads((B/'evaluation_single_seed_v2/latest.json').read_text())['path'])
S=json.loads((E/'summary.json').read_text()); rows=json.loads((E/'rows.json').read_text())
missing=json.loads((E/'missing.json').read_text())
assert missing==[dict(dataset='Quake_10x_Spleen',method='scSAMAC',seed=101,status='FAILED')],missing
P=json.loads((B/'protocol_single_seed_v2.json').read_text())
def save(p,x): p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf8')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for x in iter(lambda:f.read(8*1024*1024),b''):h.update(x)
 return h.hexdigest()
paper=D/'paper';paper.mkdir()
shutil.copytree(E,paper/'evaluation')
lines=['# 外部对比：固定单seed阶段最终结果','',
'本轮30个方法×数据组合：29成功、1失败，全部计划任务已尝试；不是全成功网格。Fusion开发3seed、主评价5seed，外部方法开发seed11、其余seed101。SD=NA表示单seed。已揭盲数据上的追加比较，不是新的未见验证；不能宣称统计显著性。',
'原独立NB不列本文比较表；其历史护栏失败审计仍保留在原共享包。','']
for scope,data in S.items():
 lines+=['## '+scope,'']
 for panel,ms in data['ranking_panels'].items():
  lines+=['### '+panel,'','| 方法 | 等权平均ARI | 平均排名 |','|---|---:|---:|']
  for m,v in ms.items():lines.append(f"| {m} | {v['mean_ari']:.6f} | {v['mean_rank']:.3f} |")
  lines.append('')
 lines+=['| 数据 | 方法 | ARI | SD | 最小群体召回 | 最差群体召回 | 碎裂度 |','|---|---|---:|---:|---:|---:|---:|']
 for d in data['datasets']:
  for m,ds in data['per_dataset'].items():
   v=ds[d]; vals=['NA' if v[k] is None else f'{v[k]:.6f}' for k in ['ari','ari_sd','smallest_group_recall','worst_group_recall','mean_fragmentation']]
   lines.append('| '+' | '.join([d,m]+vals)+' |')
 lines.append('')
lines+=['## 失败、预算和解释边界','',
'scSAMAC Spleen seed101数值失败，因此没有五主数据平均ARI和排名；另外4个历史seed失败同样保留，不改seed重试。逐项耗时和优化器步数见 evaluation/runtime_budget.json，所有单seed和历史尝试见归档。',
'scCAGN采用20×50轮batch1预训练和30轮联合训练；移除原标签NMI选轮，采用固定终轮，保留原detach z行为。scAGCL参数500实际501次更新。scSAMAC为300轮预训练、最多2000轮联合训练。作者预算不同，Fusion额外预训练预算保留，不能声称等算力。',
'Tosches attempt001中断于868/1000轮、16200352步，约10.94小时；attempt002重新完整训练，两个成本分别保留。加速验证属于额外诊断成本，不能混入正式算法预算或隐藏。',
'Muraro校正小数输入与排除5个unclear细胞的敏感性分析单列，不混入五主数据平均。',
'在五主数据上，Fusion平均ARI为0.695276，native面板平均排名1.6、统一读出面板1.8；这支持平均表现优势，但不支持每数据或所有群体指标占优。scCAGN native在Spleen和Lung更好，Fusion部分最小群体召回更差。所有读出均统一报告，不能逐数据挑赢家。',
'统一读出与原生读出结论差异较大，说明表示质量和聚类读出之间存在交互；两套面板应并列呈现。多seed不对称意味着当前比较只提供描述性证据。']
(paper/'RESULTS_TABLES.md').write_text('\n'.join(lines),encoding='utf8')
(paper/'PAPER_UPDATE_ZH.md').write_text('''# 论文问题与贡献更新

核心问题：如何通过计数建模与图表示的互补信息，在统一训练和读出规则下提高跨数据平均聚类质量，同时识别小群体与数据异质性带来的失效边界。

方法贡献应围绕冻结的融合机制及其消融证据展开，不把重新叙述本身当作算法新颖性证据。外部比较增加scSAMAC、scAGCL、scCAGN，为现有方法定位提供本地实验证据；文献新颖性仍需独立系统核查。

实验故事：先用开发集消融交代融合设计，再展示五主数据等权平均ARI与平均排名的优势；随后用原生/统一读出对照和群体召回、碎裂度揭示优势的条件。Muraro作为数据来源修订后的独立补充与敏感性分析。不是所有数据全面胜出，也不能把已揭盲数据上的后续分析称为未见验证。

主结果：Fusion五主数据平均ARI 0.695276；scCAGN原生0.677233、scAGCL原生0.555823、scGAAM原生0.665207。统一读出下分别为0.582990、0.674007、0.680740。两套固定面板中Fusion平均ARI和平均排名均最好，但不据此推断显著性；Fusion多seed、基线单seed的差异必须放在表注。

局限：Spleen和Lung的scCAGN原生ARI超过Fusion；小群体召回不保证改善。scSAMAC的Spleen失败使其主评价不完整。保留所有失败和中断成本，方法预算不等。原NB护栏审计不删除，但旧独立NB不列论文方法表。

待办：按预先固定seed补充基线多seed和配对不确定性分析；补齐scSAMAC数值失败的原因报告，任何算法改动应成为独立新协议；核验相关工作与方法新颖性；完成目标期刊格式、引用和作者审核。不得在这些已揭盲数据上调参后声称新验证。
''',encoding='utf8')
(paper/'REPRODUCTION.md').write_text('''# 复现与材料范围

在 F:/cluster-noise/nb_scgaam_comparison/experiments/external_baselines_v1 下运行：

```powershell
& 'F:/cluster-noise/scdeepcluster_author_work/pytorch_gpu/env313/Scripts/python.exe' evaluate_external_single_seed_v3.py
```

读取protocol_single_seed_v2.json确定固定seed和预算。Tosches正式结果指定attempt002，attempt001中断证据不得覆盖。不要重新启动已结束的队列来替换失败。

本包的paper/RESULTS_TABLES.md和PAPER_UPDATE_ZH.md是新增外部比较后的更新入口。原54GB包保留冻结融合、共享数据和前期消融；其旧论文表属于历史版本，应与本更新一起使用。增量tar保留external_baselines_v1全部源码、上游与适配代码、运行结果、检查点、失败、日志、验证和审计文件；排除Python缓存与.git目录。不重复共享54GB输入。

增量SHA256_MANIFEST.json逐文件校验，ARCHIVE_SHA256.json提供tar哈希。package_complete.json仅表示归档校验完成，不表示实验全成功。
''',encoding='utf8')
base=ROOT/'paper_packages/fusion_paper_v1_20261017/research_data_code_results.tar'
save(D/'BASE_PACKAGE_REFERENCE.json',{'path':str(base),'bytes':base.stat().st_size,'unchanged':True})
files=[p for p in sorted(B.rglob('*')) if p.is_file() and not {'.git','__pycache__'}.intersection(p.parts)]
manifest=[];archive=D/'external_baselines_partial_data_code_results.tar'
with tarfile.open(archive,'w') as tar:
 for i,p in enumerate(files):
  name=p.relative_to(ROOT).as_posix();digest=sha(p)
  tar.add(p,arcname=name,recursive=False)
  manifest.append({'path':name,'bytes':p.stat().st_size,'sha256':digest})
  if i%300==0:print('archived',i,len(files),flush=True)
save(D/'SHA256_MANIFEST.json',manifest)
expected={x['path']:x for x in manifest}
with tarfile.open(archive,'r') as tar:
 for member in tar:
  h=hashlib.sha256();f=tar.extractfile(member)
  for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
  assert h.hexdigest()==expected.pop(member.name)['sha256'],member.name
assert not expected
shutil.make_archive(str(D/'paper_working_materials'),'zip',paper)
save(D/'ARCHIVE_SHA256.json',{'file':archive.name,'sha256':sha(archive),'bytes':archive.stat().st_size})
status={'archive_verified':True,'all_planned_attempts_finished':True,'all_experiments_successful':False,'successful':29,'retained_failures':missing,'deferred':'external multi-seed supplement','path':str(D),'files':len(files),'utc':stamp}
save(D/'package_complete.json',status)
save(B/'package_partial_complete.json',status)
(D/'README.md').write_text('本轮单seed外部对比增量包：29成功、1失败。归档全部文件哈希已逐一回读核验。先阅读paper/PAPER_UPDATE_ZH.md、RESULTS_TABLES.md及REPRODUCTION.md。与BASE_PACKAGE_REFERENCE.json指定原54GB包配合使用。多seed附件仍待完成。',encoding='utf8')
print(json.dumps(status,ensure_ascii=False),flush=True)
