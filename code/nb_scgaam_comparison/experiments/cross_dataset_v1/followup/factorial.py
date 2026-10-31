"""Paired descriptive 2x2 edge ablation, complete grids only."""
from pathlib import Path
import sys,json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import EXP,CFG,save,utc
import numpy as np

rows=json.loads((EXP/'evaluation/rows.json').read_text())
summary=json.loads((EXP/'evaluation/summary.json').read_text())
lookup={(r['dataset'],r['method'],r['seed']):r for r in rows}
stats={r['method']:r for r in summary['complete_grid']}
variants=['none','positive','negative','both']
effects={};tables=[];groups=[]
for d in CFG['development']:
    a={v:np.array([lookup[d,'trained_'+v,s]['ari'] for s in CFG['seeds']]) for v in variants}
    e={
        'positive_when_negative_off':a['positive']-a['none'],
        'positive_when_negative_on':a['both']-a['negative'],
        'negative_when_positive_off':a['negative']-a['none'],
        'negative_when_positive_on':a['both']-a['positive'],
        'positive_main':(a['positive']-a['none']+a['both']-a['negative'])/2,
        'negative_main':(a['negative']-a['none']+a['both']-a['positive'])/2,
        'interaction':a['both']-a['positive']-a['negative']+a['none']}
    effects[d]={k:{'paired_seed_differences':v.tolist(),'mean':float(v.mean()),'sample_sd':float(v.std(ddof=1))} for k,v in e.items()}
    tables.append('| '+d+' | '+' | '.join(f'{a[v].mean():.6f}' for v in variants)+' |')
    for v in variants:
        r=[lookup[d,'trained_'+v,s] for s in CFG['seeds']]
        groups.append('| '+d+' | '+v+' | '+' | '.join(f"{np.mean([x[k] for x in r]):.6f}" for k in ['smallest_group_recall','worst_group_recall','mean_fragmentation'])+' |')
macro={k:float(np.mean([effects[d][k]['mean'] for d in effects])) for k in e}
save(EXP/'followup/R6_FACTORIAL.json',{'utc':utc(),'seeds':CFG['seeds'],'effects':effects,'dataset_macro_effects':macro,'inference':'descriptive only; seeds are not independent datasets; no significance claim'})
lines=['# R6正负边2×2完整消融','', '## Material Passport','- 四开发数据×同seed 11/22/33×四条件，共48次正式训练；使用原DEC端点预测。','- 同为float32采样配对工程版本，400+400步，正负项各保留0.5系数；无边仍保留特征重构和DEC。','- followup/audit.py已核查冻结源码哈希、完成状态、输出有限值和细胞索引；留出内容未读取。','', '| 数据 | 无边 | 仅正边 | 仅负边 | 正负边 |','|---|---:|---:|---:|---:|',*tables]
lines.append('| 跨数据等权均值 | '+' | '.join(f"{stats['trained_'+v]['mean_ari']:.6f}" for v in variants)+' |')
lines+=['', '## 主效应与交互','以每个数据同seed配对计算，再对seed和数据分别等权平均。','正边主效应 = [(仅正−无边)+(正负−仅负)]/2；负边主效应 = [(仅负−无边)+(正负−仅正)]/2；交互 = 正负−仅正−仅负+无边。','', '| 数据 | 正边平均效应 | 负边平均效应 | 交互 |','|---|---:|---:|---:|']
for d,vals in effects.items():lines.append('| '+d+' | '+' | '.join(f"{vals[k]['mean']:+.6f}" for k in ['positive_main','negative_main','interaction'])+' |')
lines.append('| 数据等权平均 | '+' | '.join(f'{macro[k]:+.6f}' for k in ['positive_main','negative_main','interaction'])+' |')
lines+=['','完整配对差值和样本SD见R6_FACTORIAL.json，不把12个seed运行当成12个独立数据集，不作显著性声明。','', '## 按预定规则判断','- 四条件均未通过跨数据ARI护栏；仅正边均值最高，但不能按每份数据选择其最优条件并拼成新方法。','- 负边的平均作用在Worm为正，在Limb、Tosches、Bach为负；说明有明确数据依赖，不能说负边总是有害。','- 正边总体主效应很小且平均为负；正负同时存在的总体交互为负，不能简单把两个损失的单独收益相加。','- 无边与仅正边仍在Tosches、Bach大幅落后强基线，因此删除边损失不是充分修复，损伤不能全部归结于非边排斥。','- 条件效应包含相应目标改变后的优化和采样轨迹，不等于已证明生物学机制。','', '## 群体指标（三seed均值）','| 数据 | 条件 | 最小群体召回 | 最差群体召回 | 碎裂度 |','|---|---|---:|---:|---:|',*groups,'','召回使用Hungarian匹配，碎裂度为1减最大单簇覆盖率；平均ARI较优不代表所有群体得益。','', '## 后续与边界','远距离负例、uniformity和锚定版本仍须完整网格后判断；这些版本同时采用温和DEC，不能仅归因关系监督，应保留相同DEC控制。400步无DEC端点与800步联合端点预算不同，不能作为等预算DEC开关证据。图融合仍有pending_controls.json规定的表达图一致性控制待做。完整方法未冻结，留出集仍封存。']
lines.insert(lines.index('- 正边总体主效应很小且平均为负；正负同时存在的总体交互为负，不能简单把两个损失的单独收益相加。'), '- Tosches存在方向反转：无正边时添加负边为+0.027258，有正边时为−0.034742；该数据的负边平均效应接近0，必须连同交互解释。')
(EXP/'followup/R6_FACTORIAL.md').write_text('\n'.join(lines),encoding='utf8')
print(json.dumps({'macro':macro,'datasets':{d:{k:v['mean'] for k,v in e.items()} for d,e in effects.items()}},indent=2))
