"""Heldout execution audit without reading evaluation labels or computing ARI."""
from pathlib import Path
import sys,json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import *
import numpy as np

def main():
    sys.addaudithook(guard)
    H=EXP/'heldout_v1';frozen=json.loads((EXP/'followup/final_method_v1/manifest.json').read_text())
    for n,h in frozen['source_sha256'].items():assert sha(ROOT/n)==h,n
    state=json.loads((H/'training_state.json').read_text())
    assert sha(EXP/'followup/heldout_train_adapter.py')==state['adapter_sha256']
    assert sha(EXP/'followup/run_heldout_v1.py')==state['supervisor_sha256']
    jobs=[];coverage={d:{m:[] for m in ['A','NB','scGAAM','NB-GCL-fusion']} for d in frozen['heldout_datasets']}
    paths=list((H/'runs').glob('*/full/*/main_seed*/attempt_001/run.json'))+list((H/'candidate_workspace/training').glob('*/*/seed*/main/run.json'))
    for p in paths:
        meta=json.loads(p.read_text());d=meta['dataset'];seed=meta['seed'];method=meta.get('method',meta.get('variant'));status=meta['status']
        jobs.append({'dataset':d,'method':method,'seed':seed,'status':status,'path':str(p)})
        if status not in ['COMPLETE','COMPLETED']:continue
        task=H/'prepared'/d/'full';ids=np.load(p.parent/'cell_positions.npy');np.testing.assert_array_equal(ids,np.load(task/'cell_positions.npy'))
        assert len(np.unique(ids))==len(ids)
        with np.load(p.parent/'outputs.npz') as f:
            assert len(f['embedding'])==len(ids) and f['prediction'].shape==(len(ids),)
            assert np.isfinite(f['embedding']).all() and np.isfinite(f['prediction']).all()
        coverage[d][method].append(seed)
    complete=[d for d,methods in coverage.items() if all(sorted(s)==frozen['seeds'] for s in methods.values())]
    report={'utc':utc(),'source_hashes_match':True,'training_state':state['status'],'jobs':jobs,'coverage':coverage,'complete_datasets':complete,'heldout_metrics_read':False,'complete_six_dataset_grid':len(complete)==6}
    save(H/'training_audit.json',report)
    lines=['# 留出训练完整性核查','','冻结源码及适配器哈希一致；已完成输出的细胞索引、长度及有限值通过。此核查不读取标签或计算ARI。','', '| 数据 | A | NB | scGAAM | 融合候选 |','|---|---:|---:|---:|---:|']
    for d,methods in coverage.items():lines.append('| '+d+' | '+' | '.join(str(len(s))+'/5' for s in methods.values())+' |')
    lines+=['','完整五seed组：'+', '.join(complete),'','Muraro计数来源问题未解决，不能把五数据结果称为完整六数据验证。完整网格后统一计算数据集等权ARI、排名、逐数据护栏和全部群体指标。','复现：env313 Python执行followup/audit_heldout_training.py。']
    (H/'TRAINING_REPORT.md').write_text('\n'.join(lines),encoding='utf8');print(json.dumps({'complete_datasets':complete,'jobs':len(jobs),'completed_jobs':sum(j['status'] in ['COMPLETE','COMPLETED'] for j in jobs)}))

if __name__=='__main__':main()
