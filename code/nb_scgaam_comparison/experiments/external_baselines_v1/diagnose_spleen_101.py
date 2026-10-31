from pathlib import Path
import sys,json
B=Path(__file__).resolve().parent;E=B.parent/'cross_dataset_v1';sys.path.insert(0,str(E))
import common as c
import numpy as np
p=E/'heldout_v1/prepared/Quake_10x_Spleen/full';ready=json.loads((p/'ready.json').read_text());report={'utc':c.utc(),'inputs':{},'frozen_files':{}}
for name in ['counts.npy','encoder_input.npy','log_input.npy','cell_positions.npy']:
 x=np.load(p/name);report['inputs'][name]={'shape':list(x.shape),'finite':bool(np.isfinite(x).all()),'min':float(x.min()),'max':float(x.max()),'matches_prepared_sha':c.sha(p/name)==ready['files'][name]}
with np.load(p/'statistics.npz') as f:
 sf=f['size_factors'];report['size_factors']={'finite':bool(np.isfinite(sf).all()),'strictly_positive':bool((sf>0).all()),'min':float(sf.min()),'max':float(sf.max())}
for r in json.loads((B/'audit/formal_launch/sha256.json').read_text(encoding='utf-8-sig')):report['frozen_files'][r['Path']]=c.sha(r['Path']).lower()==r['Hash'].lower()
P=json.loads((B/'protocol.json').read_text());src=c.ROOT/'experiments/scsamac_author_worm_v1/author_source'
report['author_source_unchanged']=all(c.sha(src/name)==h for name,h in P['source']['scSAMAC']['files'].items())
report['finding']='First nonfinite pretrain loss at epoch 74; 300 pretrain epochs executed; nonfinite embedding rejected before clustering. Finite unchanged inputs do not prove root cause; no demonstrated engineering defect. Retain as numerical failure, no algorithm change or automatic retry.'
c.save(B/'monitoring/spleen101_diagnostic.json',report);print(json.dumps(report,indent=2))
