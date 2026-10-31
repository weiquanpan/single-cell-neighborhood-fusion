"""Prepare explicitly versioned controls without editing or launching frozen jobs."""
from pathlib import Path
import sys,ast
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import EXP,save,sha,utc

target=EXP/'followup/controls_v1'
target.mkdir(exist_ok=False)
prefix='from pathlib import Path\nimport sys\nsys.path.insert(0, str(Path(__file__).resolve().parents[2]))\n'
def change(text,old,new):
    assert text.count(old)==1,(old,text.count(old))
    return text.replace(old,new)

original_gcl=(EXP/'train_gcl.py').read_text()
gcl=change(original_gcl,'from common import *',prefix+'from common import *')
gcl=change(gcl,"out=EXP/'training'/args.dataset/args.variant/", "out=EXP/'control_training'/args.dataset/'NB-GCL-expression-union-control'/")
gcl=change(gcl,'graph=GraphConstruction(data,gx,k)', '''graph=GraphConstruction(data,gx,k)
    # Sole algorithmic change: unify expression graph inference with training.
    edges=np.array(sorted({tuple(sorted(e)) for e in graph.edges() if e[0]!=e[1]}),dtype=np.int64)
    data.edge_index=torch.tensor(edges.T.copy(),dtype=torch.long)''')
gcl=change(gcl,"choices=['scGAAM-control','NB-GCL','NB-GCL-fusion','NB-GCL-context']", "choices=['NB-GCL']")

original_r6=(EXP/'train_r6.py').read_text()
r6=change(original_r6,'from common import *',prefix+'from common import *')
r6=change(r6,"out=EXP/'training'/args.dataset/args.variant/", "out=EXP/'control_training'/args.dataset/args.variant/")
r6=change(r6,"choices=['both','positive','negative','none','far','uniformity','anchored']", "choices=['both_gentle']")
# both_gentle follows the original 'both' relation branches but the existing gentle DEC branch.
assert "if args.variant not in ['negative','none']" in r6
assert "elif args.variant not in ['positive','none']" in r6
assert "if args.variant in ['far','anchored']" in r6
assert "2 if args.variant in ['both','positive','negative','none'] else 1.5" in r6
assert "if args.variant in ['both','positive','negative','none']:p=p/q.detach().sum(0)" in r6
for name,source in [('train_gcl_control.py',gcl),('train_r6_control.py',r6)]:
    ast.parse(source);(target/name).write_text(source,encoding='utf8')
save(target/'manifest.json',{'created_utc':utc(),'status':'PREPARED_NOT_RUN','posthoc_development_control':True,
    'original_sources':{name:sha(EXP/name) for name in ['train_gcl.py','train_r6.py','gcl_train.py','common.py','protocol.json']},
    'control_sources':{p.name:sha(p) for p in target.glob('*.py')},
    'datasets':['Worm_Neuron','Limb_Muscle','Tosches','Bach'],'seeds':[11,22,33],
    'outputs':str(EXP/'control_training'),
    'required_validation':'Smoke and branch checks before full execution; prepared code is not an experimental result.',
    'launch_policy':'Do not launch competing GPU jobs. Run after original queue completes and no current child remains.',
    'gcl_command':'train_gcl_control.py --dataset DATA --seed SEED --variant NB-GCL',
    'r6_command':'train_r6_control.py --dataset DATA --seed SEED --variant both_gentle',
    'heldout_access':False})
print(target)
