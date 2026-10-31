"""One-time formal freeze after all full-budget pilot checks pass (no score gate)."""
from common import *
import importlib, platform, subprocess, torch
assert not (R/'configs/frozen_manifest.json').exists(),'Freeze already exists; do not silently replace'
state=json.loads((R/'state.json').read_text());assert state['status']=='COMPLETED' and state['mode']=='pilot'
checks=['adaptation_checks.json','training_equivalence.json','sparse_adaptation_checks.json']
for name in checks:assert json.loads((R/'reports'/name).read_text())['status']=='PASSED',name
pilots={}
for method in ['A','NB','scGAAM','PCA_KMeans']:
    runs=[f.parent for f in (R/'runs/Limb_Muscle/full'/method/'pilot_seed2211').glob('attempt_*/result.json') if json.loads(f.read_text())['status']=='COMPLETED']
    assert len(runs)==1,(method,runs)
    subprocess.run([str(PYTHON),'-X','utf8',str(R/'scripts/audit_run.py'),'--run',str(runs[0])],check=True)
    pilots[method]=str(runs[0].relative_to(R))
deps={}
for name in ['torch','numpy','scipy','sklearn','h5py','scanpy','anndata','pandas','numba','matplotlib','networkx','torch_geometric','psutil']:
    module=importlib.import_module(name);deps[name]=dict(version=str(getattr(module,'__version__','unknown')),path=str(module.__file__))
env=dict(python=sys.version,executable=sys.executable,platform=platform.platform(),dependencies=deps,torch_cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),cuda_reinstalled=False,torch_replaced=False)
save(R/'provenance/environment.json',env)
save(R/'configs/runtime.json',dict(deterministic_algorithms=True,tf32=False,CUBLAS_WORKSPACE_CONFIG=os.environ['CUBLAS_WORKSPACE_CONFIG'],threads=8,scgaam_sparse_inference='deterministic_index_add_neighbor_weighted_sum',checkpoint_reload='exact',training_labels='blocked_by_audit_hook_and_absent_from_worker_inputs',main_excludes_pilot_and_smoke=True))
save(R/'reports/pilot_gate.json',dict(status='PASSED',utc=utc(),pilots=pilots,criteria=['all four full-budget jobs completed','independent metrics reproduced','checkpoint reload exact','A/NB initial shared weights and RNG identical','original/modified loop and dense gradients equivalent'],score_selection=False,retained_failed_scgaam_pilot_attempts=2))
scripts=['common.py','prepare.py','train_one.py','evaluate.py','audit_prepared.py','audit_run.py','run_queue.py','build_report.py']
files=[R/'scripts'/n for n in scripts]+list((R/'source').rglob('*.py'))+[R/'configs/experiment.json',R/'configs/runtime.json',R/'PLAN.md',R/'provenance/environment.json',R/'provenance/source_manifest.json']
save(R/'configs/frozen_manifest.json',dict(protocol_id='nb_scgaam_common_hvg2000_v1',frozen_utc=utc(),files={str(f.relative_to(R)):sha(f) for f in files},qualification_reports={n:sha(R/'reports'/n) for n in checks},pilot_gate_sha256=sha(R/'reports/pilot_gate.json')))
print('FORMAL_PLAN_FROZEN',sha(R/'configs/frozen_manifest.json'))
