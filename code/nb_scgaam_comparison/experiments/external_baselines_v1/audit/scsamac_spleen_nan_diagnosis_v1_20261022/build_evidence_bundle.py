"""Package this read-only audit; preserve all original inputs/results/source."""
from pathlib import Path
import hashlib, json, shutil, zipfile, importlib.metadata, sys, platform

HERE = Path(__file__).resolve().parent
B = HERE.parents[1]
BASE = B.parents[1]
AUTHOR = BASE / 'experiments/scsamac_author_worm_v1/author_source'
E = B.parent / 'cross_dataset_v1'

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()

def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf8')

snapshot = HERE / 'source_snapshot'
snapshot.mkdir(exist_ok=False)
copied = []
for p in sorted(AUTHOR.glob('*.py')):
    target = snapshot / 'author_source' / p.name
    target.parent.mkdir(exist_ok=True)
    shutil.copyfile(p, target)
    copied.append(dict(source=str(p), snapshot=str(target.relative_to(HERE)), sha256=sha(target)))
for p in [B/'run_one.py', B/'protocol.json', B/'protocol_single_seed_v2.json', E/'common.py', E/'followup/prepare_heldouts_v2.py']:
    target = snapshot / p.name
    shutil.copyfile(p, target)
    copied.append(dict(source=str(p), snapshot=str(target.relative_to(HERE)), sha256=sha(target)))
for seed in [101,202,303,404,505]:
    folder = B / f'runs/Quake_10x_Spleen/scSAMAC/seed{seed}/attempt_001'
    for p in [folder/'run.json', folder/'progress.json', B/f'logs/scSAMAC_Quake_10x_Spleen_{seed}.log']:
        target = HERE / 'failure_records' / f'seed{seed}' / p.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, target)
        copied.append(dict(source=str(p), snapshot=str(target.relative_to(HERE)), sha256=sha(target)))

save(HERE/'copied_evidence.json', copied)
versions={}
for name in ['numpy','scipy','torch','scanpy','anndata','h5py','scikit-learn']:
    try: versions[name]=importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError: versions[name]='not visible in this packaging process'
save(HERE/'diagnostic_environment.json', dict(python=sys.version, platform=platform.platform(), packages=versions, operator_probe_device='CPU', operator_probe_dtype='float64', optimizer_steps=0))

guard = json.loads((HERE/'protected_files_sha256.json').read_text(encoding='utf8'))
assert all(sha(p)==h for p,h in guard['before'].items())
save(HERE/'final_integrity_check.json',dict(protected_original_files=len(guard['before']), all_sha256_unchanged=True, original_results_or_papers_edited=False, training_started=False))

files = sorted(p for p in HERE.rglob('*') if p.is_file() and 'matplotlib_cache' not in p.parts and p.suffix not in ['.zip','.pyc'] and p.name not in ['bundle_manifest.json','bundle_receipt.json'])
manifest = {str(p.relative_to(HERE)).replace('\\','/'):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in files}
save(HERE/'bundle_manifest.json',manifest)
files.append(HERE/'bundle_manifest.json')
zpath = HERE/'scSAMAC_Spleen_NaN_readonly_audit_v1.zip'
with zipfile.ZipFile(zpath,'x',compression=zipfile.ZIP_DEFLATED) as z:
    for p in files: z.write(p,str(p.relative_to(HERE)))
with zipfile.ZipFile(zpath) as z:
    assert z.testzip() is None
    for name, entry in manifest.items():
        assert hashlib.sha256(z.read(name)).hexdigest()==entry['sha256']
save(HERE/'bundle_receipt.json',dict(path=str(zpath),bytes=zpath.stat().st_size,sha256=sha(zpath),files=len(files),zip_crc_and_entry_sha256_verified=True,original_files_sha256_unchanged=True))
print((HERE/'bundle_receipt.json').read_text(encoding='utf8'))
