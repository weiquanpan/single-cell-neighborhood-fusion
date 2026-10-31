from pathlib import Path
import hashlib,json,datetime
B=Path(__file__).resolve().parent;O=B/'performance_concurrency_v1';O.mkdir(exist_ok=False)
s=(B/'validate_resident_full.py').read_text()
s=s.replace("V=B/'performance_validation_v1'","V=B/'performance_concurrency_v1'")
s=s.replace('def main(mode):','def main(case_name,seed):\n mode="baseline"')
s=s.replace('case=V/mode;','case=V/case_name;')
s=s.replace("seed=101,method='scCAGN'","seed=seed,method='scCAGN'")
s=s.replace("run=case/'runs/Romanov/scCAGN/seed101/attempt_001'","run=case/'runs/Romanov/scCAGN'/f'seed{seed}'/'attempt_001'")
s=s.replace("p.add_argument('--mode',choices=['baseline','resident'],required=True);main(p.parse_args().mode)","p.add_argument('--case',required=True);p.add_argument('--seed',type=int,required=True);a=p.parse_args();main(a.case,a.seed)")
compile(s,'validate_concurrency_full.py','exec');(B/'validate_concurrency_full.py').write_text(s,encoding='utf8')
plan={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'Romanov full20x50+30 budget, CPU input unchanged, seeds101 and202','sequence':['serial202 full; serial101 reuses completed prior baseline','parallel101 and parallel202 full'],'comparisons':['each1000 epoch trajectory','final model/checkpoints/embeddings/predictions','optimizer counts and RNG','end-to-end throughput; contention caveat'],'deployment':'Not automatic. Two-seed full equality and credible speed required; representative large-data memory must be assessed. Existing production process never stopped.','source_sha256':hashlib.sha256(s.encode()).hexdigest()}
(O/'plan.json').write_text(json.dumps(plan,indent=2),encoding='utf8')
