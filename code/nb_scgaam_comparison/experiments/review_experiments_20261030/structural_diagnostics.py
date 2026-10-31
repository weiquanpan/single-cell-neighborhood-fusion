"""Descriptive reanalysis of archived development graphs; never feeds training."""
from pathlib import Path
import hashlib,json,time
import numpy as np

HERE=Path(__file__).resolve().parent
SOURCE=HERE.parents[2]/'fusion_latex_ZH_EN_v5_stable_scSAMAC/figures/mechanism_v4/inputs'

def stats(edges,y,seed):
    n=len(y);u,v=edges.T;same=y[u]==y[v]
    degree=np.bincount(edges.ravel(),minlength=n)
    inside=np.bincount(edges[same].ravel(),minlength=n)
    per=[]
    for g in np.unique(y):
        ix=y==g;vol=int(degree[ix].sum());inn=int(inside[ix].sum());cut=vol-inn
        denom=min(vol,int(degree.sum())-vol)
        per.append(dict(label=int(g),cells=int(ix.sum()),edge_endpoint_volume=vol,
                        precision=inn/vol if vol else None,cross_fraction=cut/vol if vol else None,
                        conductance=cut/denom if denom else None))
    rng=np.random.RandomState(seed);m=min(400,n);means=[];isolation=[]
    mask=np.zeros(n,dtype=bool)
    for _ in range(501):
        chosen=rng.permutation(n)[:m];mask[chosen]=True
        sub=edges[mask[u]&mask[v]]
        means.append(2*len(sub)/m)
        deg=np.bincount(sub.ravel(),minlength=n);isolation.append(float(np.mean(deg[chosen]==0)))
        mask[chosen]=False
    valid=[x['precision'] for x in per if x['precision'] is not None]
    return dict(edges=len(edges),homophily=float(same.mean()) if len(edges) else None,
                macro_endpoint_precision=float(np.mean(valid)) if valid else None,
                mean_degree=float(degree.mean()),max_degree=int(degree.max()),per_class=per,
                uniform_sample_iterations=501,uniform_sample_mean_degree=float(np.mean(means)),
                uniform_sample_degree_sd=float(np.std(means,ddof=1)),
                uniform_sample_zero_nonself_degree_fraction=float(np.mean(isolation)),
                sampling_note='Replays frozen NumPy permutation stream before stochastic edge dropping; self-loops excluded.')

def main():
    out=HERE/'structural_diagnostics';out.mkdir(exist_ok=False)
    names=sorted(SOURCE.glob('*__seed*.npz'))
    manifest=dict(script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  inputs={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in names},
                  purpose='Posthoc descriptive diagnostics only. No fitting, hyperparameter selection, or training input changes.')
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
    existing=json.loads((SOURCE/'01_development_rows.json').read_text())
    rows=[];comparisons=[]
    for path in names:
        ds,ss=path.stem.split('__seed');seed=int(ss)
        with np.load(path) as a:
            x=a['expression'];h=a['latent'];plus=a['fusion_added'];y=a['reference']
            n=len(y);xonly=x[~np.isin(x[:,0]*n+x[:,1],h[:,0]*n+h[:,1])]
            types={'expression':x,'expression_only':xonly,'latent_added':plus,'fusion':np.concatenate([x,plus]),
                   'random':np.concatenate([x,a['random_added']])}
            row=dict(dataset=ds,seed=seed,graphs={k:stats(e,y,seed) for k,e in types.items()})
        label={'Worm':'Worm_Neuron','Limb':'Limb_Muscle'}.get(ds,ds)
        er=next(r for r in existing if r['dataset']==label and r['seed']==seed and r['method']=='Expression-consistent')
        fr=next(r for r in existing if r['dataset']==label and r['seed']==seed and r['method']=='Fusion')
        assert [p['cells'] for p in row['graphs']['expression']['per_class']]==er['group_sizes']
        for j,(a,b) in enumerate(zip(row['graphs']['expression']['per_class'],row['graphs']['fusion']['per_class'])):
            comparisons.append(dict(dataset=ds,seed=seed,label=a['label'],cells=a['cells'],
                                    expression_precision=a['precision'],fusion_precision=b['precision'],
                                    delta_conductance=b['conductance']-a['conductance'] if a['conductance'] is not None and b['conductance'] is not None else None,
                                    delta_recall=fr['group_recalls'][j]-er['group_recalls'][j]))
        rows.append(row)
        (out/'progress.json').write_text(json.dumps(dict(completed=len(rows),total=len(names),last=path.name)),encoding='utf8')
    (out/'graphs.json').write_text(json.dumps(rows,indent=2,allow_nan=False),encoding='utf8')
    (out/'class_changes.json').write_text(json.dumps(comparisons,indent=2,allow_nan=False),encoding='utf8')
    (out/'status.json').write_text(json.dumps(dict(status='COMPLETE',graphs=len(rows),class_comparisons=len(comparisons))),encoding='utf8')
    print(json.dumps(dict(status='COMPLETE',graphs=len(rows),class_comparisons=len(comparisons))))

if __name__=='__main__':main()
