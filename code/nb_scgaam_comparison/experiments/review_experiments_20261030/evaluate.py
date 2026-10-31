"""Separate evaluation process: no training, graph selection, or parameter tuning."""
from pathlib import Path
import argparse,csv,json,sys
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];EXP=ROOT/'experiments/cross_dataset_v1';HOLD=EXP/'heldout_v1'
sys.path.insert(0,str(EXP))
import common as c
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score,normalized_mutual_info_score
from sklearn.metrics.cluster import contingency_matrix
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from threadpoolctl import threadpool_limits

def metrics(y,p):
    table=contingency_matrix(y,p);sizes=table.sum(1);r,j=linear_sum_assignment(-table)
    rec=np.zeros(len(table));prec=np.zeros(len(table));rec[r]=table[r,j]/sizes[r]
    prec[r]=table[r,j]/table.sum(0)[j];f1=2*rec*prec/np.maximum(rec+prec,1e-12)
    return dict(ari=float(adjusted_rand_score(y,p)),nmi=float(normalized_mutual_info_score(y,p)),
                smallest_recall=float(rec[np.argmin(sizes)]),worst_recall=float(rec.min()),
                macro_f1=float(f1.mean()),fragmentation=float((1-table.max(1)/sizes).mean()),
                class_sizes=sizes.tolist(),class_recall=rec.tolist(),class_precision=prec.tolist(),class_f1=f1.tolist(),
                confusion=table.tolist(),matched_cluster_indices=dict(zip(map(str,r),map(int,j))))

def graph_stats(edges,y):
    n=len(y);u,v=edges.T;within=y[u]==y[v]
    degree=np.bincount(edges.ravel(),minlength=n)
    same=np.bincount(edges[within].ravel(),minlength=n)
    per=[]
    for group in np.unique(y):
        ix=y==group;vol=int(degree[ix].sum());inside=int(same[ix].sum());cut=vol-inside
        denom=min(vol,int(degree.sum())-vol)
        per.append(dict(label=str(group),size=int(ix.sum()),volume=vol,
                        endpoint_precision=inside/vol if vol else None,cross_fraction=cut/vol if vol else None,
                        conductance=cut/denom if denom else None))
    return dict(edges=len(edges),homophily=float(within.mean()) if len(edges) else None,
                mean_degree=float(degree.mean()),max_degree=int(degree.max()),
                macro_endpoint_precision=float(np.mean([p['endpoint_precision'] for p in per if p['endpoint_precision'] is not None])) if len(edges) else None,
                per_class=per,expected_uniform_400_degree=float(degree.mean()*min(399,n-1)/(n-1)))

def main():
    protocol=json.loads((HERE/'protocol.json').read_text());rows=[];graphs=[];out=HERE/'evaluation';out.mkdir(exist_ok=True)
    def add(d,s,m,pred,source,labels):rows.append(dict(dataset=d,seed=s,method=m,source=str(source),**metrics(labels,pred)))
    for d in protocol['datasets']:
        task=HOLD/'prepared'/d/'full';ids=np.load(task/'cell_positions.npy')
        y=np.load(HOLD/'evaluation_only'/d/'reference_labels.npy')[ids]
        k=json.loads((task/'ready.json').read_text())['known_k']
        for s in protocol['seeds']:
            orig=HOLD/'candidate_workspace/training'/d/'NB-GCL-fusion'/f'seed{s}'/'main'
            with np.load(orig/'outputs.npz') as f:add(d,s,'Fusion_frozen',f['prediction'],orig,y)
            nb=HOLD/'candidate_workspace/cache'/d/f'nb_h_seed{s}.npy';h=np.load(nb)
            saved=out/f'{d}__H_seed{s}.npz'
            if not saved.exists():
                with threadpool_limits(limits=8):
                    raw=KMeans(n_clusters=k,n_init=20,random_state=s).fit_predict(h)
                    l2=KMeans(n_clusters=k,n_init=20,random_state=s).fit_predict(normalize(h))
                np.savez_compressed(saved,prediction_raw=raw,prediction_l2=l2)
            with np.load(saved) as f:
                add(d,s,'NB_H_KMeans20_raw',f['prediction_raw'],nb,y)
                add(d,s,'NB_H_L2KMeans20',f['prediction_l2'],nb,y)
            for m in protocol['variants']:
                p=HERE/'runs'/d/m/f'seed{s}'
                if not (p/'run.json').exists() or json.loads((p/'run.json').read_text())['status']!='COMPLETE':continue
                np.testing.assert_array_equal(ids,np.load(p/'cell_positions.npy'))
                with np.load(p/'outputs.npz') as f:add(d,s,m,f['prediction'],p,y)
                if (p/'representation.npz').exists():
                    with np.load(p/'representation.npz') as f:
                        add(d,s,m+'_direct_raw',f['prediction_raw'],p,y);add(d,s,m+'_direct_l2',f['prediction_l2'],p,y)
                with np.load(p/'graphs.npz') as f:
                    gx=f['expression'];added=f['added'];union=np.concatenate([gx,added])
                    record=dict(dataset=d,seed=s,method=m,expression=graph_stats(gx,y),added=graph_stats(added,y),union=graph_stats(union,y))
                    graphs.append(record)
            p=HERE/'scsamac'/d/f'seed{s}'
            if (p/'run.json').exists() and json.loads((p/'run.json').read_text())['status']=='COMPLETE':
                np.testing.assert_array_equal(ids,np.load(p/'cell_positions.npy'))
                with np.load(p/'outputs.npz') as f:
                    add(d,s,'scSAMAC_stable_native',f['prediction'],p,y)
                    add(d,s,'scSAMAC_stable_L2KMeans20',f['prediction_kmeans20'],p,y)
    c.save(out/'rows.json',rows);c.save(out/'graph_statistics.json',graphs)
    cols=['dataset','seed','method','ari','nmi','macro_f1','smallest_recall','worst_recall','fragmentation','source']
    with (out/'rows.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=cols,extrasaction='ignore');w.writeheader();w.writerows(rows)
    summaries=[]
    for m in sorted(set(r['method'] for r in rows)):
        rr=[r for r in rows if r['method']==m];per={}
        for d in protocol['datasets']:
            v=[r for r in rr if r['dataset']==d]
            if not v:continue
            per[d]={'n':len(v),'ari_mean':float(np.mean([r['ari'] for r in v])),
                    'ari_median':float(np.median([r['ari'] for r in v])),
                    'ari_sd':float(np.std([r['ari'] for r in v],ddof=1)) if len(v)>1 else None,
                    'nmi_mean':float(np.mean([r['nmi'] for r in v]))}
        complete=len(rr)==len(protocol['datasets'])*len(protocol['seeds'])
        summaries.append(dict(method=m,completed_runs=len(rr),complete=complete,per_dataset=per,
                              dataset_equal_ari=float(np.mean([v['ari_mean'] for v in per.values()])) if complete else None,
                              dataset_equal_nmi=float(np.mean([v['nmi_mean'] for v in per.values()])) if complete else None))
    # Paired resampling within datasets estimates seed uncertainty conditional on
    # these fixed datasets; hierarchical resampling also varies dataset selection.
    by={(r['dataset'],r['seed'],r['method']):r for r in rows};comparisons=[]
    for sm in summaries:
        m=sm['method']
        if not sm['complete'] or m=='Fusion_frozen':continue
        delta=np.array([[by[(d,s,'Fusion_frozen')]['ari']-by[(d,s,m)]['ari'] for s in protocol['seeds']] for d in protocol['datasets']])
        rng=np.random.default_rng(20261030);b=10000;D,S=delta.shape
        ix=rng.integers(S,size=(b,D,S));seed_boot=np.take_along_axis(np.broadcast_to(delta,(b,D,S)),ix,axis=2).mean((1,2))
        di=rng.integers(D,size=(b,D));si=rng.integers(S,size=(b,D,S));hier=delta[di[:,:,None],si].mean((1,2))
        comparisons.append(dict(comparator=m,mean_delta=float(delta.mean()),seed_matched_wins=int((delta>0).sum()),
                                ties=int((delta==0).sum()),n=D*S,per_dataset_mean=delta.mean(1).tolist(),
                                conditional_seed_bootstrap_95=np.quantile(seed_boot,[.025,.975]).tolist(),
                                hierarchical_bootstrap_95=np.quantile(hier,[.025,.975]).tolist(),
                                caveat='Five dataset clusters only; exploratory follow-up after primary results were inspected.'))
    c.save(out/'summary.json',dict(utc=c.utc(),status='COMPLETE_METHODS_ONLY',summaries=summaries,paired_comparisons=comparisons,
                                   protocol_sha256=c.sha(HERE/'protocol.json'),hyperparameters_selected_using_new_results=False))
    print(json.dumps({'rows':len(rows),'complete_methods':sum(s['complete'] for s in summaries),'output':str(out)},ensure_ascii=False))

if __name__=='__main__':main()
