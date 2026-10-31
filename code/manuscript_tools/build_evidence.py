"""Rebuild revision tables and vector figures from immutable experiment records."""
from pathlib import Path
import json, shutil, hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
EXP=ROOT/'nb_scgaam_comparison/experiments/review_experiments_20261030'
OLD=ROOT/'fusion_latex_ZH_EN_v5_stable_scSAMAC/en_revise'
load=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
summary=load(EXP/'evaluation/summary.json'); rows=load(EXP/'evaluation/rows.json')
oldrows=load(ROOT/'nb_scgaam_comparison/experiments/cross_dataset_v1/heldout_v1/evaluation_five/rows.json')
oldrows+= [r for r in load(ROOT/'fusion_latex_ZH_EN_v5_stable_scSAMAC/figures/clustering_demo_v3/inputs/02_existing_multiseed_rows.json') if r['method'].startswith('scAGCL')]
for r in oldrows:
    if r['method']=='scAGCL_native':r['method']='scAGCL_original'
rows+= [r for r in oldrows if r['method'] in ['scGAAM_KMeans20','scGAAM_original','scAGCL_KMeans20','scAGCL_original']]
print('Original baseline method ids:',sorted({r['method'] for r in oldrows}))
D=load(EXP/'protocol.json')['datasets']; seeds=[101,202,303,404,505]
dn=dict(zip(D,['Romanov','Adam','Bladder','Spleen','Lung']))
names={'Fusion_frozen':'Fusion (NB union)','expression':'Expression','expression_dense':'Dense expression','random':'Random union','pca_union':'PCA union','mse_union':'MSE union','fusion_noaux':'Fusion, no auxiliary loss','NB_H_KMeans20_raw':'NB latent, raw','NB_H_L2KMeans20':'NB latent, L2','pca_union_direct_raw':'PCA, raw','pca_union_direct_l2':'PCA, L2','mse_union_direct_raw':'MSE latent, raw','mse_union_direct_l2':'MSE latent, L2','scSAMAC_stable_L2KMeans20':'scSAMAC-stable, common','scSAMAC_stable_native':'scSAMAC-stable, native','scGAAM_KMeans20':'scGAAM, common','scGAAM_original':'scGAAM, native','scAGCL_KMeans20':'scAGCL, common','scAGCL_original':'scAGCL, native'}
by={(r['method'],r['dataset'],r['seed']):r for r in rows}
def arr(m,d,key='ari'):
    return np.array([by[m,d,s][key] for s in seeds])
def mean(m,key='ari'):
    return np.mean([arr(m,d,key).mean() for d in D])
def f(v):return f'{v:.4f}'
def write(part,name,txt): (OUT/part/'tables'/name).write_text(txt,encoding='utf-8')
def table(headers,body,caption,label,long=False,align=None):
    align=align or ('l'+'r'*(len(headers)-1))
    head=' & '.join(headers)+r' \\'
    data='\n'.join(' & '.join(map(str,r))+r' \\' for r in body)
    if long:
        return '\\begingroup\n\\small\\setlength{\\tabcolsep}{4pt}\n\\begin{longtable}{'+align+'}\n\\caption{'+caption+'}\\label{'+label+'}\\\\\n\\toprule\n'+head+'\n\\midrule\\endfirsthead\n\\toprule\n'+head+'\n\\midrule\\endhead\n\\bottomrule\\endfoot\n'+data+'\n\\end{longtable}\n\\endgroup\n'
    return '\\begin{table}[!htbp]\n\\centering\\small\\setlength{\\tabcolsep}{3.5pt}\\renewcommand{\\arraystretch}{1.13}\n\\caption{'+caption+'}\\label{'+label+'}\n\\begin{tabular}{'+align+'}\n\\toprule\n'+head+'\n\\midrule\n'+data+'\n\\bottomrule\n\\end{tabular}\n\\end{table}\n'

graphmethods=['Fusion_frozen','expression','expression_dense','random','pca_union','mse_union','fusion_noaux','scGAAM_KMeans20','scAGCL_KMeans20','scSAMAC_stable_L2KMeans20']
valid=set(r['method'] for r in rows)
assert set(graphmethods)<=valid,valid
hdr=['Configuration']+[dn[d] for d in D]+['Mean ARI','Mean NMI']
body=[[names[m]]+[f(arr(m,d).mean()) for d in D]+[f(mean(m)),f(mean(m,'nmi'))] for m in graphmethods]
write('manuscript_en','graph_results.tex',table(hdr,body,'Five-seed results under the common L2+KMeans20 readout. Dataset columns report mean ARI; the final columns are unweighted means across the five datasets. The first seven rows share the frozen graph-learning configuration; external architectures are contextual comparisons. scSAMAC uses the stabilized implementation in every run. Uncertainty and individual runs are reported in the supplement.','tab:graph'))
direct=['Fusion_frozen','NB_H_KMeans20_raw','NB_H_L2KMeans20','pca_union_direct_raw','pca_union_direct_l2','mse_union_direct_raw','mse_union_direct_l2','scSAMAC_stable_native','scSAMAC_stable_L2KMeans20']
body=[[names[m]]+[f(arr(m,d).mean()) for d in D]+[f(mean(m)),f(mean(m,'nmi'))] for m in direct]
write('manuscript_en','readout_results.tex',table(hdr,body,'Representation and readout sensitivity across five seeds. Raw and L2 denote KMeans20 applied before and after row-wise $\\ell_2$ normalization of the same representation; native scSAMAC uses its learned soft cluster assignments. Fusion is evaluated under its frozen L2+KMeans20 readout.','tab:readout'))

paired=[]
for m in graphmethods[1:]+[m for m in direct if m!='Fusion_frozen' and m not in graphmethods]:
    delta=np.array([arr('Fusion_frozen',d)-arr(m,d) for d in D])
    rng=np.random.default_rng(20261030);b=10000
    ix=rng.integers(5,size=(b,5,5));cond=np.take_along_axis(np.broadcast_to(delta,(b,5,5)),ix,axis=2).mean((1,2))
    di=rng.integers(5,size=(b,5));si=rng.integers(5,size=(b,5,5));hier=delta[di[:,:,None],si].mean((1,2))
    p=dict(method=m,mean=float(delta.mean()),wins=int((delta>0).sum()),ties=int((delta==0).sum()),conditional=np.quantile(cond,[.025,.975]).tolist(),hierarchical=np.quantile(hier,[.025,.975]).tolist(),without_spleen=float(np.delete(delta,3,axis=0).mean()))
    paired.append(p)
    match=next((p for p in summary['paired_comparisons'] if p['comparator']==m),None)
    if match: np.testing.assert_allclose(p['hierarchical'],match['hierarchical_bootstrap_95'],rtol=0,atol=1e-14)
(OUT/'audit/recomputed_paired_comparisons.json').write_text(json.dumps(paired,indent=2))
ci=lambda x:'['+', '.join(f(t) for t in x)+']'
write('appendix_en','paired.tex',table(['Comparator','$\Delta$ARI','Wins/ties','Seed interval','Hierarchical interval','No Spleen'],[[names[p['method']],f(p['mean']),f"{p['wins']}/{p['ties']}",ci(p['conditional']),ci(p['hierarchical']),f(p['without_spleen'])] for p in paired],'Paired Fusion-minus-comparator ARI differences. Wins and exact ties are counted among 25 pairs; other pairs are losses. Intervals are unadjusted percentile 95\\% bootstrap intervals. The last column equally weights the four remaining datasets.','tab:paired',True))
body=[]
for d in D:
    for m in graphmethods+['NB_H_KMeans20_raw','NB_H_L2KMeans20','pca_union_direct_raw','pca_union_direct_l2','mse_union_direct_raw','mse_union_direct_l2','scSAMAC_stable_native','scGAAM_original','scAGCL_original']:
        a=arr(m,d);n=arr(m,d,'nmi');body.append([dn[d],names[m],f(a.mean()),f(a.std(ddof=1)),f(np.median(a)),f(n.mean()),f(n.std(ddof=1))])
write('appendix_en','complete_results.tex',table(['Dataset','Configuration','ARI mean','SD','Median','NMI mean','SD'],body,'Complete five-seed primary summaries. SD denotes the across-seed sample standard deviation ($n=5$ seeds per dataset).','tab:complete',True))
for metric in ['ari','nmi']:
    body=[]
    for d in D:
        for m in graphmethods+direct[1:7]+['scSAMAC_stable_native','scGAAM_original','scAGCL_original']:
            body.append([dn[d],names[m]]+[f(v) for v in arr(m,d,metric)])
    write('appendix_en',metric+'_seeds.tex',table(['Dataset','Configuration']+list(map(str,seeds)),body,'Individual '+metric.upper()+' values for every primary dataset and readout. No run is excluded.','tab:'+metric+'seeds',True))

graphs=load(EXP/'evaluation/graph_statistics.json')
nbgraph=[r for r in graphs if r['method']=='fusion_noaux']
body=[]
for d in D:
    for k,label in [('expression','$E_X$'),('added','$E_+$'),('union','$E_F$')]:
        rr=[r[k] for r in nbgraph if r['dataset']==d]
        body.append([dn[d],label,f(np.mean([r['homophily'] for r in rr])),f(np.mean([r['macro_endpoint_precision'] for r in rr])),f'{np.mean([r["edges"] for r in rr]):.1f}',f'{np.mean([r["mean_degree"] for r in rr]):.2f}',f'{np.mean([r["expected_uniform_400_degree"] for r in rr]):.2f}'])
write('appendix_en','graphs.tex',table(['Dataset','Edges','Concordance','Macro precision','Edge count','Degree','Sample degree'],body,'Primary graph diagnostics, averaged over five seeds. Sample degree is the analytical expectation for a uniformly induced 400-cell subgraph before edge dropping. Non-self undirected edges are counted once. The no-auxiliary-loss run stores exactly the frozen NB union graph.','tab:graphs',True))
body=[]
for d in D:
    for m in ['fusion_noaux','expression_dense','random','pca_union','mse_union']:
        rr=[r for r in graphs if r['dataset']==d and r['method']==m]
        body.append([dn[d],names[m].replace('Fusion, no auxiliary loss','NB union')]+[f'{r["added"]["edges"]:,}' for r in rr])
write('appendix_en','edge_counts.tex',table(['Dataset','Added-edge source']+list(map(str,seeds)),body,'Added-edge counts in seed order. Dense-expression and random controls match each NB increment exactly. PCA and MSE graphs retain their natural sizes under the same neighborhood rule.','tab:edgecounts',True))
classmethods=['Fusion_frozen','expression','expression_dense','mse_union','NB_H_KMeans20_raw','pca_union_direct_raw','mse_union_direct_raw']
body=[]
for d in D:
    sizes=by['Fusion_frozen',d,101]['class_sizes']
    for c,n in enumerate(sizes):
        body.append([dn[d],f'C{c}',n]+[f(np.mean([by[m,d,s]['class_recall'][c] for s in seeds])) for m in classmethods])
write('appendix_en','class_recall.tex',table(['Dataset','Class','Cells','Fusion','Expr.','Dense','MSE union','NB raw','PCA raw','MSE raw'],body,'Per-class one-to-one matched recall, averaged over five seeds. Classes follow the archived reference-label order and retain the same identifiers across methods.','tab:recall',True))
body=[]
for d in D:
    for m in graphmethods[:7]+direct[1:]:
        body.append([dn[d],names[m]]+[f(np.mean([by[m,d,s][k] for s in seeds])) for k in ['macro_f1','smallest_recall','worst_recall','fragmentation']])
write('appendix_en','class_metrics.tex',table(['Dataset','Configuration','Macro F1','Smallest recall','Worst recall','Fragmentation'],body,'Population-level summaries (means of five seed-level metrics). One-to-one matching can assign zero recall to a class merged with a larger class, even when its cells occupy the same predicted cluster.','tab:classmetrics',True))
dev=load(ROOT/'fusion_latex_ZH_EN_v5_stable_scSAMAC/figures/mechanism_v4/inputs/01_development_rows.json')
devD=['Worm_Neuron','Limb_Muscle','Tosches','Bach']; devnames={'Worm_Neuron':'Worm','Limb_Muscle':'Limb','Tosches':'Tosches','Bach':'Bach'}
body=[]
for d in devD:
    for s in [11,22,33]:
        rr={r['method']:r for r in dev if r['dataset']==d and r['seed']==s}
        body.append([devnames[d],s]+[f(rr[m]['ari']) for m in ['Expression-consistent','Fusion','Random-equal-edges']]+[f(rr['Fusion']['ari']-rr['Expression-consistent']['ari'])])
write('appendix_en','development.tex',table(['Dataset','Seed','Expression','Fusion','Random','$\Delta$ARI'],body,'Development-set comparisons across three random seeds on the four datasets used during initial architecture selection.','tab:development',True))
devg=load(EXP/'structural_diagnostics/graphs.json');body=[]
for r in devg:r['dataset']={'Worm_Neuron':'Worm','Limb_Muscle':'Limb'}.get(r['dataset'],r['dataset'])
for d in ['Worm','Limb','Tosches','Bach']:
    for k,label in [('expression','$E_X$'),('expression_only','$E_X\\setminus E_H$'),('latent_added','$E_+$'),('fusion','$E_F$')]:
        rr=[r['graphs'][k] for r in devg if r['dataset']==d]
        body.append([d,label,f(np.mean([r['homophily'] for r in rr])),f(np.mean([r['macro_endpoint_precision'] for r in rr])),f'{np.mean([r["mean_degree"] for r in rr]):.2f}',f'{max(r["max_degree"] for r in rr):,}',f'{np.mean([r["uniform_sample_mean_degree"] for r in rr]):.3f}',f(np.mean([r['uniform_sample_zero_nonself_degree_fraction'] for r in rr]))])
write('appendix_en','dev_graphs.tex',table(['Dataset','Edges','Concord.','Macro prec.','Degree','Max degree','Sample deg.','Isolated'],body,'Development graph audit. Sample degree and isolated fraction replay 501 uniform node samples for each archived seed before edge dropping; self-loops are excluded. Max degree is the maximum over the three seeds.','tab:devgraphs',True))
protocol=load(EXP/'protocol.json');body=[]
for d in D:
    body.append([dn[d]]+[protocol['inputs'][d]['nb_pretrain_epochs'][str(s)] for s in seeds])
write('appendix_en','budgets.tex',table(['Dataset']+list(map(str,seeds)),body,'Actual NB pretraining epochs. The MSE control matches these budgets cell for cell and seed for seed; all subsequent graph runs use 501 updates.','tab:budgets'))
body=[]
for d in D:
    for m in ['expression','expression_dense','random','pca_union','mse_union','fusion_noaux']:
        rr=[load(EXP/'runs'/d/m/f'seed{s}/run.json') for s in seeds]
        body.append([dn[d],names[m],f'{np.mean([r["seconds"] for r in rr])/60:.2f}',f'{min(r["seconds"] for r in rr)/60:.2f}--{max(r["seconds"] for r in rr)/60:.2f}',f'{max(r["peak_cuda_allocated_bytes"] for r in rr)/2**30:.2f}'])
    rr=[load(EXP/'scsamac'/d/f'seed{s}/run.json') for s in seeds]
    body.append([dn[d],'scSAMAC-stable',f'{np.mean([r["seconds"] for r in rr])/60:.2f}',f'{min(r["seconds"] for r in rr)/60:.2f}--{max(r["seconds"] for r in rr)/60:.2f}',f'{max(r["peak_cuda_allocated_bytes"] for r in rr)/2**30:.2f}'])
write('appendix_en','runtime.tex',table(['Dataset','Configuration','Mean min','Range min','Peak GiB'],body,'Wall-clock execution times and peak PyTorch CUDA memory allocations for follow-up runs on an NVIDIA RTX 5070 Ti. Timings exclude cached NB pretraining and process startup overhead, whereas MSE union timings include paired MSE autoencoder pretraining.','tab:runtime',True))

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42,'axes.labelsize':10})
colors=['#215b85','#c66032','#33816d','#9467a6','#6b7380']
def savefig(name,fig,part='manuscript_en'):
    fig.savefig(OUT/part/'images'/f'{name}.pdf',bbox_inches='tight');fig.savefig(OUT/part/'images'/f'{name}.png',dpi=180,bbox_inches='tight');plt.close(fig)

fig,ax=plt.subplots(figsize=(8.4,6.4));ax.set(xlim=(-.1,10.1),ylim=(-.15,6.5));ax.axis('off')
def box(x,y,w,h,text,col='#edf3f7'):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.08',facecolor=col,edgecolor='#7790a2',linewidth=.8));ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=10)
def arrow(x1,y1,x2,y2):ax.annotate('',xy=(x2,y2),xytext=(x1,y1),arrowprops={'arrowstyle':'->','color':'#4b5963','lw':1.2})
box(3.2,5.65,3.6,.65,'Same cells / 2,000 HVGs\nObserved counts X')
box(.2,4.25,4.0,.95,'Log expression Y\nPearson kNN: '+r'$E_X$')
box(5.5,4.25,4.3,.95,'Normalized / standardized input\nNB autoencoder: 32-D H → '+r'$E_H$'+'\n≤400 epochs; early stopping')
arrow(3.4,5.55,2.2,5.3);arrow(6.6,5.55,7.65,5.3)
box(3,2.95,4,.7,'Fixed binary union: '+r'$E_F=E_X\cup E_H$')
arrow(2.2,4.15,3.5,3.75);arrow(7.65,4.15,6.5,3.75)
box(.2,1.55,2.9,.9,'Uniform sample: ≤400 cells\nInduced graph → two views\nEdge drop / feature mask')
arrow(3.3,2.85,1.65,2.55)
box(3.6,1.55,3.1,.9,'GAT + projection head\nBase/base and adversarial/\nadversarial contrastive loss')
arrow(3.2,2.0,3.5,2.0)
box(7.2,1.55,2.6,.9,'Auxiliary NB decoder\nReconstruct X from\nfirst base embedding')
arrow(6.8,2.0,7.1,2.0)
ax.text(5,1.02,'Feature-only PGD: 10 steps, step 0.01, |δ| ≤ 0.04\n501 updates; minimum-loss checkpoint',ha='center',va='center',fontsize=10)
box(.2,.0,9.6,.55,'Full-graph inference → GAT → row L2 → KMeans20 (known K)','#f5eee5')
savefig('overview',fig)

fig,axs=plt.subplots(2,2,figsize=(8.4,6.5),gridspec_kw={'height_ratios':[1,1.05]})
for ax,m,title in zip(axs.flat[:3],['expression','expression_dense','mse_union'],['A  Fusion − expression','B  Fusion − dense expression','C  Fusion − MSE union']):
    for j,d in enumerate(D):
        v=arr('Fusion_frozen',d)-arr(m,d)
        ax.scatter(j+np.linspace(-.12,.12,5),v,s=23,color=colors[j],alpha=.8)
        ax.plot([j-.21,j+.21],[np.median(v)]*2,color='black',lw=1.6)
    ax.axhline(0,color='gray',lw=.8);ax.set_xticks(range(5),list(dn.values()));ax.set_ylabel('Paired ΔARI');ax.set_title(title,loc='left',fontweight='bold');ax.grid(axis='y',alpha=.15)
ax=axs[1,1];d=D[3]
for j,m in enumerate(['Fusion_frozen','expression','expression_dense','mse_union','scSAMAC_stable_L2KMeans20']):
    v=arr(m,d);ax.scatter(j+np.linspace(-.12,.12,5),v,s=23,color=colors[j]);ax.plot([j-.21,j+.21],[np.median(v)]*2,color='black',lw=1.6)
ax.set_xticks(range(5),['Fusion','Expr.','Dense','MSE','scSAMAC']);ax.set_ylabel('ARI');ax.set_title('D  Spleen: all five seeds',loc='left',fontweight='bold');ax.grid(axis='y',alpha=.15)
fig.tight_layout(pad=1.5);savefig('paired_effects',fig)

fig,axs=plt.subplots(2,2,figsize=(8.4,6.2),sharey=True)
for ax,raw,l2,title in zip(axs.flat,['NB_H_KMeans20_raw','pca_union_direct_raw','mse_union_direct_raw','scSAMAC_stable_native'],['NB_H_L2KMeans20','pca_union_direct_l2','mse_union_direct_l2','scSAMAC_stable_L2KMeans20'],['A  NB latent','B  PCA','C  MSE latent','D  scSAMAC-stable']):
    x=np.arange(5);ax.bar(x-.18,[arr(raw,d).mean() for d in D],.35,color=colors[0],label='Raw KMeans20' if 'native' not in raw else 'Native');ax.bar(x+.18,[arr(l2,d).mean() for d in D],.35,color=colors[1],label='L2+KMeans20');ax.scatter(x,[arr('Fusion_frozen',d).mean() for d in D],color='black',marker='_',s=160,label='Frozen Fusion')
    ax.set_xticks(x,list(dn.values()));ax.set(ylim=(0,1.05),ylabel='Mean ARI');ax.set_title(title,loc='left',fontweight='bold');ax.legend(fontsize=8.5,loc='lower left');ax.grid(axis='y',alpha=.13)
fig.tight_layout();savefig('readout',fig)

fig,axs=plt.subplots(1,2,figsize=(10,5.6),gridspec_kw={'width_ratios':[1,1.05]})
for ax,d,title in zip(axs,[D[2],D[4]],['A  Bladder','B  Lung']):
    sizes=by['Fusion_frozen',d,101]['class_sizes'];vals=np.array([[np.mean([by[m,d,s]['class_recall'][c] for s in seeds]) for m in classmethods] for c in range(len(sizes))])
    im=ax.imshow(vals,vmin=0,vmax=1,cmap='Blues',aspect='auto')
    ax.set_yticks(range(len(sizes)),[f'C{c} (n={n})' for c,n in enumerate(sizes)]);ax.set_xticks(range(7),['Fusion','Expr.','Dense','MSE union','NB raw','PCA raw','MSE raw'],rotation=45,ha='right');ax.set_title(title,loc='left',fontweight='bold')
    for i in range(len(sizes)):
        for j in range(7):ax.text(j,i,f'{vals[i,j]:.2f}',ha='center',va='center',fontsize=9.5,color='white' if vals[i,j]>.6 else 'black')
fig.subplots_adjust(left=.12,right=.92,bottom=.22,top=.91,wspace=.45);cax=fig.add_axes([.94,.22,.014,.69]);fig.colorbar(im,cax=cax,label='Mean matched recall');savefig('class_recall',fig)

fig,axs=plt.subplots(1,2,figsize=(10,3.4))
for ax,key,title in zip(axs,['homophily','macro_endpoint_precision'],['A  Edge-weighted concordance','B  Class-macro endpoint precision']):
    for j,(k,label) in enumerate([('expression','Expression'),('added','NB-added'),('union','Union')]):
        ax.bar(np.arange(5)+(j-1)*.25,[np.mean([r[k][key] for r in nbgraph if r['dataset']==d]) for d in D],width=.25,label=label,color=colors[j])
    ax.set_xticks(range(5),list(dn.values()));ax.set_ylim(.5,1.01);ax.set_title(title,loc='left');ax.legend(fontsize=8);ax.grid(axis='y',alpha=.15)
fig.tight_layout();savefig('graph_precision',fig,'appendix_en')
shutil.copy2(OLD/'manuscript_en/images/primary_tsne.pdf',OUT/'appendix_en/images/archived_tsne.pdf')

# Verify control pairing and provenance without retraining.
checks=[]
for d in D:
    for s in seeds:
        rr=[load(EXP/'runs'/d/m/f'seed{s}/run.json') for m in ['expression','expression_dense','random','pca_union','mse_union','fusion_noaux']]
        assert len({r['initial_state_sha256'] for r in rr})==1
        assert all(r['optimizer_updates']==501 and r['status']=='COMPLETE' and r['checkpoint_reload_max_abs']==0 for r in rr)
        assert rr[1]['added_edges']==rr[2]['added_edges']==rr[-1]['added_edges']
        checks.append({'dataset':d,'seed':s,'paired_initialization':True,'exact_density_match':True,'all_501_updates':True})
(OUT/'audit/control_verification.json').write_text(json.dumps(checks,indent=2))
for src in [ROOT/'nb_scgaam_comparison/experiments/cross_dataset_v1/heldout_v1/evaluation_five/rows.json',ROOT/'fusion_latex_ZH_EN_v5_stable_scSAMAC/figures/mechanism_v4/inputs/01_development_rows.json']:
    shutil.copy2(src,OUT/'data'/('original_primary_rows.json' if 'evaluation_five' in str(src) else 'development_rows.json'))
print('All tables, four main figures, and supplementary graph figure generated. Paired control audits passed.')
ext=load(ROOT/'fusion_latex_ZH_EN_v5_stable_scSAMAC/figures/clustering_demo_v3/inputs/07_rows.json')
shutil.copy2(ROOT/'fusion_latex_ZH_EN_v5_stable_scSAMAC/figures/clustering_demo_v3/inputs/07_rows.json',OUT/'data/external_archived_rows.json')
shutil.copy2(ROOT/'fusion_latex_ZH_EN_v5_stable_scSAMAC/figures/clustering_demo_v3/inputs/02_existing_multiseed_rows.json',OUT/'data/original_graph_comparators.json')
body=[];vals=[]
for d in D:
    a=[by['Fusion_frozen',d,101]['ari'],by['scGAAM_original',d,101]['ari'],by['scGAAM_KMeans20',d,101]['ari'],by['scAGCL_original',d,101]['ari'],by['scAGCL_KMeans20',d,101]['ari'],by['scSAMAC_stable_native',d,101]['ari'],by['scSAMAC_stable_L2KMeans20',d,101]['ari']]
    for m in ['scCAGN_native','scCAGN_KMeans20']:
        a.append(next(r['ari'] for r in ext if r['dataset']==d and r['seed']==101 and r['method']==m))
    vals.append(a);body.append([dn[d]]+[f(v) for v in a])
body.append(['Mean']+[f(v) for v in np.mean(vals,axis=0)])
write('appendix_en','external_seed101.tex',table(['Dataset','Fusion','GAAM N','GAAM C','AGCL N','AGCL C','SAMAC N','SAMAC C','CAGN N','CAGN C'],body,'Matched seed-101 ARI context. N denotes native; C denotes common L2+KMeans20. Fusion uses its frozen common readout. scSAMAC uses the current consistently stabilized run; scCAGN retains its archived run. Every entry is a single-seed value, including the equal-dataset mean.','tab:external',False))
changes=load(EXP/'structural_diagnostics/class_changes.json');body=[]
for ds in sorted({r['dataset'] for r in changes}):
    for c in sorted({r['label'] for r in changes if r['dataset']==ds}):
        rr=[r for r in changes if r['dataset']==ds and r['label']==c]
        body.append([{'Worm_Neuron':'Worm','Limb_Muscle':'Limb'}.get(ds,ds),f'C{c}',rr[0]['cells'],f(np.mean([r['expression_precision'] for r in rr])),f(np.mean([r['fusion_precision'] for r in rr])),f(np.mean([r['delta_conductance'] for r in rr])),f(np.mean([r['delta_recall'] for r in rr]))])
write('appendix_en','conductance.tex',table(['Dataset','Class','Cells','Expr. prec.','Fusion prec.','$\\Delta$conductance','$\\Delta$recall'],body,'Class-level endpoint precision, conductance, and matched-recall shifts between aligned expression and Fusion on the development datasets (three-seed means across 39 reference classes).','tab:conductance',True))
