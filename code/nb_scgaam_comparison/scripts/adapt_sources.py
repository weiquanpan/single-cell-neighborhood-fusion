"""Reproducible, explicitly documented patch generation; originals stay unchanged."""
from common import *
import difflib, shutil
up=R/'upstream/nqm314-scGAAM-4b3c345';dst=R/'source/scgaam';dst.mkdir(parents=True,exist_ok=True)
for p in up.glob('*.py'):
    original=p.read_text(encoding='utf8');s=original
    s='\n'.join(line for line in s.split('\n') if not line.startswith('from torch_scatter import'))
    if p.name=='utils.py':
        s=s.replace('beta, principal_components):','beta, principal_components=None):')
        start=s.index('    adj_mat = np.zeros')
        end=s.index('    print("Building a "',start)
        s=s[:start]+'''    # Same upper-directed evaluation edges and union-undirected training graph.
    keep = source_nodes < target_nodes
    pairs = np.stack([source_nodes[keep], target_nodes[keep]], axis=1)
    pairs = pairs[np.lexsort((pairs[:, 1], pairs[:, 0]))]
    data.edge_index = torch.as_tensor(pairs.T.copy(), dtype=torch.long)
    G = nx.Graph()
    G.add_nodes_from(range(cell_num))
    all_pairs = np.stack([source_nodes, target_nodes], axis=1)
    all_pairs = all_pairs[np.lexsort((all_pairs[:, 1], all_pairs[:, 0]))]
    G.add_edges_from(all_pairs.tolist())
''' +s[end:]
    if p.name=='GATEncoder.py':
        s=s.replace('out_h = torch.sparse.mm(adj_sparse_h, h_prime)', '''# Equivalent weighted neighbor sum; deterministic index_add avoids
                # CUDA sparse.mm repeatability errors in full-graph inference.
                pair = adj_sparse_h.indices()
                messages = adj_sparse_h.values()[:, None] * h_prime.index_select(0, pair[1])
                out_h = h_prime.new_zeros((num_nodes, self.num_out_features))
                out_h.index_add_(0, pair[0], messages)''')
    if p.name=='train.py':
        s=s.replace(', beta, principal_components)',', beta, None)')
        s=s.replace('        Z_current = scAGCLmodel(X, full_edgeind_self_loops, None) #get embeddings',
'''        with torch.no_grad():
            Z_current = scAGCLmodel(X, full_edgeind_self_loops, None)
        pre_step_state = {k: v.detach().cpu().clone() for k,v in scAGCLmodel.state_dict().items()}''')
        start=s.index('        # logging\n');end=s.index('        now_time = t()',start)
        s=s[:start]+'''        # Reference labels and redundant per-epoch KMeans removed. Neither affects loss/RNG.
        pred = None
        loss_value = float(loss.detach())
        if not np.isfinite(loss_value):
            raise FloatingPointError('Non-finite scGAAM loss')
        if hasattr(scAGCLmodel, 'event_callback'):
            scAGCLmodel.event_callback(epoch, loss_value, float(loss1.detach()), float(loss2.detach()) if torch.is_tensor(loss2) else float(loss2))
''' +s[end:]
        s=s.replace('bestloss = loss','bestloss = loss_value').replace('if loss < bestloss:','if loss_value < bestloss:')
        s=s.replace('            Z_final=Z_current','            Z_final=Z_current.detach().clone()\n            scAGCLmodel.best_state = pre_step_state\n            scAGCLmodel.best_epoch = epoch - 1')
        s=s.replace('    return Z_final, pred_final',
'''    Z = normalize(Z_final.detach().cpu().numpy(), norm='l2')
    pred_final = KMeans(n_clusters=num_cluster, init='k-means++', n_init='auto', random_state=0).fit_predict(Z)
    return Z_final, pred_final''')
    (dst/p.name).write_text(s,encoding='utf8')
    patch=''.join(difflib.unified_diff(original.splitlines(True),s.splitlines(True),fromfile='upstream/'+p.name,tofile='source/scgaam/'+p.name))
    if patch:
        d=R/'patches';d.mkdir(exist_ok=True);(d/(p.name+'.patch')).write_text(patch,encoding='utf8')
adst=R/'source/scdeepcluster';adst.mkdir(parents=True,exist_ok=True)
for p in (OLD/'source').glob('*.py'):shutil.copyfile(p,adst/p.name)
save(R/'provenance/source_manifest.json',{'upstream_commit':json.loads((R/'provenance/scgaam_commit.json').read_text())['sha'],
    'scgaam_original':{p.name:sha(p) for p in up.rglob('*') if p.is_file()},
    'adapted':{str(p.relative_to(R)):sha(p) for p in (R/'source').rglob('*.py')},
    'A_NB_parent':str(OLD),'copied_A_NB_unchanged':True,'created_utc':utc()})
print('Sources copied, patched, and hashed')
