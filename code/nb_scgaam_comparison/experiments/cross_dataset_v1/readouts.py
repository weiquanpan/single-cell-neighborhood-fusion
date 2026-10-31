"""Label-free readouts. An exact K failure is explicit, never silently relabeled."""
import numpy as np
from scipy import sparse
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

def knn_graph(z,k=15):
    z=normalize(np.asarray(z,dtype=np.float32))
    nn=NearestNeighbors(n_neighbors=min(k+1,len(z)),metric='euclidean',n_jobs=8).fit(z)
    d,j=nn.kneighbors(z);i=np.repeat(np.arange(len(z)),j.shape[1]);j=j.ravel();d=d.ravel()
    keep=i!=j
    a=sparse.csr_matrix((np.exp(-d[keep]**2), (i[keep],j[keep])),shape=(len(z),len(z)))
    return a.maximum(a.T)

def leiden_k(a,k,seed):
    import igraph as ig, leidenalg as la
    co=sparse.triu(a,k=1).tocoo()
    g=ig.Graph(n=a.shape[0],edges=list(zip(co.row.tolist(),co.col.tolist())),directed=False)
    weights=co.data.tolist();lo=1e-6;hi=1.
    def part(res):
        p=la.find_partition(g,la.RBConfigurationVertexPartition,weights=weights,resolution_parameter=res,seed=seed,n_iterations=-1)
        return np.asarray(p.membership),len(p)
    for _ in range(20):
        y,n=part(hi)
        if n==k:return y
        if n>k:break
        hi*=2
    for _ in range(35):
        mid=(lo+hi)/2;y,n=part(mid)
        if n==k:return y
        if n<k:lo=mid
        else:hi=mid
    raise RuntimeError(f'Leiden resolution search did not attain K={k}')

def graph_readout(a,k,seed,method='spectral'):
    if method=='leiden':return leiden_k(a,k,seed)
    return SpectralClustering(n_clusters=k,affinity='precomputed',assign_labels='kmeans',n_init=20,random_state=seed,eigen_solver='arpack').fit_predict(a)

def readout(z,k,seed,method):
    z=normalize(np.asarray(z,dtype=np.float64))
    if method=='kmeans':return KMeans(n_clusters=k,n_init=20,random_state=seed).fit_predict(z)
    if method.startswith('gmm_'):
        m=GaussianMixture(n_components=k,covariance_type=method[4:],n_init=3,max_iter=300,reg_covar=1e-5,random_state=seed)
        y=m.fit_predict(z)
        if not m.converged_:raise RuntimeError('GMM did not converge')
        return y
    return graph_readout(knn_graph(z),k,seed,method)

def coassociation_knn(partitions,k=30,block=256):
    """Exact coassociation top-k, O(block*N) working memory, deterministic ties."""
    p=np.asarray(partitions);n=p.shape[1];rows=[];cols=[];vals=[]
    for start in range(0,n,block):
        stop=min(start+block,n);m=np.zeros((stop-start,n),np.float32)
        for y in p:m+=(y[start:stop,None]==y[None,:])
        m/=len(p);m[np.arange(stop-start),np.arange(start,stop)]=-1
        # Stable sorting fixes the many exact-frequency ties independently of labels.
        ix=np.argsort(-m,axis=1,kind='stable')[:,:min(k,n-1)]
        rows.extend(np.repeat(np.arange(start,stop),ix.shape[1]));cols.extend(ix.ravel());vals.extend(np.take_along_axis(m,ix,axis=1).ravel())
    a=sparse.csr_matrix((vals,(rows,cols)),shape=(n,n));a.eliminate_zeros()
    return a.maximum(a.T)

def merge_connectivity(a,y,k):
    y=np.asarray(y).copy()
    while len(np.unique(y))>k:
        _,y=np.unique(y,return_inverse=True);c=y.max()+1
        membership=sparse.csr_matrix((np.ones(len(y)),(np.arange(len(y)),y)),shape=(len(y),c))
        link=(membership.T@a@membership).toarray();sizes=np.bincount(y)
        score=link/np.sqrt(sizes[:,None]*sizes[None,:]);score[np.tril_indices(c)]=-np.inf
        i,j=np.unravel_index(np.argmax(score),score.shape);y[y==j]=i
    return np.unique(y,return_inverse=True)[1]
