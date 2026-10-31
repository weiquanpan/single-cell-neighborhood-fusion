from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import utilss as utilss
import h5py
import scipy as sp
import numpy as np
import scanpy as sc
import pandas as pd
import argparse
from itertools import chain
import os
from scipy import sparse
from scipy.spatial import distance
import scipy.sparse
import sys
import pickle
import csv
import networkx as nx
import numpy as np
from sklearn.ensemble import IsolationForest
import time
from multiprocessing import Pool
from igraph import *
from sklearn import preprocessing


# Calculate KNN graph and filter edges based on distance boundary
def calculateKNNgraphDistanceMatrixStatsSingleThread(featureMatrix, distanceType='euclidean', k=10, param=None):
    edgeList = []

    p_time = time.time()
    for i in np.arange(featureMatrix.shape[0]):
        tmp = featureMatrix[i, :].reshape(1, -1)
        distMat = distance.cdist(tmp, featureMatrix, distanceType)  # Compute pairwise distances
        res = distMat.argsort()[:k + 1]  # Select k nearest neighbors
        tmpdist = distMat[0, res[0][1:k + 1]]

        # Define boundary using mean and standard deviation
        boundary = np.mean(tmpdist) + np.std(tmpdist)
        for j in np.arange(1, k + 1):
            # Filter edges based on boundary
            if distMat[0, res[0][j]] <= boundary:
                weight = 1.0
            else:
                weight = 0.0

            edgeList.append((i, res[0][j], weight))

    return edgeList



# Calculate KNN graph (without boundary filtering)
def calculateKNNgraphDistanceMatrix(featureMatrix, distanceType='euclidean', k=10):
    distMat = distance.cdist(featureMatrix, featureMatrix, distanceType)  # Pairwise distances

    edgeList = []
    for i in np.arange(distMat.shape[0]):
        res = distMat[:, i].argsort()[:k]  # Select k nearest neighbors
        for j in np.arange(k):
            edgeList.append((i, res[j]))

    return edgeList

# Convert edge list to dictionary format
def edgeList2edgeDict(edgeList, nodesize):
    graphdict = {}
    tdict = {}

    for edge in edgeList:
        end1 = edge[0]
        end2 = edge[1]
        tdict[end1] = ""
        tdict[end2] = ""
        if end1 in graphdict:
            tmplist = graphdict[end1]
        else:
            tmplist = []
        tmplist.append(end2)
        graphdict[end1] = tmplist

    # Ensure all nodes are included in the graph dictionary
    for i in range(nodesize):
        if i not in tdict:
            graphdict[i] = []

    return graphdict



# Generate adjacency matrix and edge list
def generateAdj(featureMatrix, graphType='KNNgraph', para=None):
    """
    Generate adjacency matrix and edge list based on KNN graph.
    """
    edgeList = None
    adj = None
    edgeList = calculateKNNgraphDistanceMatrixStatsSingleThread(featureMatrix)
    graphdict = edgeList2edgeDict(edgeList, featureMatrix.shape[0])
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graphdict))  # Convert to adjacency matrix
    return adj, edgeList


# Perform Louvain clustering using igraph
def generateLouvainCluster(edgeList):
    """
    Perform Louvain clustering and return cluster assignments.
    """
    Gtmp = nx.Graph()
    Gtmp.add_weighted_edges_from(edgeList)  # Add weighted edges to the graph
    W = nx.adjacency_matrix(Gtmp)
    W = W.todense()
    graph = Graph.Weighted_Adjacency(
        W.tolist(), mode=ADJ_UNDIRECTED, attr="weight", loops=False)  # Create igraph graph
    louvain_partition = graph.community_multilevel(
        weights=graph.es['weight'], return_levels=False)  # Louvain clustering
    size = len(louvain_partition)
    hdict = {}
    count = 0
    for i in range(size):
        tlist = louvain_partition[i]
        for j in range(len(tlist)):
            hdict[tlist[j]] = i
            count += 1

    listResult = []
    for i in range(count):
        listResult.append(hdict[i])

    return listResult, size


def read_clean(data):
    assert isinstance(data, np.ndarray)
    if data.dtype.type is np.bytes_:
        data = utilss.decode(data)
    if data.size == 1:
        data = data.flat[0]
    return data


def dict_from_group(group):
    assert isinstance(group, h5py.Group)
    d = utilss.dotdict()
    for key in group:
        if isinstance(group[key], h5py.Group):
            value = dict_from_group(group[key])
        else:
            value = read_clean(group[key][...])
        d[key] = value
    return d


def nomalize_for_AF(filename,gene_num,raw, sparsify = False, skip_exprs = False): #处理数据
    with h5py.File(filename, "r") as f:
        obs = np.array(f["obs_names"][...])
        var = np.array(f["var_names"][...])
        cell_name = np.array(f["obs"]["cell_type1"])
        cell_type, cell_label = np.unique(cell_name, return_inverse=True)
        class_num = np.array(cell_label).max() + 1
        data_label = []
        data_array = []
        for i in range(cell_label.shape[0]):
            x = np.zeros(class_num)
            x[cell_label[i]] = 1
            data_label.append(x)
        data_label = np.array(data_label)
        cell_type = np.array(cell_type)
        if not skip_exprs:
            exprs_handle = f["exprs"]
            if isinstance(exprs_handle, h5py.Group):
                mat = sparse.csr_matrix((exprs_handle["data"][...], exprs_handle["indices"][...],
                                         exprs_handle["indptr"][...]), shape=exprs_handle["shape"][...])
            else:
                mat = exprs_handle[...].astype(np.float32)
                if sparsify:
                    mat = sparse.csr_matrix(mat)
        else:
            mat = sparse.csr_matrix((obs.shape[0], var.shape[0]))
        X = np.array(mat.toarray())
        X = np.ceil(X).astype(int)
        adata = sc.AnnData(X)
        adata.obs['Group'] = cell_label
        adata = normalize(adata, copy=True, highly_genes=gene_num, size_factors=True, normalize_input=raw, logtrans_input=True)
        count = adata.X
        if raw == False:
             a = pd.DataFrame(count).T
             a.to_csv("./results/adam-raw.csv")

        return count,adata.obs['Group']
    
def normalize_for_AL(filename, gene_num, raw):
    with h5py.File(filename, "r") as f:

        gene_names = np.array(f["gene_names"][...])
        cell_names = np.array(f["cell_names"][...])
        data_label = f["Y"][...].astype(int)
        X = f["X"][...].astype(np.float32)
        
        adata = sc.AnnData(X)
        adata.obs_names = cell_names
        adata.var_names = gene_names
        adata.obs['cell_type'] = data_label

   
        adata = normalize(adata, copy=True, highly_genes=gene_num, size_factors=True, normalize_input=raw, logtrans_input=True)
        
        count = adata.X        
        group_labels = adata.obs['cell_type']
        
    return count, group_labels


def read_data(filename, sparsify=False, skip_exprs=False):
    with h5py.File(filename, "r") as f:
        obs = pd.DataFrame(dict_from_group(f["obs"]), index=utilss.decode(f["obs_names"][...]))
        var = pd.DataFrame(dict_from_group(f["var"]), index=utilss.decode(f["var_names"][...]))
        uns = dict_from_group(f["uns"])
        if not skip_exprs:
            exprs_handle = f["exprs"]
            if isinstance(exprs_handle, h5py.Group):
                mat = sp.sparse.csr_matrix((exprs_handle["data"][...], exprs_handle["indices"][...],
                                            exprs_handle["indptr"][...]), shape=exprs_handle["shape"][...])
            else:
                mat = exprs_handle[...].astype(np.float32)
                if sparsify:
                    mat = sp.sparse.csr_matrix(mat)
        else:
            mat = sp.sparse.csr_matrix((obs.shape[0], var.shape[0]))
    return mat, obs, var, uns


def prepro(data_type, filename):
    if data_type == 'csv':
        data_path = 'D:/360MoveData/Users/mi/Desktop/scCAGN-main/Cluster_model/dataset/' + filename + '/data.csv'
        label_path = 'D:/360MoveData/Users/mi/Desktop/scCAGN-main/Cluster_model/dataset/' + filename + '/label.csv'
        X = pd.read_csv(data_path, header=0, index_col=0, sep=',')
        cell_label = pd.read_csv(label_path).values[:, -1]


    if data_type == 'h5':
        data_path = "D:/360MoveData/Users/mi/Desktop/scCAGN-main/Cluster_model/dataset/" + filename + "/data.h5"
        
        # data_mat = h5py.File(data_path)
        # x = np.array(data_mat['X'])
        # y = np.array(data_mat['Y'])
        # data_mat.close()
        # adata = sc.AnnData(x,dtype='float32')
        # adata.obs['Group'] = y

        #data, data_label = data_Preprocess.nomalize_for_AF(self.filepath, 1999, raw)
        X , cell_label = normalize_for_AL(data_path, 2000,raw=False)
    # return x, y
    return X, cell_label


def Selecting_highly_variable_genes(X, highly_genes):
    adata = sc.AnnData(X)
    adata.var_names_make_unique()
    sc.pp.filter_genes(adata, min_counts=1)
    sc.pp.filter_cells(adata, min_counts=1)
    sc.pp.normalize_per_cell(adata)
    sc.pp.log1p(adata)
    adata.raw = adata
    sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5, n_top_genes=highly_genes)
    adata = adata[:, adata.var['highly_variable']].copy()
    sc.pp.scale(adata)
    data = adata.X

    return data


def normalize(adata, copy=True, highly_genes=None, filter_min_counts=True, size_factors=True, normalize_input=True,
              logtrans_input=True):
    if isinstance(adata, sc.AnnData):
        if copy:
            adata = adata.copy()
    elif isinstance(adata, str):
        adata = sc.read(adata)
    else:
        raise NotImplementedError
    norm_error = 'Make sure that the dataset (adata.X) contains unnormalized count data.'
    assert 'n_count' not in adata.obs, norm_error
    if adata.X.size < 50e6: 
        if sp.sparse.issparse(adata.X):
            assert (adata.X.astype(int) != adata.X).nnz == 0, norm_error
        else:
            assert np.all(adata.X.astype(int) == adata.X), norm_error

    if filter_min_counts:
        sc.pp.filter_genes(adata, min_counts=1)
        sc.pp.filter_cells(adata, min_counts=1)
    if size_factors or normalize_input or logtrans_input:
        adata.raw = adata.copy()
    else:
        adata.raw = adata
    if size_factors:
        sc.pp.normalize_per_cell(adata)
        adata.obs['size_factors'] = adata.obs.n_counts / np.median(adata.obs.n_counts)
    else:
        adata.obs['size_factors'] = 1.0
    if logtrans_input:
        sc.pp.log1p(adata)
    if highly_genes != None:
        sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5, n_top_genes=highly_genes,
                                    subset=True)
    if normalize_input:
        sc.pp.scale(adata)
    return adata


def getcluster(x):
    adj, edgeList = generateAdj(x)
    idx = []
    for i in range(np.array(edgeList).shape[0]):
        if np.array(edgeList)[i, -1] == 1.0:
            idx.append(i)
    listResult, size = generateLouvainCluster(edgeList)
    n_clusters = len(np.unique(listResult))
    return n_clusters

# Entry point
if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Cell_cluster',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--name', type=str, default='Yan')
    parser.add_argument('--file_format', type=str, default='csv')
    args = parser.parse_args()

    filename = args.name
    if not os.path.exists("./dataset/" + filename + "/data"):
        os.system('mkdir ./dataset/' + filename + '/data')
    if not os.path.exists("./dataset/" + filename + "/graph"):
        os.system('mkdir ./dataset/' + filename + '/graph')
    if not os.path.exists("./dataset/" + filename + "/model"):
        os.system('mkdir ./dataset/' + filename + '/model')

    data_type = args.file_format
    if data_type == 'h5':
        X, Y = prepro(data_type, filename)
        X = np.ceil(X).astype(np.float32)
        count_X = X
        cluster_number = int(max(Y) - min(Y) + 1)
        adata = sc.AnnData(X)
        Y = pd.Series(Y, index=adata.obs.index)
        adata.obs['Group'] = Y
        adata = normalize(adata, copy=True, highly_genes=2000, size_factors=True, normalize_input=True,
                          logtrans_input=True)
        X = adata.X.astype(np.float32)
        Y = np.array(adata.obs["Group"])
        high_variable = np.array(adata.var.highly_variable.index, dtype=np.int32)
        count_X = count_X[:, high_variable]
        data = (count_X.astype(np.float32))
        data = preprocessing.MinMaxScaler().fit_transform(data)
        data = preprocessing.normalize(data, norm='l2')
        directory = "D:/360MoveData/Users/mi/Desktop/scCAGN-main/Cluster_model/dataset/"+filename+"/"
        np.savetxt(os.path.join(directory, f"{filename}.txt"), data, fmt="%s", delimiter=" ")
        np.savetxt(os.path.join(directory, f"{filename}_label.txt"), Y, fmt="%s", delimiter=" ")

    if data_type == 'csv':
        X, Y = prepro(data_type, filename)
        data = np.array(X).astype('float32')
        cluster_number = int(max(Y) - min(Y) + 1)
        data = Selecting_highly_variable_genes(data, 2000)
        data = preprocessing.QuantileTransformer(random_state=0).fit_transform(data) 
        data = preprocessing.normalize(data, norm='l2')
        directory = "D:/360MoveData/Users/mi/Desktop/scCAGN-main/Cluster_model/dataset/"+filename+"/"
        np.savetxt(os.path.join(directory, f"{filename}.txt"), data, fmt="%s", delimiter=" ")
        np.savetxt(os.path.join(directory, f"{filename}_label.txt"), Y, fmt="%s", delimiter=" ")

    adj, edgeList = generateAdj(data)# Generate adjacency matrix and edge list
    idx = []
    for i in range(np.array(edgeList).shape[0]):
        if np.array(edgeList)[i, -1] == 1.0:
            idx.append(i)
    directory = "D:/360MoveData/Users/mi/Desktop/scCAGN-main/Cluster_model/dataset/"+filename+"/"
    graph_directory = os.path.join(directory, "graph")

    # 2. 创建目录（如果不存在）
    os.makedirs(graph_directory, exist_ok=True)

    # 3. 构建 graph 文件路径
    graph_path = os.path.join(graph_directory, f"{filename}_graph.txt")

    # 4. 打印数据以调试
    selected_data = np.array(edgeList)[idx, 0:-1]
    print(f"Selected Data to Save: {selected_data}")

    # 5. 保存数据
    if selected_data.size > 0:
        np.savetxt(graph_path, selected_data, fmt="%d")
        print(f"Graph 文件已保存到: {graph_path}")
    else:
        print("警告：没有数据可保存")