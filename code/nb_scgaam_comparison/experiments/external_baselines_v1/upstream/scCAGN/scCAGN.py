from __future__ import print_function, division
import os
import numpy as np
import pandas as pd
from preprocess import *
from pretrain import *
import sys
import argparse
import random
from sklearn.cluster import SpectralBiclustering, KMeans, kmeans_plusplus, DBSCAN, SpectralClustering
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.optim import Adam, SGD, Adamax
from torch.utils.data import DataLoader
from torch.nn import Linear, MultiheadAttention
from utils import load_data, load_graph
from GNN import GNNLayer
import umap
from evaluation import eva, eva_pretrain
from collections import Counter
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from collections import OrderedDict
import torch.nn.init as init
 
# UMAP visualization plot
def plot(X, fig, col, size, true_labels, ann):
    """
    Create scatter plot for UMAP visualization.
    """
    ax = fig.add_subplot(1, 1, 1)
    for i, point in enumerate(X):
        ax.scatter(point[0], point[1], s=size, c=col[true_labels[i]], label=ann[i])

def plotClusters(hidden_emb, true_labels, ann):
    """
    Perform UMAP dimensionality reduction and visualize clusters.
    """
    Umap = umap.UMAP(random_state=42)
    X_umap = Umap.fit_transform(hidden_emb)  # Reduce dimensions to 2D
    fig2 = plt.figure(figsize=(10, 10), dpi=500)
    plot(X_umap, fig2, ['green', 'brown', 'purple', 'orange', 'yellow', 'hotpink', 'red', 'cyan', 'blue'], 8, true_labels, ann)
    handles, labels = fig2.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    fig2.legend(by_label.values(), by_label.keys(), loc="upper right")
    fig2.savefig("D:/360MoveData/Users/mi/Desktop/scCAGN-main/Cluster_model/dataset/" + args.name + "/UMAP.pdf")
    plt.close()

# Set random seed for reproducibility
def init_seed(seed):
    """
    Initialize random seed for reproducibility in NumPy, PyTorch, and Python.
    """
    torch.cuda.cudnn_enabled = False
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

# Pretrain clustering model
def pretarin_cluster(n_clusters, x, device):
    """
    Pretrain the autoencoder for clustering and return the best pretrain model index based on silhouette score.
    """
    Auto = args.Auto
    device = device
    silhouette_pre = []
    print("Start pretrain")
    for i in range(args.pretrain_frequency):
        print("pretrain:" + str(i))
        model = AE(
            n_enc_1=100,
            n_enc_2=200,
            n_enc_3=200,
            n_dec_1=200,
            n_dec_2=200,
            n_dec_3=100,
            n_input=2000,
            n_z=8).to(device)
        dataset = LoadDataset(x)
        epoch = args.pretrain_epoch
        silhouette = pretrain_ae(model, dataset, i, device, n_clusters, epoch, args.name, Auto=Auto)
        silhouette_pre.append(silhouette)
    silhouette_pre = np.array(silhouette_pre)
    premodel_i = np.where(silhouette_pre == np.max(silhouette_pre))[0][0]
    print("Pretrain end")
    return premodel_i

# Discriminator for adversarial learning
class Discriminator(nn.Module):
    """
    Discriminator network for adversarial learning.
    """
    def __init__(self, n_input, n_hidden, device):
        super(Discriminator, self).__init__()
        self.device = device
        self.model = nn.Sequential(
            nn.Linear(n_input, n_hidden).to(device),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(n_hidden).to(device),
            nn.Linear(n_hidden, n_hidden).to(device),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(n_hidden).to(device),
            nn.Linear(n_hidden, 1).to(device),
            nn.Sigmoid()
        )
        
    def forward(self, z):
        """
        Forward pass of the discriminator.
        """
        z = z.to(self.device)
        return self.model(z)

class AE_train(nn.Module):
    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3, n_input, n_z):
        super(AE_train, self).__init__()
        self.enc_1 = Linear(n_input, n_enc_1).to(device)
        self.enc_2 = Linear(n_enc_1, n_enc_2).to(device)
        self.enc_3 = Linear(n_enc_2, n_enc_3).to(device)
        self.z_layer = Linear(n_enc_3, n_z).to(device)

        self.dec_1 = Linear(n_z, n_dec_1).to(device)
        self.dec_2 = Linear(n_dec_1, n_dec_2).to(device)
        self.dec_3 = Linear(n_dec_2, n_dec_3).to(device)
        self.x_bar_layer = Linear(n_dec_3, n_input).to(device)

    def forward(self, x):
        enc_h1 = F.relu(self.enc_1(x))
        enc_h2 = F.relu(self.enc_2(enc_h1))
        enc_h3 = F.relu(self.enc_3(enc_h2))
        z = self.z_layer(enc_h3)

        dec_h1 = F.relu(self.dec_1(z))
        dec_h2 = F.relu(self.dec_2(dec_h1))
        dec_h3 = F.relu(self.dec_3(dec_h2))
        x_bar = self.x_bar_layer(dec_h3)
        return x_bar, enc_h1, enc_h2, enc_h3, z

# Cross Attention Mechanism
class CrossAttention(nn.Module):
    """
    Cross-attention mechanism for feature fusion between GCN and autoencoder.
    """
    def __init__(self, embed_dim, num_heads):
        super(CrossAttention, self).__init__()
        self.multihead_attention = MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)

    def forward(self, query, key, value):
        """
        Perform cross-attention using query, key, and value.
        """
        query = query.unsqueeze(1)  
        key = key.unsqueeze(1)
        value = value.unsqueeze(1)
        attention_output, _ = self.multihead_attention(query, key, value)  # Multi-head attention output
        return attention_output.squeeze(1)  # Remove extra dimension

# Clustering Model
class ClusterModel(nn.Module):
    """
    Main clustering model with GCN, AAE, and cross-attention mechanism.
    """
    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3, n_input, n_z, n_clusters, v=1):
        super(ClusterModel, self).__init__()
        # Autoencoder for feature embedding
        self.ae = AE_train(
            n_enc_1=n_enc_1,
            n_enc_2=n_enc_2,
            n_enc_3=n_enc_3,
            n_dec_1=n_dec_1,
            n_dec_2=n_dec_2,
            n_dec_3=n_dec_3,
            n_input=n_input,
            n_z=n_z)
        self.ae.load_state_dict(torch.load(args.pretrain_path, map_location=device))
        # Graph Convolutional Network (GCN)
        self.gnn_1 = GNNLayer(n_input, n_enc_1)
        self.gnn_2 = GNNLayer(n_enc_1, n_enc_2)
        self.gnn_3 = GNNLayer(n_enc_2, n_z)
        self.gnn_4 = GNNLayer(n_z, n_clusters)
        
        # Cross Attention for feature fusion
        self.cross_attention = CrossAttention(embed_dim=n_z, num_heads=4)
        
        # Cluster layer for self-supervised clustering
        self.cluster_layer = Parameter(torch.Tensor(n_clusters, n_z).to(device))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)
        self.v = v

    def forward(self, x, adj):
        """
        Forward pass for GCN, AAE, and cross-attention mechanism.
        """
        # GCN module
        h1 = self.gnn_1(x, adj)
        h2 = self.gnn_2(h1, adj)
        h3 = self.gnn_3(h2, adj)
        h4 = self.gnn_4(h3, adj)
        predict = F.softmax(h4, dim=1)

        # Autoencoder module
        enc_h1 = F.relu(self.ae.enc_1(x))
        enc_h2 = F.relu(self.ae.enc_2(enc_h1 + h1))  # Fusion with GCN layer
        enc_h3 = F.relu(self.ae.enc_3(enc_h2 + h2))
        z = self.ae.z_layer(enc_h3 + h2)

        # Cross-attention between GCN and autoencoder features
        h3_cross_attention = self.cross_attention(z, h3, h3)

        # Decoding through autoencoder
        dec_h1 = F.relu(self.ae.dec_1(h3_cross_attention + h3))
        dec_h2 = F.relu(self.ae.dec_2(dec_h1 + h2))
        dec_h3 = F.relu(self.ae.dec_3(dec_h2 + h2))
        x_bar = self.ae.x_bar_layer(dec_h3 + h1)
        
        # Self-supervised clustering
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q = q.pow((self.v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()

        return x_bar, q, predict, z

#p-distribution
def target_distribution(q, tau=0.5):
    scaled_q = q.pow(1.0 / tau)
    weight = scaled_q**2 / scaled_q.sum(0)
    return (weight.t() / weight.sum(1)).t()

def adjust_learning_rate(optimizer, epoch):
    lr = 0.001 * (0.1 ** (epoch // 20))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train_cluster(dataset, n_clusters, device):
    Auto = args.Auto
    # Dynamically adjust the number of clusters if Auto mode is enabled
    if Auto:
        if dataset.x.shape[0] < 2000:
            resolution = 0.8
        else:
            resolution = 0.5
        n_clusters = int(n_clusters * resolution) if int(n_clusters * resolution) >= 3 else 2
    else:
        n_clusters = n_clusters
    
    # Initialize the clustering model and discriminator
    model = ClusterModel(100, 200, 200, 200, 200, 100,
                         n_input=args.n_input,
                         n_z=args.n_z,
                         n_clusters=n_clusters).to(device)
    discriminator = Discriminator(n_input=args.n_z, n_hidden=64,device=device)
    
    # Optimizers for the clustering model and discriminator
    optimizer = Adamax(model.parameters(), lr=args.lr)
    optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=0.0002)
    
    # Load KNN graph adjacency matrix
    adj = load_graph(args.name)
    adj = adj.to(device)
    
    # Prepare input data and labels
    data = torch.Tensor(dataset.x).to(device)
    y = dataset.y
    with torch.no_grad():
        _, _, _, _, z = model.ae(data)
        z=z.to(device)
    kmeans = KMeans(n_clusters=n_clusters, n_init=20)
    y_pred = kmeans.fit_predict(z.data.cpu().numpy())

    model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(device)

    best_nmi = -1
    best_model_state = None
    best_epoch_result = None

    # Training loop
    for epoch in range(args.Train_epoch):
        adjust_learning_rate(optimizer, epoch)
        _, tmp_q, pred, _ = model(data, adj)
        tmp_q = tmp_q.data.detach().to(device)
        p = target_distribution(tmp_q,tau=0.5)

        res1 = tmp_q.cpu().numpy().argmax(1)  #Q
        nmi,ari,ami, completeness=eva(tmp_q.cpu().numpy(),y, res1) 

        # Save the best model based on NMI score
        if nmi > best_nmi:
            best_nmi = nmi
            best_model_state = model.state_dict()
            best_epoch_result = (nmi, ari, ami, completeness, res1, tmp_q.cpu().numpy())

        x_bar, q, pred, _ = model(data, adj)
        kl_loss = F.kl_div(q.log(), p, reduction='batchmean')
        ce_loss = F.kl_div(pred.log(), p, reduction='batchmean')

        # Create labels for adversarial learning
        def create_labels(size, value, device):
            return torch.full((size,), value, dtype=torch.float).to(device)
        real_label = create_labels(z.size(0), 1, device)
        fake_label = create_labels(z.size(0), 0, device)
        fake_z = torch.randn_like(z).to(device)
        
        # Train the discriminator
        optimizer_discriminator.zero_grad()
        real_loss = F.binary_cross_entropy(discriminator(z), real_label.unsqueeze(1))
        fake_loss = F.binary_cross_entropy(discriminator(fake_z.detach()), fake_label.unsqueeze(1))
        D_loss = (real_loss + fake_loss) / 2
        D_loss.backward()
        optimizer_discriminator.step()

        # Train the clustering model
        optimizer.zero_grad()        
        reconstruction_loss = F.mse_loss(x_bar, data)
        G_loss = F.binary_cross_entropy(discriminator(z), real_label.unsqueeze(1))
        total_loss =  0.0001 * kl_loss + 0.001 * ce_loss + 0.5 * G_loss + 0.5* reconstruction_loss
        total_loss.backward()
        optimizer.step()

    # Load the best model based on NMI score
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        nmi, ari, ami, completeness, res1, tmp_q = best_epoch_result
        print('最优模型指标:',
              'nmi {:.4f}'.format(nmi), ', ari {:.4f}'.format(ari),
              ', ami {:.4f}'.format(ami), ', completeness {:.4f}'.format(completeness))
    
    # Save pre-trained embeddings and cluster assignments
    np.savetxt("D:/360MoveData/Users/mi/Desktop/scCAGN-main/Cluster_model/dataset/" + args.name + "/pre_embedding.txt", tmp_q, fmt="%s", delimiter=",")
    np.savetxt("D:/360MoveData/Users/mi/Desktop/scCAGN-main/Cluster_model/dataset/" + args.name + "/pre_label.csv", res1, fmt="%s", delimiter=",")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Cell_cluster',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--name', type=str, default='Adam')
    parser.add_argument('--seed', type=int, default=2022, help='Random seed.')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--n_z', default=8, type=int)
    parser.add_argument('--pretrain_epoch', default=50, type=int)
    parser.add_argument('--pretrain_frequency', default=20, type=int)
    parser.add_argument('--Train_epoch', default=30, type=int)
    parser.add_argument('--n_input', default=2000, type=int)
    parser.add_argument('--pretrain_path', type=str, default='pkl') 
    parser.add_argument('--Auto', default=False)
    parser.add_argument('--pretain', default=False)
    parser.add_argument('--device', type=str, default='cuda', help="Device: 'cuda' or 'cpu'")
    args = parser.parse_args()

    # Initialize random seed
    init_seed(args.seed)
    args.cuda = torch.cuda.is_available()
    print("use cuda: {}".format(args.cuda))

    if not os.path.exists("D:/360MoveData/Users/mi/Desktop/scCAGN-main/Cluster_model/dataset/" + args.name + "/data"):
        os.system('mkdir D:/360MoveData/Users/mi/Desktop/scCAGN-main/Cluster_model/dataset/' + args.name + '/data')
    if not os.path.exists("D:/360MoveData/Users/mi/Desktop/scCAGN-main/Cluster_model/dataset/" + args.name + "/graph"):
        os.system('mkdir D:/360MoveData/Users/mi/Desktop/scCAGN-main/Cluster_model/dataset/' + args.name + '/graph')
    if not os.path.exists("D:/360MoveData/Users/mi/Desktop/scCAGN-main/Cluster_model/dataset/" + args.name + '/model'):
        os.system('mkdir D:/360MoveData/Users/mi/Desktop/scCAGN-main/Cluster_model/dataset/' + args.name + '/model')

    device = torch.device("cuda" if torch.cuda.is_available() and args.device == 'cuda' else "cpu")
    
    # Load data
    x = np.loadtxt("D:/360MoveData/Users/mi/Desktop/scCAGN-main/Cluster_model/dataset/"+args.name+"/"+args.name+".txt", dtype=float)
    y = np.loadtxt("D:/360MoveData/Users/mi/Desktop/scCAGN-main/Cluster_model/dataset/"+args.name+"/"+args.name+"_label.txt", dtype=int)#quake_smart-seq2_trachea
    if args.Auto:
        auto_clusters = getcluster(x)
        n_clusters = auto_clusters
    else:
        n_clusters = int(max(y) - min(y) + 1)

    # Pretraining if enabled
    if args.pretain:
        premodel_i = pretarin_cluster(n_clusters, x, device)
        args.pretrain_path = 'D:/360MoveData/Users/mi/Desktop/scCAGN-main/Cluster_model/dataset/' + args.name + '/model/' + args.name + str(premodel_i) + '.pkl'
    else:
        args.pretrain_path = 'D:/360MoveData/Users/mi/Desktop/scCAGN-main/Cluster_model/dataset/' + args.name + '/model/' + args.name + '.pkl'
    
    # Train clustering model
    dataset = load_data(args.name)
    train_cluster(dataset, n_clusters, device)