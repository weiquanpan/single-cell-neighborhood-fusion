from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
from scipy.optimize import linear_sum_assignment as linear
from sklearn.metrics import adjusted_mutual_info_score as ami_score
from sklearn.metrics import silhouette_score
from sklearn.metrics import completeness_score

def eva(X,y_true, y_pred):
    nmi = nmi_score(y_true, y_pred, average_method='arithmetic')
    ari = ari_score(y_true, y_pred)
    ami = ami_score(y_true, y_pred)
    completeness = completeness_score(y_true, y_pred)
    return nmi, ari, ami, completeness

def eva_pretrain(X, y_pred, epoch=0):
    silhouette = silhouette_score(X, y_pred,metric='euclidean')
    return silhouette