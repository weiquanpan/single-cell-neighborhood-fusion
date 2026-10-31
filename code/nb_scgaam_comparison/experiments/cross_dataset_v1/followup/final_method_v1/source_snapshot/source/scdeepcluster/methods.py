"""Additive research modifications of the authors' PyTorch class.

Network construction, clean DEC head, target distribution and training loops
are inherited from source/scDeepCluster.py, with separately recorded hooks.
"""
import math
import numpy as np
import torch
from torch import nn
from scDeepCluster import scDeepCluster

def cross_correlation(zc, za, valid, eps=1e-4, offdiag_weight=.005):
    a,b=zc[valid],za[valid]
    if len(a)<2: return (zc.sum()+za.sum())*0
    a=(a-a.mean(0))/torch.sqrt(a.var(0,unbiased=False)+eps)
    b=(b-b.mean(0))/torch.sqrt(b.var(0,unbiased=False)+eps)
    corr=a.T@b/len(a);diag=corr.diagonal()
    return (((1-diag)**2).sum()+offdiag_weight*(corr.square().sum()-diag.square().sum()))/zc.shape[1]

class NBLoss(nn.Module):
    """NB terms copied directly from author ZINBLoss; no zero inflation head."""
    def forward(self,x,mean,disp,pi=None,scale_factor=1.0,ridge_lambda=0.):
        eps=1e-10
        mean=mean*scale_factor[:,None]
        t1=torch.lgamma(disp+eps)+torch.lgamma(x+1.0)-torch.lgamma(x+disp+eps)
        t2=(disp+x)*torch.log(1.0+mean/(disp+eps))+x*(torch.log(disp+eps)-torch.log(mean+eps))
        return torch.mean(t1+t2)

class GeneDispersion(nn.Module):
    def __init__(self,genes):
        super().__init__();self.raw_theta=nn.Parameter(torch.full((genes,),math.log(math.expm1(1.)),dtype=torch.float64))
    def forward(self,h):
        return torch.nn.functional.softplus(self.raw_theta).clamp(1e-4,1e4).expand(h.shape[0],-1)

class Model(scDeepCluster):
    def __init__(self,input_dim,method='A',seed=2211,device='cuda',consistency_weight=.1,weight_floor=.25,retention=.8):
        super().__init__(input_dim,32,[256,64],[64,256],sigma=2.5 if method in ['A','NB','NB-G'] else 0.,gamma=1.,device=device)
        self.method=method;self.seed=seed;self.retention=retention
        self.consistency_weight=consistency_weight;self.weight_floor=weight_floor
        self.batch_generator=torch.Generator().manual_seed(seed+20000)
        self.augmentation_rng=np.random.RandomState(seed+100000)
        self.weight_rng=np.random.RandomState(seed+400000)
        self.augmentation_total_cells=0;self.augmentation_zero_cells=0;self.skipped_consistency_batches=0
        self.weight_history=[]
        if method in ['NB','NB-G']:
            del self._dec_pi
            self.zinb_loss=NBLoss().to(device)
        if method=='NB-G':self._dec_disp=GeneDispersion(input_dim).to(device)
        self.to(device)

    def configure_data(self,raw,statistics):
        self.raw_counts=raw;self.statistics=statistics
        self.dec_weights=torch.ones(len(raw),device=self.device,dtype=torch.float64)

    def transform(self,raw):
        # Identical float32 task normalization, then cast to author float64 model.
        value=np.asarray(raw,dtype=np.float32);lib=value.sum(1)
        # NumPy 2 promotes a float64 0-d reference scalar; NumPy 1.16 did not.
        # Preserve the frozen author's float32 normalization operation exactly.
        norm=np.log1p(value/(np.maximum(lib,np.float32(1))/np.float32(self.statistics['reference_total']))[:,None])
        norm-=self.statistics['gene_mean'];norm/=self.statistics['gene_scale']
        return norm

    def prepare_batch(self,raw,indices):
        super().prepare_batch(raw,indices)
        if self.method not in ['B','C','D']:return
        counts=raw.detach().cpu().numpy().astype(np.int64)
        augmented=self.augmentation_rng.binomial(counts,self.retention).astype(np.float32)
        self.view_valid=torch.as_tensor(augmented.sum(1)>0,device=self.device)
        self.augmentation_total_cells+=len(counts)
        self.augmentation_zero_cells+=int(np.sum(augmented.sum(1)==0))
        if self.method in ['C','D'] and int(self.view_valid.sum())<2:self.skipped_consistency_batches+=1
        self.aug_input=torch.as_tensor(self.transform(augmented),dtype=torch.float64,device=self.device)

    def forwardAE(self,x):
        if self.method in ['A','B0']:
            self.extra_loss=x.new_zeros(())
            return super().forwardAE(x)
        if self.method in ['NB','NB-G']:
            h=self.encoder(x+torch.randn_like(x)*self.sigma)
            z=self._enc_mu(h);h=self.decoder(z)
            z0=self._enc_mu(self.encoder(x));self.extra_loss=x.new_zeros(())
            return z0,self._dec_mean(h),self._dec_disp(h),None
        z0=self._enc_mu(self.encoder(x))
        za=self._enc_mu(self.encoder(self.aug_input))
        h=self.decoder(za)
        self.extra_loss=(self.consistency_weight*cross_correlation(z0,za,self.view_valid)
                         if self.method in ['C','D'] else x.new_zeros(()))
        return z0,self._dec_mean(h),self._dec_disp(h),self._dec_pi(h)

    def forward(self,x):
        if self.method in ['A','B0']:
            self.extra_loss=x.new_zeros(())
            return super().forward(x)
        z,mean,disp,pi=self.forwardAE(x)
        return z,self.soft_assign(z),mean,disp,pi

    def target_distribution(self,q):
        p=super().target_distribution(q)
        if self.method!='D':return p
        aug=self.weight_rng.binomial(np.asarray(self.raw_counts,dtype=np.int64),self.retention).astype(np.float32)
        with torch.no_grad():
            qa=self.soft_assign(self.encodeBatch(torch.as_tensor(self.transform(aug),dtype=torch.float64)))
            a=q.detach().clamp(1e-10,1);a=a/a.sum(1,keepdim=True)
            b=qa.clamp(1e-10,1);b=b/b.sum(1,keepdim=True);mid=(a+b)/2
            js=.5*(a*(a/mid).log()).sum(1)+.5*(b*(b/mid).log()).sum(1)
            w=self.weight_floor+(1-self.weight_floor)*(1-(js/math.log(2)).clamp(0,1))
            w[torch.as_tensor(aug.sum(1)==0,device=self.device)]=self.weight_floor
            self.dec_weights=w.detach()
        values=w.cpu().numpy();library=self.raw_counts.sum(1)
        row=dict(refresh=len(self.weight_history),min=float(values.min()),mean=float(values.mean()),median=float(np.median(values)),max=float(values.max()),zero_views=int((aug.sum(1)==0).sum()),pearson_with_log_library=float(np.corrcoef(values,np.log1p(library))[0,1]) if values.std()>0 else 0.)
        self.weight_history.append(row);self.notify('weights',**row)
        return p.detach()

    def cluster_loss(self,p,q):
        if self.method!='D':return super().cluster_loss(p,q)
        # Same author's per-cell KL, only the frozen reliability weights differ.
        per=(p.detach()*torch.log(p.detach()/(q+1e-6))).sum(-1)
        w=self.dec_weights[self.batch_indices.to(self.device)]/self.dec_weights.mean()
        return self.gamma*(w.detach()*per).mean()
