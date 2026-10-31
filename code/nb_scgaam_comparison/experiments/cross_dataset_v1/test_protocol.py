from common import *
import unittest
import numpy as np
from scipy import sparse
from readouts import coassociation_knn, merge_connectivity
from evaluate import metrics
class Tests(unittest.TestCase):
    def test_coassociation(self):
        p=np.array([[0,0,1,1,2],[5,5,6,7,7]])
        expected=np.mean(p[:,:,None]==p[:,None,:],axis=0);np.fill_diagonal(expected,0)
        np.testing.assert_allclose(coassociation_knn(p,k=4,block=2).toarray(),expected)
    def test_metrics_permutation_and_split(self):
        y=np.array([0,0,1,1,1]);m=metrics(y,np.array([9,9,7,7,7]))
        self.assertEqual(m['ari'],1);self.assertEqual(m['worst_group_recall'],1)
        m=metrics(y,np.array([0,0,1,2,2]));self.assertAlmostEqual(m['worst_max_cluster_coverage'],2/3)
    def test_merge(self):
        a=sparse.csr_matrix([[0,5,0,0],[5,0,0,0],[0,0,0,5],[0,0,5,0]])
        y=merge_connectivity(a,np.arange(4),2)
        self.assertEqual(y[0],y[1]);self.assertEqual(y[2],y[3]);self.assertNotEqual(y[0],y[2])
    def test_heldout_block(self):
        with self.assertRaises(PermissionError):prepared('Muraro')
    def test_label_guard(self):
        with self.assertRaises(PermissionError):guard('open',('data/evaluation_only/Worm/reference_labels.npy',))
    def test_context_sparse_equivalence(self):
        from train_r6 import context_graph,encode_batch,StructuralGraph
        import torch
        rng=np.random.default_rng(5);h=rng.normal(size=(40,32)).astype(np.float32)
        a,two,c,d=context_graph(h);model=StructuralGraph().float();ht=torch.tensor(h);ct=torch.tensor(c);dt=torch.tensor(d)
        with torch.no_grad():
            full=model.encode(ht,ct,dt);batched=torch.cat([encode_batch(model,ht,ct[i:i+7],dt[i:i+7]) for i in range(0,len(h),7)])
        torch.testing.assert_close(full,batched,rtol=1e-5,atol=1e-6)
        self.assertTrue(np.array_equal(c[:,0],np.arange(len(h))))
        for i in range(len(h)):
            direct=c[i,1:16];direct=direct[direct>=0];indirect=c[i,16:];indirect=indirect[indirect>=0]
            self.assertTrue(np.all(a[i,direct].toarray().ravel()>0))
            self.assertTrue(np.all(a[i,indirect].toarray().ravel()==0))
            self.assertTrue(np.all(two[i,indirect].toarray().ravel()))
if __name__=='__main__':unittest.main()
