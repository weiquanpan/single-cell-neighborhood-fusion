"""Label-free equal-count random edges, isolated from training RNG."""
import numpy as np

def sample_edges(n, expression, count, seed):
    assert 0 <= count <= n*(n-1)//2-len(expression)
    rng=np.random.default_rng(seed)
    added=set()
    while len(added)<count:
        candidates=rng.integers(0,n,size=(min(65536,max(1024,2*(count-len(added)))),2))
        for u,v in candidates:
            u,v=int(u),int(v)
            if u==v: continue
            pair=(min(u,v),max(u,v))
            if pair not in expression: added.add(pair)
            if len(added)==count: break
    return added
