import umap
import simulateProcesses as sp
import numpy as np
import matplotlib.pyplot as plt
import vgpccmbatch as gp
from sklearn.decomposition import PCA
from torch.multiprocessing import Pool
import torch.multiprocessing as mp
from joblib import Parallel, delayed

pvals = []
PPI = np.zeros(1000)
for i in range(10):
 cause = np.random.randint(1000-20)
 PPI[cause:cause+20] += 1

PPI2 = np.zeros(1000)
for i in range(10):
 cause = np.random.randint(1000-20)
 PPI2[cause:cause+30] += 1
def genManifs(i):
    eps1 = i
    x6 = np.array(sp.lorenzDrivesRossler(N=1000-1,dnoise=1e-5, h=.05, eps=eps1, eps2=0.2, PPI=PPI, PPI2=PPI2, initial=np.random.randn(6))) + 1*np.random.randn(6,1000)
    um = PCA()#umap.UMAP()
    um2 = PCA()#umap.UMAP()
    PPII = PPI 
    PPII = (PPII - PPII.mean())/(PPII.std())
    PPIr = x6[:3]*(PPII[:,None]*x6[:3].std(1)).T
    PPIl = x6[3:]*(PPII[:,None]*x6[3:].std(1)).T
    um.fit(x6[:3].T)
    um2.fit(x6[3:].T)
    PPIr1 = um.transform(PPIr.T).T[[0]]
    PPIr2 = um2.transform(PPIr.T).T[[0]]
    PPIl1 = um.transform(PPIl.T).T[[0]]
    PPIl2 = um2.transform(PPIl.T).T[[0]]
    m1 = um.transform(x6[:3].T).T[[0]]
    m2 = um2.transform(x6[3:].T).T[[0]]
    return PPIr1, PPIr2, PPIl1, PPIl2, m1, m2

print("Starting fit!")
PPIr1s, PPIr2s, PPIl1s, PPIl2s, m1s, m2s = zip(*Parallel(n_jobs=250)(delayed(genManifs)(i) for j in range(50) for i in range(5)))
print("fit!")
PPIr1s = np.array(PPIr1s).reshape(50,5,2,1000)
PPIr2s = np.array(PPIr2s).reshape(50,5,2,1000)
PPIl1s = np.array(PPIl1s).reshape(50,5,2,1000)
PPIl2s = np.array(PPIl2s).reshape(50,5,2,1000)
m1s = np.array(m1s).reshape(50,5,2,1000)
m2s = np.array(m2s).reshape(50,5,2,1000)

def test(dat,cuda):
    GP = gp.GP()
    return GP.testStateSpaceCorrelation(dat[0], dat[1][None,:], dat[2],13, tau=2, cuda=cuda)[1]

ress = []
def boot(j):#for j in range(50):
    pvals = []
    ress = []
    j = j[0]
    cuda = (mp.current_process()._identity[0] - 1)%8
    for i in range(5):
        PPIr1 = np.copy(PPIr1s[j,i])
        PPIr2 = np.copy(PPIr2s[j,i])
        PPIl1 = np.copy(PPIl1s[j,i])
        PPIl2 = np.copy(PPIl2s[j,i])
        m1 = np.copy(m1s[j,i])
        m2 = np.copy(m2s[j,i])

#        args = [[PPIr1, PPIl1, m1],[PPIr2, PPIl2, m2]]
#        args = [[m1, m2, PPIr1],[m2, m1, PPIl2]]
        args = [[PPIr1, PPIl2, m1],[PPIl2, PPIr1, m2]]
        ret = []
        for k in range(len(args)):
#            ret = p.map(test, args)
            ret+=[test(args[k],cuda)]
        r = ret[0]
        r2 = ret[1]
        res = (r[:,None] - r2).ravel()
#        ress += [res]
#        pvals += [(res[0] > res[1:]).sum()/len(res)]
#        pvals += [(res[0] > res[1:]).cpu().numpy().mean()]
        pvals += [(res[0] > res[1:]).cpu().numpy().mean()]
        ress += [res.cpu().numpy()]
    print(pvals,j)
    return np.array(pvals), np.array(ress)
resss = []
pvalss = []
with Pool(8) as p:
    args = [[i] for i in range(50)]
    for i in range(10):
        pvals, ress = zip(*p.map(boot,args))
        pvalss += [pvals]
        resss += [ress]
        np.save("PvalsLorenzManifold.npy", np.array(pvalss).reshape(i+1,50,5))
        np.save("ResLorenzManifold.npy", np.array(resss).reshape(i+1,50,5,-1))
