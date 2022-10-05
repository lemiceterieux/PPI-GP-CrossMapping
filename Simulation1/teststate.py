import umap
from sklearn.decomposition import PCA
from nilearn.glm.first_level import glover_hrf
import scipy.signal as signal
import scipy.linalg as la
import numpy as np
import matplotlib.pyplot as plt
import vgpccmbatch as gp
from torch.multiprocessing import Pool
import torch.multiprocessing as mp
from joblib import Parallel, delayed

def genManifs(i,j):
    Pop = simuls[i,500::250,:]
    Popn = np.zeros((Pop.shape[0]*13,Pop.shape[1]))
    Popn[::13] = Pop
    Pop = signal.filtfilt(b,a,Popn)
    PPI = np.load("control{0:d}.npy".format(i))[500::250]
    PPIn = np.zeros(len(PPI)*13)
    PPIn[::13] = PPI
    PPIn = np.convolve(np.ones(13),PPIn)
    PPI = PPIn[:len(PPI)*13]
    if j == 0:
        PPI = 2*((PPI > 0).astype(float)-.5)
    else:
        PPI = 2*((PPI < 0).astype(float)-.5)
    PPI = (PPI - PPI.mean())/PPI.std()
    bases = [np.ones(len(PPI))] + [np.sin(2*np.pi*(i+1)*np.arange(len(PPI))/len(PPI)) for i in range(1,480)]
    bases += [np.cos(2*np.pi*(i+1)*np.arange(len(PPI))/len(PPI)) for i in range(1,480)]
    bases = np.array(bases).T
    cm = la.convolution_matrix(hrf,len(PPI))   
    basescm = cm[:len(PPI)].dot(bases)
    B = np.linalg.inv(basescm.T.dot(basescm)).dot(basescm.T).dot(Popn)
    X = bases.dot(B)
    PPI = cm[:len(PPI)].dot((X.T*PPI).T)
    PPI = (PPI - PPI.mean(0))/PPI.std(0)
    PI = cm[:len(PPI)].dot(X)
    PI = (PI - PI.mean(0))/PI.std(0)
    Pop1 = PI[::5,50:].T
    Pop2 = PI[::5,:50].T
    um = PCA()#umap.UMAP()
    um2 = PCA()#umap.UMAP()
    PPIr = PPI[::5,50:].T#(Pop1*(PPI))
    PPIl = PPI[::5,:50].T#(Pop2*(PPI))
    um.fit(Pop1.T)
    um2.fit(Pop2.T)
    PPIr1 = um.transform(PPIr.T).T[[0]]
    PPIr2 = um2.transform(PPIr.T).T[[0]]
    PPIl1 = um.transform(PPIl.T).T[[0]]
    PPIl2 = um2.transform(PPIl.T).T[[0]]
    m1 = um.transform(Pop1.T).T[[0]]
    m2 = um2.transform(Pop2.T).T[[0]]
    return PPIr1, PPIr2, PPIl1, PPIl2, m1, m2


def test(dat,cuda):
    GP = gp.GP()
    return GP.testStateSpaceCorrelation(dat[0], dat[1][None,:], dat[2],5, tau=2, cuda=cuda)[1]

def boot(dat):#for j in range(50):
    pvals = []
    ress = []
    j = dat[0]
    cuda = (mp.current_process()._identity[0] - 1)%8
    for i in range(2):
        PPIr1 = dat[1][i]#np.copy(PPIr1s[j,i])
        PPIl2 = dat[2][i]#np.copy(PPIl2s[j,i])
        m1 = dat[3][i]#np.copy(m1s[j,i])
        m2 = dat[4][i]#np.copy(m2s[j,i])

        args = [[PPIr1, PPIl2, m1],[PPIl2, PPIr1, m2]]
#        args = [[m1, m2, PPIr1],[m2, m1, PPIl2]]
        ret = []
        for k in range(len(args)):
#            ret = p.map(test, args)
            ret+=[test(args[k],cuda)]
        r = ret[0]
        r2 = ret[1]
        res = ((r[:,None] - r2)/np.sqrt(r[:,None]**2+r2**2)).ravel()
#        ress += [res]
#        pvals += [(res[0] > res[1:]).sum()/len(res)]
#        pvals += [(res[0] > res[1:]).cpu().numpy().mean()]
        pvals += [(res[0] > res[1:]).cpu().numpy().mean()]
        ress += [np.array([r.cpu().numpy(),r2.cpu().numpy()])]#es.cpu().numpy()]
    print(pvals,j)
    return np.array(pvals), np.array(ress)

if __name__ == "__main__":
    pvals = []
    #PPI = np.zeros(1000)
    #for i in range(10):
    # cause = np.random.randint(1000-20)
    # PPI[cause:cause+20] += 1
    #
    #PPI2 = np.zeros(1000)
    #for i in range(10):
    # cause = np.random.randint(1000-20)
    # PPI2[cause:cause+30] += 1
    simuls = np.load("sampEns.npy")
    SNR = 5#float(sys.argv[1])
    simuls = simuls + .1*np.random.randn(*simuls.shape)
    hrf = glover_hrf(1*2.5/13,1,50)
    b,a = signal.butter(3,1/13,'low')

    print("Starting fit!")
    PPIr1s, PPIr2s, PPIl1s, PPIl2s, m1s, m2s = zip(*Parallel(n_jobs=250)(delayed(genManifs)(i,j) for i in range(100) for j in range(2)))
    PPIr1s = np.array(PPIr1s).reshape(100,2,1,-1)
    PPIr2s = np.array(PPIr2s).reshape(100,2,1,-1)
    PPIl1s = np.array(PPIl1s).reshape(100,2,1,-1)
    PPIl2s = np.array(PPIl2s).reshape(100,2,1,-1)
    m1s = np.array(m1s).reshape(100,2,1,-1)
    m2s = np.array(m2s).reshape(100,2,1,-1)

    ress = []
    resss = []
    pvalss = []

    mp.set_start_method('spawn',force=True)
    with Pool(8) as p:
        args = [[i,PPIr1s[i],PPIl2s[i],m1s[i],m2s[i]] for i in range(100)]
        for i in range(10):
            pvals, ress = zip(*p.map(boot,args))
            pvalss += [pvals]
            resss += [ress]
            np.save("PvalsPPIdeconv.npy", np.array(pvalss).reshape(i+1,100,2))
            np.save("ResPPIdeconv.npy", np.array(resss).reshape(i+1,100,2,2,-1))
