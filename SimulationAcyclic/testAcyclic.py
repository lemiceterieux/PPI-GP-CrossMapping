import umap
import sys
from teaspoon.parameter_selection.FNN_n import FNN_n
from sklearn.decomposition import PCA
from nilearn.glm.first_level import glover_hrf
import scipy.signal as signal
import scipy.linalg as la
import numpy as np
import matplotlib.pyplot as plt
import vgpccmbatch as gp
from torch.multiprocessing import Pool
import torch
import torch.multiprocessing as mp
from joblib import Parallel, delayed

def genManifs(i,j,conn):
    Pop = simuls[i,500::20,:]
    Popn = np.zeros((Pop.shape[0]*13,Pop.shape[1]))
    Popn[::13] = Pop
    Pop = Pop#signal.filtfilt(b,a,Popn,axis=0)
    PPI = np.load("control_2Ring{0:d}_{1:d}.npy".format(i,conn))[500::20]
    PPIn = np.zeros(len(PPI)*13)
    PPIn[::13] = PPI
    PPIn = np.convolve(np.ones(13),PPIn)
    PPI = PPI#PPIn[:len(PPI)*13]
    if j == 0:
        PPItemp = np.copy(PPI)
        PPI = 2*((PPI > 0).astype(float)-.5)
    else:
        PPI = 2*((PPI < 0).astype(float)-.5)
    bases = [np.ones(len(PPI))] + [np.sin(2*np.pi*(i+1)*np.arange(len(PPI))/len(PPI)) for i in range(1,480)]
    bases += [np.cos(2*np.pi*(i+1)*np.arange(len(PPI))/len(PPI)) for i in range(1,480)]
    bases = np.array(bases).T
    cm = la.convolution_matrix(hrf,len(PPI))   
    basescm = cm[:len(PPI)].dot(bases)
    lambs = [np.array([0])]
    lambs += [np.arange(1,480)/40]
    lambs += [np.arange(1,480)/40]
    lambs = np.concatenate(lambs)    
    B = np.linalg.inv(basescm.T.dot(basescm)+lambs*np.eye(basescm.shape[-1])).dot(basescm.T).dot(Pop)
    X = bases.dot(B)
    np.save("BOLD_{0:d}_{1:d}_diff.npy".format(i,j),Pop) 
    np.save("X_{0:d}_{1:d}_diff.npy".format(i,j),X) 
    np.save("PI_{0:d}_{1:d}_diff.npy".format(i,j),PPI) 
    PPI = (PPI - PPI.mean())/PPI.std()
    PPI = (X.T*PPI).T#cm[:len(PPI)].dot((X.T*PPI).T)
    np.save("PPI_{0:d}_{1:d}_diff.npy".format(i,j),PPI) 
    PPI = (PPI - PPI.mean(0))/PPI.std(0)
    PI = X
    PI = (PI - PI.mean(0))/PI.std(0)
    Pop1 = PI[::5,:25].T
    Pop2 = PI[::5,25:50].T
    Pop3 = PI[::5,50:75].T
    Pop4 = PI[::5,75:].T
    PPI1 = PPI[::5,:25].T
    PPI2 = PPI[::5,25:50].T
    PPI3 = PPI[::5,50:75].T
    PPI4 = PPI[::5,75:].T
    um1 = PCA() 
    um2 = PCA()
    um3 = PCA() 
    um4 = PCA()

    um1.fit(Pop1.T)
    um2.fit(Pop2.T)
    um3.fit(Pop3.T)
    um4.fit(Pop4.T)

    PPIp1 = um1.transform(PPI1.T).T[[0]]
    PPIp2 = um2.transform(PPI2.T).T[[0]]
    PPIp3 = um3.transform(PPI3.T).T[[0]]
    PPIp4 = um4.transform(PPI4.T).T[[0]]
    m1 = um1.transform(Pop1.T).T[[0]]
    m2 = um2.transform(Pop2.T).T[[0]]
    m3 = um3.transform(Pop3.T).T[[0]]
    m4 = um4.transform(Pop4.T).T[[0]]
    return PPIp1, PPIp2, PPIp3, PPIp4, m1, m2,m3,m4


def test(dat,cuda):
    GP = gp.GP()
    _,m = FNN_n(np.array(dat[0]).squeeze(),1)
    m = m*3
#    print(m, "Embedding")
    return GP.testStateSpaceCorrelation(dat[0], dat[1][:], dat[2],m+1, tau=1, cuda=cuda,ind=dat[-1])[1].cpu().numpy()

def boot(dat):#for j in range(50):
    pvals = []
    ress = []
    j = dat[0]
    cuda = (mp.current_process()._identity[0] - 1)%torch.cuda.device_count()
    for i in range(2):
        args = [[dat[1][k][i], np.array(dat[1])[:,i], np.array(dat[2])[:,i],k] for k in range(4)]#[PPIp1, [PPIp1,PPIp2,PPIp4,PPIp4], m1],[PPIl2, PPIr1, m2]]
        ret = []
        for k in range(len(args)):
            ret+=[test(args[k],cuda)]
        ret = np.array(ret)
        if np.isnan(ret).any():
            print("OH NO")
        ress += [ret]

    ress = np.array(ress)
    print(j,ress.shape)
    return ress

if __name__ == "__main__":
    pvals = []
    for conn in range(4,5):
        simuls = np.load("2Ring_{0:d}.npy".format(conn))
        SNR = 40#float(sys.argv[1])
        simuls = simuls + simuls.std(0)/SNR*np.random.randn(*simuls.shape)
        hrf = glover_hrf(1*2.5/13,1,50)
        b,a = signal.butter(3,1/13,'low')
    
        print("Starting fit!")
        PPIp1, PPIp2, PPIp3, PPIp4, m1, m2,m3,m4 = zip(*Parallel(n_jobs=250)(delayed(genManifs)(i,j,conn) for i in range(100) for j in range(2)))
        print("fit Umaps!")
        PPIp1s = np.array(PPIp1).reshape(100,2,1,-1)
        PPIp2s = np.array(PPIp2).reshape(100,2,1,-1)
        PPIp3s = np.array(PPIp3).reshape(100,2,1,-1)
        PPIp4s = np.array(PPIp4).reshape(100,2,1,-1)
        m1s = np.array(m1).reshape(100,2,1,-1)
        m2s = np.array(m2).reshape(100,2,1,-1)
        m3s = np.array(m3).reshape(100,2,1,-1)
        m4s = np.array(m4).reshape(100,2,1,-1)
   
        ress = []
        resss = []
        pvalss = []
    
        mp.set_start_method('spawn',force=True)
        with Pool(torch.cuda.device_count()) as p:
            args = [[i,[PPIp1s[i],PPIp2s[i],PPIp3s[i],PPIp4s[i]],[m1s[i],m2s[i],m3s[i],m4s[i]]] for i in range(100)]
            for i in range(10):
                ress = p.map(boot,args)
                resss += [ress]
                np.save("ResManifPPI2Ring_{0:d}.npy".format(conn), np.array(resss).reshape(i+1,100,2,4,4,-1))
