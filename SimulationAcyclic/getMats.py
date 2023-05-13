import numpy as np
Res = np.mean(np.load("ResPPI2Ring_4.npy"),0)
effectMat = np.zeros((Res.shape[0],2,4,4))
pvalMat = np.zeros((Res.shape[0],2,4,4))
import scipy.stats as stats
for i in range(4):
    for j in range(4):
        diff = (Res[:,:,i,j] - Res[:,:,j,i])
        temp = (diff[...,0] - np.mean(diff[...,1:],-1))
        p = (diff[...,[0]] > diff[...,1:]).mean(-1)
        effectMat[:,:,i,j] = temp
        pvalMat[:,:,i,j] = p

np.mean(effectMat,0)
contrast = effectMat[:,1] - effectMat[:,0]
print(stats.ttest_1samp(contrast,0,axis=0,alternative='less')[1]<0.05)
