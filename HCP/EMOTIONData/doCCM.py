import numpy as np
import torch
from teaspoon.parameter_selection.FNN_n import FNN_n
from sklearn.decomposition import PCA
import scipy.signal as signal
import scipy.linalg as la
from nilearn.glm.first_level import glover_hrf
import umap
import torch
from nilearn import surface, datasets,image
import pandas as pd
from nilearn.glm.first_level.design_matrix import make_first_level_design_matrix
import glob
#import vgpccmbatch as gp
import vgpccmbatchChain2 as gp
from joblib import Parallel, delayed
import scipy
from scipy import sparse
import nilearn
import sys
import nibabel as nib
import warnings
from torch.multiprocessing import Pool
import torch.multiprocessing as mp



def analCaus(dat):
    t = 1
    _,m = FNN_n(dat[0].squeeze(),t)
    m *= 2
    m = m+1
    cuda = (mp.current_process()._identity[0] - 1)%torch.cuda.device_count()
    gpR = gp.GP()
    res = gpR.testStateSpaceCorrelation(dat[0], dat[1],dat[2], m,ind=dat[-2], tau=t, cuda=cuda)
    print(dat[-2],dat[-1], dat[0].shape)
    return res[1].cpu().numpy()
EVCl = [13,-92,-3]#np.array([[x,y,z] for x in np.arange(1,28) for y in np.arange(-104,-79) for z in np.arange(-16,14)  ])
EVCr = [-10,-93,-3]#np.array([[x,y,z] for x in np.arange(-26,0) for y in np.arange(-105,-75) for z in np.arange(-16,20) ])
OFAl = [40,-77,-11]#np.array([[x,y,z] for x in np.arange(30,54) for y in np.arange(-93,-63) for z in np.arange(-20,-1)  ])
OFAr = [-39,-78,-12]#np.array([[x,y,z] for x in np.arange(-53,-26) for y in np.arange(-93,-63) for z in np.arange(-20,1) ])
FFAl = [42,-53,-18]#np.array([[x,y,z] for x in np.arange(32,52) for y in np.arange(-70,-34) for z in np.arange(-28,-4)  ])
FFAr = [-40,-53,-18]#np.array([[x,y,z] for x in np.arange(-52,-30) for y in np.arange(-72,-35) for z in np.arange(-29,-5)])
STSl = [53,-50,12]#np.array([[x,y,z] for x in np.arange(39,69) for y in np.arange(-68,-30) for z in np.arange(-1,26)   ])
STSr = [-54,-51,12]#np.array([[x,y,z] for x in np.arange(-68,-39) for y in np.arange(-69,-29) for z in np.arange(-1,28) ]) 

def runMVPA(Num,d, act):
    argsAll = []
    ucondAll = []
    for Phase in ["LR", "RL"]:
        if sys.argv[-1] != "preload":
            print("NO PRELOAD")
            meanimg = nib.load(d + "EMOTION_" + Phase + "/tfMRI_EMOTION_"+Phase+".nii.gz")
            print("LOADED")
            meanimg = image.clean_img(meanimg,confounds=np.loadtxt(d + "EMOTION_" + Phase + "/Movement_Regressors.txt"),high_pass=0.008,t_r=0.72)
            print("CLEANED",EVCl.shape)
            np.save(d.split("/")[-2] + "{0}_MNI.npy".format(Phase),meanimg.get_fdata())
            affinv = np.linalg.inv(meanimg.affine)
            mEVCl = np.unique(nib.affines.apply_affine(affinv,EVCl).astype(int)) 
            mEVCr = np.unique(nib.affines.apply_affine(affinv,EVCr).astype(int))
            mOFAl = np.unique(nib.affines.apply_affine(affinv,OFAl).astype(int))
            mOFAr = np.unique(nib.affines.apply_affine(affinv,OFAr).astype(int))
            mFFAl = np.unique(nib.affines.apply_affine(affinv,FFAl).astype(int))
            mFFAr = np.unique(nib.affines.apply_affine(affinv,FFAr).astype(int))
            mSTSl = np.unique(nib.affines.apply_affine(affinv,STSl).astype(int))
            mSTSr = np.unique(nib.affines.apply_affine(affinv,STSr).astype(int))
            ROIs = [mEVCl,mEVCr,mOFAl,mOFAr,mFFAl,mFFAr,mSTSl,mSTSr]
            meanimg = meanimg.get_fdata()
        else:
            meanimg = nib.load(d + "EMOTION_" + Phase + "/tfMRI_EMOTION_"+Phase+".nii.gz")
            affinv = np.linalg.inv(meanimg.affine)
            mEVCl =nib.affines.apply_affine(affinv,EVCl).astype(int) 
            mEVCr =nib.affines.apply_affine(affinv,EVCr).astype(int)
            mOFAl =nib.affines.apply_affine(affinv,OFAl).astype(int)
            mOFAr =nib.affines.apply_affine(affinv,OFAr).astype(int)
            mFFAl =nib.affines.apply_affine(affinv,FFAl).astype(int)
            mFFAr =nib.affines.apply_affine(affinv,FFAr).astype(int)
            mSTSl =nib.affines.apply_affine(affinv,STSl).astype(int)
            mSTSr =nib.affines.apply_affine(affinv,STSr).astype(int)
            ROIs = [mEVCl,mEVCr,mOFAl,mOFAr,mFFAl,mFFAr,mSTSl,mSTSr]
            meanimg = nib.Nifti1Image(np.load(d.split("/")[-2] + "{0}_MNI.npy".format(Phase)),affine=meanimg.affine)
            meanimg = image.smooth_img(meanimg,4).get_fdata()
            print(Num,d.split("/")[-2] + "{0}_MNI.npy".format(Phase),meanimg.shape,mSTSr.shape)
        ucond = ["Faces", "Shapes"] 
        TR = 0.72
        n_scans = meanimg.shape[-1]
        frame_times = TR * np.arange(n_scans)
        duration = TR * np.ones(n_scans)
        face = np.loadtxt(d+"EMOTION_" + Phase + "/EVs/fear.txt")
        shapes= np.loadtxt(d+"EMOTION_" + Phase + "/EVs/neut.txt")
        onsets = [F[0] + TR*i for F in face for i in range(int(F[1]))] + [F[0] + TR*i  for F in shapes for i in range(int(F[1]))]
        trialtype= ["Faces" for F in face for i in range(int(F[1]))] + ["Shapes" for F in shapes for i in range(int(F[1]))]
        duration = [TR for F in face for i in range(int(F[1]))] + [TR for F in shapes for i in range(int(F[1]))]
        events = pd.DataFrame({'onset': onsets, 'trial_type': trialtype, 'duration': duration})
        design = make_first_level_design_matrix(frame_times,
                events,
                high_pass=0.008,
                hrf_model=None,
                )
        psych = [design[u].values for u in ucond]
        for u in range(len(psych)):
            psych[u] = (psych[u]-psych[u].mean())/psych[u].std()
    
    
        hrf = glover_hrf(1*0.75,1,50)
        meanimgds = []
        time = np.arange(meanimg.shape[-1])
        meanimgt = meanimg[...,time]
        bases = [np.ones(time.shape[0])] + [np.sin(2*np.pi*(i+1)*np.arange(time.shape[0])/time.shape[0]) for i in range(1,240)]
        bases += [np.cos(2*np.pi*(i+1)*np.arange(time.shape[0])/time.shape[0]) for i in range(1,240)]
        bases = np.array(bases).T
        cm = la.convolution_matrix(hrf,time.shape[0])
        basescm = cm[:time.shape[0]].dot(bases)
        lambs = [np.array([0])]
        lambs += [np.arange(1,240)/40]
        lambs += [np.arange(1,240)/40]
        lambs = np.concatenate(lambs)
        B = np.linalg.inv(basescm.T.dot(basescm)+lambs*np.eye(bases.shape[1])).dot(basescm.T).dot(meanimgt.reshape(-1,meanimgt.shape[-1]).T)
        meanimg = bases.dot(B).T
        meanimg = meanimg.reshape(*meanimgt.shape)
        lPPI = [ps*meanimg for ps in psych]
    
        def getL(i,j):
            lcl = []
            sigs = []
            lcl = lPPI[j][ROIs[i][0],ROIs[i][1],ROIs[i][2]]
            lmu = meanimg[ROIs[i][0],ROIs[i][1],ROIs[i][2]]
            return lcl.reshape(-1,lPPI[j].shape[-1]),lmu.reshape(-1,lPPI[j].shape[-1])
    
        lcl_claim = []
        lmu_all = []
        for j in range(len(lPPI)):
            lcl_temp, lmu_temp= zip(*[getL(i,j) for i in range(len(ROIs))])
            lcl_claim += [np.array(lcl_temp)]
            lmu_all += [np.array(lmu_temp)]
        args = [[[]for i in range(len(ROIs))] for k in range(len(lPPI))]
        lcl_claim = np.array(lcl_claim)
        lmu_all = np.array(lmu_all)
    
    
        totest = [[] for k in range(len(lPPI))]
        passedL = []
        passedR = []
        for k in range(len(lPPI)):
            
            totest = []
            ttoestN = []
            for j in range(len(ROIs)):
                if lcl_claim[k][j] is None:
                    continue
                totest += [lcl_claim[k][j]]
            for kk in range(len(lPPI)):
                if kk == k:
                    continue
                else:
                    for j in range(len(ROIs)):
                        if lcl_claim[k][j] is None:
                            continue
                        totest += [lcl_claim[kk][j]]

            for i in range(len(ROIs)):
                if lcl_claim[k][j] is None:
                    continue
                args[k][i] = [lcl_claim[k][i], totest,lmu_all[k],i,ucond[k] + "_" + Phase]
    
        argsAll += args
        ucondAll += [ucond[k] + "_" + Phase for k in range(len(ucond))]
    return [argsAll, ucondAll]


if __name__ == "__main__":
    args = []
    args,ucond = zip(*Parallel(n_jobs=25)(delayed(runMVPA)(i,d,"") for i,d in enumerate(glob.glob("./*/")[:]) if not "cache" in d))
    for i in range(0,10):
        for d in range(len(args)):
            for dd in range(len(args[d])):
                mp.set_start_method('spawn',force=True)
                with Pool(torch.cuda.device_count()) as p:
                    print(ucond[d][dd],i,d,dd)
                    res = p.map(analCaus, args[d][dd])
                    np.save("{0:d}_{1:d}_MNIparcs_{2}cause.npy".format(i,d,ucond[d][dd]),res)
    
