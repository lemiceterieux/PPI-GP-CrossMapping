import numpy as np
from nilearn.glm.first_level import FirstLevelModel
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import RidgeClassifier as RC
from sklearn.svm import SVC
import warnings
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
from nilearn.masking import compute_brain_mask
import glob
#import vgpccmbatch as gp
import vgpccmbatchChain as gp
from joblib import Parallel, delayed
import scipy
from scipy import sparse
import nilearn
import sys
import nibabel as nib
import warnings
from torch.multiprocessing import Pool
import torch.multiprocessing as mp


EVCl = [13,-92,-3]#np.array([[x,y,z] for x in np.arange(1,28) for y in np.arange(-104,-79) for z in np.arange(-16,14)  ])
EVCllimx = [1,28]
EVCllimy = [-104,-79]
EVCllimz = [-16,14]
EVCr = [-10,-93,-3]#np.array([[x,y,z] for x in np.arange(-26,0) for y in np.arange(-105,-75) for z in np.arange(-16,20) ])
EVCrlimx = [-26,0]
EVCrlimy = [-105,-75]
EVCrlimz = [-16,20]
EVCllim = [EVCllimx,EVCllimy,EVCllimz]
EVCrlim = [EVCrlimx,EVCrlimy,EVCrlimz]
EVClcords = [[x,y,z] for x in EVCllim[0] for y in EVCllim[1] for z in EVCllim[2]]
EVCrcords = [[x,y,z] for x in EVCrlim[0] for y in EVCrlim[1] for z in EVCrlim[2]]
OFAl = [40,-77,-11]#np.array([[x,y,z] for x in np.arange(30,54) for y in np.arange(-93,-63) for z in np.arange(-20,-1)  ])
OFAr = [-39,-78,-12]#np.array([[x,y,z] for x in np.arange(-53,-26) for y in np.arange(-93,-63) for z in np.arange(-20,1) ])
OFAllimx = [30,54]
OFAllimy = [-93,-63]
OFAllimz = [-20,-1]
OFArlimx = [-53,-26]
OFArlimy = [-93,-63]
OFArlimz = [-20,-1]
OFAllim = [OFAllimx,OFAllimy,OFAllimz]
OFArlim = [OFArlimx,OFArlimy,OFArlimz]
OFAlcords = [[x,y,z] for x in OFAllim[0] for y in OFAllim[1] for z in OFAllim[2]]
OFArcords = [[x,y,z] for x in OFArlim[0] for y in OFArlim[1] for z in OFArlim[2]]
FFAl = [42,-53,-18]#np.array([[x,y,z] for x in np.arange(32,52) for y in np.arange(-70,-34) for z in np.arange(-28,-4)  ])
FFAr = [-40,-53,-18]#np.array([[x,y,z] for x in np.arange(-52,-30) for y in np.arange(-72,-35) for z in np.arange(-29,-5)])
FFAllimx = [32,52]
FFAllimy = [-70,-34]
FFAllimz = [-28,-4]
FFArlimx = [-52,-30]
FFArlimy = [-72,-35]
FFArlimz = [-29,-5]
FFAllim = [FFAllimx,FFAllimy,FFAllimz]
FFArlim = [FFArlimx,FFArlimy,FFArlimz]
FFAlcords = [[x,y,z] for x in FFAllim[0] for y in FFAllim[1] for z in FFAllim[2]]
FFArcords = [[x,y,z] for x in FFArlim[0] for y in FFArlim[1] for z in FFArlim[2]]
STSl = [53,-50,12]#np.array([[x,y,z] for x in np.arange(39,69) for y in np.arange(-68,-30) for z in np.arange(-1,26)   ])
STSr = [-54,-51,12]#np.array([[x,y,z] for x in np.arange(-68,-39) for y in np.arange(-69,-29) for z in np.arange(-1,28) ])
STSllimx = [39,69]
STSllimy = [-68,-30]
STSllimz = [-1,26]
STSrlimx = [-68,-39]
STSrlimy = [-69,-29]
STSrlimz = [-1,28]
STSllim = [STSllimx,STSllimy,STSllimz]
STSrlim = [STSrlimx,STSrlimy,STSrlimz]
STSlcords = [[x,y,z] for x in STSllim[0] for y in STSllim[1] for z in STSllim[2]]
STSrcords = [[x,y,z] for x in STSrlim[0] for y in STSrlim[1] for z in STSrlim[2]]
AMGl = [21,-4,-19]#np.array([[x,y,z] for x in np.arange(39,69) for y in np.arange(-68,-30) for z in np.arange(-1,26)   ])
AMGr = [-20,-4,-20]#np.array([[x,y,z] for x in np.arange(-68,-39) for y in np.arange(-69,-29) for z in np.arange(-1,28) ])
IFGl = [28,23,19]#np.array([[x,y,z] for x in np.arange(39,69) for y in np.arange(-68,-30) for z in np.arange(-1,26)   ])
IFGr = [-46,21,18]#np.array([[x,y,z] for x in np.arange(-68,-39) for y in np.arange(-69,-29) for z in np.arange(-1,28) ])
OFCl = [6,47,-15]#np.array([[x,y,z] for x in np.arange(39,69) for y in np.arange(-68,-30) for z in np.arange(-1,26)   ])
OFCr = [-6,46,-17]#np.array([[x,y,z] for x in np.arange(-68,-39) for y in np.arange(-69,-29) for z in np.arange(-1,28) ])
ATLl = [36,-6,-37]#np.array([[x,y,z] for x in np.arange(39,69) for y in np.arange(-68,-30) for z in np.arange(-1,26)   ])
ATLr = [-36,-6,-36]#np.array([[x,y,z] for x in np.arange(-68,-39) for y in np.arange(-69,-29) for z in np.arange(-1,28) ])
PCCl = [6,-58,31]#np.array([[x,y,z] for x in np.arange(39,69) for y in np.arange(-68,-30) for z in np.arange(-1,26)   ])
PCCr = [-5,-58,33]#np.array([[x,y,z] for x in np.arange(-68,-39) for y in np.arange(-69,-29) for z in np.arange(-1,28) ])
from sklearn.model_selection import StratifiedKFold as KFold
def cross_val_score(clf, X,Y,cv=5,n_jobs=25):
    kf = KFold(n_splits=cv)
    def trainModel(train,test):
        pca = PCA(0.95)
        Ypred = clf.fit(X[train],Y[train]).predict(X[test])
        recall = np.mean(Y[test][Y[test]==1] == Ypred[Y[test]==1])
        precision = np.mean(Y[test][Ypred==1] == Ypred[Ypred==1])
        return precision,recall
    return np.array(Parallel(n_jobs=n_jobs)(delayed(trainModel)(train,test) for train,test in kf.split(X,Y)))
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
def runMVPA(d, act):
    argsAll = []
    ucondAll = []
    for Phase in ["LR", "RL"]:
        if sys.argv[-1] != "preload":
            meanimg = nib.load(d + "WM_" + Phase + "/tfMRI_WM_"+Phase+".nii.gz")
            movement = np.loadtxt(d + "WM_" + Phase + "/Movement_Regressors.txt")
            meanimg = image.clean_img(meanimg,standardize=False,confounds=np.loadtxt(d + "WM_" + Phase + "/Movement_Regressors.txt"),high_pass=0.008,t_r=0.72)
            print("CLEANED",d.split("/")[-2],Phase)
            nib.save(meanimg,d.split("/")[-2] + "{0}_MNI.nii.gz".format(Phase))#,meanimg.get_fdata())
            affinv = np.linalg.inv(meanimg.affine)
            mEVCl =nib.affines.apply_affine(affinv,EVCl).astype(int)# np.vstack(list({tuple(row) for row in nib.affines.apply_affine(affinv,EVCl).astype(int)}))
            mEVCr =nib.affines.apply_affine(affinv,EVCr).astype(int)# np.vstack(list({tuple(row) for row in nib.affines.apply_affine(affinv,EVCr).astype(int)}))
            mOFAl =nib.affines.apply_affine(affinv,OFAl).astype(int)# np.vstack(list({tuple(row) for row in nib.affines.apply_affine(affinv,OFAl).astype(int)}))
            mOFAr =nib.affines.apply_affine(affinv,OFAr).astype(int)# np.vstack(list({tuple(row) for row in nib.affines.apply_affine(affinv,OFAr).astype(int)}))
            mFFAl =nib.affines.apply_affine(affinv,FFAl).astype(int)# np.vstack(list({tuple(row) for row in nib.affines.apply_affine(affinv,FFAl).astype(int)}))
            mFFAr =nib.affines.apply_affine(affinv,FFAr).astype(int)# np.vstack(list({tuple(row) for row in nib.affines.apply_affine(affinv,FFAr).astype(int)}))
            mSTSl =nib.affines.apply_affine(affinv,STSl).astype(int)# np.vstack(list({tuple(row) for row in nib.affines.apply_affine(affinv,STSl).astype(int)}))
            mSTSr =nib.affines.apply_affine(affinv,STSr).astype(int)# np.vstack(list({tuple(row) for row in nib.affines.apply_affine(affinv,STSr).astype(int)}))
            mAMGl =nib.affines.apply_affine(affinv,AMGl).astype(int)# np.vstack(list({tuple(row) for row in nib.affines.apply_affine(affinv,AMGl).astype(int)}))
            mAMGr =nib.affines.apply_affine(affinv,AMGr).astype(int)# np.vstack(list({tuple(row) for row in nib.affines.apply_affine(affinv,AMGr).astype(int)}))
            mIFGl =nib.affines.apply_affine(affinv,IFGl).astype(int)# np.vstack(list({tuple(row) for row in nib.affines.apply_affine(affinv,IFGl).astype(int)}))
            mIFGr =nib.affines.apply_affine(affinv,IFGr).astype(int)# np.vstack(list({tuple(row) for row in nib.affines.apply_affine(affinv,IFGr).astype(int)}))
            mOFCl =nib.affines.apply_affine(affinv,OFCl).astype(int)# np.vstack(list({tuple(row) for row in nib.affines.apply_affine(affinv,OFCl).astype(int)}))
            mOFCr =nib.affines.apply_affine(affinv,OFCr).astype(int)# np.vstack(list({tuple(row) for row in nib.affines.apply_affine(affinv,OFCr).astype(int)}))
            mATLl =nib.affines.apply_affine(affinv,ATLl).astype(int)# np.vstack(list({tuple(row) for row in nib.affines.apply_affine(affinv,ATLl).astype(int)}))
            mATLr =nib.affines.apply_affine(affinv,ATLr).astype(int)# np.vstack(list({tuple(row) for row in nib.affines.apply_affine(affinv,ATLr).astype(int)}))
            mPCCl =nib.affines.apply_affine(affinv,PCCl).astype(int)# np.vstack(list({tuple(row) for row in nib.affines.apply_affine(affinv,PCCl).astype(int)}))
            mPCCr =nib.affines.apply_affine(affinv,PCCr).astype(int)# np.vstack(list({tuple(row) for row in nib.affines.apply_affine(affinv,PCCr).astype(int)}))

            mmEVCl = np.vstack(list({tuple(row) for row in nib.affines.apply_affine(affinv,EVClcords).astype(int)}))
            mmEVCr = np.vstack(list({tuple(row) for row in nib.affines.apply_affine(affinv,EVCrcords).astype(int)}))
            mmOFAl = np.vstack(list({tuple(row) for row in nib.affines.apply_affine(affinv,OFAlcords).astype(int)}))
            mmOFAr = np.vstack(list({tuple(row) for row in nib.affines.apply_affine(affinv,OFArcords).astype(int)}))
            mmFFAl = np.vstack(list({tuple(row) for row in nib.affines.apply_affine(affinv,FFAlcords).astype(int)}))
            mmFFAr = np.vstack(list({tuple(row) for row in nib.affines.apply_affine(affinv,FFArcords).astype(int)}))
            mmSTSl = np.vstack(list({tuple(row) for row in nib.affines.apply_affine(affinv,STSlcords).astype(int)}))
            mmSTSr = np.vstack(list({tuple(row) for row in nib.affines.apply_affine(affinv,STSrcords).astype(int)}))

            ROIs = [mEVCl,mEVCr,mOFAl,mOFAr,mFFAl,mFFAr,mSTSl,mSTSr,mAMGl,mAMGr,mIFGl,mIFGr,mOFCl,mOFCr,mATLl,mATLr,mPCCl,mPCCr]
            meanimg = image.smooth_img(meanimg,4)#.get_fdata()
            meanimg = meanimg#.get_fdata()
        else:
            meanimg = nib.load((d.split("/")[-2] + "{0}_MNI.nii.gz".format(Phase)))#,affine=meanimg.affine)
            movement = np.loadtxt(d + "WM_" + Phase + "/Movement_Regressors.txt")
            affinv = np.linalg.inv(meanimg.affine)
            mEVCl =nib.affines.apply_affine(affinv,EVCl).astype(int)# np.vstack(list({tuple(row) for row in nib.affines.apply_affine(affinv,EVCl).astype(int)}))
            mEVCr =nib.affines.apply_affine(affinv,EVCr).astype(int)# np.vstack(list({tuple(row) for row in nib.affines.apply_affine(affinv,EVCr).astype(int)}))
            mOFAl =nib.affines.apply_affine(affinv,OFAl).astype(int)# np.vstack(list({tuple(row) for row in nib.affines.apply_affine(affinv,OFAl).astype(int)}))
            mOFAr =nib.affines.apply_affine(affinv,OFAr).astype(int)# np.vstack(list({tuple(row) for row in nib.affines.apply_affine(affinv,OFAr).astype(int)}))
            mFFAl =nib.affines.apply_affine(affinv,FFAl).astype(int)# np.vstack(list({tuple(row) for row in nib.affines.apply_affine(affinv,FFAl).astype(int)}))
            mFFAr =nib.affines.apply_affine(affinv,FFAr).astype(int)# np.vstack(list({tuple(row) for row in nib.affines.apply_affine(affinv,FFAr).astype(int)}))
            mSTSl =nib.affines.apply_affine(affinv,STSl).astype(int)# np.vstack(list({tuple(row) for row in nib.affines.apply_affine(affinv,STSl).astype(int)}))
            mSTSr =nib.affines.apply_affine(affinv,STSr).astype(int)# np.vstack(list({tuple(row) for row in nib.affines.apply_affine(affinv,STSr).astype(int)}))
            mAMGl =nib.affines.apply_affine(affinv,AMGl).astype(int)# np.vstack(list({tuple(row) for row in nib.affines.apply_affine(affinv,AMGl).astype(int)}))
            mAMGr =nib.affines.apply_affine(affinv,AMGr).astype(int)# np.vstack(list({tuple(row) for row in nib.affines.apply_affine(affinv,AMGr).astype(int)}))
            mIFGl =nib.affines.apply_affine(affinv,IFGl).astype(int)# np.vstack(list({tuple(row) for row in nib.affines.apply_affine(affinv,IFGl).astype(int)}))
            mIFGr =nib.affines.apply_affine(affinv,IFGr).astype(int)# np.vstack(list({tuple(row) for row in nib.affines.apply_affine(affinv,IFGr).astype(int)}))
            mOFCl =nib.affines.apply_affine(affinv,OFCl).astype(int)# np.vstack(list({tuple(row) for row in nib.affines.apply_affine(affinv,OFCl).astype(int)}))
            mOFCr =nib.affines.apply_affine(affinv,OFCr).astype(int)# np.vstack(list({tuple(row) for row in nib.affines.apply_affine(affinv,OFCr).astype(int)}))
            mATLl =nib.affines.apply_affine(affinv,ATLl).astype(int)# np.vstack(list({tuple(row) for row in nib.affines.apply_affine(affinv,ATLl).astype(int)}))
            mATLr =nib.affines.apply_affine(affinv,ATLr).astype(int)# np.vstack(list({tuple(row) for row in nib.affines.apply_affine(affinv,ATLr).astype(int)}))
            mPCCl =nib.affines.apply_affine(affinv,PCCl).astype(int)# np.vstack(list({tuple(row) for row in nib.affines.apply_affine(affinv,PCCl).astype(int)}))
            mPCCr =nib.affines.apply_affine(affinv,PCCr).astype(int)# np.vstack(list({tuple(row) for row in nib.affines.apply_affine(affinv,PCCr).astype(int)}))

            mmEVCl = np.vstack(list({tuple(row) for row in nib.affines.apply_affine(affinv,EVClcords).astype(int)}))
            mmEVCr = np.vstack(list({tuple(row) for row in nib.affines.apply_affine(affinv,EVCrcords).astype(int)}))
            mmOFAl = np.vstack(list({tuple(row) for row in nib.affines.apply_affine(affinv,OFAlcords).astype(int)}))
            mmOFAr = np.vstack(list({tuple(row) for row in nib.affines.apply_affine(affinv,OFArcords).astype(int)}))
            mmFFAl = np.vstack(list({tuple(row) for row in nib.affines.apply_affine(affinv,FFAlcords).astype(int)}))
            mmFFAr = np.vstack(list({tuple(row) for row in nib.affines.apply_affine(affinv,FFArcords).astype(int)}))
            mmSTSl = np.vstack(list({tuple(row) for row in nib.affines.apply_affine(affinv,STSlcords).astype(int)}))
            mmSTSr = np.vstack(list({tuple(row) for row in nib.affines.apply_affine(affinv,STSrcords).astype(int)}))
            mmEVCl = np.array([np.min(mmEVCl,0),np.max(mmEVCl,0)]) 
            mmEVCr = np.array([np.min(mmEVCr,0),np.max(mmEVCr,0)])
            mmFFAl = np.array([np.min(mmFFAl,0),np.max(mmFFAl,0)])
            mmFFAr = np.array([np.min(mmFFAr,0),np.max(mmFFAr,0)])
            mmOFAl = np.array([np.min(mmOFAl,0),np.max(mmOFAl,0)])
            mmOFAr = np.array([np.min(mmOFAr,0),np.max(mmOFAr,0)])
            mmSTSl = np.array([np.min(mmSTSl,0),np.max(mmSTSl,0)])
            mmSTSr = np.array([np.min(mmSTSr,0),np.max(mmSTSr,0)]) 
#            ROIs = [mEVCl,mEVCr,mOFAl,mOFAr,mFFAl,mFFAr,mSTSl,mSTSr,mAMGl,mAMGr,mIFGl,mIFGr,mOFCl,mOFCr,mATLl,mATLr,mPCCl,mPCCr]
            ROIs = [mmEVCl,mmEVCr,mmOFAl,mmOFAr,mmFFAl,mmFFAr,mmSTSl,mmSTSr]
#            print(np.array(ROIs).shape)
            meanimg = image.smooth_img(meanimg,4)#.get_fdata()
            ranges = [np.arange(meanimg.shape[0]),np.arange(meanimg.shape[1]),np.arange(meanimg.shape[2])]
#            print([[np.logical_and(ranges[0] > r[0][0],ranges[0] < r[1][0]),np.logical_and(ranges[1] > r[0][1],ranges[1]<r[1][1]),np.logical_and(ranges[2] > r[0][2], ranges[2] < r[1][2])] for r in ROIs])
            maskimg = compute_brain_mask(meanimg)
            mROIS = [meanimg.get_fdata()[np.logical_and(ranges[0] > r[0][0],ranges[0] < r[1][0])][:,np.logical_and(ranges[1] > r[0][1],ranges[1]<r[1][1])][:,:,np.logical_and(ranges[2] > r[0][2], ranges[2] < r[1][2])] for r in ROIs]
            mROIS = [mmm.reshape(-1,mmm.shape[-1]) for mmm in mROIS]
            maskimg = compute_brain_mask(meanimg)
#            print(Num,d.split("/")[-2] + "{0}_MNI.npy".format(Phase),meanimg.shape,mSTSr.shape)        
        ucond = ["Faces", "Shapes"] 
        TR = 0.72
        n_scans = meanimg.shape[-1]
        frame_times = TR * np.arange(n_scans)
        duration = TR * np.ones(n_scans)
        face = []
        body = []
        place = []
        tool = []
        for string in glob.glob(d+"WM_"+Phase+"/EVs/*bk*txt"):
            if not "all" in string:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    arr = np.loadtxt(string)
                if len(arr) > 0:
                    if "face" in string and "0bk" in string: 
#                        print(string)
                        face += [np.atleast_2d(arr)]#d+"WM_" + Phase + "/EVs/fear.txt")
                    else:
                        if ("tool" in string or "body" in string or "place" in string) and "0bk" in string:
#                            print(string)
                            if "tool" in string:
                                tool += [np.atleast_2d(arr)]#d+"WM_" + Phase + "/EVs/neut.txt")
                            elif "place" in string:
                                place += [np.atleast_2d(arr)]#d+"WM_" + Phase + "/EVs/neut.txt")
                            elif "body" in string:
                                body += [np.atleast_2d(arr)]#d+"WM_" + Phase + "/EVs/neut.txt")
        face = np.concatenate(face,0)
        tool = np.concatenate(tool,0)
        body = np.concatenate(body,0)
        place = np.concatenate(place,0)

        onsets = [F[0] for F in face] + [F[0] for F in tool]+ [F[0] for F in place] + [F[0] for F in body]

        trialtype= ["Faces" for F in face] + ["Tools" for F in tool]
        trialtype += ["Places" for F in place] + ["Bodies" for F in body]
        duration = [F[1] for F in face] + [F[1] for F in tool]
        duration += [F[1] for F in place] + [F[1] for F in body]
        events = pd.DataFrame({'onset': onsets, 'trial_type': trialtype, 'duration': duration})
        design = make_first_level_design_matrix(frame_times,
                events,
                high_pass=0.008,
                hrf_model=None,
                )
        rFaces = [mmm[...,design['Faces'] > 0] for mmm in mROIS]
        rTools = [mmm[...,design['Tools'] > 0] for mmm in mROIS]
        rPlaces = [mmm[...,design['Places'] > 0] for mmm in mROIS]
        rBodies = [mmm[...,design['Bodies'] > 0] for mmm in mROIS]
        rTrain = [np.concatenate((F,T,P,B),-1) for F,T,P,B in zip(rFaces,rTools,rPlaces,rBodies)]
        rLabels = [np.concatenate((np.ones(F.shape[-1]),2*np.ones(T.shape[-1]),3*np.ones(P.shape[-1]),4*np.ones(B.shape[-1]))) for F,T,P,B in zip(rFaces,rTools,rPlaces,rBodies)]
        classifiers = [SVC() for i in range(len(rLabels))]
        scores = [np.nanmean(cross_val_score(C,T.T,L,cv=5,n_jobs=25),0) for T,L,C in zip(rTrain,rLabels,classifiers)]
        np.save(d.split("/")[-2] + "_ROI_Eff_{0}".format(Phase),scores)
        print(np.abs(scores),d.split("/")[-2],Phase,rFaces[0].shape,rTools.shape)
        argsAll += [scores]#ROIEff/3]
    return argsAll
#Parallel(n_jobs=20)(delayed(getAdj)(rad) for rad in range(3,24+3,3))

if __name__ == "__main__":
    args = []
    IsTrue = (Parallel(n_jobs=25)(delayed(runMVPA)(d,"") for d in glob.glob("./*/") if not "cache" in d))
    np.save("WhichFaceSens3",IsTrue)
    
#Parallel(n_jobs=33)(delayed(runMVPA)(d) for d in dirs)#[:])
