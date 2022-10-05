import numpy as np
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
import vgpccmbatch as gp
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
    m = 5
    t = 1
    cuda = (mp.current_process()._identity[0] - 1)%8
    gpR = gp.GP()
    res = gpR.testStateSpaceCorrelation(dat[0], dat[1],dat[2], m, tau=t, cuda=cuda)
    print(dat[-1],res[1][:,0], np.logical_or(res[1][:,0] < res[1].mean(1)-res[1].std(1)*2,res[1][:,0] > res[1].mean(1)+2*res[1].std(1)).sum())
    return res[1].cpu().numpy()
def runMVPA(d, act):

    haxby_dataset = datasets.fetch_haxby(subjects=d+1)
    meanimg = nib.load(haxby_dataset.func[d])
    meanimg = image.clean_img(meanimg)


    behavioral = pd.read_csv(haxby_dataset.session_target[d], sep=' ')
    conditions = behavioral['labels'].values
    session = behavioral['chunks'].values
    sess = np.unique(session)
    TR = 2.5
    n_scans = len(conditions)
    frame_times = TR * np.arange(n_scans)
    duration = TR * np.ones(n_scans)
    events_ = pd.DataFrame({'onset': frame_times, 'trial_type': conditions, 'duration': duration})
    events = events_[np.logical_or(events_.trial_type == "face", events_.trial_type == "house")]
    design = make_first_level_design_matrix(frame_times,
            events,
            high_pass=0.008,
            hrf_model=None,
            )
    face = design["face"].values
    face = (face - face.mean())/face.std()
    house = design["house"].values
    house = (house - house.mean())/house.std()
#    face = (face[:,None,None,None]*meanimg.get_fdata().std(-1)).transpose(1,2,3,0)
#    house = (house[:,None,None,None]*meanimg.get_fdata().std(-1)).transpose(1,2,3,0)
#    claim = nib.Nifti1Image(face*meanimg.get_fdata(),meanimg.affine)
#    noclaim = nib.Nifti1Image(house*meanimg.get_fdata(),meanimg.affine)

    # Fetch a coarse surface of the left hemisphere only for speed
    fsaverage = datasets.fetch_surf_fsaverage(mesh='fsaverage5')
    hemi = 'left'

    # Average voxels 5 mm close to the 3d pial surface
    radius = 0.
    pial_mesh = fsaverage['pial_' + hemi]	
    lh = surface.vol_to_surf(meanimg, pial_mesh, radius=radius)
    rh = surface.vol_to_surf(meanimg, fsaverage['pial_right'], radius=radius)
    lh[lh!=lh] = 0
    rh[rh!=rh] = 0

    hrf = glover_hrf(1*2.5,1,50)
    lhds = []
    rhds = []
    for i in sess:
        time = np.where(session==i)[0]
        lht = lh[:,time]
        rht = rh[:,time]
        bases = [np.ones(time.shape[0])] + [np.sin(2*np.pi*(i+1)*np.arange(time.shape[0])/time.shape[0]) for i in range(1,480)]
        bases += [np.cos(2*np.pi*(i+1)*np.arange(time.shape[0])/time.shape[0]) for i in range(1,480)]
        bases = np.array(bases).T
        cm = la.convolution_matrix(hrf,time.shape[0])
        basescm = cm[:time.shape[0]].dot(bases)
        B = np.linalg.inv(basescm.T.dot(basescm)+1*np.eye(bases.shape[1])).dot(basescm.T).dot(lht.T)
        lhds += [bases.dot(B).T]
        B = np.linalg.inv(basescm.T.dot(basescm)+1*np.eye(bases.shape[1])).dot(basescm.T).dot(rht.T)
        rhds += [bases.dot(B).T]
    lh = np.concatenate(lhds,1)
    rh = np.concatenate(rhds,1)
    lclaim = lh*face
    rclaim = rh*face
    lnoclaim = lh*house
    rnoclaim = rh*house

    destrieux_atlas = datasets.fetch_atlas_surf_destrieux()
    parcleft = destrieux_atlas.map_left
    parcright = destrieux_atlas.map_right

    def getL(i):
        lcl = []
        lclno = []
        sigs = []
        lcl = lclaim[parcleft==i]
        lclno = lnoclaim[parcleft==i]
        lmu = lh[parcleft==i]
        mask = lmu.T.sum(0) == 0
        lcl=lcl[~mask]
        lclno=lclno[~mask]
        lmu=lmu[~mask]

        temp = lcl.T
        mask = temp.std(0) == 0
        lcl=lcl[~mask]
        lclno=lclno[~mask]
        lmu=lmu[~mask]
        if lmu.shape[0] == 0:
            return None, None, None

        reducer = PCA()#umap.UMAP()
        lcl = ((lcl.T - lcl.mean(1))*lmu.std(1))
        lclno = ((lclno.T - lclno.mean(1))*lmu.std(1))
        lmu = (lmu.T - lmu.mean(1))#/lmu.std(1))
        reducer.fit(lmu)
        lcl = reducer.transform(lcl).T[[0]]
        lclno = reducer.transform(lclno).T[[0]]
        lmu = reducer.transform(lmu).T[[0]]
        return lcl.reshape(-1,lclaim.shape[-1]),lclno.reshape(-1,lnoclaim.shape[-1]),lmu.reshape(-1,lnoclaim.shape[-1])

    def getR(i):
#    for i in range(1,rcwp+1):
        rcl = []
        rclno = []
        sigs = []
        rcl = rclaim[parcright==i]
        rclno = rnoclaim[parcright==i]
        rmu = rh[parcright==i]
        mask = rmu.T.sum(0) == 0
        rcl=rcl[~mask]
        rclno=rclno[~mask]
        rmu=rmu[~mask]

        temp = rcl.T
        mask = temp.std(0) == 0
        rcl=rcl[~mask]
        rclno=rclno[~mask]
        rmu=rmu[~mask]
        if rmu.shape[0] == 0:
            return None, None, None

        #print(mask.sum())

        rcl = ((rcl.T - rcl.mean(1))*rmu.std(1))
        rclno = ((rclno.T - rclno.mean(1))*rmu.std(1))
        rmu = (rmu.T - rmu.mean(1))#/rmu.std(1))
        reducer = PCA()#umap.UMAP()
        reducer.fit(rmu)
        rcl = reducer.transform(rcl).T[[0]]
        rclno = reducer.transform(rclno).T[[0]]
        rmu = reducer.transform(rmu).T[[0]]

        return rcl.reshape(-1,rclaim.shape[-1]),rclno.reshape(-1,rclaim.shape[-1]),rmu.reshape(-1,rclaim.shape[-1])
    lcl_claim, lcl_noclaim, lmu_all = zip(*Parallel(n_jobs=45)(delayed(getL)(i) for i in range(1,1+75)))
    rcl_claim, rcl_noclaim, rmu_all = zip(*Parallel(n_jobs=45)(delayed(getR)(i) for i in range(1,1+75)))
    args = [[]for i in range(75*2)]
    argsno = [[]for i in range(75*2)]

    totest = []
    passedL = []
    passedR = []
    totestno = []
    for j in range(75):
        if lcl_claim[j] is None:
            passedL += [j]
            continue
        totest += [lcl_claim[j]]
        totestno += [lcl_noclaim[j]]
    for j in range(75):
        if rcl_claim[j] is None:
            passedR += [j]
            continue
        totest += [rcl_claim[j]]
        totestno += [rcl_noclaim[j]]
    np.save("Passed{0}.npy".format(d),[passedL,passedR])
    for i in range(75):
        if lcl_claim[j] is None:
            continue
        args[i] = [lcl_claim[i], totest,lmu_all[i],i]
        argsno[i] = [lcl_noclaim[i] , totestno,lmu_all[i],i]

    for i in range(75):
        if rcl_claim[j] is None:
            continue
        args[75+i] = [rcl_claim[i] , totest,rmu_all[i],75+i]
        argsno[75+i] = [rcl_noclaim[i] ,totestno,rmu_all[i],75+i]

    return [args, argsno]

def getAdj(rad):
    # Anatomical MAsk
    fsaverage = datasets.fetch_surf_fsaverage(mesh='fsaverage5')
    infl_mesh = fsaverage['infl_left']
    coords, _ = surface.load_surf_mesh(infl_mesh)
    radius = rad
    nn = neighbors.NearestNeighbors(radius=radius, n_jobs=20)
    adj = nn.fit(coords).radius_neighbors_graph(coords).tolil()
    np.save("adjL"+str(rad)+".npy", adj)

    # Anatomical MAsk
    fsaverage = datasets.fetch_surf_fsaverage(mesh='fsaverage5')
    infl_mesh = fsaverage['infl_right']
    coords, _ = surface.load_surf_mesh(infl_mesh)
    radius = rad
    nn = neighbors.NearestNeighbors(radius=radius, n_jobs=20)
    adj = nn.fit(coords).radius_neighbors_graph(coords).tolil()
    np.save("adjR"+str(rad)+".npy", adj)

#Parallel(n_jobs=20)(delayed(getAdj)(rad) for rad in range(3,24+3,3))

if __name__ == "__main__":
    args = []
    args = Parallel(n_jobs=6)(delayed(runMVPA)(d,"") for d in range(6))
    print(len(args))
    for i in range(10):
        for d in range(len(args)):
            print(i,d)
            mp.set_start_method('spawn',force=True)
            with Pool(8) as p:
                res = p.map(analCaus, args[d][0])
                np.save("{0:d}_{1:d}__parcs_facescause.npy".format(i,d),res)
                resno = p.map(analCaus,args[d][1])
                np.save("{0:d}_{1:d}__parcs_housescause.npy".format(i,d),resno)

#Parallel(n_jobs=33)(delayed(runMVPA)(d) for d in dirs)#[:])
