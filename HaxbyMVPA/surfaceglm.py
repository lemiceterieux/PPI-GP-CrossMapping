import numpy as np
import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from nilearn import surface, datasets
from sklearn import neighbors
#from dask.distributed import Client
import glob
from sklearn.model_selection import ParameterGrid
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from joblib import Parallel, delayed
import scipy
from scipy import sparse
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.model_selection import GridSearchCV
import nilearn
from sklearn.preprocessing import StandardScaler
from nilearn.decoding.searchlight import search_light
from sklearn.model_selection import KFold
import sys
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

from sklearn.svm import LinearSVC
from sklearn.gaussian_process import GaussianProcessClassifier as GPC
from sklearn.decomposition import PCA
import nibabel as nib
import warnings



def runMVPA(d):
    TR = 2.5
    haxby_dataset = datasets.fetch_haxby(subjects=d)
    # Load target information as string and give a numerical identifier to each
    behavioral = pd.read_csv(haxby_dataset.session_target[d-1], sep=' ')
    conditions = behavioral['labels'].values
    func_filename = haxby_dataset.func[d-1]
    # Record these as an array of sessions
    sessions = behavioral['chunks'].values
    unique_sessions = behavioral['chunks'].unique()
    
    # fMRI data: a unique file for each session
    fmri_filename = haxby_dataset.func[d-1]
    print(fmri_filename)
    events = {}
    # events will take  the form of a dictionary of Dataframes, one per session
    for session in unique_sessions:
        # get the condition label per session
        conditions_session = conditions[sessions == session]
        # get the number of scans per session, then the corresponding
        # vector of frame times
        n_scans = len(conditions_session)
        frame_times = TR * np.arange(n_scans)
        # each event last the full TR
        duration = TR * np.ones(n_scans)
        # Define the events object
        events_ = pd.DataFrame(
            {'onset': frame_times, 'trial_type': conditions_session, 'duration': duration})
        # remove the rest condition and insert into the dictionary
        events[session] = events_[events_.trial_type != 'rest']

    z_maps = []
    conditions_label = []
    session_label = []
    
    # Instantiate the glm
    from nilearn.glm.first_level import FirstLevelModel
    
    events[session].trial_type.unique()
    from nilearn.image import index_img
    from nilearn.image import math_img
#    for session in unique_sessions:
    def runGLM(session):
        warnings.simplefilter(action='ignore', category=FutureWarning)
    
        glm = FirstLevelModel(t_r=TR,
                              mask_img=haxby_dataset.mask,
                              high_pass=.008,
                              smoothing_fwhm=0,
                              memory='nilearn_cache')

        # grab the fmri data for that particular session
        fmri_session = index_img(func_filename, sessions == session)
#        fmri_session = math_img("img + np.random.randn(*img.shape)*.1", img=fmri_session)
    
        # fit the glm
        glm.fit(fmri_session, events=events[session])
    
        # set up contrasts: one per condition
        conditions = events[session].trial_type.unique()
        z_maps = []
        conditions_label = []
        sessions_label = []
        for condition_ in conditions:
            if condition_ == "face" or condition_ == "house":
                print(condition_)
                z_maps.append(glm.compute_contrast(condition_))
                conditions_label.append(condition_)
                session_label.append(session)
        return z_maps, conditions_label, session_label
    X,Y,Z = zip(*Parallel(n_jobs=30)(delayed(runGLM)(sess) for sess in unique_sessions))
    z_maps = sum(X,[])
    conditions_label = sum(Y,[])
    session_label = sum(Z,[])
    # Fetch a coarse surface of the left hemisphere only for speed
    fsaverage = datasets.fetch_surf_fsaverage(mesh='fsaverage5')
    hemi = 'left'
    
    # Average voxels 5 mm close to the 3d pial surface
    radius = 3.
    pial_mesh = fsaverage['pial_' + hemi]
    X = surface.vol_to_surf(z_maps, pial_mesh, radius=radius).T
    print(X.shape) 
    # To define the :term:`BOLD` responses to be included within each searchlight "sphere"
    # we define an adjacency matrix based on the inflated surface vertices such
    # that nearby surfaces are concatenated within the same searchlight.
    
    infl_mesh = fsaverage['infl_' + hemi]
    coords, _ = surface.load_surf_mesh(infl_mesh)
    radius = 6.0
    nn = neighbors.NearestNeighbors(radius=radius)
    adjacency = nn.fit(coords).radius_neighbors_graph(coords).tolil()    
    from sklearn.model_selection import LeaveOneGroupOut
    cv = LeaveOneGroupOut()
    # Define Searchlight parameters, paralellizing over the number of Runs
    print(d,"Starting Searchlight!")
    # Fit the searchlights
    from sklearn.model_selection import KFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import RidgeClassifier
    from nilearn.decoding.searchlight import search_light

    estimator = make_pipeline(StandardScaler(),
                                      RidgeClassifier(alpha=10.))
    sl_map = search_light(X, conditions_label, estimator, adjacency, cv=cv,
            n_jobs=280, verbose=1, groups=session_label)
    # Save the scores
    #slScores = nib.Nifti1Image(sl_map, affine=Rnifti.affine)
    np.save(str(d)+"lh3haxby.npy",sl_map)

    hemi = 'right'
    
    # Average voxels 5 mm close to the 3d pial surface
    radius = 3.
    pial_mesh = fsaverage['pial_' + hemi]
    X = surface.vol_to_surf(z_maps, pial_mesh, radius=radius).T
    
    # To define the :term:`BOLD` responses to be included within each searchlight "sphere"
    # we define an adjacency matrix based on the inflated surface vertices such
    # that nearby surfaces are concatenated within the same searchlight.
    
    infl_mesh = fsaverage['infl_' + hemi]
    coords, _ = surface.load_surf_mesh(infl_mesh)
    radius = 6.0
    nn = neighbors.NearestNeighbors(radius=radius)
    adjacency = nn.fit(coords).radius_neighbors_graph(coords).tolil()    
    cv = LeaveOneGroupOut()
    # Define Searchlight parameters, paralellizing over the number of Runs
    print(d,"Starting Searchlight!")
    # Fit the searchlights
    sl_map = search_light(X, conditions_label, estimator, adjacency, cv=cv,
            n_jobs=280, verbose=1, groups=session_label)
    # Save the scores
    #slScores = nib.Nifti1Image(sl_map, affine=Rnifti.affine)
    np.save(str(d) + "rh3haxby.npy",sl_map)

Parallel(n_jobs=1)(delayed(runMVPA)(d+1) for d in range(6))
