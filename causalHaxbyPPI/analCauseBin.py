import numpy as np
from matplotlib import cm
import matplotlib
from glob import glob
import warnings
import math
from mpl_toolkits.axes_grid1 import make_axes_locatable
import itertools
from nilearn import surface
import networkx as nx
from joblib import Parallel, delayed
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import scipy.stats as stats
from nilearn import datasets
from nilearn import plotting
fsaverage = datasets.fetch_surf_fsaverage()
def analPs(rado,mu=0):
    rad = np.array(rado)
    length = rad.shape[-1]
    r = rad.reshape(rad.shape[0],-1)
    res = np.array(stats.ttest_1samp(r,mu)).T
    rts = res[...,0].reshape(length,length)
    rps = res[...,1].reshape(length,length)
    rts[rts!=rts]=0
    rts[rts<0]=0
    rps[rps!=rps]=1
    return rts, rps

def makePlots(rado):
    matplotlib.rcParams.update({'font.size': 14})
    first = ["face"]
    destrieux_atlas = datasets.fetch_atlas_surf_destrieux()
    atlas = destrieux_atlas
    labs = destrieux_atlas.labels[1:]
    coordinates = []
    for hemi in ['left', 'right']:
        vert = destrieux_atlas['map_%s' % hemi]
        rr, _ = surface.load_surf_mesh(fsaverage['pial_%s' % hemi])
        for k, label in enumerate(np.array(labs[:])):
            if "Unknown" not in str(label):  # Omit the Unknown label.
                # Compute mean location of vertices in label of index k
                coordinates.append(np.mean(rr[vert == k], axis=0))
    coordinates = np.array(coordinates)
#    coordinates = (-(coordinates - coordinates.mean(0))+coordinates.mean(0))
    rlabs = labs.copy()
    llabs = labs.copy()
    wallind = []
    for i in range(len(rlabs)):
        rlabs[i] = "Right " + str(rlabs[i],'utf-8')
        llabs[i] = "Left " + str(llabs[i],'utf-8')
    fulllabs = llabs + rlabs
    for i in range(len(fulllabs)):
        if "Medial_wall" in fulllabs[i]:
            wallind += [i]
    for task in ["Face"]:
        if task == "Face":
            Rado = np.copy(rado[0])
            Plot = Rado#[:,binMap=='00'] = -1
            Res,p = stats.ttest_1samp(Plot.reshape(Rado.shape[0],-1),0,axis=0)
            fdr = 0.05*np.arange(1,len(p)+1)/len(p)
            p[p!=p] = 1
            psort = np.sort(p)
            plt.close('all')
            plt.plot(psort[:20])
            plt.plot(fdr[:20])
            plt.savefig("Facepsrt")
            plt.close('all')
            below = psort <= fdr
            if np.sum(below) == 0:
                pthresh = 5e-3#0.050
            else:
                max_below = np.max(np.where(below)[0])
                pthresh = psort[max_below]

            print("FACE",np.where(below)[0],(p<pthresh).sum(),pthresh,p.min())
            Res[p > pthresh] = 0
            Plot = Res.reshape(rado[0].shape[-1],rado[0].shape[-1])
            pscore = p
        elif task == "House":
            binMap[binMap=='10'] = '00'
            binMap[binMap=='11'] = '00'
            colorStrs = ['{0}{1}'.format(i,j) for i in range(1) for j in range(2)]#np.sort(np.unique(binMap)).tolist()
            Plot = rado[1] - rado[0]
            Rado = np.copy(rado[1])
            Rad1 = np.copy(rado[0])
            Rado[:,binMap=='00'] = -1
            Rad1[:,binMap=='00'] = -1
            Plot[:,binMap=='00'] = -1
            Res,p = stats.wilcoxon(Plot.reshape(Rado.shape[0],-1),axis=0,alternative='greater')
            fdr = 0.05*np.arange(1,len(p)+1)/len(p)
            psort = np.sort(p)
            below = psort < fdr
            if np.sum(below) == 0:
                pthresh = 0.0250
            else:
                max_below = np.max(np.where(below)[0])
                pthresh = psort[max_below]

            print(fdr.max(),pthresh)
            Res[p > pthresh] = 0
            Plot = Res.reshape(rado[0].shape[-1],rado[0].shape[-1])
            pscore = p
        else:
            colorStrs = ['{0}{1}'.format(i,j) for i in range(2) for j in range(2)]#np.sort(np.unique(binMap)).tolist()
            Plot1 = evidenceplots[0]
            Plot2 = evidenceplots[1]
            Plot1[Plot1!=Plot1] = 0
            Plot2[Plot2!=Plot2] = 0
            Plot = Plot1 + Plot2
#            Plot[Plot==0]=np.nan
            Plot = rado[0] - rado[1]
            Rado = np.copy(rado[1])
            Rad1 = np.copy(rado[0])
            Rado[:,binMap=='00'] = -1
            Rad1[:,binMap=='00'] = -1
            Res,p = stats.ttest_rel(Rado.reshape(Rado.shape[0],-1),Rad1.reshape(Rado.shape[0],-1),axis=0)
            fdr = 0.05*np.arange(1,len(p)+1)/len(p)
            psort = np.sort(p)
            below = psort < fdr
            if np.sum(below) == 0:
                pthresh = 0.050
            else:
                max_below = np.max(np.where(below)[0])
                pthresh = psort[max_below]

            print(fdr.max(),pthresh)
            PRes = np.copy(Res)
            Res[p > pthresh] = 0
            Plot = Res.reshape(rado[0].shape[-1],rado[0].shape[-1])
            pscore = p

        fig,ax = plt.subplots(figsize=(12,12))

        with open("SurfBinEntCouplesParcs_{0}.txt".format(task),'w') as ff:
            for k,l in enumerate(llabs+rlabs):
                for kk,ll in enumerate(llabs+rlabs):
                    if abs(Plot[k,kk]) > 0:
                        ff.write("{0}-->{1}\n".format(l,ll))
        Plot[Plot==0] = np.nan
        Plot[Plot!=Plot]=0
        for zzz,l in enumerate(llabs+rlabs):
            if not "occipital_" in l and not "oc_" in l and not "temp" in l and not "collat" in l:
                    Plot[zzz] = np.nan
                    Plot[:,zzz] = np.nan
        Plot[Plot==0] = np.nan
        ymask = ~np.all(Plot!=Plot,axis=1)
        xmask = ~np.all(Plot!=Plot,axis=0)
        constrainplot = Plot[ymask][:,xmask]

        if "Face" in task:
            vmin = -np.nanmax(abs(Plot)+1)
            cmap = "RdBu"
        else:
            vmin = np.nanmin(abs(Plot))
            cmap = "Wistia"
        im = ax.imshow(constrainplot,cmap=cmap,vmin=vmin,vmax=np.nanmax(abs(Plot)+1))
        print(constrainplot.shape,ymask.sum(),xmask.sum(),(~(Plot!=Plot)).sum(),task)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(im, cax=cax,ticks=[vmin,  np.nanmax(abs(Plot)+1)])
        cbar.ax.set_ylabel("Wilcoxon Effect Size")
        xlabs = []
        ylabs = []
        ax.set_xticks([n for n in range(constrainplot.shape[1])])
        ax.set_yticks([n for n in range(constrainplot.shape[0])])
        for zzz,l in enumerate(llabs+rlabs):
            if ymask[zzz]:
                ylabs += [l+" -->"]
            if xmask[zzz]:
                xlabs += [l+" <--"]
        ax.set_xticklabels(xlabs)
        ax.set_yticklabels(ylabs)
        ax.tick_params(axis='x', rotation=90)
        ax.grid()
#        ax.set_title("Bootstrap Entropy Face: {2:0.4f} Bootstrap Entropy House: {3:0.4f} Ent Survivals Face = {0:d} Ent Survivals House {1:d}"
#                .format(*pthreshes,*ents))
        plt.tight_layout()
        plt.savefig("EvSurfBinEntSepColorMat_{0}".format(task))
        plt.close()




def makePs(d):#for d in dirs[:12]:
    rado = [[[[]] for l in range(1)] for m in range(1)]
    for k, rad in enumerate(["parcs"]):
        for l, cue in enumerate(["faces"]):
            for m,cl in enumerate([""]):
                x = 0
                from glob import glob
                x = np.median([np.load(s) for s in glob("*_{0:d}_*{1}*.npy".format(d,"faces"))],0).squeeze()
                y = np.median([np.load(s) for s in glob("*_{0:d}_*{1}*.npy".format(d,"houses"))],0).squeeze()
#                x = y - x
                print(x.shape)
                best = x
                best[best!=best] = 0
#                best[best<3] = 0
                rado[m][l][k] = best#*((pmat<0.1))
                print((rado[m][l][k]!=0).sum(), cl, cue, rad, d)
    return rado

dirs = [0,1,2,3,4,5]
temp = Parallel(n_jobs=30)(delayed(makePs)(d) for d in dirs)#range(0,5))
rado = np.array(temp).squeeze()[:,None]
print(rado.shape)
rado = rado.transpose(1,0,2,3)
print(rado.shape)
makePlots(rado)
