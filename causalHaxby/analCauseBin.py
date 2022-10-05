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
    first = ["face", "house"]
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
    chooser = 2
    Ps = np.arange(rado.shape[2]).tolist()
    combs = []
    for comb in itertools.combinations(Ps, rado.shape[2]-chooser):
        combs += [comb]
    np.array(combs)
    mrado = rado.mean(2)
    def combin(N,R):
        return math.factorial(N)/(math.factorial(N-R)*math.factorial(R))

    def bootstrap(rado,like,bonfact):
        # Sort our observations at each node
        sortr = np.sort(rado,0)
        # Make sure only p percentage of evidences are above 0 according to the
        # in the empirical distribution
        boot = []
        for i in np.random.randint(rado.shape[0],size=rado.shape[0]):
            i = int(i)
            boot += [sortr[i]]#*(1-frac) + rosort[i+1]*(frac)]

        boot = np.array(boot)
        boot = boot- sortr[int(np.floor((1-like)*sortr.shape[0]))]
        N = boot.shape[0]
        R = (boot>0).sum(0)
        Rperm = R
        tPerms = []
        ts = []
        for p in range(len(R)):
            tssub = []
            for pp in range(len(R)):
                combies = []
                for r in range(1,R[p,pp]+1):
                    combies += [combin(N,r)*(like)**(r)*(1-like)**(N-r)]
                tssub += [1-np.sum(combies)]
            ts += [tssub]
        ts = np.array(ts)
        toCalcEnt = ts<=0.05/bonfact

        rowEnt = (toCalcEnt.sum(0))
        if rowEnt.sum() != 0:
            rowEnt = rowEnt/rowEnt.sum()

        rrowEnt = -(rowEnt[rowEnt!=0]*np.log(rowEnt[rowEnt!=0])).sum()

        colEnt = (toCalcEnt.sum(1))
        if colEnt.sum() != 0:
            colEnt = colEnt/colEnt.sum()
        rcolEnt = -(colEnt[colEnt!=0]*np.log(colEnt[colEnt!=0])).sum()
        jointEnt = 0
        jointp = np.zeros((len(rowEnt),len(colEnt)))
        for i in range(len(colEnt)):
            for j in range(len(rowEnt)):
                jointp[i,j] = colEnt[i]*rowEnt[j]
        if jointp.sum() !=0:
            jointp = jointp/jointp.sum()
        jointEnt = -(jointp[jointp!=0]*np.log(jointp[jointp!=0])).sum()
        return rrowEnt, rcolEnt,jointEnt

    sigmap = []
    evidenceplots = []
    pthreshes = []
    ents = []
    for i in range(len(rado)):

        bonfact = 1 
        plt.close()
        pvalcols = 10
        pvalrows =10
        pvaljoints = 10
        alpha = 0.05
        multcomp = 2
        like = 0.5
        xsumm = 100
        while pvaljoints > alpha/multcomp or pvalcols > alpha/multcomp or pvalrows > alpha/multcomp:
            print(bonfact, first[i],pvalcols,pvalrows,pvaljoints)
            N = rado[i].shape[0]
            R = (rado[i]!=0).sum(0)
            Rperm = R
            tPerms = []
            ts = []
            for p in range(len(R)):
                tssub = []
                for pp in range(len(R)):
                    combies = []
                    for r in range(1,R[p,pp]+1):
                        combies += [combin(N,r)*(like)**(r)*(1-like)**(N-r)]
                    tssub += [1-np.sum(combies)]
                ts += [tssub]
            ts = np.array(ts)
            sortT = np.sort(ts.ravel())
            fdr = 0.05*np.arange(1,len(sortT)+1)/len(sortT)
            plt.plot(fdr, 'red', label="FDR")
            plt.plot(sortT,'blue', label="pVal Binomial")
            plt.legend()
            plt.ylabel("pVal")
            plt.xlabel("Sorted Index")
            plt.ylim((0,0.5))
            plt.savefig("pValBinFDR_{0}".format(first[i]))
            plt.close()
            below = sortT < fdr
            if np.sum(below) == 0:
                pthresh = 0.050
            else:
                max_below = np.max(np.where(below)[0])
                pthresh = sortT[max_below]
            print(np.min(R),np.max(R), ts.shape, pthreshes)

            toCalcEnt = ts<=0.05/bonfact#/len(ts.ravel())
            resss = np.array(Parallel(n_jobs=250)(delayed(bootstrap)(rado[i],like,bonfact) for kk in range(1000))).T
            nullrow = resss[0]
            nullcol = resss[1]
            nulljoint = resss[2]
            implot = toCalcEnt
            xsumm = (~np.all(implot==0,axis=0)).sum()
            Prop = R/N
            templot = np.median(rado[i],0)#[ymask][:,xmask]
            templot[implot==0] = np.nan

            #templot = implot#[ymask][:,xmask]
        
            toCalcEnt = implot
            rowEnt = (toCalcEnt.sum(0))
            rowEnt = rowEnt/rowEnt.sum()
            notrowEnt = 1-rowEnt
        
            colEnt = (toCalcEnt.sum(1))
            colEnt = colEnt/colEnt.sum()
            notcolEnt = 1-colEnt
        
            jointp = np.zeros((len(rowEnt),len(colEnt)))
        
            for zz in range(len(colEnt)):
                for zzz in range(len(rowEnt)):
                    jointp[zz,zzz]= rowEnt[zzz]*colEnt[zz]
            jointp = jointp/jointp.sum()
            jointEnt = -(jointp[jointp!=0]*np.log(jointp[jointp!=0])).sum()
        
            #calcEnts
            rowEnt = -(rowEnt[rowEnt!=0]*np.log(rowEnt[rowEnt!=0])).sum()
            notrowEnt = -(notrowEnt[notrowEnt!=0]*np.log(notrowEnt[notrowEnt!=0])).sum()
            pvalrows = (rowEnt > nullrow).mean()
            pvallessrows = (rowEnt < nullrow).mean()
        
            colEnt = -(colEnt[colEnt!=0]*np.log(colEnt[colEnt!=0])).sum()
            notcolEnt = -(notcolEnt[notcolEnt!=0]*np.log(notcolEnt[notcolEnt!=0])).sum()
            pvalcols = (colEnt > nullcol).mean()
            pvallesscols = (colEnt < nullcol).mean()
        
            pvaljoints = (jointEnt > nulljoint).mean()
            pvallessjoints = (jointEnt < nulljoint).mean()
        
            pvalcols = np.min([pvalcols,pvallesscols])
            pvalrows = np.min([pvalrows,pvallessrows])
            pvaljoints = np.min([pvaljoints,pvallessjoints])
            print(bonfact, first[i],"",pvalrows,pvalcols,pvaljoints)
            if pvaljoints > alpha/multcomp or pvalcols > alpha/multcomp or pvalrows > alpha/multcomp:
                bonfact *= 2

        pthreshes += [toCalcEnt.sum()]
        ents += [pvaljoints]
        sigmap += [toCalcEnt.astype(int).astype(str)]
        nodeLabs = np.array(llabs + rlabs)
        edgelist = [(u,v,Prop[u,v]) for u in range(ts.shape[0]) for v in range(ts.shape[1]) if implot[u,v]]
        G = nx.DiGraph()
        weights = []
        listBinEntCoups = []
        for u,v,t in edgelist:
            if u > len(labs):
                uu = "R{0:d}".format(1+u-len(llabs))
            else:
                uu = "L{0:d}".format(1+u)
            if v > len(labs):
                vv = "R{0:d}".format(1+v-len(llabs))
            else:
                vv = "L{0:d}".format(1+v)
            G.add_edge(uu,vv,dict={"t":t})
            listBinEntCoups += ["{0}-->{1}".format(fulllabs[u],fulllabs[v])]
            weights += [t]
        with open("BinEntCouplesParcs_{0}_{1}_{2}.txt".format(first[i],"",'parcs'),'w') as ff:
            for w,lcoups in zip(weights,listBinEntCoups):
                ff.write("{0}   {1:0.4f}\n".format(lcoups,w))
        pos = nx.spring_layout(G, k=3*1/np.sqrt(len(G.nodes())),iterations=20)
        plt.figure(3, figsize=(12, 12))                
        nx.draw_networkx(G, pos=pos,edge_color=weights,edge_cmap=plt.cm.rainbow)
        if len(weights) > 0:
            sm = plt.cm.ScalarMappable(cmap=plt.cm.rainbow, norm=plt.Normalize(vmin = np.min(weights), vmax=np.max(weights)))    
            sm._A = []
            plt.colorbar(sm)
        plt.savefig("Network_Ts_{0}_{1}_{2}.png".format(first[i],"",'parcs'))
        plt.close()

        print(rowEnt, colEnt,pvalrows,pvalcols,pvaljoints,first[i],"")
        ymask = ~np.all(implot==0,axis=1)
        xmask = ~np.all(implot==0,axis=0)
        evidenceplots += [np.copy(templot)]
        constrainplot = np.copy(templot[ymask][:,xmask])
        #templot[templot==0] = np.nan
        print(templot.shape,xmask.sum(),ymask.sum())
        fig, ax = plt.subplots(figsize=(30,30))
        plt.imshow(templot, cmap='rainbow',vmin=np.min(weights),)
        plt.colorbar()
        plt.title("{0} for {1} Joint Ent {6:0.4f} ({7:0.4f}) Incoming Ent {2:0.3f} Outgoing Ent {4:0.3f}"
                .format(first[i],"",rowEnt, pvalrows, colEnt,
                    pvalcols, jointEnt, pvaljoints))
        ax.set_xticks([n for n in range(templot.shape[1])])
        ax.set_yticks([n for n in range(templot.shape[0])])
        xlabs = []
        ylabs = []
        for zzz,l in enumerate(llabs+rlabs):
            if True:#ymask[zzz]:
                ylabs += [l+" -->"]        
            if True:#xmask[zzz]:
                xlabs += [l+" <--"]
        ax.set_xticklabels(xlabs)
        ax.set_yticklabels(ylabs)
        ax.tick_params(axis='x', rotation=90)
        ax.grid()
        plt.tight_layout()
        plt.savefig("BinEntMatrix_Ts_{0}_{1}_{2}.png".format(first[i],"",'parcs'))
        plt.close() 

        fig, ax = plt.subplots(figsize=(12,12))
        #constrainplot[constrainplot==0] = np.nan
        im = ax.imshow(constrainplot, cmap='rainbow')#,np.min(weights))
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
        ax.set_title("{0} for {1} Joint Ent {6:0.4f} ({7:0.4f}) Incoming Ent {2:0.3f} Outgoing Ent {4:0.3f}"
                .format(first[i],"",rowEnt, pvalrows, colEnt,
                    pvalcols, jointEnt, pvaljoints))
        ax.set_xticks([n for n in range(constrainplot.shape[1])])
        ax.set_yticks([n for n in range(constrainplot.shape[0])])
        xlabs = []
        ylabs = []
        for zzz,l in enumerate(llabs+rlabs):
            if ymask[zzz]:
                ylabs += [l+" -->"]        
            if xmask[zzz]:
                xlabs += [l+" <--"]
        ax.set_xticklabels(xlabs)
        ax.set_yticklabels(ylabs)
        ax.tick_params(axis='x', rotation=90)
        ax.grid()
        plt.tight_layout()
        plt.savefig("SkinnyBinEntMatrix_Ts_{0}_{1}_{2}.png".format(first[i],"",'parcs'))
        plt.close() 

    binMap = sigmap[0]
    for i in range(1,len(sigmap)):
        binMap = np.core.defchararray.add(binMap,sigmap[i])
    trueBinMap = np.copy(binMap)
    for task in ["Face", "House", "All"]:
        binMap = np.copy(trueBinMap)
        if task == "Face":
            binMap[binMap=='01'] = '00'
            binMap[binMap=='11'] = '00'
            colorStrs = ['{0}{1}'.format(i,j) for i in range(2) for j in range(1)]#np.sort(np.unique(binMap)).tolist()
            Plot = rado[0] - rado[1]
            Rado = np.copy(rado[0])
            Rad1 = np.copy(rado[1])
            Rado[:,binMap=='00'] = -1
            Rad1[:,binMap=='00'] = -1
            Plot[:,binMap=='00'] = -1
            Res,p = stats.wilcoxon(Plot.reshape(Rado.shape[0],-1),axis=0,alternative='greater')
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
                pthresh = 0.050
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

        Events = []
        Ev = [" F ", " H "]
        for cc in colorStrs[1:]:
            inds = np.where(np.array(list(cc))=='1')[0]
            strBuilt = ""
            for i in inds:
                strBuilt += Ev[i]
            Events += [strBuilt]
    
        print(colorStrs)
    
        str2Num = dict(zip(colorStrs,np.arange(len(colorStrs))))
        bin2Num = np.array([str2Num[i] for i in binMap.ravel()]).reshape(binMap.shape).astype(float)
        with open("SurfBinEntCouplesParcs_{0}.txt".format(task),'w') as ff:
            for k,l in enumerate(llabs+rlabs):
                for kk,ll in enumerate(llabs+rlabs):
                    if bin2Num[k,kk] > 0:
                        ff.write("{0}-->{1} {2}\n".format(l,ll,Events[int(bin2Num[k,kk]-1)]))
    
        cmap = (matplotlib.cm.get_cmap("rainbow",len(str2Num)-1))
        fig,ax = plt.subplots(figsize=(15,15))
        cbin2Num = np.copy(Plot)
        cbin2Num[binMap=='00'] = 0
        bin2Num[binMap=='00'] = np.nan
        Plot[Plot==0] = np.nan
        ymask = ~np.all(Plot!=Plot,axis=1)
        xmask = ~np.all(Plot!=Plot,axis=0)
        allMask = np.logical_or(ymask,xmask)
        Plot[Plot!=Plot] = 0
#        for zzz,l in enumerate(llabs+rlabs):
#            if not "occipital_" in l and not "oc_" in l and not "temp" in l and not "collat" in l:
#                    Plot[zzz] =0 
#                    Plot[:,zzz] =0 

        for ppp in range(len(rlabs)):
            if "occipital_middle" in rlabs[ppp]:
                break
        cBin2Num2 = np.zeros(cbin2Num.shape)
        cBin2Num2[ppp] = cbin2Num[ppp]
        cBin2Num2[ppp+75] = cbin2Num[ppp+75]
#        cbin2Num[ppp] = 0
#        cbin2Num[75+ppp] = 0
#        cbin2Num = cBin2Num2
#        cbin2Num[cbin2Num!=cbin2Num] = 0
        print(cBin2Num2[ppp].sum(),cBin2Num2[ppp+75].sum())
        constrainplot = bin2Num[ymask][:,xmask]
        view = plotting.view_connectome(Plot[allMask][:,allMask].T,
                coordinates[allMask],symmetric_cmap=False)#,edge_cmap=cm.hot)
        view.save_as_html("Connect_{0}.html".format(task))
        plt.figure()
        plotting.plot_connectome(Plot[allMask][:,allMask].T, coordinates[allMask],
                node_size=3,edge_kwargs={"linewidth":1})#,edge_cmap=cm.hot)
        plt.savefig("Connect_{0}.png".format(task))
        plt.close()
        fig,ax = plt.subplots(figsize=(15,15))
        im = ax.imshow(bin2Num[ymask][:,xmask],vmin=1,vmax=len(str2Num),cmap=cmap)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
        cbar = plt.colorbar(im,cax,ticks=np.arange(1,len(colorStrs)))
        cbar.ax.set_yticklabels(Events)
        cbar.ax.set_ylabel('Event', rotation=270)
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
        ax.set_title("Bootstrap Entropy Face: {2:0.4f} Bootstrap Entropy House: {3:0.4f} Ent Survivals Face = {0:d} Ent Survivals House {1:d}"
                .format(*pthreshes,*ents))
        plt.tight_layout()
        plt.savefig("SurfBinEntSepColorMat_{0}".format(task))
        plt.close()

        fig,ax = plt.subplots(figsize=(12,12))

        Plot[Plot==0] = np.nan
#        Plot[Plot!=Plot]=0
#        for zzz,l in enumerate(llabs+rlabs):
#            if not "occipital_" in l and not "oc_" in l and not "temp" in l and not "collat" in l:
#                    Plot[zzz] = np.nan
#                    Plot[:,zzz] = np.nan
        ymask = ~np.all(Plot!=Plot,axis=1)
        xmask = ~np.all(Plot!=Plot,axis=0)
        Plot[Plot==0] = np.nan
        constrainplot = Plot[ymask][:,xmask]

        if "All" in task:
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
    rado = [[[[]] for l in range(2)] for m in range(1)]
    for k, rad in enumerate(["parcs"]):
        for l, cue in enumerate(["faces","houses"]):
            for m,cl in enumerate([""]):
                x = 0
                from glob import glob
                x = np.median([np.load(s) for s in glob("*_{0:d}_*{1}*.npy".format(d,cue))],0)
                pmat = np.zeros((x.shape[0],x.shape[1]))
                best = np.zeros((x.shape[0],x.shape[1]))
                for i in range(x.shape[0]):
                    for j in range(x.shape[1]):
                        res = ((x[j,i,:,None] - x[i,j])).ravel()
                        stat = res[0]
                        pval =  (stat > res[1:]).mean()
#                        res = []
#                        for r in x:
#                            res += [(r[j,i,:,None] - r[i,j]).ravel()]
#                        res = np.array(res)
#                        stat = res[:,[0]]
#                        res[res!=res] = np.nanmax(res)
#                        pval = np.max((stat >res[:,1:]).mean(1))
#                        rasort = np.argsort(res)
#                        cum = np.arange(len(res))/len(res)
#                        pval = cum[rasort==0]
                        pmat[i,j] = pval
                        best[i,j] = -(stat-res[1:].mean())
                best[best!=best] = 0
                best[best==np.inf] = 0
                rado[m][l][k] = best*((pmat<0.1))
                print((rado[m][l][k]!=0).sum(), cl, cue, rad, d)
    return rado

dirs = [0,1,2,3,4,5]
temp = Parallel(n_jobs=30)(delayed(makePs)(d) for d in dirs)#range(0,5))
rado = np.array(temp).squeeze()
print(rado.shape)
rado = rado.transpose(1,0,2,3)
print(rado.shape)
makePlots(rado)
