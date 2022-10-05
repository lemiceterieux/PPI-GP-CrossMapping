import numpy as np
import warnings
import math
from mpl_toolkits.axes_grid1 import make_axes_locatable
import itertools
import networkx as nx
from joblib import Parallel, delayed
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import scipy.stats as stats
from nilearn import datasets

def analPs(rado,mu=0):
    rad = np.array(rado)
    length = rad.shape[-1]
    r = rad.reshape(rad.shape[0],-1)
    r[r == 0] = -1
    res = []
    res = stats.wilcoxon(r,axis=0,alternative='greater')
    res = np.array(res).T
    rts = res[:,0].reshape(length,length)
    rps = res[:,1].reshape(length,length)
    return rts, rps

def makePlots(rado):
    matplotlib.rcParams.update({'font.size': 12})
    first = ["face", "house"]
    destrieux_atlas = datasets.fetch_atlas_surf_destrieux()
    labs = destrieux_atlas.labels[1:]
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
    print(rado.shape)
    mrado = rado.mean(2)
    def combin(N,R):
        return math.factorial(N)/(math.factorial(N-R)*math.factorial(R))

    def bootstrap(ro, mro,tthresh):
        rosort = np.sort(ro,0)
        cum = np.arange(ro.shape[0])/ro.shape[0]
        boot = []
#        for frac, i in zip(*np.modf(np.random.rand(ro.shape[0])*(ro.shape[0]-1))):
        for i in np.random.randint(ro.shape[0],size=ro.shape[0]):
            i = int(i)
            boot += [rosort[i]]#*(1-frac) + rosort[i+1]*(frac)]

        boot = np.array(boot)
        boot = boot - np.median(ro,0)# + mro
#        ts = boot.mean(0)/(boot.std(0)/np.sqrt(boot.shape[0]-1))
        ts, ps = analPs(boot)
#        implot = ts*(ts>tthresh)
        implot = ps <= 0.05/tthresh
        toCalcEnt = implot
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
    pthreshes = []
    evidenceplots = []
    ents = []
    for i in range(len(rado)):

        bonfact =1.5
        plt.close()
        pvalcols = 10
        pvalrows =10
        pvaljoints = 10
        alpha = 0.05
        multcomp = 1
        like = 0.5
        while pvaljoints > alpha/multcomp or pvalcols > alpha/multcomp or pvalrows > alpha/multcomp:
            print(bonfact, first[i],pvalcols,pvalrows,pvaljoints)
            ts, ps = analPs(rado[i])
            sortT = np.sort(ps.ravel())
            print(np.unique(sortT))
            fdr = 0.05*np.arange(1,len(sortT)+1)/len(sortT)
            below = sortT < fdr
            if np.sum(below) == 0:
                pthresh = 0.050
            else:
                max_below = np.max(np.where(below)[0])
                pthresh = sortT[max_below]
            plt.plot(fdr, 'red', label="FDR")
            plt.plot(sortT,'blue', label="pVal Wilcoxon")
            plt.legend()
            plt.ylabel("pVal")
            plt.xlabel("Sorted Index")
            plt.ylim((0,0.5))
            plt.savefig("pValWilcFDR_{0}".format(first[i]))
            plt.close()
            toCalcEnt = ps<=0.05/bonfact#/len(ts.ravel())

            tscorenull = stats.t.ppf(1-0.05/bonfact,rado.shape[2]-1)
            tscore = stats.t.ppf(1-0.05/bonfact,rado.shape[2]-1)
            resss = np.array(Parallel(n_jobs=250)(delayed(bootstrap)(rado[i],mrado[i],bonfact) for z in range(1000))).T

            nullrow = resss[0]
            nullcol = resss[1]
            nulljoint = resss[2]
            implot = toCalcEnt

            Prop = (rado[i]!=0).sum(0)/rado[i].shape[0]
            templot = np.median(rado[i],0)*implot#[ymask][:,xmask]
            templot[templot==0] = np.nan

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
                bonfact += .5

        pthreshes += [toCalcEnt.sum()]
        ents += [pvaljoints]
        sigmap += [toCalcEnt.astype(int).astype(str)]
        nodeLabs = np.array(llabs + rlabs)
        edgelist = [(u,v,Prop[u,v]) for u in range(ts.shape[0]) for v in range(ts.shape[1]) if implot[u,v]]
        G = nx.DiGraph()
        weights = []
        listEntWilcCoups = []
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
            listEntWilcCoups += ["{0}-->{1}".format(fulllabs[u],fulllabs[v])]
            weights += [t]
        with open("EntWilcCouplesParcs_{0}_{1}_{2}.txt".format(first[i],"",'parcs'),'w') as ff:
            for w,lcoups in zip(weights,listEntWilcCoups):
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
        plt.imshow(templot, cmap='rainbow',vmin=np.min(weights))
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
        plt.savefig("EntWilcMatrix_Ts_{0}_{1}_{2}.png".format(first[i],"",'parcs'))
        plt.close() 

        fig, ax = plt.subplots(figsize=(12,12))
        #constrainplot[constrainplot==0] = np.nan
        im = ax.imshow(constrainplot, cmap='rainbow',vmin=np.min(weights))
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
        plt.savefig("SkinnyEntWilcMatrix_Ts_{0}_{1}_{2}.png".format(first[i],"",'parcs'))
        plt.close() 

    binMap = sigmap[0]
    print(len(sigmap))
    for i in range(1,len(sigmap)):
        binMap = np.core.defchararray.add(binMap,sigmap[i])
    colorStrs = ['{0}{1}'.format(i,j) for i in range(2) for j in range(2)]#np.sort(np.unique(binMap)).tolist()
    Plot = evidenceplots[0]
    Plot = rado[0] - rado[1]
    Plot[:,binMap=='00'] = 0
    Res,p = stats.ttest_1samp(Plot.reshape(Plot.shape[0],-1),0,axis=0,alternative='greater')
    fdr = 0.05*np.arange(1,len(p)+1)/len(p)
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

    print(fdr.max(),pthresh)
    Res[p > pthresh] = 0
    Plot = Res.reshape(rado[0].shape[-1],rado[0].shape[-1])
    Plot[binMap != '10']=0
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
    with open("SurfEntWilcCouplesParcs_All.txt",'w') as ff:
        for k,l in enumerate(llabs+rlabs):
            for kk,ll in enumerate(llabs+rlabs):
                if bin2Num[k,kk] > 0:
                    ff.write("{0}-->{1} {2}\n".format(l,ll,Events[int(bin2Num[k,kk]-1)]))

    cmap = (matplotlib.cm.get_cmap("rainbow",4-1))
    fig,ax = plt.subplots(figsize=(15,15))
    bin2Num[binMap=='00'] = np.nan
    ymask = ~np.all(binMap=='00',axis=1)
    xmask = ~np.all(binMap=='00',axis=0)
    constrainplot = bin2Num[ymask][:,xmask]
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
    plt.savefig("SurfEntWilcSepColorMat")
    plt.close()
    fig,ax = plt.subplots(figsize=(15,15))
    Plot[Plot!=Plot]=0
    for zzz,l in enumerate(llabs+rlabs):
        if not "occipital_" in l and not "oc_" in l and not "temp" in l and not "collat" in l:
                Plot[zzz] = np.nan
                Plot[:,zzz] = np.nan
    ymask = ~np.all(Plot!=Plot,axis=1)
    xmask = ~np.all(Plot!=Plot,axis=0)
    Plot[Plot==0] = np.nan
    constrainplot = Plot[ymask][:,xmask]
    im = ax.imshow(constrainplot,cmap="hot")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
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
    ax.set_title("Bootstrap Entropy Face: {2:0.4f} Bootstrap Entropy House: {3:0.4f} Ent Survivals Face = {0:d} Ent Survivals House {1:d}" .format(*pthreshes,*ents))
    plt.tight_layout()
    plt.savefig("EvWilcSurfBinEntSepColorMat_Face")
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
                        res = (x[j,i,:,None] - x[i,j]).ravel()
                        stat = res[0]
                        pval = (stat >res[1:]).mean()
                        pmat[i,j] = pval
                        best[i,j] = -(stat-res[1:].mean())
                best[best!=best] = 0
                best[best==np.inf] = 0
                rado[m][l][k] = best*((pmat<0.1))
                print((rado[m][l][k]!=0).sum(),pmat.min(), cl, cue, rad, d)
    return rado

temp = Parallel(n_jobs=30)(delayed(makePs)(d) for d in range(6))
rado = np.array(temp).squeeze()
print(rado.shape)
rado = rado.transpose(1,0,2,3)
print(rado.shape)
makePlots(rado)
