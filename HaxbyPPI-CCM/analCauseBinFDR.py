import numpy as np
from glob import glob
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
    res = np.array(stats.ttest_1samp(r,mu)).T
    rts = res[...,0].reshape(length,length)
    rps = res[...,1].reshape(length,length)
    rts[rts!=rts]=0
    rts[rts<0]=0
    rps[rps!=rps]=1
    return rts, rps

def makePlots(rado):
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
                for r in range(R[p,pp]+1):
                    combies += [combin(N,r)*(like)**(r)*(1-like)**(N-r)]
                tssub += [1-np.sum(combies)]
            ts += [tssub]
        ts = np.array(ts)
        toCalcEnt = ts<=bonfact#0.05/bonfact

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

    bonfact = 1 
    sigmap = []
    pthreshes = []
    ents = []
    for i in range(len(rado)):
        plt.close()
        pvalcols = 10
        pvalrows =10
        pvaljoints = 10
        alpha = 0.05
        multcomp = 1
        like = 0.5
        for j in range(1):#while pvaljoints > alpha/multcomp or pvalcols > alpha/multcomp or pvalrows > alpha/multcomp:
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
                    for r in range(R[p,pp]+1):
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
                pthresh = 0.00
            else:
                max_below = np.max(np.where(below)[0])
                pthresh = sortT[max_below]
            print(np.min(R),np.max(R), ts.shape, pthreshes)

            toCalcEnt = ts<=pthresh#0.05/bonfact#/len(ts.ravel())
            resss = np.array(Parallel(n_jobs=250)(delayed(bootstrap)(rado[i],like,pthresh) for kk in range(1000))).T
            nullrow = resss[0]
            nullcol = resss[1]
            nulljoint = resss[2]
            implot = toCalcEnt

            Prop = R/N
            templot = Prop*implot#[ymask][:,xmask]
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
                bonfact *= 2

        pthreshes += [toCalcEnt.sum()]
        ents += [pvaljoints]
        sigmap += [toCalcEnt.astype(int).astype(str)]
        nodeLabs = np.array(llabs + rlabs)
        edgelist = [(u,v,Prop[u,v]) for u in range(ts.shape[0]) for v in range(ts.shape[1]) if implot[u,v]]
        G = nx.DiGraph()
        weights = []
        listBinFDRCoups = []
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
            listBinFDRCoups += ["{0}-->{1}".format(fulllabs[u],fulllabs[v])]
            weights += [t]
        with open("BinFDRCouplesParcs_{0}_{1}_{2}.txt".format(first[i],"",'parcs'),'w') as ff:
            for w,lcoups in zip(weights,listBinFDRCoups):
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
        plt.savefig("BinFDRMatrix_Ts_{0}_{1}_{2}.png".format(first[i],"",'parcs'))
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
        plt.savefig("SkinnyBinFDRMatrix_Ts_{0}_{1}_{2}.png".format(first[i],"",'parcs'))
        plt.close() 

    binMap = sigmap[0]
    print(len(sigmap))
    for i in range(1,len(sigmap)):
        binMap = np.core.defchararray.add(binMap,sigmap[i])
    colorStrs = ['{0}{1}'.format(i,j) for i in range(2) for j in range(2)]#np.sort(np.unique(binMap)).tolist()
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
    with open("SurfBinFDRCouplesParcs_All.txt",'w') as ff:
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
    ax.set_title("Bootstrap Entropy Face: {2:0.4f} Bootstrap Entropy House: {3:0.4f} FDR Survivals Face = {0:d} FDR Survivals House {1:d}"
            .format(*pthreshes,*ents))
    plt.tight_layout()
    plt.savefig("SurfBinFDRSepColorMat")

def makePs(d):#for d in dirs[:12]:
    rado = [[[[]] for l in range(2)] for m in range(1)]
    for k, rad in enumerate(["parcs"]):
        for l, cue in enumerate(["faces","houses"]):
            for m,cl in enumerate([""]):
                x = [np.load(s) for s in glob("*_{0:d}_*{1}*.npy".format(d,cue))]#"taste_9_claimcause.npy")#,allow_pickle=True).tolist()
                x = np.array(x)
                print(x.shape,l,d)
#                x = np.nanmean(x,0)
#                if d < 1:
#                    x = np.median([x] + [np.load(str(ppp)+"_{3}__{1}_{0}cause.npy".format(cue,rad,cl,d)) for ppp in range(1,10)],0)
#                    for ppp in range(3):
#                        x += np.load(str(ppp)+"_{3}__{1}_{0}cause.npy".format(cue,rad,cl,d))#"taste_9_claimcause.npy")#,allow_pickle=True).tolist()
                pmat = np.zeros((x.shape[1],x.shape[2]))
                best = np.zeros((x.shape[1],x.shape[2]))
                x = np.median(x,0)
                for i in range(x.shape[0]):
                    for j in range(x.shape[1]):
                        res = (x[j,i,:] - x[i,j]).ravel()
                        stat = res[0]
                        pval = (stat >res[1:]).mean()
#                        rasort = np.argsort(res)
#                        cum = np.arange(len(res))/len(res)
#                        pval = cum[rasort==0]
                        pmat[i,j] = 1-pval
                        best[i,j] = stat
                rado[m][l][k] += [best*((pmat<0.1))] 
                print(pmat.min(), cl, cue, rad, d)
    return rado

temp = Parallel(n_jobs=30)(delayed(makePs)(d) for d in range(6))
rado = np.array(temp).squeeze()
print(rado.shape)
rado = rado.transpose(1,0,2,3)
print(rado.shape)
makePlots(rado)
