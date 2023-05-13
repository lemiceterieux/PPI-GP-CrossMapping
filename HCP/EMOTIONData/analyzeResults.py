import numpy as np
import pandas as pd
import ptitprince as pt
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
import jax.numpy as jnp
import jax
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
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

def makePlots(rado,umrado):
    conTree = ["Shapes"]
    matplotlib.rcParams.update({'font.size': 14})
    first = ["face"]+ conTree
    fulllabs = ["EVCr","EVCl","OFAr","OFAl","FFAr","FFAl","STSr","STSl"]
    nROIS = len(fulllabs)
    chooser = 2
    Ps = np.arange(rado.shape[2]).tolist()
    combs = []
    for comb in itertools.combinations(Ps, rado.shape[2]-chooser):
        combs += [comb]
    np.array(combs)
    mrado = rado.mean(2)

    def combin(N,R):
        return math.factorial(N)/(math.factorial(N-R)*math.factorial(R))

    def binom_test(R,N):
        combies = []
        like = 0.5
#        def fori(r):#for r in range(1,R+1):
#            combies += [combin(N,r)*(like)**(r)*(1-like)**(N-r)]
        nn1 = lambda r: (r[0]*(r[1]-1),r[1]-1)
        check = lambda r: r[1] == 0
        factorial = lambda N: jax.lax.while_loop(check,nn1,(N,N))[0]
        combin = lambda n,r: factorial(n)/(factorial(n-r)*factorial(r))
        def fori(c,r): 
            ret = jax.lax.cond(r>R,lambda: combin(N,r)*(like)**(r)*(1-like)**(N-r), lambda: 0.)
            return c,ret
        return 1-jnp.sum(jax.lax.scan(fori,None,jnp.arange(N))[1])

    vf = np.vectorize(lambda a: stats.binomtest(a,rado.shape[1]+1,alternative='greater').pvalue)
    vff = np.vectorize(lambda b: vf(b))
#    vf = jax.vmap(lambda a: binom_test(a,rado.shape[1]))
#    vff = jax.jit(jax.vmap(lambda b: vf(b)))

    def sampWithRep(rado):
        sortr = np.sort(rado,0)
        boot = []
        for i in (np.random.randint(rado.shape[0],size=rado.shape[0])):
            i = int(i)
            boot = [sortr[i]]
        boot = np.array(boot) - np.mean(sortr,0)
        return (boot.mean(0))/(boot.std(0)/np.sqrt(boot.shape[0]))

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

        ts = vff(R)
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
        multcomp = 1
        like = 0.5
        xsumm = 100
        while pvaljoints > alpha/multcomp or pvalcols > alpha/multcomp or pvalrows > alpha/multcomp:
            print(bonfact, first[i],pvalcols,pvalrows,pvaljoints)
            N = rado[i].shape[0]
            R = (rado[i]>0).sum(0)
            Rperm = R
            tPerms = []
            ts = []

            ts = vff(R)#np.array(ts)
            print(ts.shape)
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
            resss = np.array(Parallel(n_jobs=250)(delayed(bootstrap)(rado[i],like,bonfact) for kk in range(250))).T
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
        print(rowEnt, colEnt,pvalrows,pvalcols,pvaljoints,first[i],"")
        ymask = ~np.all(implot==0,axis=1)
        xmask = ~np.all(implot==0,axis=0)
        evidenceplots += [np.copy(templot)]
        constrainplot = np.copy(templot[ymask][:,xmask])
        #templot[templot==0] = np.nan
        print(templot.shape,xmask.sum(),ymask.sum())

    binMap = sigmap[0]
    for i in range(1,len(sigmap)):
        binMap = np.core.defchararray.add(binMap,sigmap[i])
    trueBinMap = np.copy(binMap)
    for conK in range(1):
        if conK <1:
            contrast = conTree[conK]
        else:
            contrast = "All"
        for task in ["Face"]:#, "House", "All"]:
            binMap = np.copy(trueBinMap)
            if task == "Face":
                binMap[np.logical_and(binMap!='11',binMap!='10')] = '00'
                colorStrs = ['{0}{1}'.format(i,j) for i in range(2) for j in range(2)]#np.sort(np.unique(binMap)).tolist()
                if conK < 1:
                    Plot = umrado[0] - umrado[1+conK]
                    CleanPlot = umrado[0] - umrado[1+conK]
                    print(CleanPlot.shape,"LOL")
                    Plot1 = umrado[0]
                    Plot2 = umrado[1+conK]
                else:
                    Plot = rado[0] -      np.mean(rado[1:],0)
                    CleanPlot = umrado[0] - np.mean(umrado[1:],0)
                    Plot1 = umrado[0]
                    Plot2 = umrado[1:].mean(0)
                Plot[:,binMap=='00'] = 0 
#                Res,p = stats.wilcoxon(Plot.reshape(Plot.shape[0],-1),axis=0,alternative="greater")
                Res,p = stats.ttest_1samp(Plot.reshape(Plot.shape[0],-1),0,axis=0,alternative="greater")
#                Res = Plot.mean(0)/(Plot.std(0)/np.sqrt(Plot.shape[0]))
#                p = np.array(Parallel(n_jobs=250)(delayed(sampWithRep)(Plot) for kkk in range(1000)))
#                p = (Res < p).mean(0)
                print(p.shape)
                p[p!=p] = 1
                fdr = 0.1*np.arange(0,1,len(p))
                p[p!=p] = 1
                psort = np.sort(p)
                plt.close('all')
                plt.plot(psort[:20])
                plt.plot(fdr[:20])
                plt.savefig("Facepsrt")
                plt.close('all')
                below = psort <= fdr
                if np.sum(below) == 0:
                    pthresh = 0.0250#/100
                else:
                    max_below = np.max(np.where(below)[0])
                    print(psort[max_below])
                    pthresh = psort[max_below]
    
                print("FACE",np.where(below)[0],(p<pthresh).sum(),pthresh,p.min())
                chanstoPlot = [[i] for i in np.arange(nROIS)[1::2]]
                toPlot =  [np.mean(CleanPlot[:,i][:,:,j],(-1,-2)) for i in chanstoPlot for j in chanstoPlot]
                ttoPlot =  [np.mean(Plot    [:,i][:,:,j],(-1,-2)) for i in chanstoPlot for j in chanstoPlot]
                tttoPlot =  [np.mean(Res.reshape(nROIS,nROIS)[i][:,j],(-1,-2)) for i in chanstoPlot for j in chanstoPlot]
                toPlot1 = [np.mean(Plot1    [:,i][:,:,j],(-1,-2)) for i in chanstoPlot for j in chanstoPlot]
                toPlot2 = [np.mean(Plot2    [:,i][:,:,j],(-1,-2)) for i in chanstoPlot for j in chanstoPlot]

                print(len(toPlot),len(ttoPlot))
                LabPlot = fulllabs
                fig,ax = plt.subplots(1,1,sharey=True)
                a = ax
                width = 0.225
                for i in range(len(chanstoPlot)):#,a  in enumerate(ax.ravel()): 
                    box = a.bar(np.arange(len(chanstoPlot))+.6+i*width,np.array(toPlot[i*len(chanstoPlot):i*len(chanstoPlot)+len(chanstoPlot)]).mean(-1),width,capsize=10,yerr=abs(stats.t.ppf(0.05/len(chanstoPlot)**2,df=CleanPlot.shape[0]-1))*np.array(toPlot[i*len(chanstoPlot):i*len(chanstoPlot)+len(chanstoPlot)]).std(-1)/np.sqrt(CleanPlot.shape[0]),label=fulllabs[1::2][i])#, patch_artist=True)
#                    box = a.boxplot(toPlot[i*nROIS:i*nROIS+nROIS], patch_artist=True)
                    print(np.array(toPlot[i*len(chanstoPlot):i*len(chanstoPlot)+len(chanstoPlot)]).mean(-1),abs(stats.t.ppf(0.05/len(chanstoPlot)**2,df=CleanPlot.shape[0]-1))*np.array(toPlot[i*len(chanstoPlot):i*len(chanstoPlot)+len(chanstoPlot)]).std(-1)/np.sqrt(CleanPlot.shape[0]))
                    print(np.array(toPlot[i*len(chanstoPlot):i*len(chanstoPlot)+len(chanstoPlot)]).mean(-1),abs(stats.t.ppf(0.05,df=CleanPlot.shape[0]-1))*np.array(toPlot[i*len(chanstoPlot):i*len(chanstoPlot)+len(chanstoPlot)]).std(-1)/np.sqrt(CleanPlot.shape[0]))

                    a.set_xticks([n+1 for n in range(len(chanstoPlot))])
                    labB = []
                    a.set_xticklabels(fulllabs[1::2])
                plt.tight_layout()
                a.set_ylabel("Connectivity Strength")
                a.legend(bbox_to_anchor=(1.1-.2, 1.05),title=r"Explanatory Variable") 
                plt.savefig("LeftEvSurfBinEntSepColorMat_{0}_Box".format(contrast),bbox_inches='tight')
                plt.close('all')
                fig,ax = plt.subplots(1,4,sharey=True)
                for i,a  in enumerate(ax.ravel()): 
                    box = a.bar(np.arange(len(chanstoPlot))+1,np.array(toPlot1[i*len(chanstoPlot):i*len(chanstoPlot)+len(chanstoPlot)]).mean(-1),capsize=10,yerr=abs(stats.t.ppf(0.05/len(chanstoPlot)**2,df=CleanPlot.shape[0]-1))*np.array(toPlot1[i*len(chanstoPlot):i*len(chanstoPlot)+len(chanstoPlot)]).std(-1)/np.sqrt(900))#, patch_artist=True)
#                    box = a.boxplot(toPlot1[i*nROIS:i*nROIS+nROIS], patch_artist=True)
#                    a.set(xticklabels=[])
#                    a.tick_params(bottom=False)
                    a.set_xticks([n+1 for n in range(len(chanstoPlot))])
                    labB = []
                    for T,TT in zip(toPlot1[i*len(chanstoPlot):i*len(chanstoPlot)+len(chanstoPlot)],tttoPlot[i*len(chanstoPlot):i*len(chanstoPlot)+len(chanstoPlot)]):
                        T[T==0] = -1
#                        if stats.wilcoxon(T,alternative='greater')[1] < 0.05:
                        if stats.ttest_1samp(T,0,alternative='greater')[0] >abs(stats.t.ppf(0.05,df=Plot1.shape[0]-1)):
                            labB += ["*"]
                        else:
                            labB += [""]
                    a.set_xticklabels(labB)       

                    colors = ['cyan', 'lightblue', 'lightgreen', 'tan', 'pink', 'orange']
#                    for patch, color in zip(box['boxes'], colors):
#                        patch.set_facecolor(color)
        
                plt.tight_layout()
                plt.savefig("LeftFaceSurfBinEntSepColorMat_{0}_Box".format(contrast),bbox_inches='tight')
                plt.close('all')
                fig,ax = plt.subplots(1,4,sharey=True)
                for i,a  in enumerate(ax.ravel()): 
                    box = a.bar(np.arange(len(chanstoPlot))+1,np.array(toPlot2[i*len(chanstoPlot):i*len(chanstoPlot)+len(chanstoPlot)]).mean(-1),capsize=10,yerr=abs(stats.t.ppf(0.05/len(chanstoPlot)**2,df=CleanPlot.shape[0]-1))*np.array(toPlot2[i*len(chanstoPlot):i*len(chanstoPlot)+len(chanstoPlot)]).std(-1)/np.sqrt(900))#, patch_artist=True)
#                    box = a.boxplot(toPlot2[i*len(chanstoPlot):i*len(chanstoPlot)+len(chanstoPlot)], patch_artist=True)
#                    a.set(xticklabels=[])
#                    a.tick_params(bottom=False)
                    a.set_xticks([n+1 for n in range(len(chanstoPlot))])
                    labB = []
                    for T in toPlot2[i*len(chanstoPlot):i*len(chanstoPlot)+len(chanstoPlot)]:
                        if stats.ttest_1samp(T,0,alternative='greater')[1] < 0.05:
                            labB += ["*"]
                        else:
                            labB += [""]
                    a.set_xticklabels(labB)       

                    colors = ['cyan', 'lightblue', 'lightgreen', 'tan', 'pink', 'orange']
        
                plt.tight_layout()
                plt.savefig("LeftObjSurfBinEntSepColorMat_{0}_Box".format(contrast),bbox_inches='tight')

                chanstoPlot = [[i] for i in np.arange(nROIS)[::2]]
                toPlot =  [np.mean(CleanPlot[:,i][:,:,j],(-1,-2)) for i in chanstoPlot for j in chanstoPlot]
                ttoPlot =  [np.mean(Plot    [:,i][:,:,j],(-1,-2)) for i in chanstoPlot for j in chanstoPlot]
                tttoPlot =  [np.mean(Res.reshape(nROIS,nROIS)[i][:,j],(-1,-2)) for i in chanstoPlot for j in chanstoPlot]
                toPlot1 = [np.mean(Plot1    [:,i][:,:,j],(-1,-2)) for i in chanstoPlot for j in chanstoPlot]
                toPlot2 = [np.mean(Plot2    [:,i][:,:,j],(-1,-2)) for i in chanstoPlot for j in chanstoPlot]

                print(len(toPlot),len(ttoPlot))
                LabPlot = fulllabs
                fig,ax = plt.subplots(1,1,sharey=True)
                width = .225
                a = ax
                for i in range(len(chanstoPlot)):#,a  in enumerate(ax.ravel()): 
                    box = a.bar(np.arange(len(chanstoPlot))+.6+i*width,np.array(toPlot[i*len(chanstoPlot):i*len(chanstoPlot)+len(chanstoPlot)]).mean(-1),width,capsize=10,yerr=abs(stats.t.ppf(0.05/len(chanstoPlot)**2,df=CleanPlot.shape[0]-1))*np.array(toPlot[i*len(chanstoPlot):i*len(chanstoPlot)+len(chanstoPlot)]).std(-1)/np.sqrt(CleanPlot.shape[0]),label=fulllabs[::2][i])#, patch_artist=True)
#                    box = a.boxplot(toPlot[i*nROIS:i*nROIS+nROIS], patch_artist=True)
                    print(np.array(toPlot[i*len(chanstoPlot):i*len(chanstoPlot)+len(chanstoPlot)]).mean(-1),abs(stats.t.ppf(0.05/len(chanstoPlot)**2,df=CleanPlot.shape[0]-1))*np.array(toPlot[i*len(chanstoPlot):i*len(chanstoPlot)+len(chanstoPlot)]).std(-1)/np.sqrt(CleanPlot.shape[0]))
                    print(np.array(toPlot[i*len(chanstoPlot):i*len(chanstoPlot)+len(chanstoPlot)]).mean(-1),abs(stats.t.ppf(0.05,df=CleanPlot.shape[0]-1))*np.array(toPlot[i*len(chanstoPlot):i*len(chanstoPlot)+len(chanstoPlot)]).std(-1)/np.sqrt(CleanPlot.shape[0]))

                    a.set_xticks([n+1 for n in range(len(chanstoPlot))])
                    labB = []

#                    print(np.array(toPlot[i*nROIS:i*nROIS+nROIS]).mean(1), 4.nROIS18*np.array(toPlot[i*nROIS:i*nROIS+nROIS]).std(1)/np.sqrt(900-1))
                    for T in toPlot[i*len(chanstoPlot):i*len(chanstoPlot)+len(chanstoPlot)]:
                        T[T==0] = -1
#                        if stats.wilcoxon(T,alternative='greater')[1] < 0.05:
                        if stats.ttest_1samp(T,0,alternative='greater')[0] >abs(stats.t.ppf(0.05,df=Plot1.shape[0]-1)):
                            labB += ["*"]
                        else:
                            labB += [""]
                    a.set_xticklabels(fulllabs[::2])       
                    colors = ['cyan', 'lightblue', 'lightgreen', 'tan', 'pink', 'orange']
#                    for patch, color in zip(box['boxes'], colors):
#                        patch.set_facecolor(color)
                plt.tight_layout()
                a.set_ylabel("Connectivity Strength")
                a.legend(bbox_to_anchor=(1.1-.2, 1.05),title="Explanatory Variable") 
                plt.savefig("RightEvSurfBinEntSepColorMat_{0}_Box".format(contrast),bbox_inches='tight')
                plt.close('all')
                fig,ax = plt.subplots(1,4,sharey=True)
                for i,a  in enumerate(ax.ravel()): 
                    box = a.bar(np.arange(len(chanstoPlot))+1,np.array(toPlot1[i*len(chanstoPlot):i*len(chanstoPlot)+len(chanstoPlot)]).mean(-1),capsize=10,yerr=abs(stats.t.ppf(0.05/len(chanstoPlot)**2,df=CleanPlot.shape[0]-1))*np.array(toPlot1[i*len(chanstoPlot):i*len(chanstoPlot)+len(chanstoPlot)]).std(-1)/np.sqrt(CleanPlot.shape[0]))#, patch_artist=True)
#                    box = a.boxplot(toPlot1[i*nROIS:i*nROIS+nROIS], patch_artist=True)
#                    a.set(xticklabels=[])
#                    a.tick_params(bottom=False)
                    a.set_xticks([n+1 for n in range(len(chanstoPlot))])
                    labB = []
                    for T,TT in zip(toPlot1[i*len(chanstoPlot):i*len(chanstoPlot)+len(chanstoPlot)],tttoPlot[i*len(chanstoPlot):i*len(chanstoPlot)+len(chanstoPlot)]):
                        T[T==0] = -1
#                        if stats.wilcoxon(T,alternative='greater')[1] < 0.05:
                        if stats.ttest_1samp(T,0,alternative='greater')[0] >abs(stats.t.ppf(0.05,df=Plot1.shape[0]-1)):
                            labB += ["*"]
                        else:
                            labB += [""]
                    a.set_xticklabels(labB)       

                    colors = ['cyan', 'lightblue', 'lightgreen', 'tan', 'pink', 'orange']
#                    for patch, color in zip(box['boxes'], colors):
#                        patch.set_facecolor(color)
        
                plt.tight_layout()
                plt.savefig("RightFaceSurfBinEntSepColorMat_{0}_Box".format(contrast),bbox_inches='tight')
                plt.close('all')
                fig,ax = plt.subplots(1,4,sharey=True)
                for i,a  in enumerate(ax.ravel()): 
                    box = a.bar(np.arange(len(chanstoPlot))+1,np.array(toPlot2[i*len(chanstoPlot):i*len(chanstoPlot)+len(chanstoPlot)]).mean(-1),capsize=10,yerr=abs(stats.t.ppf(0.05/len(chanstoPlot)**2,df=CleanPlot.shape[0]-1))*np.array(toPlot2[i*len(chanstoPlot):i*len(chanstoPlot)+len(chanstoPlot)]).std(-1)/np.sqrt(CleanPlot.shape[0]))#, patch_artist=True)
#                    box = a.boxplot(toPlot2[i*len(chanstoPlot):i*len(chanstoPlot)+len(chanstoPlot)], patch_artist=True)
#                    a.set(xticklabels=[])
#                    a.tick_params(bottom=False)
                    a.set_xticks([n+1 for n in range(len(chanstoPlot))])
                    labB = []
                    for T in toPlot2[i*len(chanstoPlot):i*len(chanstoPlot)+len(chanstoPlot)]:
                        if stats.ttest_1samp(T,0,alternative='greater')[1] < 0.05:
                            labB += ["*"]
                        else:
                            labB += [""]
                    a.set_xticklabels(labB)       

                    colors = ['cyan', 'lightblue', 'lightgreen', 'tan', 'pink', 'orange']
        
                plt.tight_layout()
                plt.savefig("RightObjSurfBinEntSepColorMat_{0}_Box".format(contrast),bbox_inches='tight')
    
#                if (p<pthresh).sum() == 0:
#                    continue
                Res[p > pthresh] = 0
                p[p>pthresh] = 1
                Plot = Res.reshape(rado[0].shape[-1],rado[0].shape[-1])
                pscore = p.reshape(rado[0].shape[-1],rado[0].shape[-1])
    
            Events = []
    
            fig,ax = plt.subplots()#figsize=(12,12))
    
            Plot[Plot==0] = np.nan
            pscore[pscore==1] = np.nan
            print("PSCORE", np.nanmax(pscore))
            Plot[Plot!=Plot]=0
            ymask = ~np.all(Plot!=Plot,axis=1)
            xmask = ~np.all(Plot!=Plot,axis=0)
            print("N Parcels {0:d}".format(np.sum(ymask)))
            Plot[Plot==0] = np.nan
            constrainplot = Plot[ymask][:,xmask]
            constrainplot = CleanPlot.mean(0)/(CleanPlot.std(0)/np.sqrt(CleanPlot.shape[0]))
   
            if "All" in task:
                vmin = -np.nanmax(abs(Plot)+1)
                cmap = "RdBu"
            else:
                vmin = 1.63#np.nanmin(abs(Plot))
                vmax = 10#np.nanmax(constrainplot)#22
#                vmin = 2#np.nanmin(abs(Plot))
#                vmax = 8#np.nanmax(abs(Plot))#8#22
                cmap = "Wistia"
#            constrainplot[constrainplot < abs(stats.t.ppf(0.05/(len(chanstoPlot))**2,df=CleanPlot.shape[0]-1))] = np.nan
            im = ax.imshow(constrainplot[::2,::2],cmap=cmap,vmin=vmin,vmax=vmax)
            print(constrainplot.shape,ymask.sum(),xmask.sum(),(~(Plot!=Plot)).sum(),task)
#            if (~(Plot!=Plot)).sum() == 0:
#                continue
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = plt.colorbar(im, cax=cax,ticks=[vmin,  vmax])
            cbar.ax.set_ylabel("T Score")
            xlabs = []
            ylabs = []
            ax.set_xticks([n for n in range(constrainplot.shape[1]//2)])
            ax.set_yticks([n for n in range(constrainplot.shape[0]//2)])
            for zzz,l in enumerate(fulllabs[::2]):
                if ymask[zzz]:
                    ylabs += [l+" -->"]
                if xmask[zzz]:
                    xlabs += [l+" <--"]
            ax.set_xticklabels(xlabs)
            ax.set_yticklabels(ylabs)
            ax.tick_params(axis='x', rotation=90)
            ax.grid()
            plt.tight_layout()
            plt.savefig("RightEvSurfBinEntSepColorMat_{0}".format(contrast))
            plt.close()
            fig,ax = plt.subplots()#figsize=(12,12))
            im = ax.imshow(constrainplot[1::2,1::2],cmap=cmap,vmin=vmin,vmax=vmax)
            print(constrainplot.shape,ymask.sum(),xmask.sum(),(~(Plot!=Plot)).sum(),task)
#            if (~(Plot!=Plot)).sum() == 0:
#                continue
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = plt.colorbar(im, cax=cax,ticks=[vmin,  vmax])
            cbar.ax.set_ylabel("T Score")
            xlabs = []
            ylabs = []
            ax.set_xticks([n for n in range(constrainplot.shape[1]//2)])
            ax.set_yticks([n for n in range(constrainplot.shape[0]//2)])
            for zzz,l in enumerate(fulllabs[1::2]):
                if ymask[zzz]:
                    ylabs += [l+" -->"]
                if xmask[zzz]:
                    xlabs += [l+" <--"]
            ax.set_xticklabels(xlabs)
            ax.set_yticklabels(ylabs)
            ax.tick_params(axis='x', rotation=90)
            ax.grid()
            plt.tight_layout()
            plt.savefig("LeftEvSurfBinEntSepColorMat_{0}".format(contrast))
            plt.close()
            fig,ax = plt.subplots()#figsize=(12,12))
            constrainplot = CleanPlot.mean(0)/(CleanPlot.std(0)/np.sqrt(CleanPlot.shape[0]))
#            constrainplot[constrainplot < abs(stats.t.ppf(0.05/(2*len(chanstoPlot))**2,df=CleanPlot.shape[0]-1))] = np.nan
            im = ax.imshow(constrainplot[:,:],cmap=cmap,vmin=vmin,vmax=vmax)
            print(constrainplot.shape,ymask.sum(),xmask.sum(),(~(Plot!=Plot)).sum(),task)
#            if (~(Plot!=Plot)).sum() == 0:
#                continue
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = plt.colorbar(im, cax=cax,ticks=[vmin,  vmax])
            cbar.ax.set_ylabel("T Score")
            xlabs = []
            ylabs = []
            ax.set_xticks([n for n in range(constrainplot.shape[1])])
            ax.set_yticks([n for n in range(constrainplot.shape[0])])
            for zzz,l in enumerate(fulllabs[:]):
                if ymask[zzz]:
                    ylabs += [l+" -->"]
                if xmask[zzz]:
                    xlabs += [l+" <--"]
            ax.set_xticklabels(xlabs)
            ax.set_yticklabels(ylabs)
            ax.tick_params(axis='x', rotation=90)
            ax.grid()
            plt.tight_layout()
            plt.savefig("EvSurfBinEntSepColorMat_{0}".format(contrast))
            plt.close()


def makePs(d):#for d in dirs[:12]:
    rado = [[[[]] for l in range(2)] for m in range(1)]
    umrado = [[[[]] for l in range(2)] for m in range(1)]
    contrast = ["Faces","Shapes"]
    for k, rad in enumerate(["parcs"]):
        for l, cue in enumerate(contrast):
            for m,cl in enumerate([""]):
                x = 0
                nums = 0
                x = [np.load(s) for s in sorted(glob("0_{0:d}_*{1}*LRcau*.npy".format(d,cue)))]
                x = x[:]
                perms = np.concatenate(np.array(x)[:,:,:,1:],-1)
                xl = np.nanmedian(x[:],0)#[:,:,0]

                x = [np.load(s) for s in sorted(glob("0_{0:d}_*{1}*RLcau*.npy".format(d,cue)))]
                x = x[:]
                perms = np.concatenate(np.array(x)[:,:,:,1:],-1)
                x = (xl + np.nanmedian(x[:],0))/2#[:,:,0]

                pmat = np.zeros((x.shape[0],x.shape[1]))
                best = np.zeros((x.shape[0],x.shape[1]))
                x[x!=x] = 0
                for i in range(x.shape[0]):
                    for j in range(x.shape[1]):
                        res = ((x[j,i,:,None] - x[i,j]))
                        stat = res[0,0]
                        perm = res[1:,1:].ravel()
                        pval =  (stat > perm).mean()

                        best[i,j] = -(stat - np.median(perm.ravel()[:]))
                        pmat[i,j] = pval
                best[best!=best] = 0
                best[best==np.inf] = 0
                rado[m][l][k] = best*(pmat<0.5)

                umrado[m][l][k] = best
    return rado,umrado

# Filter samples based on working memory task
X = np.load("../WMGLM/WhichFaceSens3.npy")
X = X[...,1]
X[X!=X] = 0
X = (X[...,::2]>0.8250).sum(-1).mean(-1)
XX = (X >= 0.95)
Z = np.array([s.split("/")[-2] for s in glob("../WM/*/") if not "cache" in s])[XX]
ZZ = np.array([s.split("/")[-2] for s in glob("../EMOTION/*/") if not "cache" in s])
mask = np.zeros(len(ZZ)).astype(bool)
for i in range(len(ZZ)):
    for j in range(len(Z)):
        if ZZ[i] == Z[j]:
            mask[i] = True
            break
dirs = np.arange(len(glob("0_*parcs*.npy"))//4)[mask]
temp,um = zip(*Parallel(n_jobs=200)(delayed(makePs)(d) for d in dirs))
rado = np.array(temp).squeeze()
umrado = np.array(um).squeeze()
print(rado.shape)
rado = rado.transpose(1,0,2,3)
umrado = umrado.transpose(1,0,2,3)
print(rado.shape)
makePlots(rado,umrado)
