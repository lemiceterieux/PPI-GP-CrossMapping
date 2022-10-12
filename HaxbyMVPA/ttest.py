import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("agg")
from nibabel import freesurfer as fs
import numpy as np
import scipy.stats as stats
from nilearn import plotting, surface, datasets
from glob import glob
fsaverage = datasets.fetch_surf_fsaverage(mesh='fsaverage5')
rhsulc = fsaverage['sulc_right']
lhsulc = fsaverage['sulc_left']


rh = np.array([np.load(f) for f in glob("*rh12haxby.npy")])
lh = np.array([np.load(f) for f in glob("*lh12haxby.npy")]) 
lh[lh!=lh] = 0
rh[rh!=rh] = 0
#lh[lh==0] = -1
#rh[rh==0] = -1


#for i in range(len(lh)):
#    display = plotting.plot_surf_stat_map(fsaverage['infl_right'],  cmap='rainbow',
#            bg_on_data=False, stat_map=rh[i],view='medial', colorbar=True,
#            threshold=.6,bg_map=rhsulc,title='Subject ' + str(i) + ', T map, lateral, right hemisphere')
#    display.savefig(str(i) + "_accuracyRightlateral.png")
#    plt.close(display) 
#
#    display = plotting.plot_surf_stat_map(fsaverage['infl_right'],  cmap='rainbow',
#            bg_on_data=False, stat_map=rh[i],view='lateral', colorbar=True,
#            threshold=.6,bg_map=rhsulc,title='Subject ' + str(i) + ', T map, medial, right hemisphere')
#    display.savefig(str(i) + "_accuracyRightmedial.png")
#    plt.close(display) 
#
#    display = plotting.plot_surf_stat_map(fsaverage['infl_left'],  cmap='rainbow',
#            bg_on_data=False, stat_map=lh[i],view='medial', colorbar=True,
#            threshold=.6,bg_map=rhsulc,title='Subject ' + str(i) + ', T map, medial, left hemisphere')
#    display.savefig(str(i) + "_accuracyLeftmedial.png")
#    plt.close(display) 
#
#    display = plotting.plot_surf_stat_map(fsaverage['infl_left'],  cmap='rainbow',
#            bg_on_data=False, stat_map=lh[i],view='lateral', colorbar=True,
#            threshold=.6,bg_map=rhsulc,title='Subject ' + str(i) + ', T map, lateral, left hemisphere')
#    display.savefig(str(i) + "_accuracyLeftlateral.png")
#    plt.close(display) 

lh = lh.T
rh = rh.T
#print(np.min(lh))
#print(np.min(rh))
lh[lh!=lh] = .0
rh[rh!=rh] = .0
print(lh.shape)
#lh += np.random.rand(*lh.shape)*.4
#rh += np.random.rand(*rh.shape)*.4

results = np.array(stats.ttest_1samp(rh.T,0.5,alternative='greater',axis=0)).T
print(results.shape)

rts = results[...,0]
rps = results[...,1]
rps[rps!=rps] = 1
rps[rps == 0] = rps[rps!=0].min()
rts[rts!=rts] = 0
#rts[rts > 5.0] = rts[rts<= 5.0].max()
rps[rts<0] = 1
rts[rts<0] = 0
rps1mn = 1-rps
from nibabel import freesurfer as fs
fs.io.write_morph_data("rh.psom3", rps1mn)
fs.io.write_morph_data("rh.plog103", -np.log10(rps))
fs.io.write_morph_data("rh.tval3", rts)

#display = plotting.plot_surf_stat_map(fsaverage['infl_right'],  cmap='rainbow',
#        stat_map=rh[...,0],view='medial', colorbar=True,
#        threshold=.1,bg_map=rhsulc,title='Accuracy, medial, right hemisphere')
#display.savefig("1subj.png")
#print("Finish one subject")

results = np.array(stats.ttest_1samp(lh.T,0.5,alternative='greater',axis=0)).T
lts = results[...,0]
lps = results[...,1]
lps[lps!=lps] = 1
lts[lts!=lts] = 0
lps[lts<0] = 1
lps[lps == 0] = lps[lps!=0].min()
lts[lts<0] = 0
#lts[lts > 5.0] = lts[lts<= 5.0].max()
lps1mn = 1-lps
fs.io.write_morph_data("lh.psom3", lps1mn)
fs.io.write_morph_data("lh.plog103", -np.log10(lps))
fs.io.write_morph_data("lh.tval3", lts)
#lts = fs.io.read_morph_data("lh.thvalcor3.w")
#rts = fs.io.read_morph_data("rh.thvalcor3.w")
destrieux_atlas = datasets.fetch_atlas_surf_destrieux()
lhparcellation = destrieux_atlas['map_left']
rhparcellation = destrieux_atlas['map_right']
lps = 10**(-nib.load("lhcwsig.mgh").get_fdata()[...,0].squeeze())
lcind = nib.load("lhcwsig.mgh").get_fdata()[...,-1].squeeze()
lcwp = []
with open("lhtval.txt") as f:
 for i,l in enumerate(f):
     if i > 36:
       lcwp += [float(l.split()[7])]
lcwp = np.array(lcwp)
lcwp = (lcwp < .05).sum()
lmask = np.logical_or(lcind > lcwp, lcind == 0)
lps[lmask] = 1
print((lps==1).sum() == lmask.sum())
lps1mn = 1 - lps
rps = 10**(-nib.load("rhcwsig.mgh").get_fdata()[...,0].squeeze())
rcind = nib.load("rhcwsig.mgh").get_fdata()[...,-1].squeeze()
rcwp = []
with open("rhtval.txt") as f:
 for i,l in enumerate(f):
     if i > 36:
       rcwp += [float(l.split()[7])]
rcwp = np.array(rcwp)
rcwp = (rcwp < .05).sum()
print(rcwp,lcwp)
rmask = np.logical_or(rcind > rcwp, rcind == 0)
rps[rmask] = 1
print((rps==1).sum() == rmask.sum())
rps1mn = 1 - rps
print(rps.shape,rts.shape)
rts[rmask] = 0
lts[lmask] = 0
rts[rts!=rts] = 0
lts[lts!=lts] = 0
rtsm = min(rts[~rmask].min(),lts[~lmask].min())
ltsm = rtsm#lts[~lmask].min()
rts[rts>100] = 100
lts[lts>100] = 100
labs = destrieux_atlas.labels[1:]
for hemi in ['left', 'right']:
    vert = destrieux_atlas['map_%s' % hemi]
    for k, label in enumerate(np.array(labs[:])):
        if "Unknown" not in str(label):  # Omit the Unknown label.
            if not "occipital_" in str(label) and not "oc_" in str(label) and not "temp" in str(label) and not "collat" in str(label):            
                if hemi == "left":
                    lts[vert==k] = 0
                elif hemi == "right":
                    rts[vert==k] = 0
for hemi in ['left', 'right']:
    vert = destrieux_atlas['map_%s' % hemi]
    for k, label in enumerate(np.array(labs[:])):
        if hemi == "left":
            if lts[vert==k].sum()>0:
                print("Left",str(label,'utf-8'),lts[vert==k].max())
        if hemi == "right":
            if rts[vert==k].sum()>0:
                print("Right",str(label,'utf-8'),rts[vert==k].max())

print(ltsm, rtsm)

def check(lab, out):
    for l in destrieux_atlas['labels']:
        if lab in str(l):
            out += [l]
    return out

def makereg(atlas,labels,p):
    mask = ~p# != 1
    ind = []
    for a in atlas[mask]:
        if a not in ind:
            ind += [a]
    labs = [str(labels[a]) for a in ind]
    return ind, labs
deslabs = destrieux_atlas['labels']
lregions, llabs = makereg(lhparcellation, deslabs, lmask)
rregions, rlabs = makereg(rhparcellation, deslabs, rmask)

pth = .995

input("Waiting to Plot")

display = plotting.plot_surf_stat_map(fsaverage['infl_right'],  cmap='rainbow',
        bg_on_data=False, stat_map=rts,view='medial', colorbar=False,
        threshold=rtsm,vmax=15,bg_map=rhsulc,title='T map, lateral, right hemisphere')
#plotting.plot_surf_contours(fsaverage.infl_right, rhparcellation, labels=rlabs,
#                                    levels=rregions, figure=display,
#                                    legend=True, threshold=3, avg_method='lateraln')
display.tight_layout()
display.savefig("rhlateraltscoresdestrieux3noLabCorTransHaxby.png")
#display = plotting.plot_surf_stat_map(fsaverage['infl_right'],  cmap='rainbow', bg_on_data=False, stat_map=rps1mn,view='laterall', colorbar=False, threshold=pth,bg_map=rhsulc,title='1-P map, laterall, right hemisphere')
#plotting.plot_surf_contours(fsaverage.infl_right, rhparcellation, labels=rlabs,
#                                    levels=rregions, figure=display,
#                                    legend=True, threshold=3, avg_method='median')
#display.tight_layout()
#display.savefig("rhmedialps1mndestrieux3noLabCorTransHaxby.png")
display = plotting.plot_surf_stat_map(fsaverage['infl_right'],  cmap='rainbow',
        bg_on_data=False, stat_map=rts,view='lateral', colorbar=False,
        threshold=rtsm,vmax=15, bg_map=rhsulc,title='T map, medial, right hemisphere')
#plotting.plot_surf_contours(fsaverage.infl_right, rhparcellation, labels=rlabs,
#                                    levels=rregions, figure=display,
#                                    legend=True, threshold=3, avg_method='median')
display.tight_layout()
display.savefig("rhmedialtscoresdestrieux3noLabCorTransHaxby.png")
display = plotting.plot_surf_stat_map(fsaverage['infl_right'],  cmap='rainbow',
        bg_on_data=False, stat_map=rts,view='ventral', colorbar=True,
        threshold=rtsm,vmax=15, bg_map=rhsulc,title='T map, ventral, right hemisphere')
#plotting.plot_surf_contours(fsaverage.infl_right, rhparcellation, labels=rlabs,
#                                    levels=rregions, figure=display,
#                                    legend=True, threshold=3, avg_method='median')
display.tight_layout()
display.savefig("rhventraltscoresdestrieux3noLabCorTransHaxby.png")

#display = plotting.plot_surf_stat_map(fsaverage['infl_right'],  cmap='rainbow', bg_on_data=False, stat_map=rps1mn,view='lateral', colorbar=True, threshold=pth,bg_map=rhsulc,title='1-P map, lateral, right hemisphere')
#plotting.plot_surf_contours(fsaverage.infl_right, rhparcellation, labels=rlabs,
#                                    levels=rregions, figure=display,
#                                    legend=True, threshold=3, avg_method='median')
#display.tight_layout()
#display.savefig("rhlateralps1mndestrieux3noLabCorTransHaxby.png")


display = plotting.plot_surf_stat_map(fsaverage['infl_left'],  cmap='rainbow',
        bg_on_data=False, stat_map=lts,view='medial', colorbar=False,
        threshold=rtsm,vmax=15,bg_map=lhsulc,title='T map, medial, left hemisphere')
#plotting.plot_surf_contours(fsaverage.infl_left, lhparcellation, labels=llabs,
#                                    levels=lregions, figure=display,
#                                    legend=True, threshold=3, avg_method='median')
display.tight_layout()
display.savefig("lhmedialtscoresdestrieux3noLabCorTransHaxby.png")
#display = plotting.plot_surf_stat_map(fsaverage['infl_left'],  cmap='rainbow', bg_on_data=False, stat_map=lps1mn,view='medial', colorbar=False, threshold=pth,bg_map=lhsulc,title='1-P map, medial, left hemisphere')
#plotting.plot_surf_contours(fsaverage.infl_left, lhparcellation, labels=llabs,
#                                    levels=lregions, figure=display,
#                                    legend=True, threshold=3, avg_method='median')
#display.tight_layout()
#display.savefig("lhmedialps1mndestrieux3noLabCorTransHaxby.png")
display = plotting.plot_surf_stat_map(fsaverage['infl_left'],  cmap='rainbow',
        bg_on_data=False, stat_map=lts,view='lateral', colorbar=False,
        threshold=rtsm,vmax=15,bg_map=lhsulc,title='T map, lateral, left hemisphere')
#plotting.plot_surf_contours(fsaverage.infl_left, lhparcellation, labels=llabs,
#                                    levels=lregions, figure=display,
#                                    legend=True, threshold=3, avg_method='median')
display.tight_layout()
display.savefig("lhlateraltscoresdestrieux3noLabCorTransHaxby.png")
display = plotting.plot_surf_stat_map(fsaverage['infl_left'],  cmap='rainbow',
        bg_on_data=False, stat_map=lts,view='ventral', colorbar=False,
        threshold=rtsm,vmax=15,bg_map=lhsulc,title='T map, ventral, left hemisphere')
#plotting.plot_surf_contours(fsaverage.infl_left, lhparcellation, labels=llabs,
#                                    levels=lregions, figure=display,
#                                    legend=True, threshold=3, avg_method='median')
display.tight_layout()
display.savefig("lhventraltscoresdestrieux3noLabCorTransHaxby.png")

#display = plotting.plot_surf_stat_map(fsaverage['infl_left'],  cmap='rainbow', bg_on_data=False, stat_map=lps1mn,view='lateral', colorbar=True, threshold=pth,bg_map=lhsulc,title='1-P map, lateral, left hemisphere')
#plotting.plot_surf_contours(fsaverage.infl_left, lhparcellation, labels=llabs,
#                                    levels=lregions, figure=display,
#                                    legend=False, threshold=3, avg_method='median')
#display.tight_layout()
#display.savefig("lhlateralps1mndestrieux3noLabOpaq.png")

